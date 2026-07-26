"""Cross-size QK-LSTM: one model that transfers across problem sizes.

The original :class:`~qklstm.optimizer.QKLSTMQAOAOptimizer` ties its input
dimension to ``n_qubits``, so a model trained on 6-node graphs cannot even be
*evaluated* on a 10-node graph -- the tensor shapes disagree.  Every size needs
its own model, which undercuts the "train once, predict instantly" premise.

:class:`CrossSizeQKLSTMOptimizer` removes that limit by requiring a
**size-agnostic** encoder (:class:`~qklstm.encoding.SpectralEncoder`).  Because
the feature vector has the same length for every graph, a single model can be

* trained on a *range* of small sizes (cheap to label), and
* asked for QAOA parameters on **larger, unseen** sizes (expensive to label).

Quality is reported as the **approximation ratio** ``<C> / C_max`` using the
exact optimum from :func:`~qklstm.qaoa.brute_force_maxcut`, so results are
comparable across sizes (raw cut values are not -- bigger graphs cut more edges).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .encoding import ProblemEncoder, SpectralEncoder
from .qaoa import (
    BETA_PERIOD,
    GAMMA_PERIOD,
    QAOA,
    Edge,
    brute_force_maxcut,
    canonicalize_qaoa_params,
)
from .qklstm_model import QuantumKernelLSTM


class PeriodicAngleLoss(nn.Module):
    """Loss that respects the periodicity of the QAOA angles.

    Plain MSE is the wrong metric for angles: ``gamma = 0.01`` and
    ``gamma = pi - 0.01`` describe almost the same circuit but sit at opposite
    ends of the interval, so MSE reports a large error for a near-perfect
    prediction.  This loss instead uses

    ``1 - cos(2*pi * (pred - target) / period)``

    which is zero exactly when the prediction matches the target up to a full
    period, and is smooth everywhere.  ``gamma`` uses period ``pi`` and ``beta``
    period ``pi/2``, matching the symmetries of the MaxCut objective.
    """

    def __init__(self, p: int):
        super().__init__()
        self.p = p

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = prediction - target
        gamma_delta = delta[:, : self.p] * (2 * np.pi / GAMMA_PERIOD)
        beta_delta = delta[:, self.p :] * (2 * np.pi / BETA_PERIOD)
        return (
            (1 - torch.cos(gamma_delta)).mean()
            + (1 - torch.cos(beta_delta)).mean()
        )


class CrossSizeQKLSTMOptimizer:
    """Train one QK-LSTM across several graph sizes and transfer to new ones.

    Parameters
    ----------
    train_sizes:
        Graph sizes (node counts) sampled during training.
    qaoa_depth:
        QAOA depth ``p``; the model predicts ``2p`` parameters.
    encoder:
        Must be size-agnostic.  Defaults to :class:`SpectralEncoder`.
    kernel_qubits:
        Qubits used by the quantum kernel (independent of the problem size).
    """

    def __init__(
        self,
        train_sizes: Sequence[int] = (4, 5, 6),
        qaoa_depth: int = 1,
        lstm_hidden: int = 64,
        use_quantum: bool = True,
        encoder: Optional[ProblemEncoder] = None,
        kernel_qubits: int = 4,
        lr: float = 1e-3,
        seed: Optional[int] = 42,
        device: Optional[str] = None,
    ):
        encoder = encoder if encoder is not None else SpectralEncoder()
        if not encoder.is_size_agnostic:
            raise ValueError(
                "CrossSizeQKLSTMOptimizer requires a size-agnostic encoder "
                f"(e.g. SpectralEncoder); got {type(encoder).__name__}, whose "
                "dimension depends on the graph size."
            )

        self.encoder = encoder
        self.train_sizes = tuple(sorted(set(train_sizes)))
        if min(self.train_sizes) < 2:
            raise ValueError("train_sizes must all be >= 2")

        self.qaoa_depth = qaoa_depth
        self.use_quantum = use_quantum
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # One QAOA simulator per size, created on demand and reused.
        self._qaoa_cache: Dict[int, QAOA] = {}

        self.model = QuantumKernelLSTM(
            input_dim=self.encoder.dim,
            hidden_dim=lstm_hidden,
            output_dim=2 * qaoa_depth,
            n_qubits=kernel_qubits,
            use_quantum=use_quantum,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.transfer_results: Optional[Dict[int, Dict[str, float]]] = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def qaoa_for(self, n_nodes: int) -> QAOA:
        """Return (and cache) a :class:`QAOA` simulator for ``n_nodes`` qubits."""
        if n_nodes not in self._qaoa_cache:
            self._qaoa_cache[n_nodes] = QAOA(n_qubits=n_nodes, p=self.qaoa_depth)
        return self._qaoa_cache[n_nodes]

    def random_graph(self, n_nodes: int) -> List[Edge]:
        """Sample a random graph on ``n_nodes`` vertices.

        Edge count is drawn from ``[n, 2n]`` and capped at ``C(n, 2)`` so the
        sampling loop always terminates for small graphs.
        """
        max_edges = n_nodes * (n_nodes - 1) // 2
        low = min(n_nodes, max_edges)
        high = min(2 * n_nodes, max_edges)
        n_edges = np.random.randint(low, high + 1) if high >= low else max_edges

        edges: List[Edge] = []
        while len(edges) < n_edges:
            i = np.random.randint(0, n_nodes)
            j = np.random.randint(0, n_nodes)
            if i != j:
                edge = (min(i, j), max(i, j))
                if edge not in edges:
                    edges.append(edge)
        return edges

    def _trajectory(self, problem_graph: Sequence[Edge], n_nodes: int,
                    n_steps: int) -> np.ndarray:
        """Repeat the invariant encoding across the LSTM's time axis."""
        encoding = self.encoder.encode(problem_graph, n_nodes)
        return np.stack([encoding] * n_steps)

    # ------------------------------------------------------------------ #
    # Data generation
    # ------------------------------------------------------------------ #
    def generate_training_data(
        self,
        n_problems_per_size: int = 20,
        n_steps: int = 6,
        n_restarts: int = 4,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Label random graphs of every training size with optimal QAOA angles.

        Labels are produced with multiple restarts (so they are near-global
        rather than whichever local optimum COBYLA happened to find) and then
        pushed through :func:`~qklstm.qaoa.canonicalize_qaoa_params`, which folds
        the symmetry orbit onto a single representative.  Without both steps the
        targets are multimodal and the regression cannot converge.
        """
        X, y = [], []
        if verbose:
            print(f"Generating {n_problems_per_size} problems for each of "
                  f"sizes {list(self.train_sizes)} "
                  f"({n_restarts} restarts, canonicalised labels)...")

        for n_nodes in self.train_sizes:
            qaoa = self.qaoa_for(n_nodes)
            for _ in range(n_problems_per_size):
                edges = self.random_graph(n_nodes)
                optimal_params, _ = qaoa.optimize(edges, n_restarts=n_restarts)
                X.append(self._trajectory(edges, n_nodes, n_steps))
                y.append(canonicalize_qaoa_params(optimal_params, self.qaoa_depth))
            if verbose:
                print(f"  size {n_nodes}: {n_problems_per_size} problems labelled")

        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32),
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(
        self,
        n_problems_per_size: int = 20,
        n_epochs: int = 30,
        batch_size: int = 8,
        val_split: float = 0.2,
        n_reference: int = 12,
        n_restarts: int = 4,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the shared model over all ``train_sizes``."""
        if verbose:
            print("\n" + "=" * 64)
            print("TRAINING CROSS-SIZE QK-LSTM")
            print("=" * 64)

        X, y = self.generate_training_data(
            n_problems_per_size=n_problems_per_size,
            n_restarts=n_restarts,
            verbose=verbose,
        )
        # Shuffle so the train/val split mixes sizes instead of splitting by size.
        perm = torch.randperm(len(X))
        X, y = X[perm].to(self.device), y[perm].to(self.device)

        n_val = max(1, int(len(X) * val_split))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        if self.use_quantum:
            self.model.set_reference_data(X_train[: min(n_reference, len(X_train))])

        # Angles are periodic, so a periodic loss is the correct objective.
        criterion = PeriodicAngleLoss(self.qaoa_depth)
        if verbose:
            print(f"\nTraining samples: {len(X_train)}  |  Validation: {len(X_val)}")
            print(f"Encoding dim: {self.encoder.dim} (independent of graph size)")
            print(f"\nTraining for {n_epochs} epochs...")

        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss, n_batches = 0.0, 0

            indices = torch.randperm(len(X_train))
            for start in range(0, len(X_train), batch_size):
                batch_idx = indices[start : start + batch_size]
                predictions = self.model(X_train[batch_idx])
                loss = criterion(predictions, y_train[batch_idx])

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val), y_val).item()

            self.history["train_loss"].append(epoch_loss / max(1, n_batches))
            self.history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"Train Loss: {self.history['train_loss'][-1]:.6f} - "
                      f"Val Loss: {val_loss:.6f}")

        if verbose:
            print("\nTraining complete!")
        return self.history

    # ------------------------------------------------------------------ #
    # Inference at any size
    # ------------------------------------------------------------------ #
    def predict_parameters(
        self, problem_graph: Sequence[Edge], n_nodes: int, n_steps: int = 6
    ) -> np.ndarray:
        """Predict ``(gamma, beta)`` for a graph of **any** size in one pass."""
        self.model.eval()
        trajectory = self._trajectory(problem_graph, n_nodes, n_steps)
        X = torch.tensor(np.array([trajectory]), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X).cpu().numpy()[0]

    def approximation_ratio(
        self, problem_graph: Sequence[Edge], n_nodes: int, params: np.ndarray
    ) -> float:
        """Expected cut under ``params`` divided by the exact optimum."""
        qaoa = self.qaoa_for(n_nodes)
        gamma, beta = params[: self.qaoa_depth], params[self.qaoa_depth :]
        expected = qaoa.compute_expectation(gamma, beta, list(problem_graph))
        best, _ = brute_force_maxcut(list(problem_graph), n_nodes)
        return expected / best if best > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Transfer evaluation
    # ------------------------------------------------------------------ #
    def evaluate_transfer(
        self,
        test_sizes: Sequence[int],
        n_problems: int = 5,
        refine_maxiter: int = 20,
        verbose: bool = True,
    ) -> Dict[int, Dict[str, float]]:
        """Measure generalisation to (possibly unseen) sizes.

        For each size, compares three strategies by mean approximation ratio:

        ``random``    -- random angles, the no-knowledge baseline;
        ``predicted`` -- a single QK-LSTM forward pass (no circuit optimisation);
        ``refined``   -- the prediction warm-starting a short QAOA run.

        Sizes outside :attr:`train_sizes` are genuine extrapolation and are
        flagged as such in the printed table.
        """
        results: Dict[int, Dict[str, float]] = {}

        if verbose:
            print("\n" + "=" * 64)
            print("CROSS-SIZE TRANSFER EVALUATION")
            print("=" * 64)
            print(f"Trained on sizes: {list(self.train_sizes)}")
            print(f"\n{'size':>5} {'seen':>6} {'random':>9} {'predicted':>11} "
                  f"{'refined':>9}")
            print("-" * 64)

        for n_nodes in test_sizes:
            qaoa = self.qaoa_for(n_nodes)
            rand_r, pred_r, ref_r = [], [], []

            for _ in range(n_problems):
                edges = self.random_graph(n_nodes)

                random_params = np.random.uniform(0, 2 * np.pi, 2 * self.qaoa_depth)
                rand_r.append(self.approximation_ratio(edges, n_nodes, random_params))

                predicted = self.predict_parameters(edges, n_nodes)
                pred_r.append(self.approximation_ratio(edges, n_nodes, predicted))

                refined, _ = qaoa.optimize(
                    edges, initial_params=predicted, maxiter=refine_maxiter
                )
                ref_r.append(self.approximation_ratio(edges, n_nodes, refined))

            seen = n_nodes in self.train_sizes
            results[n_nodes] = {
                "random": float(np.mean(rand_r)),
                "predicted": float(np.mean(pred_r)),
                "refined": float(np.mean(ref_r)),
                "predicted_std": float(np.std(pred_r)),
                "seen_in_training": seen,
            }

            if verbose:
                r = results[n_nodes]
                print(f"{n_nodes:>5} {'yes' if seen else 'NEW':>6} "
                      f"{r['random']:>9.4f} {r['predicted']:>11.4f} "
                      f"{r['refined']:>9.4f}")

        if verbose:
            unseen = [n for n in test_sizes if n not in self.train_sizes]
            if unseen:
                gain = np.mean([results[n]["predicted"] - results[n]["random"]
                                for n in unseen])
                print("-" * 64)
                print(f"Mean gain over random on UNSEEN sizes {unseen}: {gain:+.4f}")

        self.transfer_results = results
        return results
