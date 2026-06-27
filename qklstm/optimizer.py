"""End-to-end QK-LSTM meta-optimiser for QAOA.

``QKLSTMQAOAOptimizer`` ties the three pieces together:

* :class:`~qklstm.qaoa.QAOA` provides the ground-truth (slow) optimiser used to
  generate training labels and to refine predictions.
* :class:`~qklstm.qklstm_model.QuantumKernelLSTM` learns the mapping
  ``problem -> optimal (gamma, beta)`` across many problems.

The payoff is *meta-learning*: instead of optimising every new problem from
scratch (~100 circuit evaluations), the trained model predicts good parameters
in a single forward pass, optionally followed by a few refinement steps.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .qaoa import QAOA, Edge
from .qklstm_model import QuantumKernelLSTM


class QKLSTMQAOAOptimizer:
    """Train a QK-LSTM to predict QAOA parameters for MaxCut problems."""

    def __init__(
        self,
        n_qubits: int,
        qaoa_depth: int = 1,
        lstm_hidden: int = 128,
        use_quantum: bool = True,
        lr: float = 1e-3,
        seed: Optional[int] = 42,
        device: Optional[str] = None,
    ):
        self.n_qubits = n_qubits
        self.qaoa_depth = qaoa_depth
        self.use_quantum = use_quantum
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.qaoa = QAOA(n_qubits=n_qubits, p=qaoa_depth)

        # Input  : problem encoding (degree sequence + edge density + #edges).
        # Output : 2 * depth QAOA parameters [gamma..., beta...].
        input_dim = n_qubits + 2
        output_dim = 2 * qaoa_depth

        self.model = QuantumKernelLSTM(
            input_dim=input_dim,
            hidden_dim=lstm_hidden,
            output_dim=output_dim,
            n_qubits=min(4, n_qubits),
            use_quantum=use_quantum,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.comparison_results: Optional[Dict] = None

    # ------------------------------------------------------------------ #
    # Problem encoding
    # ------------------------------------------------------------------ #
    def encode_problem(self, problem_graph: List[Edge]) -> np.ndarray:
        """Encode a graph as a fixed-length feature vector.

        Features = ``[degree_0, ..., degree_{n-1}, edge_density, n_edges]`` with
        length ``n_qubits + 2``.
        """
        adj = np.zeros((self.n_qubits, self.n_qubits))
        for i, j in problem_graph:
            adj[i, j] = adj[j, i] = 1.0

        degree = adj.sum(axis=1)
        max_edges = self.n_qubits * (self.n_qubits - 1) / 2
        edge_density = len(problem_graph) / max_edges if max_edges else 0.0

        features = np.concatenate([degree, [edge_density, float(len(problem_graph))]])
        return features[: self.n_qubits + 2].astype(np.float32)

    # ------------------------------------------------------------------ #
    # Random problem generation
    # ------------------------------------------------------------------ #
    def _random_graph(self) -> List[Edge]:
        """Sample a random MaxCut graph.

        The number of edges is capped at the maximum number of *distinct* edges
        ``C(n, 2)``; without this cap small graphs (e.g. ``n_qubits=4``) could
        request more unique edges than exist and the sampling loop would never
        terminate.
        """
        max_edges = self.n_qubits * (self.n_qubits - 1) // 2
        low = min(self.n_qubits, max_edges)
        high = min(2 * self.n_qubits, max_edges)
        n_edges = np.random.randint(low, high + 1)

        edges: List[Edge] = []
        while len(edges) < n_edges:
            i_node = np.random.randint(0, self.n_qubits)
            j_node = np.random.randint(0, self.n_qubits)
            if i_node != j_node:
                edge = (min(i_node, j_node), max(i_node, j_node))
                if edge not in edges:
                    edges.append(edge)
        return edges

    def generate_training_data(
        self, n_problems: int = 100, n_steps: int = 10, verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate ``(X, y)`` training data by solving random MaxCut problems.

        For each problem we solve it with classical QAOA to obtain the optimal
        parameters, then build a length-``n_steps`` "trajectory" of the problem
        encoding (the temporal axis the LSTM consumes).  ``X`` has shape
        ``(n_problems, n_steps, input_dim)`` and ``y`` has shape
        ``(n_problems, 2 * depth)``.
        """
        X_train, y_train = [], []
        if verbose:
            print(f"Generating {n_problems} training problems...")

        for i in range(n_problems):
            edges = self._random_graph()
            optimal_params, _ = self.qaoa.optimize(edges)

            encoding = self.encode_problem(edges)
            trajectory = [encoding for _ in range(n_steps)]

            X_train.append(trajectory)
            y_train.append(optimal_params)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{n_problems} problems")

        X = torch.tensor(np.array(X_train), dtype=torch.float32)
        y = torch.tensor(np.array(y_train), dtype=torch.float32)
        return X, y

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(
        self,
        n_problems: int = 100,
        n_epochs: int = 50,
        batch_size: int = 16,
        val_split: float = 0.2,
        n_reference: int = 20,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the QK-LSTM to map trajectories to optimal QAOA parameters."""
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING QK-LSTM FOR QAOA ENHANCEMENT")
            print("=" * 60)

        X, y = self.generate_training_data(n_problems=n_problems, verbose=verbose)
        X, y = X.to(self.device), y.to(self.device)

        n_val = max(1, int(len(X) * val_split))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        if verbose:
            print(f"\nTraining samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")

        # Configure the quantum kernel reference set (bug-fixed projection).
        if self.use_quantum:
            n_ref = min(n_reference, len(X_train))
            self.model.set_reference_data(X_train[:n_ref])

        criterion = nn.MSELoss()
        if verbose:
            print(f"\nTraining for {n_epochs} epochs...")

        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            indices = torch.randperm(len(X_train))
            for start in range(0, len(X_train), batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val), y_val).item()

            avg_train_loss = epoch_loss / max(1, n_batches)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs} - "
                    f"Train Loss: {avg_train_loss:.6f} - "
                    f"Val Loss: {val_loss:.6f}"
                )

        if verbose:
            print("\nTraining complete!")
        return self.history

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict_parameters(
        self, problem_graph: List[Edge], n_steps: int = 10
    ) -> np.ndarray:
        """Predict ``(gamma, beta)`` for a problem in a single forward pass."""
        self.model.eval()
        encoding = self.encode_problem(problem_graph)
        trajectory = [encoding for _ in range(n_steps)]
        X = torch.tensor(np.array([trajectory]), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X).cpu().numpy()[0]

    def solve_with_prediction(
        self, problem_graph: List[Edge], verbose: bool = True
    ) -> Dict[str, Dict]:
        """Compare baseline QAOA, raw QK-LSTM prediction, and refinement."""
        if verbose:
            print("\n" + "=" * 60)
            print("SOLVING MAXCUT PROBLEM")
            print("=" * 60)
            print(f"Problem: {len(problem_graph)} edges on {self.n_qubits} qubits")

        # Baseline: full classical QAOA optimisation.
        params_baseline, value_baseline = self.qaoa.optimize(problem_graph)
        if verbose:
            print(f"\n[Baseline] Optimal value: {value_baseline:.4f}")

        # QK-LSTM prediction (single forward pass).
        params_predicted = self.predict_parameters(problem_graph)
        gamma_pred = params_predicted[: self.qaoa_depth]
        beta_pred = params_predicted[self.qaoa_depth :]
        value_predicted = self.qaoa.compute_expectation(
            gamma_pred, beta_pred, problem_graph
        )
        if verbose:
            print(f"[QK-LSTM]  Predicted value: {value_predicted:.4f}")

        # QK-LSTM prediction + classical refinement.
        params_refined, value_refined = self.qaoa.optimize(
            problem_graph, initial_params=params_predicted
        )
        if verbose:
            print(f"[Refined]  Refined value:   {value_refined:.4f}")

        results = {
            "baseline": {"params": params_baseline, "value": value_baseline},
            "predicted": {"params": params_predicted, "value": value_predicted},
            "refined": {"params": params_refined, "value": value_refined},
        }

        if verbose:
            denom = value_baseline if value_baseline else 1e-9
            print("\n" + "=" * 60)
            print("PERFORMANCE COMPARISON")
            print("=" * 60)
            print(f"QK-LSTM prediction:   {value_predicted / denom * 100:.1f}% of baseline")
            print(f"QK-LSTM + refinement: {value_refined / denom * 100:.1f}% of baseline")

        self.comparison_results = results
        return results
