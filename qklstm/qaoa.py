"""QAOA solver for the MaxCut problem.

The Quantum Approximate Optimization Algorithm (QAOA) prepares a parameterised
quantum state by alternating a *cost* unitary and a *mixer* unitary::

    |psi(gamma, beta)> = prod_{l=1..p} U_M(beta_l) U_C(gamma_l) |+>^{n}

and tunes ``(gamma, beta)`` so that measuring the state yields a high-quality
cut.  For MaxCut the cost of a bitstring ``z`` is the number of edges whose
endpoints fall in different partitions.

Analogy: QAOA is like searching a landscape for the highest peak while exploring
many paths simultaneously thanks to quantum superposition.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

Edge = Tuple[int, int]


# Periods of the QAOA angles for unweighted MaxCut (verified numerically in
# tests/test_symmetry.py).  ``exp(-i*gamma*sum Z_i Z_j)`` has an integer spectrum
# so gamma is pi-periodic; the mixer is pi-periodic on its own, and the extra
# factor of two comes from X^(x)n, which commutes with the cost Hamiltonian and
# stabilises |+>^n, making beta pi/2-periodic.
GAMMA_PERIOD = np.pi
BETA_PERIOD = np.pi / 2


def canonicalize_qaoa_params(params: np.ndarray, p: int) -> np.ndarray:
    """Map QAOA angles onto a canonical representative of their symmetry orbit.

    A MaxCut QAOA objective is invariant under

    * ``gamma_l -> gamma_l + pi``           (integer cost spectrum),
    * ``beta_l  -> beta_l + pi/2``          (spin-flip symmetry ``X^(x)n``),
    * ``(gamma, beta) -> (-gamma, -beta)``  (time reversal; applies jointly to
      every layer because the state is real).

    A classical optimiser started from a random point therefore returns angles
    scattered over many equivalent branches.  Regressing on such labels is
    ill-posed -- the mean of a multimodal target is not itself a good solution.
    Folding every label into the domain ``gamma in [0, pi)``, ``beta in [0, pi/4]``
    makes the regression target single-valued without changing any objective
    value.
    """
    gamma = np.asarray(params[:p], dtype=float) % GAMMA_PERIOD
    beta = np.asarray(params[p:], dtype=float) % BETA_PERIOD

    # Time reversal is a joint operation across all layers, so the decision is
    # taken once (using the first layer) and applied to the whole vector.
    if len(beta) and beta[0] > BETA_PERIOD / 2:
        gamma = (-gamma) % GAMMA_PERIOD
        beta = (-beta) % BETA_PERIOD

    return np.concatenate([gamma, beta])


def brute_force_maxcut(
    problem_graph: List[Edge], n_nodes: int
) -> Tuple[int, str]:
    """Exactly solve MaxCut by enumerating all ``2^n`` bipartitions.

    Returns ``(best_cut, best_bitstring)``.  This is exponential and intended
    only for the small instances used here (``n_nodes`` up to ~16); it provides
    the denominator ``C_max`` for the **approximation ratio**
    ``<C> / C_max``, which is the standard way to compare QAOA quality across
    problems of different sizes.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")

    best_cut, best_bits = -1, "0" * n_nodes
    for idx in range(2 ** n_nodes):
        bits = format(idx, f"0{n_nodes}b")
        cut = sum(1 for i, j in problem_graph if bits[i] != bits[j])
        if cut > best_cut:
            best_cut, best_bits = cut, bits

    return best_cut, best_bits


class QAOA:
    """QAOA for MaxCut via exact statevector simulation.

    Parameters
    ----------
    n_qubits:
        Number of graph vertices / qubits.
    p:
        Number of QAOA layers (circuit depth).
    """

    def __init__(self, n_qubits: int, p: int = 1):
        self.n_qubits = n_qubits
        self.p = p

    # ------------------------------------------------------------------ #
    # Circuit construction
    # ------------------------------------------------------------------ #
    def create_qaoa_circuit(
        self, gamma: np.ndarray, beta: np.ndarray, problem_graph: List[Edge]
    ) -> QuantumCircuit:
        """Build the depth-``p`` QAOA circuit for ``problem_graph``."""
        qr = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(qr)

        # Uniform superposition |+>^n.
        qc.h(range(self.n_qubits))

        for layer in range(self.p):
            # Cost Hamiltonian: exp(-i gamma sum_{(i,j)} Z_i Z_j).
            for i, j in problem_graph:
                qc.rzz(2.0 * gamma[layer], i, j)
            # Mixer Hamiltonian: exp(-i beta sum_i X_i).
            for i in range(self.n_qubits):
                qc.rx(2.0 * beta[layer], i)

        return qc

    # ------------------------------------------------------------------ #
    # Objective
    # ------------------------------------------------------------------ #
    def compute_expectation(
        self, gamma: np.ndarray, beta: np.ndarray, problem_graph: List[Edge]
    ) -> float:
        """Return the expected MaxCut value ``<psi| C |psi>``.

        ``C`` counts cut edges, so larger is better.
        """
        qc = self.create_qaoa_circuit(gamma, beta, problem_graph)
        state = Statevector.from_instruction(qc)
        probs = np.abs(state.data) ** 2

        expectation = 0.0
        for idx, prob in enumerate(probs):
            if prob == 0.0:
                continue
            # Qiskit orders qubit 0 as the least significant bit; reverse so that
            # bitstring[i] corresponds to vertex i.
            bitstring = format(idx, f"0{self.n_qubits}b")[::-1]
            cut = sum(1 for i, j in problem_graph if bitstring[i] != bitstring[j])
            expectation += cut * prob

        return float(expectation)

    # ------------------------------------------------------------------ #
    # Classical optimisation of the variational parameters
    # ------------------------------------------------------------------ #
    def optimize(
        self,
        problem_graph: List[Edge],
        initial_params: Optional[np.ndarray] = None,
        maxiter: int = 100,
        n_restarts: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """Optimise ``(gamma, beta)`` with COBYLA to *maximise* the cut.

        Returns ``(optimal_params, optimal_value)`` where ``optimal_params`` is a
        length-``2p`` vector laid out as ``[gamma_1..gamma_p, beta_1..beta_p]``.

        ``n_restarts`` runs the optimiser from several random starts and keeps the
        best result.  The QAOA landscape is non-convex, so a single COBYLA run
        regularly settles in a local optimum; restarts matter when the output is
        used as a *training label*, since inconsistent labels are unlearnable.
        Ignored when ``initial_params`` is supplied (that is a deliberate warm
        start, so it is honoured exactly).
        """

        def objective(flat_params: np.ndarray) -> float:
            gamma = flat_params[: self.p]
            beta = flat_params[self.p :]
            # Negate because scipy minimises and we want to maximise the cut.
            return -self.compute_expectation(gamma, beta, problem_graph)

        if initial_params is not None:
            starts = [np.asarray(initial_params, dtype=float).copy()]
        else:
            starts = [
                np.random.uniform(0, 2 * np.pi, 2 * self.p)
                for _ in range(max(1, n_restarts))
            ]

        best_params, best_value = None, -np.inf
        for start in starts:
            result = minimize(
                objective, start, method="COBYLA", options={"maxiter": maxiter}
            )
            if -result.fun > best_value:
                best_params, best_value = result.x, float(-result.fun)

        return best_params, best_value
