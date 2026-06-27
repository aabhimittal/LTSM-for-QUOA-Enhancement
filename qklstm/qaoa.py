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
    ) -> Tuple[np.ndarray, float]:
        """Optimise ``(gamma, beta)`` with COBYLA to *maximise* the cut.

        Returns ``(optimal_params, optimal_value)`` where ``optimal_params`` is a
        length-``2p`` vector laid out as ``[gamma_1..gamma_p, beta_1..beta_p]``.
        """
        if initial_params is None:
            params = np.random.uniform(0, 2 * np.pi, 2 * self.p)
        else:
            params = np.asarray(initial_params, dtype=float).copy()

        def objective(flat_params: np.ndarray) -> float:
            gamma = flat_params[: self.p]
            beta = flat_params[self.p :]
            # Negate because scipy minimises and we want to maximise the cut.
            return -self.compute_expectation(gamma, beta, problem_graph)

        result = minimize(
            objective, params, method="COBYLA", options={"maxiter": maxiter}
        )
        return result.x, float(-result.fun)
