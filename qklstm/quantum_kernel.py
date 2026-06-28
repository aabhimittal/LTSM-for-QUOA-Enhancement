"""Quantum kernel module for QK-LSTM.

The quantum kernel maps classical feature vectors into the exponentially large
Hilbert space of an ``n``-qubit system using a *ZZ-feature map* and measures the
similarity between two inputs as the squared overlap of their quantum states::

    K_Q(x1, x2) = |<phi(x1) | phi(x2)>|^2

Analogy: like a prism that splits white light into a rainbow, the feature map
spreads classical data across quantum superposition states where correlations
that are invisible to classical methods become apparent.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector


class QuantumKernel:
    """Quantum kernel based on a ZZ-feature map.

    Parameters
    ----------
    n_qubits:
        Number of qubits used to encode each input vector.
    n_layers:
        Number of repetitions of the (Hadamard -> RZ encoding -> ZZ entangling)
        block.  More layers create a richer, more entangled feature map.
    """

    def __init__(self, n_qubits: int, n_layers: int = 2):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    # ------------------------------------------------------------------ #
    # Feature map
    # ------------------------------------------------------------------ #
    def feature_map(self, x: np.ndarray) -> QuantumCircuit:
        """Build the feature-map circuit ``U_phi(x)|0...0>``.

        The data ``x`` is padded/truncated (wrap mode) to match ``n_qubits``.
        Each layer applies:

        1. A Hadamard on every qubit (superposition).
        2. ``RZ(2 * x_i)`` data-encoding rotations.
        3. ``CX -> RZ(2 (pi - x_i)(pi - x_{i+1})) -> CX`` entangling blocks that
           inject pairwise (ZZ) correlations.
        """
        x = np.asarray(x, dtype=float).ravel()
        # Match the data length to the qubit count (wrap then truncate).
        x_padded = np.pad(
            x, (0, max(0, self.n_qubits - len(x))), mode="wrap"
        )[: self.n_qubits]

        qr = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(qr)

        for _ in range(self.n_layers):
            # Superposition layer.
            for i in range(self.n_qubits):
                qc.h(i)

            # Data-encoding layer.
            for i in range(self.n_qubits):
                qc.rz(2.0 * x_padded[i], i)

            # Entangling (ZZ) layer.
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(
                    2.0 * (np.pi - x_padded[i]) * (np.pi - x_padded[i + 1]),
                    i + 1,
                )
                qc.cx(i, i + 1)

        return qc

    # ------------------------------------------------------------------ #
    # Kernel evaluation
    # ------------------------------------------------------------------ #
    def compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Return ``K_Q(x1, x2) = |<phi(x1)|phi(x2)>|^2`` in ``[0, 1]``."""
        state1 = Statevector.from_instruction(self.feature_map(x1))
        state2 = Statevector.from_instruction(self.feature_map(x2))
        overlap = np.abs(state1.inner(state2)) ** 2
        return float(overlap)

    def compute_kernel_matrix(
        self, X1: np.ndarray, X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the kernel (Gram) matrix between two datasets.

        Returns an array of shape ``(len(X1), len(X2))``.  If ``X2`` is ``None``
        the symmetric matrix ``K(X1, X1)`` is computed (and symmetry is used to
        halve the number of circuit evaluations).
        """
        X1 = np.asarray(X1, dtype=float)
        symmetric = X2 is None
        X2 = X1 if symmetric else np.asarray(X2, dtype=float)

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        if symmetric:
            for i in range(n1):
                K[i, i] = 1.0  # a state always perfectly overlaps with itself
                for j in range(i + 1, n2):
                    val = self.compute_kernel(X1[i], X2[j])
                    K[i, j] = K[j, i] = val
        else:
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.compute_kernel(X1[i], X2[j])

        return K
