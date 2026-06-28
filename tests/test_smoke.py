"""Fast smoke tests for the QK-LSTM / QAOA pipeline.

These run on tiny problems so they finish quickly while still exercising both
the classical and the quantum (Qiskit) code paths end to end.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qklstm import QAOA, QKLSTMQAOAOptimizer, QuantumKernel
from qklstm.qklstm_model import QuantumKernelLSTM


# ---------------------------------------------------------------------- #
# Quantum kernel
# ---------------------------------------------------------------------- #
def test_kernel_self_similarity_is_one():
    qk = QuantumKernel(n_qubits=3, n_layers=2)
    x = np.array([0.5, 0.7, 0.2])
    assert qk.compute_kernel(x, x) == pytest.approx(1.0, abs=1e-6)


def test_kernel_bounds_and_symmetry():
    qk = QuantumKernel(n_qubits=3, n_layers=1)
    x1, x2 = np.array([0.1, 0.4, 0.9]), np.array([0.8, 0.2, 0.3])
    k12, k21 = qk.compute_kernel(x1, x2), qk.compute_kernel(x2, x1)
    assert 0.0 <= k12 <= 1.0
    assert k12 == pytest.approx(k21, abs=1e-9)


def test_kernel_matrix_shape_and_diagonal():
    qk = QuantumKernel(n_qubits=2, n_layers=1)
    X = np.random.RandomState(0).rand(4, 2)
    K = qk.compute_kernel_matrix(X)
    assert K.shape == (4, 4)
    assert np.allclose(np.diag(K), 1.0, atol=1e-6)
    assert np.allclose(K, K.T, atol=1e-9)


# ---------------------------------------------------------------------- #
# QAOA
# ---------------------------------------------------------------------- #
def test_qaoa_expectation_in_range():
    qaoa = QAOA(n_qubits=3, p=1)
    edges = [(0, 1), (1, 2)]
    val = qaoa.compute_expectation(np.array([0.5]), np.array([0.5]), edges)
    assert 0.0 <= val <= len(edges)


def test_qaoa_optimize_improves_or_matches_random():
    qaoa = QAOA(n_qubits=4, p=1)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # square, optimal cut = 4
    params, value = qaoa.optimize(edges, maxiter=50)
    assert params.shape == (2,)
    assert value > 1.5  # should comfortably beat a trivial guess


# ---------------------------------------------------------------------- #
# QK-LSTM model
# ---------------------------------------------------------------------- #
def test_model_forward_classical_shape():
    model = QuantumKernelLSTM(input_dim=6, hidden_dim=16, output_dim=2,
                              use_quantum=False)
    x = torch.randn(3, 5, 6)
    assert model(x).shape == (3, 2)


def test_model_forward_quantum_shape():
    """The projection-dimension bug fix means quantum mode must not crash."""
    model = QuantumKernelLSTM(input_dim=6, hidden_dim=16, output_dim=2,
                              n_qubits=3, use_quantum=True)
    X_ref = torch.randn(4, 5, 6)
    model.set_reference_data(X_ref)
    out = model(torch.randn(2, 5, 6))
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------- #
# End-to-end optimiser
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("use_quantum", [False, True])
def test_optimizer_end_to_end(use_quantum):
    opt = QKLSTMQAOAOptimizer(
        n_qubits=4, qaoa_depth=1, lstm_hidden=16,
        use_quantum=use_quantum, seed=0,
    )
    history = opt.train(n_problems=6, n_epochs=2, batch_size=2,
                        n_reference=4, verbose=False)
    assert len(history["train_loss"]) == 2
    assert all(np.isfinite(history["train_loss"]))

    results = opt.solve_with_prediction([(0, 1), (1, 2), (2, 3), (3, 0)],
                                        verbose=False)
    for key in ("baseline", "predicted", "refined"):
        assert key in results
        assert np.isfinite(results[key]["value"])
    # Refinement starts from the prediction, so it can only help.
    assert results["refined"]["value"] >= results["predicted"]["value"] - 1e-6


def test_encode_problem_length():
    opt = QKLSTMQAOAOptimizer(n_qubits=5, qaoa_depth=1, use_quantum=False, seed=0)
    enc = opt.encode_problem([(0, 1), (1, 2)])
    assert enc.shape == (5 + 2,)
