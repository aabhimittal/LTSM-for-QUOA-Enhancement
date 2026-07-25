"""End-to-end tests for cross-size transfer.

Kept deliberately tiny (few problems, few epochs, classical kernel by default)
so the suite stays fast; the quantum path is exercised in one focused test.
"""

from __future__ import annotations

import numpy as np
import pytest

from qklstm import (
    CrossSizeQKLSTMOptimizer,
    DegreeEncoder,
    QKLSTMQAOAOptimizer,
    SpectralEncoder,
)


def _tiny_optimizer(use_quantum=False, train_sizes=(4, 5)):
    return CrossSizeQKLSTMOptimizer(
        train_sizes=train_sizes,
        qaoa_depth=1,
        lstm_hidden=16,
        use_quantum=use_quantum,
        seed=0,
    )


# ---------------------------------------------------------------------- #
# Construction
# ---------------------------------------------------------------------- #
def test_rejects_size_dependent_encoder():
    """A degree encoder cannot support transfer, so it must be refused."""
    with pytest.raises(ValueError, match="size-agnostic"):
        CrossSizeQKLSTMOptimizer(train_sizes=(4, 5), encoder=DegreeEncoder(4))


def test_model_input_dim_matches_encoder():
    opt = _tiny_optimizer()
    assert opt.model.input_dim == opt.encoder.dim
    assert opt.encoder.is_size_agnostic


# ---------------------------------------------------------------------- #
# The core capability: predict on a size never trained on
# ---------------------------------------------------------------------- #
def test_predicts_on_unseen_larger_size():
    """Train on {4, 5}; ask for parameters on 7 nodes -- shapes must just work."""
    opt = _tiny_optimizer(train_sizes=(4, 5))
    opt.train(n_problems_per_size=3, n_epochs=2, batch_size=2, verbose=False)

    unseen_n = 7
    edges = opt.random_graph(unseen_n)
    params = opt.predict_parameters(edges, unseen_n)

    assert params.shape == (2 * opt.qaoa_depth,)
    assert np.isfinite(params).all()


def test_approximation_ratio_is_a_valid_fraction():
    opt = _tiny_optimizer()
    opt.train(n_problems_per_size=3, n_epochs=2, batch_size=2, verbose=False)

    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    ratio = opt.approximation_ratio(edges, 4, opt.predict_parameters(edges, 4))
    assert 0.0 <= ratio <= 1.0


def test_evaluate_transfer_reports_all_sizes_and_flags_unseen():
    opt = _tiny_optimizer(train_sizes=(4,))
    opt.train(n_problems_per_size=4, n_epochs=3, batch_size=2, verbose=False)

    results = opt.evaluate_transfer([4, 6], n_problems=2, refine_maxiter=5,
                                    verbose=False)

    assert set(results) == {4, 6}
    assert results[4]["seen_in_training"] is True
    assert results[6]["seen_in_training"] is False   # genuine extrapolation
    for stats in results.values():
        for key in ("random", "predicted", "refined"):
            assert 0.0 <= stats[key] <= 1.0


def test_refinement_never_hurts_the_prediction():
    """Refinement warm-starts from the prediction, so it can only improve it."""
    opt = _tiny_optimizer()
    opt.train(n_problems_per_size=3, n_epochs=2, batch_size=2, verbose=False)

    results = opt.evaluate_transfer([5], n_problems=3, refine_maxiter=20,
                                    verbose=False)
    assert results[5]["refined"] >= results[5]["predicted"] - 1e-6


def test_training_history_is_finite():
    opt = _tiny_optimizer()
    history = opt.train(n_problems_per_size=3, n_epochs=3, batch_size=2,
                        verbose=False)
    assert len(history["train_loss"]) == 3
    assert all(np.isfinite(history["train_loss"]))
    assert all(np.isfinite(history["val_loss"]))


@pytest.mark.parametrize("n", [3, 4, 6])
def test_random_graph_is_valid(n):
    opt = _tiny_optimizer()
    edges = opt.random_graph(n)
    assert len(edges) == len(set(edges))            # no duplicates
    for i, j in edges:
        assert 0 <= i < j < n                       # canonical, no self-loops


# ---------------------------------------------------------------------- #
# Quantum path (slower -- a single small case)
# ---------------------------------------------------------------------- #
def test_cross_size_with_quantum_kernel():
    opt = _tiny_optimizer(use_quantum=True, train_sizes=(4,))
    opt.train(n_problems_per_size=3, n_epochs=1, batch_size=2, n_reference=2,
              verbose=False)

    edges = opt.random_graph(5)                     # unseen size
    params = opt.predict_parameters(edges, 5)
    assert np.isfinite(params).all()


# ---------------------------------------------------------------------- #
# Backward compatibility of the original single-size optimiser
# ---------------------------------------------------------------------- #
def test_default_encoder_preserves_original_behaviour():
    opt = QKLSTMQAOAOptimizer(n_qubits=5, qaoa_depth=1, use_quantum=False, seed=0)
    encoding = opt.encode_problem([(0, 1), (1, 2)])

    assert encoding.shape == (5 + 2,)
    # [degree_0..degree_4, density, n_edges]
    assert np.allclose(encoding[:5], [1, 2, 1, 0, 0])
    assert encoding[-1] == 2


def test_original_optimizer_accepts_spectral_encoder():
    """The pluggable encoder also works in the single-size optimiser."""
    enc = SpectralEncoder()
    opt = QKLSTMQAOAOptimizer(n_qubits=4, qaoa_depth=1, use_quantum=False,
                              seed=0, encoder=enc)
    assert opt.model.input_dim == enc.dim
    assert opt.encode_problem([(0, 1), (1, 2)]).shape == (enc.dim,)
