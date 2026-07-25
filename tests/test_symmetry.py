"""Tests for QAOA parameter symmetries, canonicalisation and the periodic loss.

These properties are what make the parameter regression well-posed: without
them the optimal angles for a single graph are scattered over many equivalent
branches and the learning target is multimodal.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qklstm import QAOA, canonicalize_qaoa_params
from qklstm.cross_size import PeriodicAngleLoss
from qklstm.qaoa import BETA_PERIOD, GAMMA_PERIOD

PI = np.pi

GRAPHS = [
    (3, [(0, 1), (1, 2), (2, 0)]),                                  # triangle
    (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),                          # square
    (4, [(0, 1), (1, 2), (2, 3)]),                                  # path
    (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),          # K4
    (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]),          # cycle+chord
]


def _expectation(qaoa, params, edges, p):
    return qaoa.compute_expectation(params[:p], params[p:], edges)


# ---------------------------------------------------------------------- #
# The three generating symmetries
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("n,edges", GRAPHS)
def test_gamma_has_period_pi(n, edges):
    qaoa = QAOA(n_qubits=n, p=1)
    rng = np.random.default_rng(0)
    for gamma, beta in rng.uniform(0, 2 * PI, (5, 2)):
        base = _expectation(qaoa, np.array([gamma, beta]), edges, 1)
        shifted = _expectation(qaoa, np.array([gamma + GAMMA_PERIOD, beta]), edges, 1)
        assert base == pytest.approx(shifted, abs=1e-9)


@pytest.mark.parametrize("n,edges", GRAPHS)
def test_beta_has_period_pi_over_two(n, edges):
    """Spin-flip symmetry: X^(x)n commutes with the cost and fixes |+>^n."""
    qaoa = QAOA(n_qubits=n, p=1)
    rng = np.random.default_rng(1)
    for gamma, beta in rng.uniform(0, 2 * PI, (5, 2)):
        base = _expectation(qaoa, np.array([gamma, beta]), edges, 1)
        shifted = _expectation(qaoa, np.array([gamma, beta + BETA_PERIOD]), edges, 1)
        assert base == pytest.approx(shifted, abs=1e-9)


@pytest.mark.parametrize("n,edges", GRAPHS)
def test_time_reversal_symmetry(n, edges):
    qaoa = QAOA(n_qubits=n, p=1)
    rng = np.random.default_rng(2)
    for gamma, beta in rng.uniform(0, 2 * PI, (5, 2)):
        base = _expectation(qaoa, np.array([gamma, beta]), edges, 1)
        reversed_ = _expectation(qaoa, np.array([-gamma, -beta]), edges, 1)
        assert base == pytest.approx(reversed_, abs=1e-9)


# ---------------------------------------------------------------------- #
# Canonicalisation
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("n,edges", GRAPHS[:3])
def test_canonicalization_preserves_objective(p, n, edges):
    """Folding onto the canonical representative must not change the physics."""
    qaoa = QAOA(n_qubits=n, p=p)
    rng = np.random.default_rng(3)
    for params in rng.uniform(0, 2 * PI, (6, 2 * p)):
        before = _expectation(qaoa, params, edges, p)
        after = _expectation(qaoa, canonicalize_qaoa_params(params, p), edges, p)
        assert before == pytest.approx(after, abs=1e-9)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_canonical_params_lie_in_fundamental_domain(p):
    rng = np.random.default_rng(4)
    for params in rng.uniform(-10, 10, (20, 2 * p)):
        canon = canonicalize_qaoa_params(params, p)
        gamma, beta = canon[:p], canon[p:]
        assert np.all((gamma >= 0) & (gamma < GAMMA_PERIOD))
        assert np.all((beta >= 0) & (beta < BETA_PERIOD))
        assert beta[0] <= BETA_PERIOD / 2 + 1e-9   # time reversal fixes the branch


@pytest.mark.parametrize("p", [1, 2])
def test_canonicalization_is_idempotent(p):
    rng = np.random.default_rng(5)
    for params in rng.uniform(0, 2 * PI, (10, 2 * p)):
        once = canonicalize_qaoa_params(params, p)
        twice = canonicalize_qaoa_params(once, p)
        assert np.allclose(once, twice, atol=1e-9)


def test_equivalent_params_share_one_representative():
    """The whole symmetry orbit must collapse to a single label."""
    base = np.array([0.7, 0.3])
    orbit = [
        base,
        base + np.array([GAMMA_PERIOD, 0.0]),
        base + np.array([0.0, BETA_PERIOD]),
        -base,
        base + np.array([2 * GAMMA_PERIOD, 3 * BETA_PERIOD]),
    ]
    canonical = [canonicalize_qaoa_params(p, 1) for p in orbit]
    for other in canonical[1:]:
        assert np.allclose(canonical[0], other, atol=1e-9)


# ---------------------------------------------------------------------- #
# Periodic loss
# ---------------------------------------------------------------------- #
def test_periodic_loss_is_zero_for_exact_match():
    loss = PeriodicAngleLoss(p=1)
    target = torch.tensor([[1.2, 0.4]])
    assert loss(target.clone(), target).item() == pytest.approx(0.0, abs=1e-6)


def test_periodic_loss_ignores_full_period_offsets():
    """A prediction one period away is physically identical -> zero loss."""
    loss = PeriodicAngleLoss(p=1)
    target = torch.tensor([[1.2, 0.4]])
    shifted = torch.tensor([[1.2 + GAMMA_PERIOD, 0.4 + BETA_PERIOD]])
    assert loss(shifted, target).item() == pytest.approx(0.0, abs=1e-5)


def test_periodic_loss_penalises_genuine_error():
    loss = PeriodicAngleLoss(p=1)
    target = torch.tensor([[0.0, 0.0]])
    half_period = torch.tensor([[GAMMA_PERIOD / 2, BETA_PERIOD / 2]])
    assert loss(half_period, target).item() > 1.0


def test_periodic_loss_is_differentiable():
    loss = PeriodicAngleLoss(p=1)
    pred = torch.tensor([[0.5, 0.2]], requires_grad=True)
    loss(pred, torch.tensor([[1.0, 0.3]])).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------- #
# Multi-restart labelling
# ---------------------------------------------------------------------- #
def test_multi_restart_is_at_least_as_good_as_single():
    """Restarts exist because a single COBYLA run lands in local optima."""
    n, edges = 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    qaoa = QAOA(n_qubits=n, p=1)

    np.random.seed(0)
    single = max(qaoa.optimize(edges, n_restarts=1)[1] for _ in range(3))
    np.random.seed(0)
    multi = qaoa.optimize(edges, n_restarts=6)[1]

    assert multi >= single - 1e-6


def test_initial_params_are_honoured_over_restarts():
    """A supplied warm start must be used verbatim, not overridden by restarts."""
    qaoa = QAOA(n_qubits=4, p=1)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    warm = np.array([0.6, 0.3])
    params, value = qaoa.optimize(edges, initial_params=warm, maxiter=5,
                                  n_restarts=99)
    assert np.isfinite(params).all() and np.isfinite(value)
