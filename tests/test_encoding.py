"""Tests for the graph encoders.

The two properties that make cross-size transfer possible are asserted directly:
**permutation invariance** and **size-agnosticism**.
"""

from __future__ import annotations

import numpy as np
import pytest

from qklstm import DegreeEncoder, SpectralEncoder
from qklstm.encoding import normalized_laplacian_spectrum, adjacency_matrix
from qklstm.qaoa import brute_force_maxcut


def _cycle(n: int):
    return [(i, (i + 1) % n) for i in range(n)]


# ---------------------------------------------------------------------- #
# Permutation invariance -- the key property
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(5))
def test_spectral_encoding_is_permutation_invariant(seed):
    """Relabelling vertices must not change the encoding."""
    enc = SpectralEncoder()
    n = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]

    baseline = enc.encode(edges, n)

    perm = np.random.RandomState(seed).permutation(n)
    relabelled = [(int(perm[i]), int(perm[j])) for i, j in edges]
    permuted = enc.encode(relabelled, n)

    assert np.allclose(baseline, permuted, atol=1e-5)


def test_degree_encoder_is_not_permutation_invariant():
    """Contrast: the original encoding does depend on the labelling."""
    enc = DegreeEncoder(n_nodes=4)
    star = [(0, 1), (0, 2), (0, 3)]          # hub at vertex 0
    relabelled = [(1, 0), (1, 2), (1, 3)]    # same graph, hub at vertex 1
    assert not np.allclose(enc.encode(star, 4), enc.encode(relabelled, 4))


# ---------------------------------------------------------------------- #
# Size-agnosticism -- what enables transfer
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [3, 4, 6, 10, 16])
def test_spectral_dim_is_constant_across_sizes(n):
    enc = SpectralEncoder()
    vec = enc.encode(_cycle(n), n)
    assert vec.shape == (enc.dim,)
    assert enc.is_size_agnostic


def test_degree_encoder_dim_grows_with_size():
    assert DegreeEncoder(4).dim == 6
    assert DegreeEncoder(10).dim == 12
    assert not DegreeEncoder(4).is_size_agnostic


def test_size_hint_toggle_changes_dim_by_one():
    assert SpectralEncoder(include_size_hint=True).dim == \
        SpectralEncoder(include_size_hint=False).dim + 1


# ---------------------------------------------------------------------- #
# Numerical sanity
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 4, 8, 12])
def test_encoding_is_finite_and_bounded(n):
    vec = SpectralEncoder().encode(_cycle(n), n)
    assert np.isfinite(vec).all()
    assert np.abs(vec).max() <= 3.0  # spectrum <= 2, stats normalised


@pytest.mark.parametrize("graph,n", [([], 1), ([], 5), ([(0, 1)], 2)])
def test_degenerate_graphs_do_not_produce_nan(graph, n):
    """Empty graphs and isolated vertices must not leak NaN into the network."""
    vec = SpectralEncoder().encode(graph, n)
    assert np.isfinite(vec).all()


def test_normalized_laplacian_spectrum_within_zero_two():
    """The bound [0, 2] is what makes spectra comparable across sizes."""
    for n in (4, 7, 11):
        adj = adjacency_matrix(_cycle(n), n)
        eig = normalized_laplacian_spectrum(adj)
        assert eig.min() >= -1e-9 and eig.max() <= 2.0 + 1e-9
        # A connected graph has exactly one zero eigenvalue.
        assert np.sum(eig < 1e-8) == 1


def test_disconnected_graph_has_multiple_zero_eigenvalues():
    adj = adjacency_matrix([(0, 1), (2, 3)], 4)  # two components
    eig = normalized_laplacian_spectrum(adj)
    assert np.sum(eig < 1e-8) == 2


def test_encoder_rejects_invalid_size():
    with pytest.raises(ValueError):
        SpectralEncoder().encode([], 0)


# ---------------------------------------------------------------------- #
# Exact MaxCut optimum (denominator of the approximation ratio)
# ---------------------------------------------------------------------- #
def test_brute_force_maxcut_on_known_graphs():
    # A square: the optimal bipartition cuts all 4 edges.
    assert brute_force_maxcut([(0, 1), (1, 2), (2, 3), (3, 0)], 4)[0] == 4
    # A triangle is odd, so one edge must stay uncut.
    assert brute_force_maxcut([(0, 1), (1, 2), (2, 0)], 3)[0] == 2
    # A single edge is always cuttable.
    assert brute_force_maxcut([(0, 1)], 2)[0] == 1


def test_brute_force_bitstring_realises_reported_cut():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    best, bits = brute_force_maxcut(edges, 4)
    realised = sum(1 for i, j in edges if bits[i] != bits[j])
    assert realised == best
