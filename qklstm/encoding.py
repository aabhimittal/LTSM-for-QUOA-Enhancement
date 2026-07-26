"""Problem encoders: turning a graph into a feature vector for the QK-LSTM.

Two encoders are provided:

* :class:`DegreeEncoder` -- the original encoding
  ``[degree_0, ..., degree_{n-1}, edge_density, n_edges]``.  Its dimension is
  ``n_nodes + 2``, so a model trained with it is welded to one problem size, and
  relabelling the vertices changes the vector.

* :class:`SpectralEncoder` -- a **permutation-invariant** and **size-agnostic**
  descriptor built from the normalized Laplacian spectrum plus scale-free
  structural statistics.  Its dimension is a constant independent of ``n_nodes``,
  which is what allows a single trained model to transfer across graph sizes
  (train on small graphs, predict for larger ones).

Why the *normalized* Laplacian?  ``L = I - D^{-1/2} A D^{-1/2}`` has all of its
eigenvalues in ``[0, 2]`` for **any** graph size, whereas the combinatorial
Laplacian ``L = D - A`` has a spectrum that grows with the degrees.  Bounding the
spectrum is precisely what makes the features comparable across sizes.

Only NumPy is required -- no extra graph library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int]

# Fixed quantile positions used to summarise the spectrum with a constant number
# of features regardless of how many eigenvalues there are.
_SPECTRUM_QUANTILES: Tuple[float, ...] = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))


def adjacency_matrix(problem_graph: Sequence[Edge], n_nodes: int) -> np.ndarray:
    """Build the symmetric 0/1 adjacency matrix of an undirected graph."""
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    for i, j in problem_graph:
        if i == j:
            continue  # ignore self-loops: they carry no cut information
        adj[i, j] = adj[j, i] = 1.0
    return adj


def normalized_laplacian_spectrum(adj: np.ndarray) -> np.ndarray:
    """Eigenvalues of ``L = I - D^{-1/2} A D^{-1/2}``, sorted ascending.

    Isolated vertices (degree 0) use the standard convention ``D^{-1/2} -> 0``,
    which contributes an eigenvalue of 1 for that vertex's row.  All eigenvalues
    lie in ``[0, 2]`` regardless of graph size.
    """
    degrees = adj.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)

    normalized = adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    laplacian = np.eye(len(adj)) - normalized
    # eigvalsh is exact for symmetric matrices and returns ascending eigenvalues.
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return np.clip(eigenvalues, 0.0, 2.0)


class ProblemEncoder(ABC):
    """Interface for turning ``(problem_graph, n_nodes)`` into a feature vector."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Length of the vector produced by :meth:`encode`."""

    @property
    def is_size_agnostic(self) -> bool:
        """Whether :attr:`dim` is independent of the graph size."""
        return False

    @abstractmethod
    def encode(self, problem_graph: Sequence[Edge], n_nodes: int) -> np.ndarray:
        """Return a ``float32`` feature vector of length :attr:`dim`."""


class DegreeEncoder(ProblemEncoder):
    """Original size-dependent encoding (kept for backward compatibility).

    Produces ``[degree_0, ..., degree_{n-1}, edge_density, n_edges]`` of length
    ``n_nodes + 2``.  Note this is *not* permutation invariant: relabelling the
    vertices permutes the degree block.
    """

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes

    @property
    def dim(self) -> int:
        return self.n_nodes + 2

    def encode(self, problem_graph: Sequence[Edge], n_nodes: int = None) -> np.ndarray:
        n = self.n_nodes if n_nodes is None else n_nodes
        adj = adjacency_matrix(problem_graph, n)
        degree = adj.sum(axis=1)

        max_edges = n * (n - 1) / 2
        edge_density = len(problem_graph) / max_edges if max_edges else 0.0

        features = np.concatenate(
            [degree, [edge_density, float(len(problem_graph))]]
        )
        return features[: self.dim].astype(np.float32)


class SpectralEncoder(ProblemEncoder):
    """Permutation-invariant, size-agnostic spectral graph encoding.

    Every feature is either a function of the (unordered) normalized-Laplacian
    spectrum or a normalised aggregate over vertices/edges, so relabelling the
    vertices leaves the vector unchanged.  Every feature is also scale-free, so
    the same vector layout describes a 4-node and a 40-node graph.

    Feature layout (``dim`` = 24, or 25 with ``include_size_hint``):

    ==========  ====================================================
    indices     meaning
    ==========  ====================================================
    0-10        11 fixed quantiles of the normalized Laplacian spectrum
    11-14       spectrum mean, std, spectral gap, zero-eigenvalue fraction
    15-19       degree mean/std/min/max (normalised by ``n-1``) and skewness
    20-23       edge density, avg clustering, triangle density, assortativity
    24          ``log(n) / 10`` size hint (only if ``include_size_hint``)
    ==========  ====================================================

    Parameters
    ----------
    include_size_hint:
        When ``True`` (default) a gentle ``log(n)/10`` feature is appended so the
        model *may* condition on problem size.  Trade-off: it helps interpolation
        within the training range but can mislead extrapolation far outside it.
        Set ``False`` for a strictly size-blind encoding.
    """

    _N_SPECTRUM_QUANTILES = len(_SPECTRUM_QUANTILES)
    _N_SPECTRUM_STATS = 4
    _N_DEGREE_STATS = 5
    _N_STRUCTURAL = 4

    def __init__(self, include_size_hint: bool = True):
        self.include_size_hint = include_size_hint

    @property
    def dim(self) -> int:
        return (
            self._N_SPECTRUM_QUANTILES
            + self._N_SPECTRUM_STATS
            + self._N_DEGREE_STATS
            + self._N_STRUCTURAL
            + (1 if self.include_size_hint else 0)
        )

    @property
    def is_size_agnostic(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Feature groups
    # ------------------------------------------------------------------ #
    @staticmethod
    def _spectrum_features(eigenvalues: np.ndarray) -> List[float]:
        """Quantiles + summary statistics of the spectrum (order independent)."""
        quantiles = list(np.quantile(eigenvalues, _SPECTRUM_QUANTILES))

        # lambda_2 is the algebraic connectivity: 0 iff the graph is disconnected.
        spectral_gap = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        # A zero eigenvalue per connected component.
        zero_fraction = float(np.mean(eigenvalues < 1e-8))

        return quantiles + [
            float(np.mean(eigenvalues)),
            float(np.std(eigenvalues)),
            spectral_gap,
            zero_fraction,
        ]

    @staticmethod
    def _degree_features(degrees: np.ndarray, n_nodes: int) -> List[float]:
        """Degree distribution moments, normalised by the maximum degree n-1."""
        scale = max(n_nodes - 1, 1)
        normalized = degrees / scale

        std = float(np.std(normalized))
        if std > 1e-12:
            skewness = float(np.mean(((normalized - normalized.mean()) / std) ** 3))
        else:
            skewness = 0.0  # a regular graph has no degree skew

        return [
            float(np.mean(normalized)),
            std,
            float(np.min(normalized)),
            float(np.max(normalized)),
            skewness,
        ]

    @staticmethod
    def _structural_features(
        adj: np.ndarray, degrees: np.ndarray, problem_graph: Sequence[Edge], n_nodes: int
    ) -> List[float]:
        """Density, clustering, triangle density and degree assortativity."""
        max_edges = n_nodes * (n_nodes - 1) / 2
        edge_density = len(problem_graph) / max_edges if max_edges else 0.0

        # Triangles through each vertex: diag(A^3) / 2.
        adj_cubed_diag = np.diag(adj @ adj @ adj)
        triangles_per_node = adj_cubed_diag / 2.0
        n_triangles = float(adj_cubed_diag.sum() / 6.0)

        # Local clustering = triangles / possible pairs of neighbours.
        possible_pairs = degrees * (degrees - 1) / 2.0
        local_clustering = np.divide(
            triangles_per_node,
            possible_pairs,
            out=np.zeros_like(triangles_per_node),
            where=possible_pairs > 0,
        )
        avg_clustering = float(np.mean(local_clustering))

        max_triangles = n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6.0
        triangle_density = n_triangles / max_triangles if max_triangles else 0.0

        return [
            edge_density,
            avg_clustering,
            triangle_density,
            SpectralEncoder._assortativity(degrees, problem_graph),
        ]

    @staticmethod
    def _assortativity(degrees: np.ndarray, problem_graph: Sequence[Edge]) -> float:
        """Pearson correlation of the degrees at each edge's endpoints.

        Positive means high-degree vertices attach to high-degree vertices.
        Returns 0 when undefined (no edges, or every endpoint has equal degree).
        """
        if not len(problem_graph):
            return 0.0

        # Each undirected edge contributes both orientations.
        left = np.array([degrees[i] for i, _ in problem_graph] +
                        [degrees[j] for _, j in problem_graph], dtype=float)
        right = np.array([degrees[j] for _, j in problem_graph] +
                         [degrees[i] for i, _ in problem_graph], dtype=float)

        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def encode(self, problem_graph: Sequence[Edge], n_nodes: int) -> np.ndarray:
        """Encode ``problem_graph`` into a fixed-length invariant descriptor."""
        if n_nodes < 1:
            raise ValueError("n_nodes must be >= 1")

        adj = adjacency_matrix(problem_graph, n_nodes)
        degrees = adj.sum(axis=1)
        eigenvalues = normalized_laplacian_spectrum(adj)

        features: List[float] = []
        features += self._spectrum_features(eigenvalues)
        features += self._degree_features(degrees, n_nodes)
        features += self._structural_features(adj, degrees, problem_graph, n_nodes)
        if self.include_size_hint:
            features.append(float(np.log(n_nodes) / 10.0))

        vector = np.asarray(features, dtype=np.float32)
        # Guard against NaN/inf leaking into the network from degenerate graphs.
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
