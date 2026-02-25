"""Graph utilities and helper functions matching the Robotarium Python API."""

from __future__ import annotations

import numpy as np

# ── Graph Laplacian utilities ───────────────────────────────────────────────


def cycle_GL(n: int) -> np.ndarray:
    """Return the graph Laplacian for a cycle graph on *n* vertices."""
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = 2
        L[i, (i + 1) % n] = -1
        L[i, (i - 1) % n] = -1
    return L


def lineGL(n: int) -> np.ndarray:
    """Return the graph Laplacian for a line/path graph on *n* vertices."""
    L = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            L[i, i] += 1
            L[i, i - 1] = -1
        if i < n - 1:
            L[i, i] += 1
            L[i, i + 1] = -1
    return L


def completeGL(n: int) -> np.ndarray:
    """Return the graph Laplacian for the complete graph K_n."""
    L = n * np.eye(n) - np.ones((n, n))
    return L


def random_connectedGL(n: int, edge_probability: float = 0.5) -> np.ndarray:
    """Return a graph Laplacian for a random connected graph.

    Starts with a spanning tree (path) and adds random edges.
    """
    A = np.zeros((n, n))
    # Spanning tree (path)
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1

    # Random additional edges
    rng = np.random.default_rng()
    for i in range(n):
        for j in range(i + 2, n):
            if rng.random() < edge_probability:
                A[i, j] = 1
                A[j, i] = 1

    D = np.diag(A.sum(axis=1))
    return D - A


def topological_neighbors(L: np.ndarray, agent: int) -> np.ndarray:
    """Return the topological neighbors of *agent* given graph Laplacian *L*."""
    row = L[agent, :]
    neighbors = np.where((row < 0))[0]
    return neighbors


# ── Convergence checkers ────────────────────────────────────────────────────


def at_pose(
    states: np.ndarray,
    targets: np.ndarray,
    position_error: float = 0.05,
    rotation_error: float = 0.2,
) -> np.ndarray:
    """Check which robots are at their target pose (position + heading).

    Returns a 1D boolean array of length N.
    """
    n = states.shape[1]
    close = np.zeros(n, dtype=bool)
    for i in range(n):
        pos_err = np.linalg.norm(states[:2, i] - targets[:2, i])
        rot_err = abs(((states[2, i] - targets[2, i]) + np.pi) % (2 * np.pi) - np.pi)
        close[i] = pos_err < position_error and rot_err < rotation_error
    return close


def at_position(
    states: np.ndarray,
    targets: np.ndarray,
    position_error: float = 0.05,
) -> np.ndarray:
    """Check which robots are at their target position (ignoring heading).

    Returns a 1D boolean array of length N.
    """
    n = states.shape[1]
    close = np.zeros(n, dtype=bool)
    for i in range(n):
        close[i] = np.linalg.norm(states[:2, i] - targets[:2, i]) < position_error
    return close


def determine_marker_size(robotarium, marker_size_meters: float = 0.08) -> float:
    """Compute the matplotlib marker size for a given physical size.

    Returns a value suitable for use as the *s* parameter in scatter plots.
    """
    return (72.0 * marker_size_meters) ** 2
