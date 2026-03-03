import numpy as np
from scipy.spatial import cKDTree


def resample_by_arclength(P: np.ndarray, ds: float = 1.0) -> np.ndarray:
    """Resample polyline to approximately uniform arc-length spacing."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("P must be an Nx2 array")
    if P.shape[0] == 0:
        raise ValueError("P must contain at least one point")

    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    L = float(s[-1])
    if L == 0.0:
        return P[[0]].copy()
    s_new = np.arange(0.0, L + 1e-9, ds)
    px = np.interp(s_new, s, P[:, 0])
    py = np.interp(s_new, s, P[:, 1])
    return np.column_stack([px, py])


def directed_nn_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each point in A, distance to nearest point in B."""
    tree = cKDTree(B)
    d, _ = tree.query(A, k=1, workers=-1)
    return d


def distance_distribution(A: np.ndarray, B: np.ndarray, ds: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute symmetric nearest-neighbor distance distribution between two polylines."""
    Au = resample_by_arclength(A, ds=ds)
    Bu = resample_by_arclength(B, ds=ds)
    dAB = directed_nn_distances(Au, Bu)
    dBA = directed_nn_distances(Bu, Au)
    D = np.concatenate([dAB, dBA])
    return D, dAB, dBA
