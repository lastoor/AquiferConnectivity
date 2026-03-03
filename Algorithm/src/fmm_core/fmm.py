import heapq
import math

import numpy as np

FAR, NARROW, ACCEPTED = 0, 1, 2


def _solve_quadratic_update(ax: float, ay: float, f: float, h: float) -> float:
    """Upwind isotropic update for |∇T| = f where f = 1/speed."""
    if ax > ay:
        ax, ay = ay, ax
    fh = f * h

    if (ay - ax) >= fh:
        return ax + fh

    disc = 2.0 * fh * fh - (ay - ax) * (ay - ax)
    if disc < 0:
        disc = 0.0
    return 0.5 * (ax + ay + math.sqrt(disc))


def fast_marching_2d_regions(
    speed: np.ndarray,
    source_mask: np.ndarray,
    h: float = 1.0,
    traversable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve |∇T| = 1/speed for isotropic medium with FMM, with T=0 on source_mask.

    Parameters
    ----------
    speed : (ny, nx) array, positive values
    source_mask : (ny, nx) bool array, True where source region is
    h : grid spacing
    """
    speed = np.asarray(speed, dtype=float)
    if speed.ndim != 2:
        raise ValueError("speed must be a 2D array")
    if np.any(speed <= 0):
        raise ValueError("All speed values must be > 0")

    ny, nx = speed.shape
    source_mask = np.asarray(source_mask, dtype=bool)
    if source_mask.shape != speed.shape:
        raise ValueError("source_mask must have same shape as speed")
    if traversable_mask is None:
        traversable_mask = np.ones_like(source_mask, dtype=bool)
    else:
        traversable_mask = np.asarray(traversable_mask, dtype=bool)
        if traversable_mask.shape != speed.shape:
            raise ValueError("traversable_mask must have same shape as speed")

    source_mask = source_mask & traversable_mask
    if not np.any(source_mask):
        raise ValueError("source_mask is empty")

    f = 1.0 / speed
    T = np.full((ny, nx), np.inf, dtype=float)
    state = np.full((ny, nx), FAR, dtype=np.int8)
    heap: list[tuple[float, int, int]] = []

    def push(i: int, j: int, val: float) -> None:
        if val < T[i, j]:
            T[i, j] = val
            heapq.heappush(heap, (val, i, j))
            state[i, j] = NARROW

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    src_idx = np.argwhere(source_mask)
    for i, j in src_idx:
        T[i, j] = 0.0
        state[i, j] = ACCEPTED

    for i, j in src_idx:
        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if not (0 <= ni < ny and 0 <= nj < nx):
                continue
            if not traversable_mask[ni, nj]:
                continue
            if state[ni, nj] == ACCEPTED:
                continue

            ax = np.inf
            ay = np.inf

            if nj - 1 >= 0 and state[ni, nj - 1] == ACCEPTED:
                ax = min(ax, T[ni, nj - 1])
            if nj + 1 < nx and state[ni, nj + 1] == ACCEPTED:
                ax = min(ax, T[ni, nj + 1])

            if ni - 1 >= 0 and state[ni - 1, nj] == ACCEPTED:
                ay = min(ay, T[ni - 1, nj])
            if ni + 1 < ny and state[ni + 1, nj] == ACCEPTED:
                ay = min(ay, T[ni + 1, nj])

            if np.isinf(ax) and np.isinf(ay):
                continue
            push(ni, nj, _solve_quadratic_update(ax, ay, f[ni, nj], h))

    while heap:
        t, i, j = heapq.heappop(heap)
        if state[i, j] == ACCEPTED:
            continue
        state[i, j] = ACCEPTED
        T[i, j] = t

        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if not (0 <= ni < ny and 0 <= nj < nx):
                continue
            if not traversable_mask[ni, nj]:
                continue
            if state[ni, nj] == ACCEPTED:
                continue

            ax = np.inf
            ay = np.inf

            if nj - 1 >= 0 and state[ni, nj - 1] == ACCEPTED:
                ax = min(ax, T[ni, nj - 1])
            if nj + 1 < nx and state[ni, nj + 1] == ACCEPTED:
                ax = min(ax, T[ni, nj + 1])
            if ni - 1 >= 0 and state[ni - 1, nj] == ACCEPTED:
                ay = min(ay, T[ni - 1, nj])
            if ni + 1 < ny and state[ni + 1, nj] == ACCEPTED:
                ay = min(ay, T[ni + 1, nj])

            if np.isinf(ax) and np.isinf(ay):
                continue
            push(ni, nj, _solve_quadratic_update(ax, ay, f[ni, nj], h))

    return T, state


def extract_path_source_to_target(
    T: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    max_steps: int = 200000,
    traversable_mask: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """
    Backtrack from best target to source by descending T on 4-neighborhood.

    Returns path in index coordinates as [(i,j), ...] from target -> source.
    """
    T = np.asarray(T, dtype=float)
    if T.ndim != 2:
        raise ValueError("T must be a 2D array")

    ny, nx = T.shape
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    if source_mask.shape != (ny, nx) or target_mask.shape != (ny, nx):
        raise ValueError("source_mask and target_mask must match T shape")
    if traversable_mask is None:
        traversable_mask = np.ones_like(source_mask, dtype=bool)
    else:
        traversable_mask = np.asarray(traversable_mask, dtype=bool)
        if traversable_mask.shape != (ny, nx):
            raise ValueError("traversable_mask must match T shape")

    source_mask = source_mask & traversable_mask
    target_mask = target_mask & traversable_mask
    if not np.any(target_mask):
        raise ValueError("target_mask is empty")

    Ti = T.copy()
    Ti[~target_mask] = np.inf
    flat = int(np.argmin(Ti))
    best_t = float(Ti.flat[flat])
    if not np.isfinite(best_t):
        raise RuntimeError("No reachable target cell (T=inf on target_mask)")

    i, j = np.unravel_index(flat, T.shape)
    chosen_target = (int(i), int(j))

    path: list[tuple[int, int]] = [(i, j)]
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(max_steps):
        if source_mask[i, j]:
            break

        best_val = T[i, j]
        best_ij = (i, j)
        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx and traversable_mask[ni, nj] and T[ni, nj] < best_val:
                best_val = T[ni, nj]
                best_ij = (ni, nj)

        if best_ij == (i, j):
            break

        i, j = best_ij
        path.append((i, j))

    return path, chosen_target
