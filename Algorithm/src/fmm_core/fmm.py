import heapq
import math

import numpy as np
import skfmm

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


def solve_travel_time_custom(
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


def solve_travel_time_scikit(
    speed: np.ndarray,
    source_mask: np.ndarray,
    h: float = 1.0,
    traversable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve |∇T| = 1/speed using scikit-fmm with T=0 on source_mask.
    """
    speed = np.asarray(speed, dtype=float)
    if speed.ndim != 2:
        raise ValueError("speed must be a 2D array")
    if np.any(speed <= 0):
        raise ValueError("All speed values must be > 0")

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

    phi = np.ones_like(speed, dtype=float)
    phi[source_mask] = 0.0
    phi_masked = np.ma.MaskedArray(phi, mask=~traversable_mask)
    speed_masked = np.ma.MaskedArray(speed, mask=~traversable_mask)
    T_ma = skfmm.travel_time(phi_masked, speed_masked, dx=float(h), order=2)
    T = np.asarray(np.ma.filled(T_ma, np.inf), dtype=float)

    state = np.full(speed.shape, FAR, dtype=np.int8)
    state[np.isfinite(T) & traversable_mask] = ACCEPTED
    state[source_mask] = ACCEPTED
    return T, state


def backtrack_path_discrete(
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


def backtrack_path_continuous(
    T: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    step_size: float = 0.5,
    max_steps: int = 200000,
    traversable_mask: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], tuple[int, int]]:
    """
    Backtrack from target to source using continuous gradient descent.
    Returns path as [(y, x), ...] in floating point coordinates.
    """
    T = np.asarray(T, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    ny, nx = T.shape

    if source_mask.shape != (ny, nx) or target_mask.shape != (ny, nx):
        raise ValueError("source_mask and target_mask must match T shape")
    if traversable_mask is None:
        traversable_mask = np.ones_like(T, dtype=bool)
    else:
        traversable_mask = np.asarray(traversable_mask, dtype=bool)
        if traversable_mask.shape != (ny, nx):
            raise ValueError("traversable_mask must match T shape")

    source_mask = source_mask & traversable_mask
    target_mask = target_mask & traversable_mask
    if not np.any(target_mask):
        raise ValueError("target_mask is empty")

    # 1. Find the starting pixel (lowest T in the target region)
    Ti = T.copy()
    Ti[~target_mask] = np.inf
    flat = int(np.argmin(Ti))
    if not np.isfinite(Ti.flat[flat]):
        raise RuntimeError("No reachable target cell.")

    start_y, start_x = np.unravel_index(flat, T.shape)
    chosen_target = (int(start_y), int(start_x))

    # 2. Gradient Descent Setup
    # Path tracks floating point coordinates (y, x)
    cy, cx = float(start_y), float(start_x)
    path = [(cy, cx)]

    for _ in range(max_steps):
        # Stop if we've entered the source region
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < ny and 0 <= ix < nx and source_mask[iy, ix]:
            break

        # Integer bounds for bilinear interpolation
        y0, x0 = int(np.floor(cy)), int(np.floor(cx))
        y1, x1 = y0 + 1, x0 + 1

        # Check boundary conditions
        if y0 < 0 or y1 >= ny or x0 < 0 or x1 >= nx:
            break
            
        # Stop if we are interpolating into non-traversable space
        if not (traversable_mask[y0, x0] and traversable_mask[y0, x1] and 
                traversable_mask[y1, x0] and traversable_mask[y1, x1]):
            break

        # Grab the 4 corner T values
        T00, T01 = T[y0, x0], T[y0, x1]
        T10, T11 = T[y1, x0], T[y1, x1]

        # Fractional components
        dy, dx = cy - y0, cx - x0

        # Calculate partial derivatives using bilinear interpolation
        grad_x = (1.0 - dy) * (T01 - T00) + dy * (T11 - T10)
        grad_y = (1.0 - dx) * (T10 - T00) + dx * (T11 - T01)

        # Normalize the gradient vector
        norm = np.hypot(grad_y, grad_x)
        if norm < 1e-8:
            break  # Reached a local minimum or flat region

        grad_y /= norm
        grad_x /= norm

        # Take a step downhill (opposite the gradient)
        cy -= step_size * grad_y
        cx -= step_size * grad_x

        path.append((cy, cx))

    return path, chosen_target


def solve_travel_time(
    speed: np.ndarray,
    source_mask: np.ndarray,
    h: float = 1.0,
    traversable_mask: np.ndarray | None = None,
    method: str = "custom",
) -> tuple[np.ndarray, np.ndarray]:
    if method == "custom":
        return solve_travel_time_custom(speed, source_mask, h=h, traversable_mask=traversable_mask)
    if method == "scikit":
        return solve_travel_time_scikit(speed, source_mask, h=h, traversable_mask=traversable_mask)
    raise ValueError(f"Unsupported FMM method: {method}")


def extract_path(
    T: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    method: str = "zigzag",
    max_steps: int = 200000,
    step_size: float = 0.5,
    traversable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if method in {"zigzag", "discrete"}:
        path_yx, chosen_target = backtrack_path_discrete(
            T,
            source_mask,
            target_mask,
            max_steps=max_steps,
            traversable_mask=traversable_mask,
        )
    elif method == "continuous":
        path_yx, chosen_target = backtrack_path_continuous(
            T,
            source_mask,
            target_mask,
            step_size=step_size,
            max_steps=max_steps,
            traversable_mask=traversable_mask,
        )
    else:
        raise ValueError(f"Unsupported backtracking method: {method}")

    path_xy = np.asarray(path_yx, dtype=float)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2:
        raise RuntimeError("Extracted path has invalid shape")
    return path_xy[:, [1, 0]][::-1], chosen_target


def compute_fmm_path(
    speed: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    h: float = 1.0,
    traversable_mask: np.ndarray | None = None,
    solver: str = "custom",
    backtracking: str = "discrete",
    continuous_step_size: float = 0.5,
    max_steps: int = 200000,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    T, _state = solve_travel_time(
        speed,
        source_mask,
        h=h,
        traversable_mask=traversable_mask,
        method=solver,
    )
    path_xy, chosen_target = extract_path(
        T,
        source_mask,
        target_mask,
        method=backtracking,
        max_steps=max_steps,
        step_size=continuous_step_size,
        traversable_mask=traversable_mask,
    )
    if path_xy.shape[0] < 2:
        raise RuntimeError("FMM path extraction failed")
    travel_value = float(T[chosen_target[0], chosen_target[1]])
    return path_xy, travel_value, chosen_target


def fast_marching_2d_regions(
    speed: np.ndarray,
    source_mask: np.ndarray,
    h: float = 1.0,
    traversable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return solve_travel_time_custom(speed, source_mask, h=h, traversable_mask=traversable_mask)


def fast_marching_2d_scikit(
    speed: np.ndarray,
    source_mask: np.ndarray,
    h: float = 1.0,
    traversable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return solve_travel_time_scikit(speed, source_mask, h=h, traversable_mask=traversable_mask)


def extract_path_source_to_target(
    T: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    max_steps: int = 200000,
    traversable_mask: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    return backtrack_path_discrete(
        T,
        source_mask,
        target_mask,
        max_steps=max_steps,
        traversable_mask=traversable_mask,
    )


def extract_path_continuous(
    T: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    step_size: float = 0.5,
    max_steps: int = 200000,
    traversable_mask: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], tuple[int, int]]:
    return backtrack_path_continuous(
        T,
        source_mask,
        target_mask,
        step_size=step_size,
        max_steps=max_steps,
        traversable_mask=traversable_mask,
    )
