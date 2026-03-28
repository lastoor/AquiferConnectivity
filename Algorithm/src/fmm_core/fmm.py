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


import math
import numpy as np


def backtrack_path_continuous(
    T: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    step_size: float = 0.5,
    max_steps: int = 200000,
    traversable_mask: np.ndarray | None = None,
    stop_threshold: float = 1e-3,
    descent_tol: float = 1e-10,
    max_step_retries: int = 12,
) -> tuple[list[tuple[float, float]], tuple[int, int]]:
    """
    Backtrack from a target region to a source region by integrating the
    negative gradient of the bilinearly interpolated travel-time field.

    The path is computed in continuous coordinates (row, column) using RK4.
    At each step, the code enforces monotone descent in T by shrinking the step
    size if necessary.

    Parameters
    ----------
    T : (ny, nx) array
        Travel-time field computed by FMM.
    source_mask : (ny, nx) array of bool
        Source region. Backtracking stops when the path enters this region.
    target_mask : (ny, nx) array of bool
        Target region. The starting point is the target cell with minimal T.
    step_size : float
        Nominal spatial step in grid-index coordinates.
    max_steps : int
        Hard iteration cap.
    traversable_mask : (ny, nx) array of bool, optional
        True for traversable cells. If omitted, all cells are traversable.
    stop_threshold : float
        Additional stopping threshold on T, useful near the source.
    descent_tol : float
        Minimum required decrease in T for accepting a step.
    max_step_retries : int
        Maximum number of halvings used to enforce monotone descent.

    Returns
    -------
    path : list[tuple[float, float]]
        Continuous backtracked path as (row, col) coordinates.
    chosen_target : tuple[int, int]
        The discrete target cell from which backtracking started.
    """
    T = np.asarray(T, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)

    if T.ndim != 2:
        raise ValueError("T must be a 2D array.")
    if source_mask.shape != T.shape or target_mask.shape != T.shape:
        raise ValueError("source_mask and target_mask must have the same shape as T.")

    ny, nx = T.shape
    if traversable_mask is None:
        traversable_mask = np.ones_like(T, dtype=bool)
    else:
        traversable_mask = np.asarray(traversable_mask, dtype=bool)
        if traversable_mask.shape != T.shape:
            raise ValueError("traversable_mask must have the same shape as T.")

    if ny < 2 or nx < 2:
        raise ValueError("Bilinear backtracking requires at least a 2x2 grid.")

    # Start from the target cell with the smallest travel time.
    Ti = np.where(target_mask, T, np.inf)
    flat = int(np.argmin(Ti))
    if not np.isfinite(Ti.flat[flat]):
        raise RuntimeError("No reachable target cell found.")

    start_y, start_x = np.unravel_index(flat, T.shape)
    chosen_target = (int(start_y), int(start_x))

    def bilinear_T_and_unit_descent(y: float, x: float) -> tuple[float, float, float]:
        """
        Return (T_interp, unit_descent_y, unit_descent_x) at a continuous location.

        The gradient is the exact derivative of the bilinear interpolant over the
        cell containing (y, x). The returned vector is the unit vector pointing
        in the direction of decreasing T.
        """
        # Clamp point into the valid interpolation range.
        y = float(np.clip(y, 0.0, ny - 1.0 - 1e-12))
        x = float(np.clip(x, 0.0, nx - 1.0 - 1e-12))

        y0 = int(math.floor(y))
        x0 = int(math.floor(x))
        y0 = min(max(y0, 0), ny - 2)
        x0 = min(max(x0, 0), nx - 2)
        y1 = y0 + 1
        x1 = x0 + 1

        # If the cell is not fully traversable, stop rather than interpolate across it.
        if not (
            traversable_mask[y0, x0]
            and traversable_mask[y0, x1]
            and traversable_mask[y1, x0]
            and traversable_mask[y1, x1]
        ):
            return T[y0, x0], 0.0, 0.0

        dy = y - y0
        dx = x - x0

        T00 = T[y0, x0]
        T01 = T[y0, x1]
        T10 = T[y1, x0]
        T11 = T[y1, x1]

        # Bilinear interpolation.
        T_interp = (
            (1.0 - dy) * ((1.0 - dx) * T00 + dx * T01)
            + dy * ((1.0 - dx) * T10 + dx * T11)
        )

        # Exact derivative of the bilinear interpolant.
        dT_dx = (1.0 - dy) * (T01 - T00) + dy * (T11 - T10)
        dT_dy = (1.0 - dx) * (T10 - T00) + dx * (T11 - T01)

        # Backtracking follows steepest descent.
        norm = math.hypot(dT_dy, dT_dx)
        if norm < 1e-12:
            return T_interp, 0.0, 0.0

        return T_interp, -dT_dy / norm, -dT_dx / norm

    def rk4_step(y: float, x: float, ds: float) -> tuple[float, float, float]:
        """
        One RK4 step for dX/ds = unit_descent(X).
        Returns (y_next, x_next, T_next_at_start_of_step).
        """
        T0, k1y, k1x = bilinear_T_and_unit_descent(y, x)
        if k1y == 0.0 and k1x == 0.0:
            return y, x, T0

        _, k2y, k2x = bilinear_T_and_unit_descent(
            y + 0.5 * ds * k1y, x + 0.5 * ds * k1x
        )
        _, k3y, k3x = bilinear_T_and_unit_descent(
            y + 0.5 * ds * k2y, x + 0.5 * ds * k2x
        )
        _, k4y, k4x = bilinear_T_and_unit_descent(
            y + ds * k3y, x + ds * k3x
        )

        y_next = y + (ds / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        x_next = x + (ds / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        return y_next, x_next, T0

    cy, cx = float(start_y), float(start_x)
    path: list[tuple[float, float]] = [(cy, cx)]

    for _ in range(max_steps):
        iy = int(math.floor(cy))
        ix = int(math.floor(cx))

        if 0 <= iy < ny and 0 <= ix < nx:
            if source_mask[iy, ix]:
                break

        T_current, gy, gx = bilinear_T_and_unit_descent(cy, cx)
        if T_current <= stop_threshold:
            break
        if gy == 0.0 and gx == 0.0:
            break

        # Try to take a step; if it does not decrease T, shrink it.
        ds = step_size
        accepted = False

        for _retry in range(max_step_retries):
            ny_, nx_, T_start = rk4_step(cy, cx, ds)
            if not (np.isfinite(ny_) and np.isfinite(nx_)):
                ds *= 0.5
                continue

            # Reject steps that leave the domain.
            if not (0.0 <= ny_ <= ny - 1.0 and 0.0 <= nx_ <= nx - 1.0):
                ds *= 0.5
                continue

            T_next, _, _ = bilinear_T_and_unit_descent(ny_, nx_)
            if not np.isfinite(T_next):
                ds *= 0.5
                continue

            # Enforce monotone descent in the travel-time field.
            if T_next > T_current - descent_tol:
                ds *= 0.5
                continue

            # Accept the step.
            cy, cx = ny_, nx_
            path.append((cy, cx))
            accepted = True
            break

        if not accepted:
            break

        # Stop if we have effectively entered the source after the move.
        iy = int(math.floor(cy))
        ix = int(math.floor(cx))
        if 0 <= iy < ny and 0 <= ix < nx and source_mask[iy, ix]:
            break

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
