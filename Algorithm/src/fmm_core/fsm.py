"""Fast Sweeping Method (FSM) travel-time solver and path backtracker.

Solves the advection equation  v · ∇T = 1  using upwind finite differences
with a 4-direction Gauss-Seidel sweep strategy, then traces the fastest path
backward via RK45 integration of  d[x,y]/dτ = -v(x,y).

Grid convention: arrays are (n_y, n_x) — first axis = y (rows), second = x (cols).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator


def compute_fsm_residual_field(
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    t_field: np.ndarray,
    source_mask: np.ndarray,
) -> np.ndarray:
    """Return the discrete FSM PDE residual field v·∇T - 1.

    The residual uses the same sign-split upwind discretization as the FSM
    update step so it is directly compatible with the solver's numerical
    scheme. Cells where the residual is not meaningful are returned as NaN.
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    t_field = np.asarray(t_field, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)

    n_y, n_x = vx.shape
    residual = np.full((n_y, n_x), np.nan, dtype=float)

    vx_pos = np.maximum(vx, 0.0)
    vx_neg = np.minimum(vx, 0.0)
    vy_pos = np.maximum(vy, 0.0)
    vy_neg = np.minimum(vy, 0.0)

    for j in range(n_y):
        for i in range(n_x):
            if source_mask[j, i] or not np.isfinite(t_field[j, i]):
                continue
            if vx[j, i] == 0.0 and vy[j, i] == 0.0:
                continue

            t_left = t_field[j, i - 1] if i > 0 else np.inf
            t_right = t_field[j, i + 1] if i < n_x - 1 else np.inf
            t_down = t_field[j - 1, i] if j > 0 else np.inf
            t_up = t_field[j + 1, i] if j < n_y - 1 else np.inf

            term_x = 0.0
            denom_x = 0.0
            if vx_pos[j, i] > 0.0 and np.isfinite(t_left):
                term_x += vx_pos[j, i] * t_left / dx
                denom_x += vx_pos[j, i] / dx
            if vx_neg[j, i] < 0.0 and np.isfinite(t_right):
                term_x -= vx_neg[j, i] * t_right / dx
                denom_x -= vx_neg[j, i] / dx

            term_y = 0.0
            denom_y = 0.0
            if vy_pos[j, i] > 0.0 and np.isfinite(t_down):
                term_y += vy_pos[j, i] * t_down / dy
                denom_y += vy_pos[j, i] / dy
            if vy_neg[j, i] < 0.0 and np.isfinite(t_up):
                term_y -= vy_neg[j, i] * t_up / dy
                denom_y -= vy_neg[j, i] / dy

            denominator = denom_x + denom_y
            if denominator <= 0.0:
                continue

            residual[j, i] = denominator * t_field[j, i] - term_x - term_y - 1.0

    return residual


def compute_fsm_residual_mse(
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    t_field: np.ndarray,
    source_mask: np.ndarray,
) -> float:
    """Return the mean squared discrete residual of v·∇T - 1."""
    residual = compute_fsm_residual_field(vx, vy, dx, dy, t_field, source_mask)
    valid = np.isfinite(residual)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(residual[valid] ** 2))

import numpy as np


def solve_fsm_travel_time(
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    source_mask: np.ndarray,
    max_sweeps: int = 100,
    tol: float = 1e-6,
    mse_tol: float | None = None,
    convergence_mode: str = "update_or_mse",
    mse_check_every: int = 5,
    mse_activation_factor: float = 10.0,
    consecutive_passes: int = 1,
    order: int = 1,
    neighbors: int = 4,
) -> np.ndarray:
    """Solve v·∇T = 1 via upwind finite differences + fast sweeping.

    The default configuration uses the original first-order monotone
    4-neighbor update. Wider 8-neighbor updates and second-order
    extrapolation can be enabled independently where supported.

    Arrays follow numpy convention: shape (n_y, n_x), i.e. first axis = y
    (rows), second axis = x (cols).

    Parameters
    ----------
    vx, vy : ndarray, shape (n_y, n_x)
        x- and y-components of the velocity field (cell-centred).
    dx, dy : float
        Grid spacing in x and y.
    source_mask : bool ndarray, shape (n_y, n_x)
        Cells where T = 0 (the starting boundary).
    max_sweeps : int
        Maximum number of 4-direction sweep cycles before giving up.
    tol : float
        Convergence tolerance on the maximum per-sweep change in T.
    mse_tol : float or None
        Optional mean-squared residual tolerance on the discrete PDE residual
        ``v·∇T - 1``. When None, only the update-based criterion is used.
    convergence_mode : {"update", "mse", "update_or_mse", "update_and_mse"}
        Convergence rule when ``mse_tol`` is provided.
    mse_check_every : int
        Evaluate the residual-MSE criterion at most every N sweeps when MSE
        checking is active.
    mse_activation_factor : float
        Start checking residual MSE only after ``max_change`` falls below
        ``mse_activation_factor * tol`` for modes that also use update
        screening.
    consecutive_passes : int
        Require the selected convergence rule to hold for this many consecutive
        eligible checks before returning.
    order : {1, 2}
        Local update order. ``1`` uses first-order causal propagation.
        ``2`` enables second-order extrapolation where supported.
    neighbors : {4, 8}
        Local stencil width. ``4`` uses only axis-aligned neighbors.
        ``8`` also enables diagonal ray updates.

    Returns
    -------
    T : ndarray, shape (n_y, n_x)
        Travel-time field; np.inf where unreachable.
    """
    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have the same shape.")
    if source_mask.shape != vx.shape:
        raise ValueError("source_mask must have the same shape as vx and vy.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")

    n_y, n_x = vx.shape

    T = np.full((n_y, n_x), np.inf, dtype=float)
    T[source_mask] = 0.0

    allowed_modes = {"update", "mse", "update_or_mse", "update_and_mse"}
    allowed_orders = {1, 2}
    allowed_neighbors = {4, 8}
    if convergence_mode not in allowed_modes:
        raise ValueError(f"convergence_mode must be one of {sorted(allowed_modes)}.")
    if order not in allowed_orders:
        raise ValueError(f"order must be one of {sorted(allowed_orders)}.")
    if neighbors not in allowed_neighbors:
        raise ValueError(f"neighbors must be one of {sorted(allowed_neighbors)}.")
    if order == 2 and neighbors == 4:
        raise ValueError("order=2 with neighbors=4 is not supported.")
    if mse_tol is not None and mse_tol < 0.0:
        raise ValueError("mse_tol must be non-negative when provided.")
    if mse_check_every < 1:
        raise ValueError("mse_check_every must be at least 1.")
    if mse_activation_factor <= 0.0:
        raise ValueError("mse_activation_factor must be positive.")
    if consecutive_passes < 1:
        raise ValueError("consecutive_passes must be at least 1.")

    # Split velocities for the original monotone upwind update.
    vx_pos = np.maximum(vx, 0.0)
    vx_neg = np.minimum(vx, 0.0)
    vy_pos = np.maximum(vy, 0.0)
    vy_neg = np.minimum(vy, 0.0)

    pass_count = 0
    eps = 1e-14

    # 4 sweep directions: (step_j along y-axis, step_i along x-axis)
    sweep_directions = [
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ]

    # Eight ray directions for the wider stencil.
    # Each entry is (di, dj, step_length).
    ray_directions = [
        (-1,  0, dx),
        ( 1,  0, dx),
        ( 0, -1, dy),
        ( 0,  1, dy),
        (-1, -1, float(np.hypot(dx, dy))),
        (-1,  1, float(np.hypot(dx, dy))),
        ( 1, -1, float(np.hypot(dx, dy))),
        ( 1,  1, float(np.hypot(dx, dy))),
    ]

    def _first_order_monotone_candidate(j: int, i: int) -> float:
        """Original local monotone candidate from upwind FD."""
        vxp = vx_pos[j, i]
        vxn = vx_neg[j, i]
        vyp = vy_pos[j, i]
        vyn = vy_neg[j, i]

        abs_vx = abs(vx[j, i])
        abs_vy = abs(vy[j, i])
        if abs_vx == 0.0 and abs_vy == 0.0:
            return np.inf

        T_left = T[j, i - 1] if i > 0 else np.inf
        T_right = T[j, i + 1] if i < n_x - 1 else np.inf
        T_down = T[j - 1, i] if j > 0 else np.inf
        T_up = T[j + 1, i] if j < n_y - 1 else np.inf

        term_x = 0.0
        denom_x = 0.0
        if vxp > 0.0 and np.isfinite(T_left):
            term_x += vxp * T_left / dx
            denom_x += vxp / dx
        if vxn < 0.0 and np.isfinite(T_right):
            term_x -= vxn * T_right / dx
            denom_x -= vxn / dx

        term_y = 0.0
        denom_y = 0.0
        if vyp > 0.0 and np.isfinite(T_down):
            term_y += vyp * T_down / dy
            denom_y += vyp / dy
        if vyn < 0.0 and np.isfinite(T_up):
            term_y -= vyn * T_up / dy
            denom_y -= vyn / dy

        denominator = denom_x + denom_y
        if denominator <= 0.0:
            return np.inf

        return (1.0 + term_x + term_y) / denominator

    def _wide_stencil_candidate(j: int, i: int) -> float:
        """Wide-stencil candidate using causal rays with optional 2nd order."""
        vx_ = vx[j, i]
        vy_ = vy[j, i]
        speed = float(np.hypot(vx_, vy_))
        if speed == 0.0:
            return np.inf

        best = np.inf

        for di, dj, h in ray_directions:
            ii1 = i + di
            jj1 = j + dj
            if ii1 < 0 or ii1 >= n_x or jj1 < 0 or jj1 >= n_y:
                continue

            T1 = T[jj1, ii1]
            if not np.isfinite(T1):
                continue

            # Unit vector from current cell to the neighbor.
            ux = (di * dx) / h
            uy = (dj * dy) / h

            # Velocity component in the direction from neighbor -> current.
            align = -(vx_ * ux + vy_ * uy)
            if align <= eps:
                continue

            # First-order causal propagation along this ray.
            ray_cand = np.inf
            cand1 = T1 + h / align
            if np.isfinite(cand1):
                ray_cand = cand1

            if order == 2:
                # Second-order upwind (BDF2) along the same ray
                ii2 = i + 2 * di
                jj2 = j + 2 * dj
                if 0 <= ii2 < n_x and 0 <= jj2 < n_y:
                    T2 = T[jj2, ii2]
                    if np.isfinite(T2) and T2 < T1:
                        # Correct PDE-informed BDF2 formula
                        cand2 = (4.0 * T1 - T2 + 2.0 * (h / align)) / 3.0
                        
                        # If valid and causal, it overrides cand1 for THIS ray
                        if np.isfinite(cand2) and cand2 >= T1:
                            ray_cand = cand2

            # Finally, compare this ray's best candidate against the global best
            if ray_cand < best:
                best = ray_cand

        return best

    for sweep in range(max_sweeps):
        max_change = 0.0

        for step_j, step_i in sweep_directions:
            j_range = range(n_y) if step_j == 1 else range(n_y - 1, -1, -1)
            i_range = range(n_x) if step_i == 1 else range(n_x - 1, -1, -1)

            for j in j_range:
                for i in i_range:
                    if source_mask[j, i]:
                        continue

                    T_old = T[j, i]
                    if not np.isfinite(T_old):
                        T_old = np.inf

                    cand1 = _first_order_monotone_candidate(j, i)
                    if neighbors == 8:
                        cand2 = _wide_stencil_candidate(j, i)
                        T_new = min(cand1, cand2)
                    else:
                        T_new = cand1

                    if not np.isfinite(T_new):
                        continue

                    if T_new < T[j, i]:
                        if np.isfinite(T[j, i]):
                            change = T[j, i] - T_new
                            if change > max_change:
                                max_change = change
                        else:
                            # Inf -> finite is a real update, so prevent premature convergence.
                            max_change = np.inf
                        T[j, i] = T_new

        if sweep == 0:
            continue

        update_ok = max_change < tol

        # If no residual criterion is requested, use update-based stopping.
        if mse_tol is None:
            if update_ok:
                pass_count += 1
                if pass_count >= consecutive_passes:
                    return T
            else:
                pass_count = 0
            continue

        if convergence_mode == "update":
            converged = update_ok

        elif convergence_mode == "mse":
            if (sweep % mse_check_every) != 0:
                pass_count = 0
                continue
            mse_value = compute_fsm_residual_mse(vx, vy, dx, dy, T, source_mask)
            converged = np.isfinite(mse_value) and mse_value < mse_tol

        elif convergence_mode == "update_or_mse":
            if update_ok:
                pass_count += 1
                if pass_count >= consecutive_passes:
                    return T
                continue

            if not (max_change < (mse_activation_factor * tol) and (sweep % mse_check_every) == 0):
                pass_count = 0
                continue

            mse_value = compute_fsm_residual_mse(vx, vy, dx, dy, T, source_mask)
            converged = np.isfinite(mse_value) and mse_value < mse_tol

        else:  # update_and_mse
            if not update_ok:
                pass_count = 0
                continue
            if (sweep % mse_check_every) != 0:
                pass_count = 0
                continue

            mse_value = compute_fsm_residual_mse(vx, vy, dx, dy, T, source_mask)
            converged = np.isfinite(mse_value) and mse_value < mse_tol and update_ok

        if converged:
            pass_count += 1
            if pass_count >= consecutive_passes:
                return T
        else:
            pass_count = 0

    return T

def solve_fsm_travel_time_legacy(
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    source_mask: np.ndarray,
    max_sweeps: int = 100,
    tol: float = 1e-6,
    mse_tol: float | None = None,
    convergence_mode: str = "update_or_mse",
    mse_check_every: int = 5,
    mse_activation_factor: float = 10.0,
    consecutive_passes: int = 1,
) -> np.ndarray:
    """Solve v·∇T = 1 via upwind finite differences + fast sweeping.

    Uses the 4-direction Gauss-Seidel sweep strategy. Arrays follow numpy
    convention: shape (n_y, n_x), i.e. first axis = y (rows), second = x (cols).

    Parameters
    ----------
    vx, vy : ndarray, shape (n_y, n_x)
        x- and y-components of the velocity field (cell-centred).
    dx, dy : float
        Grid spacing in x and y.
    source_mask : bool ndarray, shape (n_y, n_x)
        Cells where T = 0 (the starting boundary).
    max_sweeps : int
        Maximum number of 4-direction sweep cycles before giving up.
    tol : float
        Convergence tolerance on the maximum per-sweep change in T.
    mse_tol : float or None
        Optional mean-squared residual tolerance on the discrete PDE residual
        ``v·∇T - 1``. When None, only the update-based criterion is used.
    convergence_mode : {"update", "mse", "update_or_mse", "update_and_mse"}
        Convergence rule when ``mse_tol`` is provided.
    mse_check_every : int
        Evaluate the residual-MSE criterion at most every N sweeps when MSE
        checking is active.
    mse_activation_factor : float
        Start checking residual MSE only after ``max_change`` falls below
        ``mse_activation_factor * tol`` for modes that also use update
        screening.
    consecutive_passes : int
        Require the selected convergence rule to hold for this many consecutive
        eligible checks before returning.

    Returns
    -------
    T : ndarray, shape (n_y, n_x)
        Travel-time field; np.inf where unreachable.
    """
    n_y, n_x = vx.shape

    T = np.full((n_y, n_x), np.inf)
    T[source_mask] = 0.0

    vx_pos = np.maximum(vx, 0.0)
    vx_neg = np.minimum(vx, 0.0)
    vy_pos = np.maximum(vy, 0.0)
    vy_neg = np.minimum(vy, 0.0)
    allowed_modes = {"update", "mse", "update_or_mse", "update_and_mse"}
    if convergence_mode not in allowed_modes:
        raise ValueError(f"convergence_mode must be one of {sorted(allowed_modes)}.")
    if mse_tol is not None and mse_tol < 0.0:
        raise ValueError("mse_tol must be non-negative when provided.")
    if mse_check_every < 1:
        raise ValueError("mse_check_every must be at least 1.")
    if mse_activation_factor <= 0.0:
        raise ValueError("mse_activation_factor must be positive.")
    if consecutive_passes < 1:
        raise ValueError("consecutive_passes must be at least 1.")
    pass_count = 0

    # 4 sweep directions: (step_j along y-axis, step_i along x-axis)
    sweep_directions = [
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ]

    for sweep in range(max_sweeps):
        max_change = 0.0

        for step_j, step_i in sweep_directions:
            j_range = range(n_y) if step_j == 1 else range(n_y - 1, -1, -1)
            i_range = range(n_x) if step_i == 1 else range(n_x - 1, -1, -1)

            for j in j_range:
                for i in i_range:
                    if source_mask[j, i]:
                        continue

                    vxp = vx_pos[j, i]
                    vxn = vx_neg[j, i]
                    vyp = vy_pos[j, i]
                    vyn = vy_neg[j, i]
                    abs_vx = abs(vx[j, i])
                    abs_vy = abs(vy[j, i])

                    if abs_vx == 0.0 and abs_vy == 0.0:
                        continue  # stagnation point — T stays inf

                    T_left  = T[j, i - 1] if i > 0       else np.inf
                    T_right = T[j, i + 1] if i < n_x - 1 else np.inf
                    T_down  = T[j - 1, i] if j > 0       else np.inf
                    T_up    = T[j + 1, i] if j < n_y - 1 else np.inf

                    # Dimensional reduction: only include a direction when its
                    # upwind neighbour already has a finite T.  This prevents
                    # inf × velocity from blocking propagation.
                    term_x = 0.0
                    denom_x = 0.0
                    if vxp > 0.0 and T_left != np.inf:
                        term_x += vxp * T_left / dx
                        denom_x += vxp / dx
                    if vxn < 0.0 and T_right != np.inf:
                        term_x -= vxn * T_right / dx
                        denom_x -= vxn / dx

                    term_y = 0.0
                    denom_y = 0.0
                    if vyp > 0.0 and T_down != np.inf:
                        term_y += vyp * T_down / dy
                        denom_y += vyp / dy
                    if vyn < 0.0 and T_up != np.inf:
                        term_y -= vyn * T_up / dy
                        denom_y -= vyn / dy

                    denominator = denom_x + denom_y
                    if denominator <= 0.0:
                        continue

                    T_new = (1.0 + term_x + term_y) / denominator

                    if T_new < T[j, i]:
                        if T[j, i] != np.inf:
                            change = T[j, i] - T_new
                            if change > max_change:
                                max_change = change
                        T[j, i] = T_new

        if sweep <= 0:
            continue

        update_ok = max_change < tol
        if mse_tol is None:
            if update_ok:
                pass_count += 1
                if pass_count >= consecutive_passes:
                    return T
            else:
                pass_count = 0
            continue

        if convergence_mode == "update":
            converged = update_ok
        else:
            should_check_mse = False
            if convergence_mode == "mse":
                should_check_mse = (sweep % mse_check_every) == 0
            elif convergence_mode == "update_or_mse":
                if update_ok:
                    converged = True
                    pass_count += 1
                    if pass_count >= consecutive_passes:
                        return T
                    continue
                should_check_mse = (
                    max_change < (mse_activation_factor * tol)
                    and (sweep % mse_check_every) == 0
                )
            else:  # update_and_mse
                if not update_ok:
                    pass_count = 0
                    continue
                should_check_mse = (sweep % mse_check_every) == 0

            if not should_check_mse:
                pass_count = 0
                continue

            mse_value = compute_fsm_residual_mse(vx, vy, dx, dy, T, source_mask)
            mse_ok = np.isfinite(mse_value) and mse_value < mse_tol

            if convergence_mode == "mse":
                converged = mse_ok
            else:
                converged = update_ok and mse_ok if convergence_mode == "update_and_mse" else mse_ok

        if converged:
            pass_count += 1
            if pass_count >= consecutive_passes:
                return T
        else:
            pass_count = 0

    return T


def backtrack_path_rk45(
    target_xy: tuple[float, float],
    vx: np.ndarray,
    vy: np.ndarray,
    t_field: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    dt: float = 0.5,
    max_steps: int = 10_000,
    stop_x: float | None = None,
) -> np.ndarray:
    """Trace the fastest path backward from target to source by integrating -v.

    Uses scipy RK45 with adaptive step size and terminal event detection.
    The ODE is d[x,y]/dτ = [-vx(x,y), -vy(x,y)].

    Parameters
    ----------
    target_xy : (x, y) float tuple
        Continuous coordinates of the starting (target) point.
    vx, vy : ndarray, shape (n_y, n_x)
        Velocity field (cell-centred). First axis = y, second = x.
    t_field : ndarray, shape (n_y, n_x)
        Travel-time field from solve_fsm_travel_time.
    x_coords, y_coords : 1D ndarrays
        Grid coordinate arrays (length n_x and n_y respectively).
    dt : float
        Maximum RK45 step size (in the same units as the grid coordinates).
        The integrator may use smaller steps; this is an upper bound only.
    max_steps : int
        Maximum pseudotime span = max_steps * dt.
    stop_x : float or None
        Terminal x-boundary: stop when x_val <= stop_x (source column).
        Prevents the path from overshooting into the invalid T region left of
        the source.

    Returns
    -------
    path : ndarray, shape (N, 2)
        Path in (x, y) order, going from source to target (reversed backtrack).
    """
    interp_kwargs = dict(method="linear", bounds_error=False, fill_value=None)
    axes = (y_coords, x_coords)
    interp_vx = RegularGridInterpolator(axes, vx, **interp_kwargs)
    interp_vy = RegularGridInterpolator(axes, vy, **interp_kwargs)
    t_finite = np.where(np.isfinite(t_field), t_field, 0.0)
    interp_T  = RegularGridInterpolator(axes, t_finite, **interp_kwargs)

    def rhs(tau: float, state: np.ndarray) -> list[float]:
        x, y = float(state[0]), float(state[1])
        return [-float(interp_vx((y, x))), -float(interp_vy((y, x)))]

    def event_t_zero(tau: float, state: np.ndarray) -> float:
        return float(interp_T((float(state[1]), float(state[0])))) - 1e-3
    event_t_zero.terminal  = True   # type: ignore[attr-defined]
    event_t_zero.direction = -1     # type: ignore[attr-defined]  # fire only when T decreasing

    events: list = [event_t_zero]
    if stop_x is not None:
        def event_source_x(tau: float, state: np.ndarray) -> float:
            return state[0] - stop_x
        event_source_x.terminal  = True   # type: ignore[attr-defined]
        event_source_x.direction = -1     # type: ignore[attr-defined]  # fire only when x decreasing
        events.append(event_source_x)

    sol = solve_ivp(
        rhs,
        t_span=[0.0, max_steps * dt],
        y0=[float(target_xy[0]), float(target_xy[1])],
        method="RK45",
        events=events,
        max_step=dt,
        rtol=1e-4,
        atol=1e-6,
        dense_output=False,
    )

    # sol.y shape: (2, N) — reverse so path goes source → target
    path_arr = sol.y.T[::-1]
    return path_arr


def compute_fsm_path(
    vx: np.ndarray,
    vy: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    dx: float | None = None,
    dy: float | None = None,
    max_sweeps: int = 100,
    tol: float = 1e-6,
    mse_tol: float | None = None,
    convergence_mode: str = "update_or_mse",
    mse_check_every: int = 5,
    mse_activation_factor: float = 10.0,
    consecutive_passes: int = 1,
    order: int = 1,
    neighbors: int = 8,
    dt: float = 0.5,
    max_steps: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the FSM travel-time field and fastest path via RK45 backtracking.

    Parameters
    ----------
    vx, vy : ndarray, shape (n_y, n_x)
        Cell-centred Darcy velocity components.
    source_mask : bool ndarray, shape (n_y, n_x)
        Cells where T = 0 (flow origin).
    target_mask : bool ndarray, shape (n_y, n_x)
        Cells that define the arrival region.
    x_coords, y_coords : 1D ndarrays
        Cell-centre coordinate arrays.
    dx, dy : float or None
        Grid spacing; inferred from coordinate arrays if not provided.
    max_sweeps, tol : int, float
        FSM solver parameters (see solve_fsm_travel_time).
    mse_tol, convergence_mode
        Optional residual-MSE convergence controls (see solve_fsm_travel_time).
    mse_check_every, mse_activation_factor, consecutive_passes
        Efficiency/robustness controls for residual-MSE checking.
    order : {1, 2}
        Travel-time update order passed to ``solve_fsm_travel_time``.
    neighbors : {4, 8}
        Travel-time stencil width passed to ``solve_fsm_travel_time``.
    dt : float
        RK45 max step size (same units as coordinates).
    max_steps : int
        Max RK45 integration steps.

    Returns
    -------
    path_xy : ndarray, shape (N, 2)
        Path in (x, y) order, source → target.
    t_field : ndarray, shape (n_y, n_x)
        Travel-time field (np.inf where unreachable).
    arrival_time : float
        Minimum finite T value at any target cell.
    """
    if dx is None:
        dx = float(x_coords[1] - x_coords[0])
    if dy is None:
        dy = float(y_coords[1] - y_coords[0])

    traversable = np.isfinite(vx) & np.isfinite(vy)
    source_mask = source_mask & traversable
    target_mask = target_mask & traversable

    if not np.any(source_mask):
        raise RuntimeError("FSM: no traversable source cells.")
    if not np.any(target_mask):
        raise RuntimeError("FSM: no traversable target cells.")

    t_field = solve_fsm_travel_time(
        vx,
        vy,
        dx,
        dy,
        source_mask,
        max_sweeps=max_sweeps,
        tol=tol,
        mse_tol=mse_tol,
        convergence_mode=convergence_mode,
        mse_check_every=mse_check_every,
        mse_activation_factor=mse_activation_factor,
        consecutive_passes=consecutive_passes,
        order=order,
        neighbors=neighbors,
    )

    valid_target = target_mask & np.isfinite(t_field)
    if not np.any(valid_target):
        raise RuntimeError("FSM: no target cell reached a finite travel time.")

    target_indices = np.argwhere(valid_target)
    target_vals    = t_field[valid_target]
    best_pos       = int(np.argmin(target_vals))
    best_j, best_i = int(target_indices[best_pos, 0]), int(target_indices[best_pos, 1])
    arrival_time   = float(target_vals[best_pos])

    target_x_cont = float(x_coords[best_i])
    target_y_cont = float(y_coords[best_j])

    src_cols = np.argwhere(source_mask)[:, 1]
    stop_x = float(x_coords[src_cols.min()]) if src_cols.size > 0 else None

    path_xy = backtrack_path_rk45(
        (target_x_cont, target_y_cont),
        vx, vy, t_field,
        x_coords, y_coords,
        dt=dt,
        max_steps=max_steps,
        stop_x=stop_x,
    )
    return path_xy, t_field, arrival_time
