"""Fast Sweeping Method (FSM) travel-time solver and path backtracker.

Solves the advection equation  v · ∇T = 1  using upwind finite differences
with a 4-direction Gauss-Seidel sweep strategy, then traces the fastest path
backward via RK45 integration of  d[x,y]/dτ = -v(x,y).

Grid convention: arrays are (n_y, n_x) — first axis = y (rows), second = x (cols).
"""

from __future__ import annotations

import math
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    def _njit(fn=None, *, cache: bool = False, **kwargs):  # type: ignore[misc]
        if fn is not None:
            return fn
        def decorator(f):
            return f
        return decorator
    _NUMBA_AVAILABLE = False


class FiPyConfigurationError(ValueError):
    """Raised when the FiPy backend cannot represent the requested geometry."""


class FSMSourceConfigurationError(ValueError):
    """Raised when the current FSM solver cannot represent the source layout."""


# ---------------------------------------------------------------------------
# Numba-accelerated inner kernels
# ---------------------------------------------------------------------------

@_njit(cache=True)
def _fsm_residual_kernel(
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    t_field: np.ndarray,
    source_mask: np.ndarray,
) -> np.ndarray:
    n_y, n_x = vx.shape
    residual = np.full((n_y, n_x), np.nan)
    for j in range(n_y):
        for i in range(n_x):
            if source_mask[j, i]:
                continue
            tji = t_field[j, i]
            if not math.isfinite(tji):
                continue
            vx_ = vx[j, i]
            vy_ = vy[j, i]
            if vx_ == 0.0 and vy_ == 0.0:
                continue

            vxp = vx_ if vx_ > 0.0 else 0.0
            vxn = vx_ if vx_ < 0.0 else 0.0
            vyp = vy_ if vy_ > 0.0 else 0.0
            vyn = vy_ if vy_ < 0.0 else 0.0

            t_left  = t_field[j, i - 1] if i > 0       else math.inf
            t_right = t_field[j, i + 1] if i < n_x - 1 else math.inf
            t_down  = t_field[j - 1, i] if j > 0       else math.inf
            t_up    = t_field[j + 1, i] if j < n_y - 1 else math.inf

            term_x  = 0.0
            denom_x = 0.0
            if vxp > 0.0 and math.isfinite(t_left):
                term_x  += vxp * t_left / dx
                denom_x += vxp / dx
            if vxn < 0.0 and math.isfinite(t_right):
                term_x  -= vxn * t_right / dx
                denom_x -= vxn / dx

            term_y  = 0.0
            denom_y = 0.0
            if vyp > 0.0 and math.isfinite(t_down):
                term_y  += vyp * t_down / dy
                denom_y += vyp / dy
            if vyn < 0.0 and math.isfinite(t_up):
                term_y  -= vyn * t_up / dy
                denom_y -= vyn / dy

            denominator = denom_x + denom_y
            if denominator > 0.0:
                residual[j, i] = denominator * tji - term_x - term_y - 1.0

    return residual


@_njit(cache=True)
def _first_order_update(
    T: np.ndarray,
    vx_: float,
    vy_: float,
    dx: float,
    dy: float,
    n_y: int,
    n_x: int,
    j: int,
    i: int,
) -> float:
    # Dimensional reduction: only include a direction when its upwind neighbor
    # already has a finite T. This lets propagation bootstrap from a 1D source
    # (e.g. inlet cut) instead of stalling on inf × velocity.
    denom = 0.0
    rhs   = 1.0

    if vx_ > 0.0 and i > 0:
        T_left = T[j, i - 1]
        if math.isfinite(T_left):
            denom += vx_ / dx
            rhs   += vx_ * T_left / dx
    elif vx_ < 0.0 and i < n_x - 1:
        T_right = T[j, i + 1]
        if math.isfinite(T_right):
            denom += (-vx_) / dx
            rhs   += (-vx_) * T_right / dx

    if vy_ > 0.0 and j > 0:
        T_down = T[j - 1, i]
        if math.isfinite(T_down):
            denom += vy_ / dy
            rhs   += vy_ * T_down / dy
    elif vy_ < 0.0 and j < n_y - 1:
        T_up = T[j + 1, i]
        if math.isfinite(T_up):
            denom += (-vy_) / dy
            rhs   += (-vy_) * T_up / dy

    if denom <= 0.0:
        return math.inf
    return rhs / denom


@_njit(cache=True)
def _second_order_update(
    T: np.ndarray,
    vx_: float,
    vy_: float,
    dx: float,
    dy: float,
    n_y: int,
    n_x: int,
    j: int,
    i: int,
) -> float:
    # Dimensional reduction: when the upwind stencil for a direction is
    # unavailable, drop that direction instead of returning inf so propagation
    # can bootstrap from a 1D source.
    denom       = 0.0
    rhs         = 1.0
    lower_bound = -math.inf
    used_any    = False

    if vx_ > 0.0:
        if i >= 2 and math.isfinite(T[j, i - 1]) and math.isfinite(T[j, i - 2]):
            denom += 3.0 * vx_ / (2.0 * dx)
            rhs   += vx_ * (4.0 * T[j, i - 1] - T[j, i - 2]) / (2.0 * dx)
            if T[j, i - 1] > lower_bound: lower_bound = T[j, i - 1]
            if T[j, i - 2] > lower_bound: lower_bound = T[j, i - 2]
            used_any = True
        elif i >= 1 and math.isfinite(T[j, i - 1]):
            denom += vx_ / dx
            rhs   += vx_ * T[j, i - 1] / dx
            if T[j, i - 1] > lower_bound: lower_bound = T[j, i - 1]
            used_any = True
    elif vx_ < 0.0:
        if i <= n_x - 3 and math.isfinite(T[j, i + 1]) and math.isfinite(T[j, i + 2]):
            denom += 3.0 * (-vx_) / (2.0 * dx)
            rhs   += (-vx_) * (4.0 * T[j, i + 1] - T[j, i + 2]) / (2.0 * dx)
            if T[j, i + 1] > lower_bound: lower_bound = T[j, i + 1]
            if T[j, i + 2] > lower_bound: lower_bound = T[j, i + 2]
            used_any = True
        elif i <= n_x - 2 and math.isfinite(T[j, i + 1]):
            denom += (-vx_) / dx
            rhs   += (-vx_) * T[j, i + 1] / dx
            if T[j, i + 1] > lower_bound: lower_bound = T[j, i + 1]
            used_any = True

    if vy_ > 0.0:
        if j >= 2 and math.isfinite(T[j - 1, i]) and math.isfinite(T[j - 2, i]):
            denom += 3.0 * vy_ / (2.0 * dy)
            rhs   += vy_ * (4.0 * T[j - 1, i] - T[j - 2, i]) / (2.0 * dy)
            if T[j - 1, i] > lower_bound: lower_bound = T[j - 1, i]
            if T[j - 2, i] > lower_bound: lower_bound = T[j - 2, i]
            used_any = True
        elif j >= 1 and math.isfinite(T[j - 1, i]):
            denom += vy_ / dy
            rhs   += vy_ * T[j - 1, i] / dy
            if T[j - 1, i] > lower_bound: lower_bound = T[j - 1, i]
            used_any = True
    elif vy_ < 0.0:
        if j <= n_y - 3 and math.isfinite(T[j + 1, i]) and math.isfinite(T[j + 2, i]):
            denom += 3.0 * (-vy_) / (2.0 * dy)
            rhs   += (-vy_) * (4.0 * T[j + 1, i] - T[j + 2, i]) / (2.0 * dy)
            if T[j + 1, i] > lower_bound: lower_bound = T[j + 1, i]
            if T[j + 2, i] > lower_bound: lower_bound = T[j + 2, i]
            used_any = True
        elif j <= n_y - 2 and math.isfinite(T[j + 1, i]):
            denom += (-vy_) / dy
            rhs   += (-vy_) * T[j + 1, i] / dy
            if T[j + 1, i] > lower_bound: lower_bound = T[j + 1, i]
            used_any = True

    if not used_any or denom <= 0.0:
        return math.inf

    cand = rhs / denom
    if not math.isfinite(cand):
        return math.inf
    if cand + 1e-12 < lower_bound:
        return math.inf
    return cand


@_njit(cache=True)
def _diagonal_update(
    T: np.ndarray,
    vx_: float,
    vy_: float,
    dx: float,
    dy: float,
    n_y: int,
    n_x: int,
    j: int,
    i: int,
    eps: float,
) -> float:
    # diag_directions = [(-1,-1), (-1,1), (1,-1), (1,1)]
    # di = -1 + 2*(k//2), dj = -1 + 2*(k%2)
    best = math.inf
    for k in range(4):
        di = -1 + 2 * (k // 2)
        dj = -1 + 2 * (k  % 2)
        ii = i + di
        jj = j + dj
        if ii < 0 or ii >= n_x or jj < 0 or jj >= n_y:
            continue
        T1 = T[jj, ii]
        if not math.isfinite(T1):
            continue
        h    = math.sqrt((di * dx) ** 2 + (dj * dy) ** 2)
        ux   = (di * dx) / h
        uy   = (dj * dy) / h
        align = -(vx_ * ux + vy_ * uy)
        if align <= eps:
            continue
        cand = T1 + h / align
        if cand < best:
            best = cand
    return best


@_njit(cache=True)
def _second_order_update_no_guard(
    T: np.ndarray,
    vx_: float,
    vy_: float,
    dx: float,
    dy: float,
    n_y: int,
    n_x: int,
    j: int,
    i: int,
) -> float:
    # Dimensional reduction: when the upwind stencil for a direction is
    # unavailable, drop that direction instead of returning inf so propagation
    # can bootstrap from a 1D source.
    denom    = 0.0
    rhs      = 1.0
    used_any = False

    if vx_ > 0.0:
        if i >= 2 and math.isfinite(T[j, i - 1]) and math.isfinite(T[j, i - 2]):
            denom += 3.0 * vx_ / (2.0 * dx)
            rhs   += vx_ * (4.0 * T[j, i - 1] - T[j, i - 2]) / (2.0 * dx)
            used_any = True
        elif i >= 1 and math.isfinite(T[j, i - 1]):
            denom += vx_ / dx
            rhs   += vx_ * T[j, i - 1] / dx
            used_any = True
    elif vx_ < 0.0:
        if i <= n_x - 3 and math.isfinite(T[j, i + 1]) and math.isfinite(T[j, i + 2]):
            denom += 3.0 * (-vx_) / (2.0 * dx)
            rhs   += (-vx_) * (4.0 * T[j, i + 1] - T[j, i + 2]) / (2.0 * dx)
            used_any = True
        elif i <= n_x - 2 and math.isfinite(T[j, i + 1]):
            denom += (-vx_) / dx
            rhs   += (-vx_) * T[j, i + 1] / dx
            used_any = True

    if vy_ > 0.0:
        if j >= 2 and math.isfinite(T[j - 1, i]) and math.isfinite(T[j - 2, i]):
            denom += 3.0 * vy_ / (2.0 * dy)
            rhs   += vy_ * (4.0 * T[j - 1, i] - T[j - 2, i]) / (2.0 * dy)
            used_any = True
        elif j >= 1 and math.isfinite(T[j - 1, i]):
            denom += vy_ / dy
            rhs   += vy_ * T[j - 1, i] / dy
            used_any = True
    elif vy_ < 0.0:
        if j <= n_y - 3 and math.isfinite(T[j + 1, i]) and math.isfinite(T[j + 2, i]):
            denom += 3.0 * (-vy_) / (2.0 * dy)
            rhs   += (-vy_) * (4.0 * T[j + 1, i] - T[j + 2, i]) / (2.0 * dy)
            used_any = True
        elif j <= n_y - 2 and math.isfinite(T[j + 1, i]):
            denom += (-vy_) / dy
            rhs   += (-vy_) * T[j + 1, i] / dy
            used_any = True

    if not used_any or denom <= 0.0:
        return math.inf

    cand = rhs / denom
    if not math.isfinite(cand):
        return math.inf
    return cand


@_njit(cache=True)
def _fsm_sweep_kernel(
    T: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    source_mask: np.ndarray,
    order: int,
    use_diag: bool,
) -> float:
    """Run one full FSM sweep cycle (all 4 directions). Returns max T change."""
    n_y, n_x = vx.shape
    max_change = 0.0
    eps = 1e-14

    # Four Gauss-Seidel sweep directions: (step_j, step_i)
    step_js = (1, -1, -1,  1)
    step_is = (1,  1, -1, -1)

    for sd in range(4):
        step_j = step_js[sd]
        step_i = step_is[sd]

        for jj in range(n_y):
            j = jj if step_j == 1 else n_y - 1 - jj
            for ii in range(n_x):
                i = ii if step_i == 1 else n_x - 1 - ii

                if source_mask[j, i]:
                    continue

                T_old = T[j, i]
                vx_   = vx[j, i]
                vy_   = vy[j, i]

                if vx_ == 0.0 and vy_ == 0.0:
                    continue

                T_new = math.inf

                if order == 2:
                    T_new = _second_order_update(T, vx_, vy_, dx, dy, n_y, n_x, j, i)

                if not math.isfinite(T_new):
                    T_new = _first_order_update(T, vx_, vy_, dx, dy, n_y, n_x, j, i)

                if use_diag and not math.isfinite(T_new):
                    T_new = _diagonal_update(T, vx_, vy_, dx, dy, n_y, n_x, j, i, eps)

                if not math.isfinite(T_new):
                    continue

                if T_new < T_old:
                    if math.isfinite(T_old):
                        change = T_old - T_new
                        if change > max_change:
                            max_change = change
                    else:
                        max_change = math.inf
                    T[j, i] = T_new

    return max_change


@_njit(cache=True)
def _fsm_sweep_kernel_new(
    T: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    pinned: np.ndarray,
    monotone: bool,
    use_second_order: bool,
) -> float:
    """Run one full FSM sweep cycle for the auto-inflow solver."""
    n_y, n_x = vx.shape
    max_change = 0.0

    step_js = (1, -1, -1, 1)
    step_is = (1, 1, -1, -1)

    for sd in range(4):
        step_j = step_js[sd]
        step_i = step_is[sd]

        for jj in range(n_y):
            j = jj if step_j == 1 else n_y - 1 - jj
            for ii in range(n_x):
                i = ii if step_i == 1 else n_x - 1 - ii

                if pinned[j, i]:
                    continue

                vx_ = vx[j, i]
                vy_ = vy[j, i]
                if vx_ == 0.0 and vy_ == 0.0:
                    continue

                T_old = T[j, i]
                if use_second_order:
                    T_new = _second_order_update_no_guard(T, vx_, vy_, dx, dy, n_y, n_x, j, i)
                else:
                    T_new = _first_order_update(T, vx_, vy_, dx, dy, n_y, n_x, j, i)

                if not math.isfinite(T_new):
                    continue

                if monotone:
                    if T_new < T_old:
                        T[j, i] = T_new
                else:
                    T[j, i] = T_new

                if math.isfinite(T_old) and math.isfinite(T[j, i]):
                    change = abs(T[j, i] - T_old)
                    if change > max_change:
                        max_change = change

    return max_change


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    vx          = np.asarray(vx,          dtype=float)
    vy          = np.asarray(vy,          dtype=float)
    t_field     = np.asarray(t_field,     dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)
    return _fsm_residual_kernel(vx, vy, dx, dy, t_field, source_mask)


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


def _pin_inflow_boundary(
    vx: np.ndarray,
    vy: np.ndarray,
    T_inflow: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize travel time by pinning inflow boundary nodes from velocity sign."""
    n_y, n_x = vx.shape
    t_init = np.full((n_y, n_x), np.inf, dtype=float)
    pinned = np.zeros((n_y, n_x), dtype=bool)

    mask = vx[:, 0] > 0.0
    t_init[mask, 0] = T_inflow
    pinned[mask, 0] = True

    mask = vx[:, -1] < 0.0
    t_init[mask, -1] = T_inflow
    pinned[mask, -1] = True

    mask = vy[0, :] > 0.0
    t_init[0, mask] = T_inflow
    pinned[0, mask] = True

    mask = vy[-1, :] < 0.0
    t_init[-1, mask] = T_inflow
    pinned[-1, mask] = True

    return t_init, pinned


def _pin_source_mask(
    source_mask: np.ndarray,
    T_inflow: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize travel time from an explicit pinned source region."""
    source_mask = np.asarray(source_mask, dtype=bool)
    t_init = np.full(source_mask.shape, np.inf, dtype=float)
    t_init[source_mask] = T_inflow
    return t_init, source_mask.copy()


def solve_fsm_travel_time_new(
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    source_mask: np.ndarray | None = None,
    T_inflow: float = 0.0,
    max_sweeps: int = 200,
    tol: float = 1e-10,
    verbose: bool = False,
    second_order: bool = False,
    *,
    use_inlet_cut: bool = False,
) -> np.ndarray:
    """Solve ``v · ∇T = 1`` from cell-centered velocities.

    Parameters
    ----------
    vx, vy : ndarray, shape (n_y, n_x)
        Cell-centered velocity components.
    dx, dy : float
        Grid spacing.
    source_mask : bool ndarray or None, default None
        Explicit pinned region where ``T = T_inflow``. When omitted or empty,
        the solver falls back to auto-detected inflow boundary pinning unless
        ``use_inlet_cut=True``.
    T_inflow : float, default 0.0
        Value imposed on pinned source cells or inflow boundary nodes.
    max_sweeps : int, default 200
        Maximum number of full 4-direction sweep cycles per phase.
    tol : float, default 1e-10
        Sweep convergence tolerance.
    verbose : bool, default False
        When True, print per-sweep progress.
    second_order : bool, default False
        When True, run a second-order non-monotone phase after the first-order
        monotone solve converges.
    use_inlet_cut : bool, default False
        When True, reinterpret ``source_mask`` as a single inlet-cut column,
        solve only on the strictly downwind subdomain, set the cut column to
        ``T_inflow``, and return ``np.inf`` upwind.
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have the same shape.")
    if vx.ndim != 2:
        raise ValueError("vx and vy must be 2D arrays.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be at least 1.")
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")
    if source_mask is not None:
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != vx.shape:
            raise ValueError("source_mask must have the same shape as vx and vy.")

    has_source_mask = source_mask is not None and np.any(source_mask)

    if use_inlet_cut:
        if not has_source_mask:
            raise ValueError("use_inlet_cut=True requires a non-empty source_mask.")
        n_y, n_x = vx.shape
        source_col, downwind_start = _extract_inlet_cut(source_mask)
        full_t = np.full((n_y, n_x), np.inf, dtype=float)
        full_t[:, source_col] = T_inflow

        if downwind_start >= n_x:
            return full_t

        vx_downwind = vx[:, downwind_start:]
        vy_downwind = vy[:, downwind_start:]
        n_y_downwind, n_x_downwind = vx_downwind.shape
        vx_augmented = np.empty((n_y_downwind, n_x_downwind + 1), dtype=float)
        vy_augmented = np.empty((n_y_downwind, n_x_downwind + 1), dtype=float)
        vx_augmented[:, 0] = vx_downwind[:, 0]
        vx_augmented[:, 1:] = vx_downwind
        vy_augmented[:, 0] = vy_downwind[:, 0]
        vy_augmented[:, 1:] = vy_downwind
        source_mask_downwind = np.zeros_like(vx_augmented, dtype=bool)
        source_mask_downwind[:, 0] = True

        t_downwind = solve_fsm_travel_time_new(
            vx_augmented,
            vy_augmented,
            dx,
            dy,
            source_mask=source_mask_downwind,
            T_inflow=T_inflow,
            max_sweeps=max_sweeps,
            tol=tol,
            verbose=verbose,
            second_order=second_order,
            use_inlet_cut=False,
        )
        full_t[:, downwind_start:] = t_downwind[:, 1:]
        return full_t

    if has_source_mask:
        T, pinned = _pin_source_mask(source_mask, T_inflow)
    else:
        T, pinned = _pin_inflow_boundary(vx, vy, T_inflow)

    vx_f = np.ascontiguousarray(vx, dtype=np.float64)
    vy_f = np.ascontiguousarray(vy, dtype=np.float64)
    pinned_b = np.ascontiguousarray(pinned, dtype=np.bool_)

    for sweep in range(max_sweeps):
        change = _fsm_sweep_kernel_new(
            T,
            vx_f,
            vy_f,
            dx,
            dy,
            pinned_b,
            True,
            False,
        )
        if verbose:
            print(
                f"  [1st] sweep {sweep + 1:3d}: max|dT| = {change:.3e}, "
                f"reached {np.isfinite(T).sum()}/{T.size} nodes"
            )
        if sweep > 0 and change < tol and np.isfinite(T).all():
            break

    if not second_order:
        return T

    if verbose:
        print("  --- switching to 2nd-order ---")

    for sweep in range(max_sweeps):
        change = _fsm_sweep_kernel_new(
            T,
            vx_f,
            vy_f,
            dx,
            dy,
            pinned_b,
            False,
            True,
        )
        if verbose:
            print(f"  [2nd] sweep {sweep + 1:3d}: max|dT| = {change:.3e}")
        if sweep > 0 and change < tol:
            break

    return T


def solve_fd_travel_time(
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    source_mask: np.ndarray,
    T_inflow: float = 0.0,
    max_iters: int = 10000,
    tol: float = 1e-6,
    verbose: bool = False,
) -> np.ndarray:
    """Solve ``v · ∇T = 1`` via upwind FD Jacobi iteration with PDE-loss convergence.

    Uses a simple Jacobi iteration: at each step every free cell is updated
    simultaneously from the **previous** iterate using the one-sided upwind
    finite-difference formula::

        T^{k+1}[j,i] = (1 + Σ_upwind |v_d| * T^k[nbr] / h_d)
                        / (Σ_upwind |v_d| / h_d)

    Convergence is declared when the RMS PDE residual
    ``||v · ∇_upwind T − 1||_rms`` (the *loss*) falls below ``tol``.

    **Key differences from the FSM sweep solvers:**

    * **Update order** — all cells are updated from the *old* iterate
      simultaneously (Jacobi), not via multi-directional Gauss-Seidel
      sweeping that exploits characteristic information.
    * **Convergence criterion** — the loss is the discrete PDE residual
      ``||v · ∇T − 1||_rms``, not the maximum change in T between sweeps.

    Parameters
    ----------
    vx, vy : ndarray, shape (n_y, n_x)
        Cell-centered velocity components.
    dx, dy : float
        Grid spacing.
    source_mask : bool ndarray, shape (n_y, n_x)
        Cells pinned to ``T_inflow``.
    T_inflow : float, default 0.0
        Value imposed on pinned source cells.
    max_iters : int, default 10000
        Maximum number of Jacobi iterations.
    tol : float, default 1e-6
        RMS PDE residual convergence tolerance.
    verbose : bool, default False
        When True, print progress every 200 iterations.

    Returns
    -------
    T : ndarray, shape (n_y, n_x)
        Travel-time field.
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)

    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have the same shape.")
    if source_mask.shape != vx.shape:
        raise ValueError("source_mask must have the same shape as vx and vy.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1.")
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")

    n_y, n_x = vx.shape

    T = np.zeros((n_y, n_x), dtype=float)
    T[source_mask] = T_inflow

    free = ~source_mask

    vx_pos = np.maximum(vx, 0.0)   # max(vx, 0) — for backward (upwind) x-diff
    vx_neg = np.minimum(vx, 0.0)   # min(vx, 0) — for forward  (upwind) x-diff
    vy_pos = np.maximum(vy, 0.0)
    vy_neg = np.minimum(vy, 0.0)

    # Cells where at least one upwind neighbor exists — used for residual RMS.
    col_idx = np.arange(n_x)
    row_idx = np.arange(n_y)
    has_upwind = (
        ((vx > 0.0) & (col_idx > 0)[np.newaxis, :])
        | ((vx < 0.0) & (col_idx < n_x - 1)[np.newaxis, :])
        | ((vy > 0.0) & (row_idx > 0)[:, np.newaxis])
        | ((vy < 0.0) & (row_idx < n_y - 1)[:, np.newaxis])
    ) & free

    for it in range(max_iters):
        T_old = T.copy()

        # ---- Jacobi update: all free cells updated from T_old ----
        rhs   = np.ones((n_y, n_x), dtype=float)
        denom = np.zeros((n_y, n_x), dtype=float)

        # vx > 0 → backward diff: upwind neighbor is T_old[:,i-1]
        rhs[:, 1:]   += vx_pos[:, 1:]    * T_old[:, :-1] / dx
        denom[:, 1:] += vx_pos[:, 1:]    / dx
        # vx < 0 → forward diff: upwind neighbor is T_old[:,i+1]
        rhs[:, :-1]  += (-vx_neg[:, :-1]) * T_old[:, 1:]  / dx
        denom[:, :-1]+= (-vx_neg[:, :-1]) / dx
        # vy > 0 → backward diff: upwind neighbor is T_old[j-1,:]
        rhs[1:, :]   += vy_pos[1:, :]    * T_old[:-1, :] / dy
        denom[1:, :] += vy_pos[1:, :]    / dy
        # vy < 0 → forward diff: upwind neighbor is T_old[j+1,:]
        rhs[:-1, :]  += (-vy_neg[:-1, :]) * T_old[1:, :]  / dy
        denom[:-1, :]+= (-vy_neg[:-1, :]) / dy

        update = free & (denom > 0.0)
        T[update] = rhs[update] / denom[update]

        # ---- PDE residual (loss function) using upwind FD on updated T ----
        dTdx = np.zeros((n_y, n_x), dtype=float)
        dTdy = np.zeros((n_y, n_x), dtype=float)

        # For vx > 0 at cells i >= 1: backward diff (T[j,i] - T[j,i-1]) / dx
        m = vx[:, 1:] > 0.0
        dTdx[:, 1:][m]  = (T[:, 1:] - T[:, :-1])[m]  / dx
        # For vx < 0 at cells i <= n_x-2: forward diff (T[j,i+1] - T[j,i]) / dx
        m = vx[:, :-1] < 0.0
        dTdx[:, :-1][m] = (T[:, 1:] - T[:, :-1])[m]  / dx

        # For vy > 0 at cells j >= 1: backward diff
        m = vy[1:, :] > 0.0
        dTdy[1:, :][m]  = (T[1:, :] - T[:-1, :])[m]  / dy
        # For vy < 0 at cells j <= n_y-2: forward diff
        m = vy[:-1, :] < 0.0
        dTdy[:-1, :][m] = (T[1:, :] - T[:-1, :])[m]  / dy

        residual = vx * dTdx + vy * dTdy - 1.0

        n_up = int(has_upwind.sum())
        rms = float(np.sqrt(np.mean(residual[has_upwind] ** 2))) if n_up > 0 else 0.0

        log_interval = max(200, max_iters // 20)
        if verbose and it % log_interval == 0:
            print(f"  [fd] iter {it:6d}/{max_iters}: rms residual = {rms:.3e}", flush=True)

        if rms < tol:
            print(f"  [fd] converged at iter {it}/{max_iters}: rms residual = {rms:.3e} (tol={tol:.3e})", flush=True)
            break

    else:
        print(f"  [fd] stopped at max_iters={max_iters}: rms residual = {rms:.3e} (tol={tol:.3e})", flush=True)

    return T


def compute_fsm_residual_field_new(
    vx: np.ndarray,
    vy: np.ndarray,
    t_field: np.ndarray,
    dx: float,
    dy: float,
    scheme: str = "central",
) -> np.ndarray:
    """Return ``v·∇T - 1`` using either a central or first-order upwind stencil."""
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    t_field = np.asarray(t_field, dtype=float)

    if vx.shape != vy.shape or vx.shape != t_field.shape:
        raise ValueError("vx, vy, and t_field must have the same shape.")
    if vx.ndim != 2:
        raise ValueError("vx, vy, and t_field must be 2D arrays.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")

    n_y, n_x = t_field.shape
    grad_x = np.zeros((n_y, n_x), dtype=float)
    grad_y = np.zeros((n_y, n_x), dtype=float)
    invalid = ~np.isfinite(t_field) | ((vx == 0.0) & (vy == 0.0))

    if scheme == "central":
        grad_x[:, 1:-1] = (t_field[:, 2:] - t_field[:, :-2]) / (2.0 * dx)
        grad_y[1:-1, :] = (t_field[2:, :] - t_field[:-2, :]) / (2.0 * dy)
        boundary = np.zeros((n_y, n_x), dtype=bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        invalid |= boundary
    elif scheme == "upwind1":
        mask = vx[:, 1:] > 0.0
        grad_x[:, 1:][mask] = (t_field[:, 1:] - t_field[:, :-1])[mask] / dx
        mask = vx[:, :-1] < 0.0
        grad_x[:, :-1][mask] = (t_field[:, 1:] - t_field[:, :-1])[mask] / dx
        mask = vy[1:, :] > 0.0
        grad_y[1:, :][mask] = (t_field[1:, :] - t_field[:-1, :])[mask] / dy
        mask = vy[:-1, :] < 0.0
        grad_y[:-1, :][mask] = (t_field[1:, :] - t_field[:-1, :])[mask] / dy

        cols = np.arange(n_x)
        rows = np.arange(n_y)
        no_x = ((vx > 0.0) & (cols == 0)[np.newaxis, :]) | ((vx < 0.0) & (cols == n_x - 1)[np.newaxis, :])
        no_y = ((vy > 0.0) & (rows == 0)[:, np.newaxis]) | ((vy < 0.0) & (rows == n_y - 1)[:, np.newaxis])
        invalid |= no_x | no_y
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    residual = vx * grad_x + vy * grad_y - 1.0
    residual[invalid] = np.nan
    return residual

def _face_to_cell_centered_velocity(
    vx: np.ndarray,
    vy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average MAC face velocities onto cell centers."""
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    if vx.ndim != 2 or vy.ndim != 2:
        raise ValueError("vx and vy must be 2D arrays.")
    if vx.shape[0] + 1 != vy.shape[0] or vx.shape[1] != vy.shape[1] + 1:
        raise ValueError(
            "MAC face velocities must satisfy vx.shape == (ny, nx + 1) and vy.shape == (ny + 1, nx)."
        )
    vx_cell = 0.5 * (vx[:, :-1] + vx[:, 1:])
    vy_cell = 0.5 * (vy[:-1, :] + vy[1:, :])
    return vx_cell, vy_cell


def _extract_inlet_cut(source_mask: np.ndarray) -> tuple[int, int]:
    """Interpret ``source_mask`` as a single inlet cut column.

    Returns
    -------
    source_col : int
        The x-column occupied by the inlet cut.
    downwind_start : int
        First strictly downwind column index (`source_col + 1`).
    """
    source_mask = np.asarray(source_mask, dtype=bool)
    source_indices = np.argwhere(source_mask)
    if source_indices.size == 0:
        raise FSMSourceConfigurationError("source_mask is empty.")

    source_cols = np.unique(source_indices[:, 1])
    if source_cols.size != 1:
        raise FSMSourceConfigurationError(
            "Current FSM solver requires source_mask to occupy exactly one source column."
        )

    source_col = int(source_cols[0])
    return source_col, source_col + 1


def _solve_fsm_travel_time_sweep(
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
    """
    Solve v · ∇T = 1 via fast sweeping and upwind finite differences.

    Order 1:
        Standard monotone first-order upwind update.

    Order 2:
        Coordinate-wise second-order upwind update:
            vx * D_x^{up,2} T + vy * D_y^{up,2} T = 1
        with fallback to first order in any direction where the second
        upwind point is unavailable.

    Notes
    -----
    - This is an advection equation, not an eikonal equation.
    - The second-order scheme is not monotone, so in difficult fields it
      may be less robust than the first-order solve.
    - `neighbors=4` is the proper second-order mode here.
    - `neighbors=8` is retained only as an optional extra first-order
      diagonal probe; the second-order scheme itself is based on the
      4-neighbor coordinate stencil.
    """
    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have the same shape.")
    if source_mask.shape != vx.shape:
        raise ValueError("source_mask must have the same shape as vx and vy.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")

    allowed_modes    = {"update", "mse", "update_or_mse", "update_and_mse"}
    allowed_orders   = {1, 2}
    allowed_neighbors = {4, 8}

    if convergence_mode not in allowed_modes:
        raise ValueError(f"convergence_mode must be one of {sorted(allowed_modes)}.")
    if order not in allowed_orders:
        raise ValueError(f"order must be one of {sorted(allowed_orders)}.")
    if neighbors not in allowed_neighbors:
        raise ValueError(f"neighbors must be one of {sorted(allowed_neighbors)}.")
    if mse_tol is not None and mse_tol < 0.0:
        raise ValueError("mse_tol must be non-negative when provided.")
    if mse_check_every < 1:
        raise ValueError("mse_check_every must be at least 1.")
    if mse_activation_factor <= 0.0:
        raise ValueError("mse_activation_factor must be positive.")
    if consecutive_passes < 1:
        raise ValueError("consecutive_passes must be at least 1.")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be at least 1.")

    n_y, n_x = vx.shape
    T = np.full((n_y, n_x), np.inf, dtype=float)
    T[source_mask] = 0.0

    # Ensure contiguous float64 arrays for the njit kernel.
    vx_f          = np.ascontiguousarray(vx,          dtype=np.float64)
    vy_f          = np.ascontiguousarray(vy,          dtype=np.float64)
    source_mask_b = np.ascontiguousarray(source_mask, dtype=np.bool_)

    use_diag  = neighbors == 8
    pass_count = 0

    for sweep in range(max_sweeps):
        max_change = _fsm_sweep_kernel(
            T, vx_f, vy_f, dx, dy, source_mask_b, order, use_diag
        )

        if sweep == 0:
            continue

        update_ok = max_change < tol

        # If no residual criterion is requested, stop on the update criterion.
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
    use_inlet_cut: bool = False,
    solver: str = "fipy",
) -> np.ndarray:
    """Solve ``v · ∇T = 1`` from MAC-face velocities.

    Parameters
    ----------
    vx : ndarray, shape (ny, nx + 1)
        x-component on vertical MAC faces.
    vy : ndarray, shape (ny + 1, nx)
        y-component on horizontal MAC faces.
    dx, dy : float
        Grid spacing.
    source_mask : bool ndarray, shape (ny, nx)
        Cells where ``T = 0``.
    use_inlet_cut : bool, default False
        When supported by the selected backend, reinterpret ``source_mask`` as
        a single inlet-cut column, solve only on the strictly downwind domain,
        set the cut column to zero, and return ``np.inf`` upwind.
    solver : {"fipy", "sweep", "legacy", "new", "fd"}
        Numerical backend. ``"fipy"`` is the default and falls back to
        ``"sweep"`` automatically when FiPy / PETSc is unavailable.
        ``"legacy"`` routes through ``solve_fsm_travel_time_legacy()`` after
        averaging MAC-face velocities to cell centers. ``"new"`` routes
        through ``solve_fsm_travel_time_new()`` after averaging MAC-face
        velocities to cell centers and supports either explicit pinned
        ``source_mask`` cells or auto-detected inflow boundary pinning when
        ``source_mask`` is empty.  ``"fd"`` routes through
        ``solve_fd_travel_time()``, which uses **ordinary central finite
        differences** (not sign-split upwind) and converges by minimising the
        PDE residual ``||v · ∇T - 1||_rms`` via pseudo-time marching.

    Other parameters are forwarded to the sweep backend when it is selected
    explicitly or used as the fallback.
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)

    if source_mask.ndim != 2:
        raise ValueError("source_mask must be 2D.")
    ny, nx = source_mask.shape
    expected_vx = (ny, nx + 1)
    expected_vy = (ny + 1, nx)
    if vx.shape != expected_vx:
        raise ValueError(f"vx must have shape {expected_vx}, got {vx.shape}.")
    if vy.shape != expected_vy:
        raise ValueError(f"vy must have shape {expected_vy}, got {vy.shape}.")

    allowed_solvers = {"fipy", "sweep", "legacy", "new", "fd"}
    if solver not in allowed_solvers:
        raise ValueError(f"solver must be one of {sorted(allowed_solvers)}.")

    if solver == "fipy":
        try:
            return solve_fipy_travel_time(vx, vy, dx, dy, source_mask)
        except (ImportError, FiPyConfigurationError) as exc:
            warnings.warn(
                f"FiPy backend failed; falling back to the sweep FSM solver. Original error: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )

    vx_cell, vy_cell = _face_to_cell_centered_velocity(vx, vy)
    if solver == "legacy":
        return solve_fsm_travel_time_legacy(
            vx_cell,
            vy_cell,
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
            use_inlet_cut=use_inlet_cut,
        )

    if solver == "new":
        if mse_tol is not None:
            warnings.warn(
                "mse_tol is ignored by solver='new'.",
                RuntimeWarning,
                stacklevel=2,
            )
        if convergence_mode != "update_or_mse":
            warnings.warn(
                "convergence_mode is ignored by solver='new'.",
                RuntimeWarning,
                stacklevel=2,
            )
        if mse_check_every != 5:
            warnings.warn(
                "mse_check_every is ignored by solver='new'.",
                RuntimeWarning,
                stacklevel=2,
            )
        if mse_activation_factor != 10.0:
            warnings.warn(
                "mse_activation_factor is ignored by solver='new'.",
                RuntimeWarning,
                stacklevel=2,
            )
        if consecutive_passes != 1:
            warnings.warn(
                "consecutive_passes is ignored by solver='new'.",
                RuntimeWarning,
                stacklevel=2,
            )
        if neighbors != 4:
            raise ValueError("solver='new' only supports neighbors=4.")
        if order not in {1, 2}:
            raise ValueError("solver='new' only supports order=1 or order=2.")
        return solve_fsm_travel_time_new(
            vx_cell,
            vy_cell,
            dx,
            dy,
            source_mask=source_mask,
            max_sweeps=max_sweeps,
            tol=tol,
            second_order=(order == 2),
            use_inlet_cut=use_inlet_cut,
        )

    if solver == "fd":
        if use_inlet_cut:
            source_col, downwind_start = _extract_inlet_cut(source_mask)
            full_t = np.full((ny, nx), np.inf, dtype=float)
            full_t[:, source_col] = 0.0
            if downwind_start >= nx:
                return full_t
            vx_dw = vx_cell[:, downwind_start:]
            vy_dw = vy_cell[:, downwind_start:]
            ny_dw, nx_dw = vx_dw.shape
            vx_aug = np.empty((ny_dw, nx_dw + 1), dtype=float)
            vy_aug = np.empty((ny_dw, nx_dw + 1), dtype=float)
            vx_aug[:, 0] = vx_dw[:, 0]
            vx_aug[:, 1:] = vx_dw
            vy_aug[:, 0] = vy_dw[:, 0]
            vy_aug[:, 1:] = vy_dw
            sm_dw = np.zeros((ny_dw, nx_dw + 1), dtype=bool)
            sm_dw[:, 0] = True
            t_dw = solve_fd_travel_time(
                vx_aug, vy_aug, dx, dy, sm_dw,
                T_inflow=0.0, max_iters=max_sweeps * 50, tol=tol, verbose=True,
            )
            full_t[:, downwind_start:] = t_dw[:, 1:]
            return full_t
        return solve_fd_travel_time(
            vx_cell,
            vy_cell,
            dx,
            dy,
            source_mask=source_mask,
            T_inflow=0.0,
            max_iters=max_sweeps * 50,
            tol=tol,
            verbose=True,
        )

    if use_inlet_cut:
        source_col, downwind_start = _extract_inlet_cut(source_mask)
        full_t = np.full((ny, nx), np.inf, dtype=float)
        full_t[:, source_col] = 0.0

        if downwind_start >= nx:
            return full_t

        vx_downwind = vx_cell[:, downwind_start:]
        vy_downwind = vy_cell[:, downwind_start:]
        ny_downwind, nx_downwind = vx_downwind.shape
        vx_augmented = np.empty((ny_downwind, nx_downwind + 1), dtype=float)
        vy_augmented = np.empty((ny_downwind, nx_downwind + 1), dtype=float)
        vx_augmented[:, 0] = vx_downwind[:, 0]
        vx_augmented[:, 1:] = vx_downwind
        vy_augmented[:, 0] = vy_downwind[:, 0]
        vy_augmented[:, 1:] = vy_downwind
        source_mask_downwind = np.zeros_like(vx_augmented, dtype=bool)
        source_mask_downwind[:, 0] = True

        t_downwind = _solve_fsm_travel_time_sweep(
            vx_augmented,
            vy_augmented,
            dx,
            dy,
            source_mask_downwind,
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
        full_t[:, downwind_start:] = t_downwind[:, 1:]
        return full_t

    return _solve_fsm_travel_time_sweep(
        vx_cell,
        vy_cell,
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
    *,
    use_inlet_cut: bool = True,
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
    use_inlet_cut : bool, default False
        When True, reinterpret ``source_mask`` as an inlet cut on a single
        source column, solve only on the strictly downwind domain, set the
        source column to zero, and return ``np.inf`` upwind.

    Returns
    -------
    T : ndarray, shape (n_y, n_x)
        Travel-time field; np.inf where unreachable.
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)

    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have the same shape.")
    if source_mask.shape != vx.shape:
        raise ValueError("source_mask must have the same shape as vx and vy.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")

    n_y, n_x = vx.shape
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

    if use_inlet_cut:
        source_col, downwind_start = _extract_inlet_cut(source_mask)
        full_t = np.full((n_y, n_x), np.inf, dtype=float)
        full_t[:, source_col] = 0.0

        if downwind_start >= n_x:
            return full_t

        vx_downwind = vx[:, downwind_start:]
        vy_downwind = vy[:, downwind_start:]
        n_y_downwind, n_x_downwind = vx_downwind.shape
        vx_augmented = np.empty((n_y_downwind, n_x_downwind + 1), dtype=float)
        vy_augmented = np.empty((n_y_downwind, n_x_downwind + 1), dtype=float)
        vx_augmented[:, 0] = vx_downwind[:, 0]
        vx_augmented[:, 1:] = vx_downwind
        vy_augmented[:, 0] = vy_downwind[:, 0]
        vy_augmented[:, 1:] = vy_downwind
        source_mask_downwind = np.zeros_like(vx_augmented, dtype=bool)
        source_mask_downwind[:, 0] = True

        t_downwind = solve_fsm_travel_time_legacy(
            vx_augmented,
            vy_augmented,
            dx,
            dy,
            source_mask_downwind,
            max_sweeps=max_sweeps,
            tol=tol,
            mse_tol=mse_tol,
            convergence_mode=convergence_mode,
            mse_check_every=mse_check_every,
            mse_activation_factor=mse_activation_factor,
            consecutive_passes=consecutive_passes,
            use_inlet_cut=False,
        )
        full_t[:, downwind_start:] = t_downwind[:, 1:]
        return full_t

    T = np.full((n_y, n_x), np.inf)
    T[source_mask] = 0.0

    vx_pos = np.maximum(vx, 0.0)
    vx_neg = np.minimum(vx, 0.0)
    vy_pos = np.maximum(vy, 0.0)
    vy_neg = np.minimum(vy, 0.0)
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
    neighbors: int = 4,
    use_inlet_cut: bool = False,
    solver: str = "fipy",
    dt: float = 0.5,
    max_steps: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the FSM travel-time field and fastest path via RK45 backtracking.

    Parameters
    ----------
    vx : ndarray, shape (ny, nx + 1)
        x-component on vertical MAC faces.
    vy : ndarray, shape (ny + 1, nx)
        y-component on horizontal MAC faces.
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
        Sweep-backend update order passed to ``solve_fsm_travel_time``.
    neighbors : {4, 8}
        Sweep-backend stencil width passed to ``solve_fsm_travel_time``.
    use_inlet_cut : bool, default False
        Reinterpret ``source_mask`` as a single inlet-cut column when supported
        by the selected backend.
    solver : {"fipy", "sweep"}
        Travel-time backend passed to ``solve_fsm_travel_time``.
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

    vx_cell, vy_cell = _face_to_cell_centered_velocity(vx, vy)
    traversable = np.isfinite(vx_cell) & np.isfinite(vy_cell)
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
        use_inlet_cut=use_inlet_cut,
        solver=solver,
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
        vx_cell, vy_cell, t_field,
        x_coords, y_coords,
        dt=dt,
        max_steps=max_steps,
        stop_x=stop_x,
    )
    return path_xy, t_field, arrival_time

def _pack_mac_velocity_to_fipy_face_variable(
    mesh,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
):
    """
    Pack MAC-grid face velocities into a FiPy rank-1 FaceVariable.

    Assumed input layout (numpy convention, y first):
        vx.shape == (ny, nx + 1)  # x-velocity on vertical faces
        vy.shape == (ny + 1, nx)  # y-velocity on horizontal faces

    If your arrays are transposed, swap the indexing used below.
    """
    from fipy import FaceVariable

    fcx, fcy = np.asarray(mesh.faceCenters)
    n_faces = mesh.numberOfFaces

    # On a Cartesian grid with origin at (0, 0):
    # vertical faces:   x/dx is integer, y/dy is half-integer
    # horizontal faces: x/dx is half-integer, y/dy is integer
    x_over_dx = fcx / dx
    y_over_dy = fcy / dy

    vertical = np.isclose(np.mod(x_over_dx, 1.0), 0.0) & np.isclose(np.mod(y_over_dy, 1.0), 0.5)
    horizontal = np.isclose(np.mod(x_over_dx, 1.0), 0.5) & np.isclose(np.mod(y_over_dy, 1.0), 0.0)

    if int(vertical.sum()) + int(horizontal.sum()) != n_faces:
        raise RuntimeError("Could not classify all faces as vertical or horizontal.")

    vel = np.zeros((2, n_faces), dtype=float)

    # Vertical faces -> x-component from vx[y, x_face]
    xv = np.rint(x_over_dx[vertical]).astype(int)
    yv = np.rint(y_over_dy[vertical] - 0.5).astype(int)
    if np.any(xv < 0) or np.any(xv >= vx.shape[1]) or np.any(yv < 0) or np.any(yv >= vx.shape[0]):
        raise ValueError("vx does not match the grid size implied by source_mask and dy/dx.")
    vel[0, vertical] = vx[yv, xv]

    # Horizontal faces -> y-component from vy[y_face, x]
    xh = np.rint(x_over_dx[horizontal] - 0.5).astype(int)
    yh = np.rint(y_over_dy[horizontal]).astype(int)
    if np.any(xh < 0) or np.any(xh >= vy.shape[1]) or np.any(yh < 0) or np.any(yh >= vy.shape[0]):
        raise ValueError("vy does not match the grid size implied by source_mask and dy/dx.")
    vel[1, horizontal] = vy[yh, xh]

    velocity = FaceVariable(mesh=mesh, rank=1, value=0.0)
    velocity.setValue(vel)
    return velocity, vel


def solve_fipy_travel_time(
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
    source_mask: np.ndarray,
):
    """
    Solve v · ∇T = 1 on a uniform 2D grid with FiPy + PETSc.

    Parameters
    ----------
    vx : ndarray, shape (ny, nx + 1)
        x-component of velocity on vertical faces.
    vy : ndarray, shape (ny + 1, nx)
        y-component of velocity on horizontal faces.
    dx, dy : float
        Grid spacing.
    source_mask : bool ndarray, shape (ny, nx)
        Release region. The FiPy backend interprets this as an inlet cut:
        the source column is assigned ``T = 0``, the solve is performed only
        on the strictly downwind subdomain, and upwind cells are returned as
        ``np.inf``.

    Returns
    -------
    T : ndarray, shape (ny, nx)
        Cell-centered solution.
    """
    try:
        from fipy import CellVariable, Grid2D, ImplicitSourceTerm, UpwindConvectionTerm
        from fipy.solvers.petsc import LinearLUSolver
    except Exception as exc:
        raise ImportError(
            "FiPy with PETSc is required. Install FiPy and petsc4py in the same environment."
        ) from exc

    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)

    if source_mask.ndim != 2:
        raise ValueError("source_mask must be 2D.")
    ny, nx = source_mask.shape

    expected_vx = (ny, nx + 1)
    expected_vy = (ny + 1, nx)
    if vx.shape != expected_vx:
        raise ValueError(f"vx must have shape {expected_vx}, got {vx.shape}.")
    if vy.shape != expected_vy:
        raise ValueError(f"vy must have shape {expected_vy}, got {vy.shape}.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")

    source_indices = np.argwhere(source_mask)
    if source_indices.size == 0:
        raise FiPyConfigurationError("source_mask is empty.")

    source_cols = np.unique(source_indices[:, 1])
    if source_cols.size != 1:
        raise FiPyConfigurationError(
            "FiPy backend requires source_mask to occupy exactly one source column."
        )

    source_col = int(source_cols[0])
    full_t = np.full((ny, nx), np.inf, dtype=float)
    full_t[:, source_col] = 0.0

    downwind_start = source_col + 1
    if downwind_start >= nx:
        return full_t

    vx_sub = vx[:, downwind_start:]
    vy_sub = vy[:, downwind_start:]
    nx_sub = vx_sub.shape[1] - 1

    mesh = Grid2D(dx=dx, dy=dy, nx=nx_sub, ny=ny)

    # Cell-centered unknown on the strictly downwind domain.
    T = CellVariable(mesh=mesh, name="T", value=0.0)
    T.constrain(0.0, where=mesh.facesLeft)

    # Pack MAC face velocities into a FiPy face vector coefficient.
    velocity, velocity_values = _pack_mac_velocity_to_fipy_face_variable(
        mesh, vx_sub, vy_sub, dx, dy
    )

    face_normals = np.asarray(mesh.faceNormals)
    normal_flux = (
        np.asarray(velocity_values[0], dtype=float) * np.asarray(face_normals[0], dtype=float)
        + np.asarray(velocity_values[1], dtype=float) * np.asarray(face_normals[1], dtype=float)
    )
    exterior_mask = np.asarray(mesh.exteriorFaces.value, dtype=bool)
    inlet_mask = np.asarray(mesh.facesLeft.value, dtype=bool)
    outflow_mask = exterior_mask & (~inlet_mask) & (normal_flux > 1e-14)
    if np.any(outflow_mask):
        T.faceGrad.constrain(((0.0,), (0.0,)), where=outflow_mask)

    # Conservative form:
    #   ∇·(v T) - (∇·v) T = 1
    # UpwindConvectionTerm approximates ∇·(v T).
    eq = UpwindConvectionTerm(coeff=velocity) - ImplicitSourceTerm(coeff=velocity.divergence) == 1.0

    # PETSc direct solve.
    solver = LinearLUSolver()
    eq.solve(var=T, solver=solver)

    t_sub = np.asarray(T.value, dtype=float).reshape((ny, nx_sub))
    full_t[:, downwind_start:] = t_sub
    return full_t
