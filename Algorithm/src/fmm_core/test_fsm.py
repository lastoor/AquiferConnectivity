import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "./FlowSimulation"))

"""
Solve  v(x) . grad T(x) = 1  on a regular 2D grid.

Method: first-order upwind finite differences + Gauss-Seidel sweeping
in the 4 coordinate-sweep directions (fast sweeping method, FSM).
For an acyclic streamline graph this converges in O(1) sweeps;
with recirculation it still converges because the upwind operator is an M-matrix.

Inflow BC: T = 0 on the inflow boundary (v . n < 0).
Outflow BC: none (scheme determines T there).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator


# ----------------------------------------------------------------------
# Core solver
# ----------------------------------------------------------------------
def solve_travel_time(vx, vy, dx, dy, T_inflow=0.0,
                      max_sweeps=200, tol=1e-10, verbose=False,
                      second_order=False):
    """
    Solve v . grad T = 1 by upwind finite differences + fast sweeping.

    Inflow boundary condition T = T_inflow is applied automatically
    wherever v . n < 0 on the domain boundary.  No condition is imposed
    on outflow portions of the boundary -- the upwind stencil
    naturally degenerates to a one-sided update there.

    Parameters
    ----------
    vx, vy : (ny, nx) arrays, velocity components at grid nodes.
    dx, dy : grid spacing.
    T_inflow : value on inflow boundary (default 0).
    max_sweeps, tol : iteration controls.
    second_order : if True, run a two-phase solve: first-order FSM to
        convergence (establishes a valid field), then second-order
        Gauss-Seidel sweeps without the monotone-decrease guard (which
        would reject correct second-order updates that are larger than
        the first-order solution).  Falls back to first-order at nodes
        where only one upstream cell is available.

    Returns
    -------
    T : (ny, nx) array.  np.inf on nodes unreachable from inflow.
    """
    ny, nx = vx.shape
    vxp = np.maximum(vx, 0.0);  vxm = np.minimum(vx, 0.0)
    vyp = np.maximum(vy, 0.0);  vym = np.minimum(vy, 0.0)

    T = np.full((ny, nx), np.inf)
    pinned = np.zeros((ny, nx), dtype=bool)

    # Pin inflow boundary nodes (v . n < 0 on the outward normal).
    # Left edge (outward normal = -x-hat): inflow where vx > 0.
    mask = vx[:, 0] > 0;      T[mask, 0]    = T_inflow; pinned[mask, 0] = True
    mask = vx[:, -1] < 0;     T[mask, -1]   = T_inflow; pinned[mask, -1] = True
    mask = vy[0, :] > 0;      T[0, mask]    = T_inflow; pinned[0, mask] = True
    mask = vy[-1, :] < 0;     T[-1, mask]   = T_inflow; pinned[-1, mask] = True

    sweeps = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    def _sweep(T, use_second_order, monotone):
        """One full set of 4 Gauss-Seidel sweeps.  Returns max|dT|."""
        T_old = T.copy()
        for si, sj in sweeps:
            i_iter = range(ny) if si == 1 else range(ny - 1, -1, -1)
            j_iter = range(nx) if sj == 1 else range(nx - 1, -1, -1)
            for i in i_iter:
                for j in j_iter:
                    if pinned[i, j]:
                        continue
                    coeff = 0.0
                    rhs = 1.0
                    if vxp[i, j] > 0 and j > 0 and np.isfinite(T[i, j-1]):
                        if use_second_order and j > 1 and np.isfinite(T[i, j-2]):
                            coeff += 1.5 * vxp[i, j] / dx
                            rhs += (vxp[i, j] / dx) * (2.0*T[i, j-1] - 0.5*T[i, j-2])
                        else:
                            c = vxp[i, j] / dx;  coeff += c;  rhs += c * T[i, j-1]
                    if vxm[i, j] < 0 and j < nx - 1 and np.isfinite(T[i, j+1]):
                        if use_second_order and j < nx - 2 and np.isfinite(T[i, j+2]):
                            coeff += 1.5 * (-vxm[i, j]) / dx
                            rhs += ((-vxm[i, j]) / dx) * (2.0*T[i, j+1] - 0.5*T[i, j+2])
                        else:
                            c = -vxm[i, j] / dx;  coeff += c;  rhs += c * T[i, j+1]
                    if vyp[i, j] > 0 and i > 0 and np.isfinite(T[i-1, j]):
                        if use_second_order and i > 1 and np.isfinite(T[i-2, j]):
                            coeff += 1.5 * vyp[i, j] / dy
                            rhs += (vyp[i, j] / dy) * (2.0*T[i-1, j] - 0.5*T[i-2, j])
                        else:
                            c = vyp[i, j] / dy;  coeff += c;  rhs += c * T[i-1, j]
                    if vym[i, j] < 0 and i < ny - 1 and np.isfinite(T[i+1, j]):
                        if use_second_order and i < ny - 2 and np.isfinite(T[i+2, j]):
                            coeff += 1.5 * (-vym[i, j]) / dy
                            rhs += ((-vym[i, j]) / dy) * (2.0*T[i+1, j] - 0.5*T[i+2, j])
                        else:
                            c = -vym[i, j] / dy;  coeff += c;  rhs += c * T[i+1, j]
                    if coeff > 0:
                        T_new = rhs / coeff
                        if monotone:
                            if T_new < T[i, j]:
                                T[i, j] = T_new
                        else:
                            T[i, j] = T_new
        finite = np.isfinite(T) & np.isfinite(T_old)
        return np.max(np.abs(T[finite] - T_old[finite])) if finite.any() else np.inf

    # Phase 1: first-order FSM with monotone-decrease guard.
    for k in range(max_sweeps):
        change = _sweep(T, use_second_order=False, monotone=True)
        if verbose:
            print(f"  [1st] sweep {k+1:3d}: max|dT| = {change:.3e}, "
                  f"reached {np.isfinite(T).sum()}/{T.size} nodes")
        if k > 0 and change < tol and np.isfinite(T).all():
            break

    if not second_order:
        return T

    # Phase 2: second-order Gauss-Seidel without monotone guard,
    # starting from the converged first-order field.
    if verbose:
        print(f"  --- switching to 2nd-order ---")
    for k in range(max_sweeps):
        change = _sweep(T, use_second_order=True, monotone=False)
        if verbose:
            print(f"  [2nd] sweep {k+1:3d}: max|dT| = {change:.3e}")
        if k > 0 and change < tol:
            break

    return T


def pde_residual(vx, vy, T, dx, dy, scheme="central"):
    """
    Compute r = v·∇T - 1 using a chosen gradient stencil.

    scheme="upwind1": first-order one-sided upwind (matches the first-order
        solver — yields ≈0 at converged first-order T but is BIASED when
        used to score a second-order solution: it then reports O(h)
        truncation error of the residual stencil itself, not the true PDE
        error.  Use only for verifying first-order convergence.
    scheme="central": second-order central differences.  Unbiased — O(h²)
        accurate regardless of which solver produced T, so it is the right
        metric for comparing schemes against the continuous PDE.

    Returns NaN where T is unreached, velocity is zero, or the stencil is
    unavailable at the boundary.
    """
    ny, nx = T.shape
    grad_x = np.zeros((ny, nx))
    grad_y = np.zeros((ny, nx))
    invalid = ~np.isfinite(T) | ((vx == 0) & (vy == 0))

    if scheme == "central":
        grad_x[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * dx)
        grad_y[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dy)
        # central diff unavailable in the first/last row/column
        bnd = np.zeros((ny, nx), bool)
        bnd[[0, -1], :] = True
        bnd[:, [0, -1]] = True
        invalid |= bnd
    elif scheme == "upwind1":
        m = vx[:, 1:] > 0
        grad_x[:, 1:][m] = (T[:, 1:] - T[:, :-1])[m] / dx
        m = vx[:, :-1] < 0
        grad_x[:, :-1][m] = (T[:, 1:] - T[:, :-1])[m] / dx
        m = vy[1:, :] > 0
        grad_y[1:, :][m] = (T[1:, :] - T[:-1, :])[m] / dy
        m = vy[:-1, :] < 0
        grad_y[:-1, :][m] = (T[1:, :] - T[:-1, :])[m] / dy
        cols = np.arange(nx);  rows = np.arange(ny)
        no_x = ((vx > 0) & (cols == 0)[np.newaxis, :]) | \
                ((vx < 0) & (cols == nx - 1)[np.newaxis, :])
        no_y = ((vy > 0) & (rows == 0)[:, np.newaxis]) | \
                ((vy < 0) & (rows == ny - 1)[:, np.newaxis])
        invalid |= no_x | no_y
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    res = vx * grad_x + vy * grad_y - 1.0
    res[invalid] = np.nan
    return res


# ----------------------------------------------------------------------
# Reference: Lagrangian particle tracking
# ----------------------------------------------------------------------
def lagrangian_arrival_times(vx, vy, x, y, seed_points, t_max=1e3):
    """Integrate dx/dt = v(x) from each seed; return arrival time at end."""
    vx_interp = RegularGridInterpolator((y, x), vx, bounds_error=False,
                                         fill_value=0.0)
    vy_interp = RegularGridInterpolator((y, x), vy, bounds_error=False,
                                         fill_value=0.0)

    def rhs(t, p):
        return [vx_interp((p[1], p[0]))[0], vy_interp((p[1], p[0]))[0]]

    results = []
    for (x0, y0, xf, yf) in seed_points:
        def hit_target(t, p, xf=xf, yf=yf):
            return (p[0] - xf)**2 + (p[1] - yf)**2 - (0.5 * min(np.diff(x)[0],
                                                                np.diff(y)[0]))**2
        hit_target.terminal = True
        hit_target.direction = -1
        sol = solve_ivp(rhs, (0, t_max), [x0, y0],
                        events=hit_target, rtol=1e-8, atol=1e-10,
                        max_step=0.05)
        if sol.t_events[0].size > 0:
            results.append(sol.t_events[0][0])
        else:
            results.append(np.nan)
    return np.array(results)


# ----------------------------------------------------------------------
# Test 1: uniform flow, T has analytic solution
# ----------------------------------------------------------------------
def test_uniform_flow():
    print("=" * 60)
    print("TEST 1: uniform flow v = (1, 0),  analytic T(x,y) = x")
    print("=" * 60)

    errs = []
    hs = []
    for n in [33, 65, 129, 257]:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        dx = x[1] - x[0]; dy = y[1] - y[0]
        X, Y = np.meshgrid(x, y)
        vx = np.ones_like(X)
        vy = np.zeros_like(X)

        T = solve_travel_time(vx, vy, dx, dy, verbose=False)
        T_exact = X
        err = np.max(np.abs(T - T_exact))
        errs.append(err); hs.append(dx)
        print(f"  n = {n:4d},  h = {dx:.4e},  max|T-T_exact| = {err:.3e}")

    # Estimate convergence rate
    rates = np.log(np.array(errs[:-1])/np.array(errs[1:])) / \
            np.log(np.array(hs[:-1])/np.array(hs[1:]))
    print(f"  observed convergence rates: {rates}")
    return errs, hs


# ----------------------------------------------------------------------
# Test 2: radial flow from a source, analytic T = r^2 / 2
# ----------------------------------------------------------------------
def test_radial_flow():
    print("=" * 60)
    print("TEST 2: radial flow v = (x, y),  analytic T(x,y) = ln(r)  (r >= r0)")
    print("        Using v = (x, y)/r^2 so that dT/ds = 1 along rays  =>  T = r")
    print("=" * 60)
    # With v = (x,y)/r, |v|=1, streamlines are rays, T = r - r0.

    errs = []; hs = []
    for n in [65, 129, 257]:
        L = 1.0
        x = np.linspace(-L, L, n)
        y = np.linspace(-L, L, n)
        dx = x[1] - x[0]; dy = y[1] - y[0]
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        eps = 0.05
        Rsafe = np.maximum(R, eps)
        vx = X / Rsafe
        vy = Y / Rsafe

        # Inflow is... there is no inflow boundary here (all characteristics
        # emanate from the origin). Instead pin T on a small disc r < r0.
        # We handle this by a custom boundary: set T = r on the disc r < 2*eps.

        T_exact = R.copy()
        T = np.full_like(R, np.inf)
        disc = R < 3 * eps
        T[disc] = R[disc]  # Dirichlet BC on interior disc

        # Run solver but with custom pinned set
        T = _solve_with_custom_bc(vx, vy, dx, dy, T, pinned=disc)

        ring = (R > 0.3) & (R < 0.9)
        err = np.max(np.abs(T[ring] - T_exact[ring]))
        errs.append(err); hs.append(dx)
        print(f"  n = {n:4d},  h = {dx:.4e},  max|T-T_exact| on 0.3<r<0.9 = {err:.3e}")

    rates = np.log(np.array(errs[:-1])/np.array(errs[1:])) / \
            np.log(np.array(hs[:-1])/np.array(hs[1:]))
    print(f"  observed convergence rates: {rates}")
    return errs, hs


def _solve_with_custom_bc(vx, vy, dx, dy, T_init, pinned,
                          max_sweeps=200, tol=1e-9):
    """Same as solve_travel_time but with an arbitrary pinned mask."""
    ny, nx = vx.shape
    vxp = np.maximum(vx, 0.0);  vxm = np.minimum(vx, 0.0)
    vyp = np.maximum(vy, 0.0);  vym = np.minimum(vy, 0.0)
    ax = np.abs(vx) / dx; ay = np.abs(vy) / dy
    diag = ax + ay

    T = T_init.copy()
    sweeps = [(slice(None, None,  1), slice(None, None,  1)),
              (slice(None, None,  1), slice(None, None, -1)),
              (slice(None, None, -1), slice(None, None,  1)),
              (slice(None, None, -1), slice(None, None, -1))]

    for k in range(max_sweeps):
        T_old = T.copy()
        for si, sj in sweeps:
            for i in range(ny)[si]:
                for j in range(nx)[sj]:
                    if pinned[i, j] or diag[i, j] == 0.0:
                        continue
                    rhs = 1.0
                    if vxp[i, j] > 0 and j > 0:
                        rhs += vxp[i, j] / dx * T[i, j-1]
                    if vxm[i, j] < 0 and j < nx - 1:
                        rhs += (-vxm[i, j]) / dx * T[i, j+1]
                    if vyp[i, j] > 0 and i > 0:
                        rhs += vyp[i, j] / dy * T[i-1, j]
                    if vym[i, j] < 0 and i < ny - 1:
                        rhs += (-vym[i, j]) / dy * T[i+1, j]
                    T_new = rhs / diag[i, j]
                    if np.isfinite(T_new) and T_new < T[i, j]:
                        T[i, j] = T_new
        finite = np.isfinite(T) & np.isfinite(T_old)
        if finite.any() and np.max(np.abs(T[finite] - T_old[finite])) < tol:
            break
    return T


# ----------------------------------------------------------------------
# Test 3: heterogeneous Darcy-like field, cross-check vs particle tracking
# ----------------------------------------------------------------------
def test_heterogeneous():
    print("=" * 60)
    print("TEST 3: heterogeneous v-field, Eulerian vs Lagrangian arrival times")
    print("=" * 60)
    n = 201
    L = 1.0
    x = np.linspace(0, L, n); y = np.linspace(0, L, n)
    dx = x[1] - x[0]; dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # Mean flow in +x direction plus a smooth perturbation.
    rng = np.random.default_rng(0)
    # Build a smooth random stream function psi, then v = (d psi/dy, -d psi/dx)
    # ensures incompressibility.
    kx = np.fft.fftfreq(n, dx) * 2 * np.pi
    ky = np.fft.fftfreq(n, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    filt = np.exp(-K2 * 0.01)  # smooth
    noise = rng.standard_normal((n, n))
    psi_hat = np.fft.fft2(noise) * filt
    psi = np.real(np.fft.ifft2(psi_hat))
    psi *= 0.05 / np.std(psi)  # perturbation strength
    dpsi_dy, dpsi_dx = np.gradient(psi, dy, dx)
    vx = 1.0 + dpsi_dy
    vy = -dpsi_dx

    T = solve_travel_time(vx, vy, dx, dy, verbose=True)

    # Pick some target points, compare arrival times with Lagrangian tracking.
    seeds = []
    targets = [(0.5, 0.25), (0.5, 0.5), (0.5, 0.75), (0.8, 0.4), (0.8, 0.6)]
    # Streamlines start at x=0 (inflow). Launch seed with y=target_y and
    # integrate forward; then compare arrival time at target.
    # (For this perturbed field y drifts slightly; we use event-based stopping
    # on reaching the target x-coordinate.)
    vx_i = RegularGridInterpolator((y, x), vx, bounds_error=False, fill_value=1.0)
    vy_i = RegularGridInterpolator((y, x), vy, bounds_error=False, fill_value=0.0)

    print(f"\n  target (x,y)       T_Eulerian      T_Lagrangian   rel.diff")
    for (xt, yt) in targets:
        def rhs(t, p):
            return [float(vx_i((p[1], p[0]))), float(vy_i((p[1], p[0])))]
        def hit(t, p, xt=xt):
            return p[0] - xt
        hit.terminal = True; hit.direction = 1
        sol = solve_ivp(rhs, (0, 5.0), [1e-6, yt], events=hit,
                        rtol=1e-9, atol=1e-11, max_step=0.01)
        if sol.t_events[0].size == 0:
            continue
        t_lag = sol.t_events[0][0]
        # Eulerian T at (xt, y_at_hit)
        y_hit = sol.y_events[0][0][1]
        T_interp = RegularGridInterpolator((y, x), T)
        t_eul = float(T_interp((y_hit, xt)))
        rel = abs(t_lag - t_eul) / t_lag
        print(f"   ({xt:.2f}, {yt:.2f})      {t_eul:.6f}      {t_lag:.6f}    {rel:.2e}")

    return T, vx, vy


# ----------------------------------------------------------------------
# Test 4: heterogeneous Darcy flow from fBm k-field
# ----------------------------------------------------------------------
def test_darcy_flow(field_size=200, seed=0, compute_residual=True):
    """
    Generate a heterogeneous k-field via fBm, solve the Darcy head equation
    with PETSc, compute cell-centred velocity, then run the FSM solver.
    """
    print("=" * 60)
    print("TEST 4: heterogeneous Darcy flow (fBm k-field + PETSc head solve)")
    print("=" * 60)

    from simulation.grf_generation import fBmGenerator, compute_c
    from simulation.hydraulic import solve_head, compute_velocity

    hurst = 0.5
    c = compute_c(field_size, hurst, std_target=1.0)
    fbm_gen = fBmGenerator(c=c, alpha=hurst * 2, num_pt=field_size,
                            dim=2, minpadding=field_size)
    rng_state = np.random.get_state()
    np.random.seed(seed)
    logk = fbm_gen.sample()
    np.random.set_state(rng_state)
    K = np.exp(logk)

    dx = dy = 1.0
    x = np.arange(field_size, dtype=float) * dx
    y = np.arange(field_size, dtype=float) * dy

    h, msg, ok = solve_head(K, x, y, dh=0.01, rtol=1e-10, atol=1e-10)
    print(f"  Head solve: {msg}  success={ok}")
    if not ok:
        print("  WARNING: head solve did not converge; skipping FSM test.")
        return None, None, None

    vx, vy = compute_velocity(K, x, y, h, output_location="cell",
                               align_with_modflow=False)

    def _run_and_report(label, so):
        T_ = solve_travel_time(vx, vy, dx, dy, verbose=True, second_order=so)
        n = np.isfinite(T_).sum()
        print(f"\n  [{label}] {n}/{T_.size} nodes reached, "
              f"T range [{np.nanmin(T_):.3f}, {np.nanmax(T_):.3f}]")
        if compute_residual:
            for s in ("upwind1", "central"):
                res = pde_residual(vx, vy, T_, dx, dy, scheme=s)
                valid = np.isfinite(res)
                abs_r = np.abs(res[valid])
                rel_r = abs_r / np.abs(res[valid] + 1.0)
                print(f"  [{label}/{s:>7}] max|r|={abs_r.max():.3e}  "
                      f"mean|r|={abs_r.mean():.3e}  |  "
                      f"max|r/v·∇T|={rel_r.max():.3e}  "
                      f"mean|r/v·∇T|={rel_r.mean():.3e}")
        return T_

    T1 = _run_and_report("1st-order", False)
    print()
    T2 = _run_and_report("2nd-order", True)
    return T2, vx, vy


if __name__ == "__main__":
    test_uniform_flow()
    print()
    test_radial_flow()
    print()
    T, vx, vy = test_heterogeneous()
    print()
    test_darcy_flow()