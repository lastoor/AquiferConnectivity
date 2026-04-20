import numpy as np


def compute_gradients(T, dx=1.0, dy=1.0):
    """
    Return Tx, Ty, n = |grad T|.
    """
    Ty, Tx = np.gradient(T, dy, dx, edge_order=2)
    n = np.sqrt(Tx**2 + Ty**2)
    return Tx, Ty, n


def nonlinear_eikonal_residual(T, n_true, dx=1.0, dy=1.0):
    """
    Residual of the full nonlinear eikonal equation:
        |grad T| - n_true
    """
    _, _, n = compute_gradients(T, dx=dx, dy=dy)
    return n - n_true


def solve_linearized_deltaT_2d(
    T_bg,
    delta_n,
    dx=1.0,
    dy=1.0,
    source_mask=None,
    max_sweeps=200,
    tol=1e-8,
    verbose=False,
):
    """
    Solve the linearized equation

        grad(T_bg) · grad(deltaT) = n_bg * delta_n,
        n_bg = |grad(T_bg)|

    by repeated upwind Gauss-Seidel / fast sweeping.
    """
    T_bg = np.asarray(T_bg, dtype=float)
    delta_n = np.asarray(delta_n, dtype=float)

    if T_bg.shape != delta_n.shape:
        raise ValueError("T_bg and delta_n must have the same shape.")

    ny, nx = T_bg.shape
    Tx, Ty, n_bg = compute_gradients(T_bg, dx=dx, dy=dy)
    rhs = n_bg * delta_n

    deltaT = np.zeros_like(T_bg)

    if source_mask is None:
        source_mask = (T_bg == np.nanmin(T_bg))
    else:
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != T_bg.shape:
            raise ValueError("source_mask must have same shape as T_bg.")

    bad = (
        ~np.isfinite(T_bg)
        | ~np.isfinite(delta_n)
        | ~np.isfinite(Tx)
        | ~np.isfinite(Ty)
        | ~np.isfinite(rhs)
    )
    fixed = source_mask | bad
    deltaT[fixed] = 0.0

    eps = 1e-14

    sweep_orders = [
        (range(1, ny - 1), range(1, nx - 1)),
        (range(1, ny - 1), range(nx - 2, 0, -1)),
        (range(ny - 2, 0, -1), range(1, nx - 1)),
        (range(ny - 2, 0, -1), range(nx - 2, 0, -1)),
    ]

    for sweep in range(max_sweeps):
        old = deltaT.copy()

        for iy_range, ix_range in sweep_orders:
            for i in iy_range:
                for j in ix_range:
                    if fixed[i, j]:
                        continue

                    ax = Tx[i, j]
                    ay = Ty[i, j]

                    # Upwind neighbors according to sign of background transport field
                    ux = deltaT[i, j - 1] if ax >= 0.0 else deltaT[i, j + 1]
                    uy = deltaT[i - 1, j] if ay >= 0.0 else deltaT[i + 1, j]

                    alphax = abs(ax) / dx
                    alphay = abs(ay) / dy
                    denom = alphax + alphay

                    if denom < eps:
                        continue

                    deltaT[i, j] = (rhs[i, j] + alphax * ux + alphay * uy) / denom

        err = np.nanmax(np.abs(deltaT - old))
        if verbose:
            print(f"  inner sweep {sweep+1:4d}: max update = {err:.3e}")
        if err < tol:
            break

    return deltaT


def iterative_linearized_eikonal_2d(
    T_init,
    n_true,
    dx=1.0,
    dy=1.0,
    source_mask=None,
    outer_max=20,
    inner_max_sweeps=200,
    inner_tol=1e-8,
    eikonal_tol=1e-6,
    damping=1.0,
    max_deltaT=None,
    monotone_residual=True,
    verbose=False,
):
    """
    Iterative re-linearization for the nonlinear eikonal equation.

    Parameters
    ----------
    T_init : ndarray
        Initial travel time field, e.g. from FMM.
    n_true : ndarray
        Target slowness field.
    dx, dy : float
        Grid spacings.
    source_mask : bool ndarray or None
        Fixed points where deltaT = 0 and T is not updated.
    outer_max : int
        Maximum number of outer re-linearization iterations.
    inner_max_sweeps : int
        Max sweeps for each linearized solve.
    inner_tol : float
        Inner linearized-solver tolerance.
    eikonal_tol : float
        Stop when || |grad T| - n_true ||_inf < eikonal_tol.
    damping : float
        Update factor in (0, 1]. Smaller helps stability for large corrections.
    max_deltaT : float or None
        If not None, clip each outer-step correction pointwise:
            deltaT <- clip(deltaT, -max_deltaT, +max_deltaT)
    monotone_residual : bool
        If True, backtrack damping when nonlinear residual worsens.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    T : ndarray
        Final travel time field.
    history : list of dict
        Iteration history.
    """
    T = np.array(T_init, dtype=float, copy=True)
    n_true = np.asarray(n_true, dtype=float)

    if T.shape != n_true.shape:
        raise ValueError("T_init and n_true must have the same shape.")

    if source_mask is None:
        source_mask = (T == np.nanmin(T))
    else:
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != T.shape:
            raise ValueError("source_mask must have same shape as T_init.")

    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must satisfy 0 < damping <= 1")

    history = []
    iterates = [T.copy()]

    # Initial nonlinear residual
    res0 = nonlinear_eikonal_residual(T, n_true, dx=dx, dy=dy)
    res0_inf = np.nanmax(np.abs(res0))

    if verbose:
        print(f"initial nonlinear residual inf-norm = {res0_inf:.3e}")

    for k in range(outer_max):
        # Re-linearize around current T
        Tx, Ty, n_bg = compute_gradients(T, dx=dx, dy=dy)
        delta_n = n_true - n_bg

        dn_inf = np.nanmax(np.abs(delta_n))
        if dn_inf < eikonal_tol:
            if verbose:
                print(f"stopping: slowness mismatch below tolerance at iter {k}")
            break

        deltaT = solve_linearized_deltaT_2d(
            T_bg=T,
            delta_n=delta_n,
            dx=dx,
            dy=dy,
            source_mask=source_mask,
            max_sweeps=inner_max_sweeps,
            tol=inner_tol,
            verbose=False,
        )

        if max_deltaT is not None:
            deltaT = np.clip(deltaT, -max_deltaT, max_deltaT)

        # Try damped update
        alpha = damping
        accepted = False
        old_res_inf = np.nanmax(np.abs(nonlinear_eikonal_residual(T, n_true, dx=dx, dy=dy)))

        for trial in range(12):
            T_candidate = T.copy()
            T_candidate[~source_mask] += alpha * deltaT[~source_mask]

            new_res = nonlinear_eikonal_residual(T_candidate, n_true, dx=dx, dy=dy)
            new_res_inf = np.nanmax(np.abs(new_res))

            if (not monotone_residual) or (new_res_inf <= old_res_inf):
                T = T_candidate
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            # fall back: take tiny step anyway, unless it is numerically hopeless
            T[~source_mask] += alpha * deltaT[~source_mask]
            new_res_inf = np.nanmax(
                np.abs(nonlinear_eikonal_residual(T, n_true, dx=dx, dy=dy))
            )

        dT_inf = np.nanmax(np.abs(alpha * deltaT))
        history.append(
            {
                "iter": k + 1,
                "delta_n_inf": float(dn_inf),
                "deltaT_inf": float(np.nanmax(np.abs(deltaT))),
                "step_inf": float(dT_inf),
                "alpha": float(alpha),
                "residual_inf": float(new_res_inf),
            }
        )
        iterates.append(T.copy())

        if verbose:
            print(
                f"outer {k+1:2d}: "
                f"||delta_n||inf={dn_inf:.3e}, "
                f"||deltaT||inf={np.nanmax(np.abs(deltaT)):.3e}, "
                f"alpha={alpha:.3e}, "
                f"||res||inf={new_res_inf:.3e}"
            )

        if new_res_inf < eikonal_tol:
            if verbose:
                print(f"converged at outer iteration {k+1}")
            break

    return T, history, iterates


def solve_eikonal_by_relinearization(
    T0_fmm,
    n_true,
    dx=1.0,
    dy=1.0,
    source_mask=None,
    outer_max=20,
    inner_max_sweeps=200,
    inner_tol=1e-8,
    eikonal_tol=1e-6,
    damping=0.8,
    max_deltaT=None,
    monotone_residual=True,
    verbose=True,
):
    """
    Convenience wrapper.
    """
    T_final, history, iterates = iterative_linearized_eikonal_2d(
        T_init=T0_fmm,
        n_true=n_true,
        dx=dx,
        dy=dy,
        source_mask=source_mask,
        outer_max=outer_max,
        inner_max_sweeps=inner_max_sweeps,
        inner_tol=inner_tol,
        eikonal_tol=eikonal_tol,
        damping=damping,
        max_deltaT=max_deltaT,
        monotone_residual=monotone_residual,
        verbose=verbose,
    )
    return T_final, history, iterates
