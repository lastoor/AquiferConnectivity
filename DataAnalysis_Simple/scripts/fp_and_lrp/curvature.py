from __future__ import annotations

import functools
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


@dataclass(slots=True)
class CurvatureGeometry:
    s: np.ndarray
    spline_points: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    curvature: np.ndarray
    curvature_raw: np.ndarray
    curvature_circle: np.ndarray
    curvature_label: str
    spline_mode: str


def arclength_parameter(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(ds)))


def _normalize_tangent(tangent: np.ndarray) -> np.ndarray:
    tan = np.asarray(tangent, dtype=float)
    norms = np.linalg.norm(tan, axis=1)
    return tan / np.maximum(norms[:, None], 1e-12)


def _coerce_odd_window(window: int, n_points: int) -> int:
    max_size = n_points if n_points % 2 == 1 else n_points - 1
    if max_size < 1:
        return 0
    size = max(int(window), 1)
    if size % 2 == 0:
        size += 1
    size = min(size, max_size)
    if size % 2 == 0:
        size -= 1
    return max(size, 0)


def compute_raw_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    s = arclength_parameter(pts)
    xs = np.gradient(pts[:, 0], s, edge_order=2)
    ys = np.gradient(pts[:, 1], s, edge_order=2)
    xss = np.gradient(xs, s, edge_order=2)
    yss = np.gradient(ys, s, edge_order=2)
    tangent = _normalize_tangent(np.column_stack([xs, ys]))
    denom = np.maximum((xs * xs + ys * ys) ** 1.5, 1e-14)
    curvature = (xs * yss - ys * xss) / denom
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normal, np.asarray(curvature, dtype=float)


def curvature_local_poly(points: np.ndarray, window: int = 9, degree: int = 3) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    n = len(points)

    if n < 3:
        return np.full(n, np.nan)

    if window % 2 == 0:
        raise ValueError("window must be odd")

    s = arclength_parameter(points)
    kappa = np.full(n, np.nan)
    half = window // 2

    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)

        sw = s[i0:i1] - s[i]
        xw = x[i0:i1]
        yw = y[i0:i1]

        deg = min(degree, len(sw) - 1)
        if deg < 2:
            continue

        px = np.poly1d(np.polyfit(sw, xw, deg))
        py = np.poly1d(np.polyfit(sw, yw, deg))

        x1 = np.polyder(px, 1)(0.0)
        x2 = np.polyder(px, 2)(0.0)
        y1 = np.polyder(py, 1)(0.0)
        y2 = np.polyder(py, 2)(0.0)

        denom = (x1 * x1 + y1 * y1) ** 1.5
        if denom > 1e-14:
            kappa[i] = (x1 * y2 - y1 * x2) / denom

    return kappa


def curvature_smoothing_spline(
    points: np.ndarray,
    smooth: float = 0.0,
    k: int = 3,
) -> tuple[np.ndarray, UnivariateSpline | None, UnivariateSpline | None]:
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    s = arclength_parameter(points)

    keep = np.ones(len(s), dtype=bool)
    keep[1:] = np.diff(s) > 1e-14
    s2 = s[keep]
    x2 = x[keep]
    y2 = y[keep]

    if len(s2) <= k:
        return np.full_like(s, np.nan, dtype=float), None, None

    spl_x = UnivariateSpline(s2, x2, s=smooth, k=k)
    spl_y = UnivariateSpline(s2, y2, s=smooth, k=k)

    x1 = spl_x.derivative(1)(s)
    x2d = spl_x.derivative(2)(s)
    y1 = spl_y.derivative(1)(s)
    y2d = spl_y.derivative(2)(s)

    denom = (x1 * x1 + y1 * y1) ** 1.5
    kappa = np.full_like(s, np.nan, dtype=float)
    mask = denom > 1e-14
    kappa[mask] = (x1[mask] * y2d[mask] - y1[mask] * x2d[mask]) / denom[mask]
    return kappa, spl_x, spl_y


def compute_spline_geometry(
    points: np.ndarray,
    *,
    mode: str = "interpolation",
    smooth: float = 0.0,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    s_input = arclength_parameter(pts)
    if mode not in {"interpolation", "approximation"}:
        raise ValueError(f"Unsupported spline mode {mode!r}; expected 'interpolation' or 'approximation'.")
    effective_smooth = 0.0 if mode == "interpolation" else float(smooth)
    if effective_smooth < 0.0:
        raise ValueError(f"smooth must be nonnegative; received {effective_smooth}.")

    kappa, spl_x, spl_y = curvature_smoothing_spline(pts, smooth=effective_smooth, k=degree)
    if spl_x is None or spl_y is None:
        tangent, normal, curvature = compute_raw_geometry(pts)
        return arclength_parameter(pts), pts, tangent, normal, curvature

    s_dense = np.linspace(s_input[0], s_input[-1], max(200, 4 * len(s_input)))
    spline_points = np.column_stack([spl_x(s_dense), spl_y(s_dense)])
    s_eval = arclength_parameter(spline_points)
    dx = spl_x.derivative(1)(s_dense)
    dy = spl_y.derivative(1)(s_dense)
    ddx = spl_x.derivative(2)(s_dense)
    ddy = spl_y.derivative(2)(s_dense)
    tangent = _normalize_tangent(np.column_stack([dx, dy]))
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    denom = (dx * dx + dy * dy) ** 1.5
    curvature = np.full_like(s_dense, np.nan, dtype=float)
    mask = denom > 1e-14
    curvature[mask] = (dx[mask] * ddy[mask] - dy[mask] * ddx[mask]) / denom[mask]
    return s_eval, np.asarray(spline_points, dtype=float), tangent, normal, curvature


def compute_savgol_geometry(
    points: np.ndarray,
    *,
    window: int = 11,
    poly_order: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute path geometry via tangent-angle Savitzky-Golay differentiation.

    The curvature identity κ = dθ/ds (where θ is the tangent angle) reduces
    curvature estimation to differentiating a single scalar signal, avoiding the
    poorly-conditioned (x'y'' − y'x'') / |r'|³ formula that amplifies spline
    second-derivative noise.

    Returns (s, pts, tangent, normal, curvature).
    """
    pts = np.asarray(points, dtype=float)
    s = arclength_parameter(pts)
    n = len(pts)

    # Average arc-length spacing (near-uniform after resample_by_arclength)
    ds = s[-1] / (n - 1) if n > 1 else 1.0

    # Raw tangent direction via central differences
    dx = np.gradient(pts[:, 0], s, edge_order=2)
    dy = np.gradient(pts[:, 1], s, edge_order=2)
    theta_raw = np.unwrap(np.arctan2(dy, dx))

    win = _coerce_odd_window(window, n)
    if win >= poly_order + 2:
        theta_smooth = savgol_filter(theta_raw, win, poly_order, deriv=0, delta=ds)
        kappa = savgol_filter(theta_raw, win, poly_order, deriv=1, delta=ds)
    else:
        # Fallback when window cannot be satisfied (very short paths)
        theta_smooth = theta_raw
        kappa = np.gradient(theta_raw, s, edge_order=2)

    tangent = np.column_stack([np.cos(theta_smooth), np.sin(theta_smooth)])
    normal = np.column_stack([-np.sin(theta_smooth), np.cos(theta_smooth)])
    return s, pts, tangent, normal, np.asarray(kappa, dtype=float)


def _fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones(pts.shape[0], dtype=float)])
    b = x * x + y * y
    coeffs, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, d = coeffs
    radius_sq = cx * cx + cy * cy + d
    if not np.isfinite(radius_sq) or radius_sq <= 0.0:
        raise ValueError("Degenerate circle fit")
    return float(cx), float(cy), float(np.sqrt(radius_sq))


def compute_circle_curvature(points: np.ndarray, window: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n_points = pts.shape[0]
    win = _coerce_odd_window(window, n_points)
    if win < 3:
        return np.full(n_points, np.nan, dtype=float)

    half = win // 2
    curvature = np.full(n_points, np.nan, dtype=float)
    for idx in range(half, n_points - half):
        local_pts = pts[idx - half : idx + half + 1]
        try:
            _cx, _cy, radius = _fit_circle(local_pts)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.isfinite(radius) or radius <= 1e-12:
            continue
        prev_vec = pts[idx] - pts[idx - 1]
        next_vec = pts[idx + 1] - pts[idx]
        turn = prev_vec[0] * next_vec[1] - prev_vec[1] * next_vec[0]
        curvature[idx] = (1.0 if turn >= 0.0 else -1.0) / radius
    return curvature


def compute_curvature_bundle(
    points: np.ndarray,
    *,
    method: str = "spline",
    spline_mode: str = "interpolation",
    spline_smooth: float = 0.0,
    spline_degree: int = 3,
    circle_window: int = 7,
    savgol_window: int = 11,
    savgol_poly: int = 3,
) -> CurvatureGeometry:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    if pts.shape[0] < 3:
        raise ValueError("At least 3 points are required to compute curvature.")

    if method not in {"spline", "savgol"}:
        raise ValueError(f"Unsupported curvature method {method!r}; expected 'spline' or 'savgol'.")

    if method == "savgol":
        s, spline_points, tangent, normal, curvature = compute_savgol_geometry(
            pts,
            window=savgol_window,
            poly_order=savgol_poly,
        )
    else:
        s, spline_points, tangent, normal, curvature = compute_spline_geometry(
            pts,
            mode=spline_mode,
            smooth=spline_smooth,
            degree=spline_degree,
        )

    tangent_raw, normal_raw, curvature_raw = compute_raw_geometry(spline_points)
    curvature_circle = compute_circle_curvature(spline_points, window=circle_window)
    finite = np.isfinite(curvature)
    if not np.all(finite):
        curvature = np.where(finite, curvature, curvature_raw)
        tangent = np.where(finite[:, None], tangent, tangent_raw)
        normal = np.where(finite[:, None], normal, normal_raw)

    return CurvatureGeometry(
        s=s,
        spline_points=spline_points,
        tangent=tangent,
        normal=normal,
        curvature=curvature,
        curvature_raw=curvature_raw,
        curvature_circle=curvature_circle,
        curvature_label=method,
        spline_mode=spline_mode,
    )


@functools.lru_cache(maxsize=1)
def _build_analytical_curvature_func():
    import sympy as sp

    t = sp.Symbol("t")
    y = (
        sp.Rational(35, 100) * sp.sin(2 * sp.pi * t)
        + sp.Rational(18, 100) * sp.exp(-((t - sp.Rational(55, 100)) / sp.Rational(7, 100)) ** 2)
        + sp.Rational(5, 100) * sp.sin(16 * sp.pi * t)
        * sp.exp(-((t - sp.Rational(55, 100)) / sp.Rational(16, 100)) ** 2)
    )
    dy = sp.diff(y, t)
    ddy = sp.diff(dy, t)
    kappa_expr = ddy / (1 + dy**2) ** sp.Rational(3, 2)
    return sp.lambdify(t, kappa_expr, modules="numpy")


def curvature_analytical(t_values: np.ndarray) -> np.ndarray:
    f = _build_analytical_curvature_func()
    return np.asarray(f(np.asarray(t_values, dtype=float)), dtype=float)


def make_test_curve(n: int = 400, noise: float = 0.0):
    t = np.linspace(0.0, 1.0, n)
    x = t
    y = (
        0.35 * np.sin(2.0 * np.pi * t)
        + 0.18 * np.exp(-((t - 0.55) / 0.07) ** 2)
        + 0.05 * np.sin(16.0 * np.pi * t) * np.exp(-((t - 0.55) / 0.16) ** 2)
    )
    points = np.column_stack([x, y])
    if noise > 0:
        rng = np.random.default_rng(1234)
        points = points + noise * rng.standard_normal(points.shape)
    return points, t


def compare_methods(
    points: np.ndarray,
    local_window: int = 11,
    local_degree: int = 3,
    spline_smooth: float = 1e-4,
    spline_degree: int = 3,
    kappa_true: np.ndarray | None = None,
):
    points = np.asarray(points, dtype=float)
    s = arclength_parameter(points)
    kappa_local = curvature_local_poly(points, window=local_window, degree=local_degree)
    kappa_spline, spl_x, spl_y = curvature_smoothing_spline(points, smooth=spline_smooth, k=spline_degree)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    ax = axes[0]
    ax.plot(points[:, 0], points[:, 1], "o", ms=3, alpha=0.6, label="data")
    if spl_x is not None and spl_y is not None:
        s_dense = np.linspace(s.min(), s.max(), 1500)
        ax.plot(spl_x(s_dense), spl_y(s_dense), "-", lw=2, label="smoothing spline")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    ax = axes[1]
    if kappa_true is not None:
        ax.plot(s, kappa_true, "--k", lw=2, label="analytical (exact)")
    ax.plot(s, kappa_local, lw=2, label=f"local poly (window={local_window}, deg={local_degree})", alpha=0.7)
    ax.plot(s, kappa_spline, lw=2, label=f"spline (smooth={spline_smooth:g})", alpha=0.7)
    ax.set_title("Curvature comparison")
    ax.set_xlabel("arc length s")
    ax.set_ylabel("curvature")
    ax.legend()

    plt.show()
    return {
        "s": s,
        "kappa_local": kappa_local,
        "kappa_spline": kappa_spline,
        "kappa_true": kappa_true,
        "spl_x": spl_x,
        "spl_y": spl_y,
    }
