from __future__ import annotations

from scipy import constants
from scipy.optimize import brentq
from scipy.special import erfc, expi
import numpy as np


def compute_effective_length(field_size: float) -> float:
    field_size_int = int(field_size)
    margin = field_size_int // 20
    return float(field_size_int - 2 * margin)


def _g_tau(t: float, *, length: float, velocity: float, i_y: float, sigma_y_sq: float, euler_const: float) -> float:
    tiny = np.finfo(float).tiny
    t_prime = max((t * velocity) / i_y, tiny)
    term_bracket = expi(-t_prime) + (np.exp(-t_prime) * (1.0 + t_prime) - 1.0) / (t_prime**2)
    f_norm = 2.0 * t_prime - 3.0 * np.log(t_prime) + 1.5 - 3.0 * euler_const + 3.0 * term_bracket
    x11 = max(sigma_y_sq * (i_y**2) * f_norm, tiny)
    x1 = velocity * t
    return 0.5 * erfc((length - x1) / np.sqrt(2.0 * x11))


def _find_root_bracket(
    *,
    sigma_y_sq: float,
    length: float,
    velocity: float,
    i_y: float,
    percentile: float,
    euler_const: float,
    lo_factor: float,
    hi_factor: float,
    max_expand_steps: int = 12,
) -> tuple[float, float] | None:
    base_time = length / velocity
    lo = max(base_time * lo_factor, np.finfo(float).tiny)
    hi = base_time * hi_factor
    objective = lambda t: _g_tau(t, length=length, velocity=velocity, i_y=i_y, sigma_y_sq=sigma_y_sq, euler_const=euler_const) - percentile
    f_lo = objective(lo)
    f_hi = objective(hi)
    if np.signbit(f_lo) != np.signbit(f_hi):
        return lo, hi

    for _ in range(max_expand_steps):
        lo = max(lo / 2.0, np.finfo(float).tiny)
        hi = hi * 2.0
        f_lo = objective(lo)
        f_hi = objective(hi)
        if np.signbit(f_lo) != np.signbit(f_hi):
            return lo, hi
    return None


def compute_analytical_tmin_curve(
    var: np.ndarray,
    *,
    length: float,
    velocity: float,
    i_y: float,
    percentile: float,
    lo_factor: float = 0.1,
    hi_factor: float = 2.0,
) -> tuple[np.ndarray, list[str]]:
    euler_const = float(getattr(constants, "euler_gamma", np.euler_gamma))
    values = np.asarray(var, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    warnings: list[str] = []
    if length <= 0 or velocity <= 0 or i_y <= 0:
        warnings.append("Analytical tmin skipped because length, velocity, and i_y must be positive.")
        return out, warnings
    if not (0.0 < percentile < 1.0):
        warnings.append("Analytical tmin skipped because percentile must be in (0, 1).")
        return out, warnings

    for idx, sigma_y_sq in enumerate(values):
        if sigma_y_sq <= 0:
            warnings.append(f"Analytical tmin skipped for non-positive variance sigma2={sigma_y_sq}.")
            continue
        bracket = _find_root_bracket(
            sigma_y_sq=float(sigma_y_sq),
            length=length,
            velocity=velocity,
            i_y=i_y,
            percentile=percentile,
            euler_const=euler_const,
            lo_factor=lo_factor,
            hi_factor=hi_factor,
        )
        if bracket is None:
            warnings.append(f"Analytical tmin root bracket failed for sigma2={sigma_y_sq}.")
            continue
        objective = lambda t: _g_tau(
            t,
            length=length,
            velocity=velocity,
            i_y=i_y,
            sigma_y_sq=float(sigma_y_sq),
            euler_const=euler_const,
        ) - percentile
        try:
            t_solution = brentq(objective, bracket[0], bracket[1])
        except ValueError:
            warnings.append(f"Analytical tmin root solve failed for sigma2={sigma_y_sq}.")
            continue
        out[idx] = t_solution * velocity / i_y
    return out, warnings
