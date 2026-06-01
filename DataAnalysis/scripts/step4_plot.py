#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cmcrameri.cm as cmc
import numpy as np
from scipy import constants
from scipy.optimize import brentq
from scipy.special import erfc, expi

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **kwargs):  # type: ignore[override]
        return iterable

from step1_index import VisualizationConfig, default_cache_dir, load_config

pathcm = cmc.davos.resampled(6)
COLORS = {
    "fp": pathcm(0),
    "lrp": pathcm(4), #"#FC1212",
    "lrpFMM": pathcm(4),
    "fpFMM": pathcm(3), #"#31830B",
    "fpDijkstra": pathcm(3), #"#131EE2",
    "fpFSM": "#2CA02C",
    "fpFmAsr": pathcm(2),
    "fpFmAsrIso": "#17BECF",
    "fpFmAsrConic": "#9467BD",
    "fpPypar2": pathcm(1),
}

LINESTYLES = {
    "fp": "-",
    "lrp": "-",
    "lrpFMM": "-.",
    "fpFMM": "--",
    "fpDijkstra": ":",
    "fpFSM": (0, (3, 1, 1, 1)),
    "fpFmAsr": (0, (5, 2)),
    "fpFmAsrIso": (0, (1, 1)),
    "fpFmAsrConic": (0, (7, 2, 1, 2)),
    "fpPypar2": (0, (3, 1, 1, 1, 1, 1)),
}

LINEWIDTH = 2

@dataclass(slots=True)
class FigureArtifact:
    id: str
    filename: str
    path: Path
    section: str
    metadata: dict[str, Any]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except AttributeError as exc:
            if "__main__" not in str(exc):
                raise
            import __main__
            import step2_metrics as step2

            setattr(__main__, "VarianceResults", step2.VarianceResults)
            setattr(__main__, "StdResults", step2.StdResults)
            setattr(__main__, "SummaryStats", step2.SummaryStats)
            f.seek(0)
            return pickle.load(f)


def configure_style(cfg: VisualizationConfig) -> None:
    for path in glob.glob(cfg.font.cmu_font_glob):
        fm.fontManager.addfont(path)
    plt.rcParams["font.family"] = cfg.font.family
    plt.rcParams["mathtext.fontset"] = cfg.font.mathtext_fontset
    plt.rcParams["axes.unicode_minus"] = cfg.font.unicode_minus


def compute_effective_length(field_size: float) -> float:
    field_size_int = int(field_size)
    margin = field_size_int // 20
    return float(field_size_int - 2 * margin)


def _compute_x11(
    t: float,
    *,
    velocity: float,
    i_y: float,
    sigma_y_sq: float,
    euler_const: float,
) -> float:
    tiny = np.finfo(float).tiny
    eps = np.finfo(float).eps
    t_prime = max((t * velocity) / i_y, eps)
    # Use expm1 to avoid catastrophic cancellation for small t_prime.
    num = np.expm1(-t_prime) + t_prime * np.exp(-t_prime)
    term_bracket = expi(-t_prime) + num / (t_prime * t_prime)
    f_norm = 2.0 * t_prime - 3.0 * np.log(t_prime) + 1.5 - 3.0 * euler_const + 3.0 * term_bracket
    return float(max(sigma_y_sq * (i_y**2) * f_norm, tiny))


def _g_tau(t: float, *, length: float, velocity: float, i_y: float, sigma_y_sq: float, euler_const: float) -> float:
    x11 = _compute_x11(
        t,
        velocity=velocity,
        i_y=i_y,
        sigma_y_sq=sigma_y_sq,
        euler_const=euler_const,
    )
    x1 = velocity * t
    value = 0.5 * erfc((length - x1) / np.sqrt(2.0 * x11))
    return float(np.clip(value, 0.0, 1.0))


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
    lo = max(base_time * lo_factor, np.finfo(float).eps)
    hi = base_time * hi_factor
    objective = lambda t: _g_tau(t, length=length, velocity=velocity, i_y=i_y, sigma_y_sq=sigma_y_sq, euler_const=euler_const) - percentile
    f_lo = objective(lo)
    f_hi = objective(hi)
    if np.isfinite(f_lo) and np.isfinite(f_hi) and np.signbit(f_lo) != np.signbit(f_hi):
        return lo, hi

    for _ in range(max_expand_steps):
        lo = max(lo / 2.0, np.finfo(float).eps)
        hi = hi * 2.0
        f_lo = objective(lo)
        f_hi = objective(hi)
        if np.isfinite(f_lo) and np.isfinite(f_hi) and np.signbit(f_lo) != np.signbit(f_hi):
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
        f0 = objective(bracket[0])
        f1 = objective(bracket[1])
        if not (np.isfinite(f0) and np.isfinite(f1)):
            warnings.append(f"Analytical tmin objective non-finite for sigma2={sigma_y_sq}.")
            continue
        try:
            t_solution = brentq(objective, bracket[0], bracket[1])
        except ValueError:
            warnings.append(f"Analytical tmin root solve failed for sigma2={sigma_y_sq}.")
            continue
        if not np.isfinite(t_solution):
            warnings.append(f"Analytical tmin root solve returned non-finite value for sigma2={sigma_y_sq}.")
            continue
        out[idx] = t_solution * velocity / i_y
    return out, warnings


def plot_characteristic_time(
    var: np.ndarray,
    stats_fpt: np.ndarray,
    stats_mhr: np.ndarray,
    stats_mhr_fmm: np.ndarray,
    stats_fpt_fmm: np.ndarray,
    stats_fpt_dijkstra: np.ndarray,
    t_ref: float,
    dh: float,
    analytical_tmin: np.ndarray | None = None,
    analytical_label: str = "Analytical $t_{\\mathrm{min}}$",
    stats_fpt_fsm: np.ndarray | None = None,
    stats_fpt_fm_asr: np.ndarray | None = None,
    stats_fpt_fm_asr_iso: np.ndarray | None = None,
    stats_fpt_fm_asr_conic: np.ndarray | None = None,
    stats_fpt_pypar2: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=[6, 4])
    # fig, ax = plt.subplots(figsize=[5, 4])

    # if stats_fpt_fsm is not None and np.any(np.isfinite(stats_fpt_fsm[:, 0])):
        # ax.errorbar(var, stats_fpt_fsm[:, 0] / t_ref, stats_fpt_fsm[:, 1] / t_ref, color=COLORS["fpFSM"], linestyle="-", marker="o", markerfacecolor="None", label="FP (FSM)")
    ax.plot(var, stats_fpt[:, 0] / t_ref, color=COLORS["fp"], linestyle="-", label=r"$\gamma^{\!\star}_{\mathcal{S}}$ (Pollock)")
    if analytical_tmin is not None:
        ax.plot(var[:5], analytical_tmin[:5], color=COLORS["fp"], linestyle=":", linewidth=1.8, marker="o", markersize=6, markerfacecolor="None", label=analytical_label)
    if stats_fpt_pypar2 is not None and np.any(np.isfinite(stats_fpt_pypar2[:, 0])):
        ax.errorbar(
            var,
            stats_fpt_pypar2[:, 0] / t_ref,
            color=COLORS["fpPypar2"],
            linestyle="-",
            label=r"$\gamma^{\!\star}_{\mathcal{D}}$ (PyPar2)",
        )
    if stats_fpt_fm_asr is not None and np.any(np.isfinite(stats_fpt_fm_asr[:, 0])):
        ax.errorbar(
            var,
            stats_fpt_fm_asr[:, 0] / t_ref,
            color=COLORS["fpFmAsr"],
            linestyle="-",
            label=r"$\gamma^{\!\star}_{\mathcal{F}}$ (FM-ASR)",
        )
    ax.errorbar(var, stats_fpt_fmm[:, 0] / t_ref, color=COLORS["fpFMM"], linestyle="-", label=r"$\gamma^{\!\star}_{\mathcal{E}}$ (FMM)")
    ax.errorbar(var, stats_fpt_dijkstra[:, 0] / t_ref, color=COLORS["fpDijkstra"], linestyle=":", label=r"$\gamma^{\!\star}_{\mathcal{E}}$ (Dijkstra)")
    # if stats_fpt_fm_asr_iso is not None and np.any(np.isfinite(stats_fpt_fm_asr_iso[:, 0])):
        # ax.errorbar(
        #     var,
        #     stats_fpt_fm_asr_iso[:, 0] / t_ref,
        #     color=COLORS["fpFmAsrIso"],
        #     linestyle="--",
        #     marker="x",
        #     markersize=4,
        #     markerfacecolor="None",
        #     label=r"FP (FM-ASR\, iso)",
        # )
    # if stats_fpt_fm_asr_conic is not None and np.any(np.isfinite(stats_fpt_fm_asr_conic[:, 0])):
    #     ax.errorbar(
    #         var,
    #         stats_fpt_fm_asr_conic[:, 0] / t_ref,
    #         color=COLORS["fpFmAsrConic"],
    #         linestyle="--",
    #         marker="x",
    #         markersize=4,
    #         markerfacecolor="None",
    #         label=r"FP (FM-ASR\, conic)",
    #     )
    
    ax.errorbar(var, stats_mhr_fmm[:, 0] / (t_ref * dh), color=COLORS["lrpFMM"], linestyle="-", label=r"$\gamma^{\!\star}_{\mathcal{R}}$ (FMM)")
    ax.errorbar(var, stats_mhr[:, 0] / (t_ref * dh), color=COLORS["lrp"], linestyle=":", label=r"$\gamma^{\!\star}_{\mathcal{R}}$ (Dijkstra)")
    # ax.errorbar(var, stats_fpt[:, 0] / t_ref, stats_fpt[:, 1] / t_ref, color=COLORS["fp"], linestyle="-", marker="o", markerfacecolor="None", label="FP")
    # ax.errorbar(var, stats_mhr[:, 0] / (t_ref * dh), stats_mhr[:, 1] / (t_ref * dh), color=COLORS["lrp"], linestyle="-", marker="o", markerfacecolor="None", label="LRP")
    # ax.errorbar(var, stats_mhr_fmm[:, 0] / (t_ref * dh), stats_mhr_fmm[:, 1] / (t_ref * dh), color=COLORS["lrpFMM"], linestyle="-", marker="o", markerfacecolor="None", label="LRP (FMM)")
    # ax.errorbar(var, stats_fpt_fmm[:, 0] / t_ref, stats_fpt_fmm[:, 1] / t_ref, color=COLORS["fpFMM"], linestyle="-", marker="o", markerfacecolor="None", label="FP (FMM)")
    # ax.errorbar(var, stats_fpt_dijkstra[:, 0] / t_ref, stats_fpt_dijkstra[:, 1] / t_ref, color=COLORS["fpDijkstra"], linestyle="-", marker="o", markerfacecolor="None", label="FP (Dijkstra)")
    # ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathbb{E}\left[ t\!^\star \right] U / I_Y$")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "characteristic_time"}


def plot_scatter_tmin_vs_rmin(
    stats_fpt: np.ndarray,
    stats_fpt_fmm: np.ndarray,
    stats_fpt_dijkstra: np.ndarray,
    stats_mhr: np.ndarray,
    stats_mhr_fmm: np.ndarray,
    t_ref: float,
    dh: float,
    stats_fpt_fsm: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stats_fpt[:, 0] / t_ref, stats_fpt_fmm[:, 0] / t_ref, color=COLORS["fpFMM"], label="FP (FMM)", alpha=0.6)
    ax.scatter(stats_fpt[:, 0] / t_ref, stats_fpt_dijkstra[:, 0] / t_ref, color=COLORS["fpDijkstra"], label="$FP (Dijkstra)", alpha=0.6)
    if stats_fpt_fsm is not None and np.any(np.isfinite(stats_fpt_fsm[:, 0])):
        ax.scatter(stats_fpt[:, 0] / t_ref, stats_fpt_fsm[:, 0] / t_ref, color=COLORS["fpFSM"], label="FP (FSM)", alpha=0.6)
    ax.scatter(stats_fpt[:, 0] / t_ref, stats_mhr[:, 0] / (t_ref * dh), color=COLORS["lrp"], label="LRP", alpha=0.6)
    ax.scatter(stats_fpt[:, 0] / t_ref, stats_mhr_fmm[:, 0] / (t_ref * dh), color=COLORS["lrpFMM"], label="LRP (FMM)", alpha=0.6)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("$t_{\\text{min}} U / I_Y$ (Particle Tracking)")
    ax.set_ylabel("$t_{\\text{min}} U / I_Y$ (Other Methods)")
    ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    # ax.set_title("Characteristic Time Scale")
    ax.legend()
    fig.tight_layout()
    return fig, {"kind": "scatter_tmin_vs_rmin"}


def _coerce_series(values: np.ndarray | list[float], expected_len: int, *, label: str, warnings: list[str]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == expected_len:
        return arr
    out = np.full(expected_len, np.nan, dtype=float)
    n = min(expected_len, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    warnings.append(
        f"{label} has length {arr.size}, expected {expected_len}; "
        f"used first {n} values and filled the rest with NaN."
    )
    return out


def _aligned_ratio_moments(num: np.ndarray, den: np.ndarray, *, dh: float, label: str, warnings: list[str]) -> tuple[float, float]:
    n = np.asarray(num, dtype=float).ravel()
    d = np.asarray(den, dtype=float).ravel()
    m = min(n.size, d.size)
    if m == 0:
        warnings.append(f"{label}: empty samples; returning NaN.")
        return float("nan"), float("nan")
    n = n[:m]
    d_norm = d[:m] / dh
    valid = np.isfinite(n) & np.isfinite(d_norm) & (d_norm != 0.0)
    if not np.any(valid):
        warnings.append(f"{label}: no valid samples after finite/nonzero filtering; returning NaN.")
        return float("nan"), float("nan")
    ratio = n[valid] / d_norm[valid]
    return float(np.mean(ratio)), float(np.var(ratio))


def _compute_ratio_moments_series(
    var: np.ndarray,
    *,
    scatter_samples_by_var: dict[str, Any] | None,
    dh: float,
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_var = len(np.asarray(var, dtype=float).ravel())
    mean_fp = np.full(n_var, np.nan, dtype=float)
    mean_fmm = np.full(n_var, np.nan, dtype=float)
    mean_dijkstra = np.full(n_var, np.nan, dtype=float)
    mean_fsm = np.full(n_var, np.nan, dtype=float)
    var_fp = np.full(n_var, np.nan, dtype=float)
    var_fmm = np.full(n_var, np.nan, dtype=float)
    var_dijkstra = np.full(n_var, np.nan, dtype=float)
    var_fsm = np.full(n_var, np.nan, dtype=float)

    if not isinstance(scatter_samples_by_var, dict):
        warnings.append("Ratio plots require scatter_time_samples from Stage 3; received missing/invalid payload.")
        return mean_fp, mean_fmm, mean_dijkstra, mean_fsm, var_fp, var_fmm, var_dijkstra, var_fsm

    for idx, variance_value in enumerate(np.asarray(var, dtype=float).ravel()):
        key = _resolve_sample_key(scatter_samples_by_var, float(variance_value))
        if key is None:
            warnings.append(f"Ratio plots missing scatter samples for sigma2={variance_value}; setting NaN.")
            continue
        sample_dict = scatter_samples_by_var.get(key, {})
        required = ("fpt", "fpt_fmm", "fpt_dijkstra", "mhr")
        if not all(rk in sample_dict for rk in required):
            warnings.append(f"Ratio plots have incomplete scatter samples for sigma2={variance_value}; setting NaN.")
            continue

        mean_fp[idx], var_fp[idx] = _aligned_ratio_moments(
            sample_dict["fpt"],
            sample_dict["mhr"],
            dh=dh,
            label=f"FP ratio moments at sigma2={variance_value}",
            warnings=warnings,
        )
        mean_fmm[idx], var_fmm[idx] = _aligned_ratio_moments(
            sample_dict["fpt_fmm"],
            sample_dict["mhr"],
            dh=dh,
            label=f"FP(FMM) ratio moments at sigma2={variance_value}",
            warnings=warnings,
        )
        mean_dijkstra[idx], var_dijkstra[idx] = _aligned_ratio_moments(
            sample_dict["fpt_dijkstra"],
            sample_dict["mhr"],
            dh=dh,
            label=f"FP(Dijkstra) ratio moments at sigma2={variance_value}",
            warnings=warnings,
        )
        if "fpt_fsm" in sample_dict:
            mean_fsm[idx], var_fsm[idx] = _aligned_ratio_moments(
                sample_dict["fpt_fsm"],
                sample_dict["mhr"],
                dh=dh,
                label=f"FP(FSM) ratio moments at sigma2={variance_value}",
                warnings=warnings,
            )

    return mean_fp, mean_fmm, mean_dijkstra, mean_fsm, var_fp, var_fmm, var_dijkstra, var_fsm


def _value_slug(value: float) -> str:
    return str(f"{float(value):g}").replace("-", "m").replace(".", "p")


def _empirical_cdf(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(samples, dtype=float).ravel())
    n = x.size
    y = np.arange(1, n + 1, dtype=float) / float(n)
    return x, y


def _parse_var_from_key(key: str) -> float | None:
    if not key.startswith("var_"):
        return None
    try:
        return float(key[4:])
    except ValueError:
        return None


def _resolve_requested_plot_variances(var: np.ndarray, requested: list[float] | None) -> list[float]:
    available = [float(v) for v in np.asarray(var, dtype=float).ravel()]
    if not available:
        return []
    if not requested:
        return [available[0]]
    out: list[float] = []
    for req in requested:
        match = next((v for v in available if abs(v - float(req)) < 1e-12), None)
        if match is None:
            raise ValueError(f"Requested plot variance {req} is not available. Available values: {available}")
        if not any(abs(match - prev) < 1e-12 for prev in out):
            out.append(match)
    return out


def _resolve_sample_key(samples_by_var: dict[str, Any], variance_value: float) -> str | None:
    for key in samples_by_var:
        parsed = _parse_var_from_key(key)
        if parsed is not None and abs(parsed - float(variance_value)) < 1e-12:
            return key
    return None


def _parse_plot_variances(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    tokens = [tok.strip() for tok in raw.split(",")]
    values: list[float] = []
    for tok in tokens:
        if not tok:
            continue
        values.append(float(tok))
    return values or None


def _parse_scatter_modes(raw: str | None) -> list[str]:
    if raw is None:
        return ["mean"]
    tokens = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        return ["mean"]
    allowed = {"mean", "all_points"}
    out: list[str] = []
    for token in tokens:
        if token not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            raise ValueError(f"Invalid scatter mode '{token}'. Allowed values: {allowed_str}")
        if token not in out:
            out.append(token)
    return out


def plot_scatter_tmin_vs_rmin_all_points(samples: dict[str, np.ndarray], *, t_ref: float, dh: float, variance_value: float):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.asarray(samples.get("fpt", []), dtype=float).ravel() / t_ref
    y_fmm = np.asarray(samples.get("fpt_fmm", []), dtype=float).ravel() / t_ref
    y_dij = np.asarray(samples.get("fpt_dijkstra", []), dtype=float).ravel() / t_ref
    y_fsm = np.asarray(samples.get("fpt_fsm", []), dtype=float).ravel() / t_ref
    y_mhr = np.asarray(samples.get("mhr", []), dtype=float).ravel() / (t_ref * dh)
    y_mhr_fmm = np.asarray(samples.get("mhr_fmm", []), dtype=float).ravel() / (t_ref * dh)
    n = min(x.size, y_fmm.size, y_dij.size, y_mhr.size, y_mhr_fmm.size)
    if n == 0:
        raise ValueError(f"No valid scatter samples for variance {variance_value}")
    x = x[:n]
    y_fmm = y_fmm[:n]
    y_dij = y_dij[:n]
    y_fsm = y_fsm[:n] if y_fsm.size >= n else np.asarray([], dtype=float)
    y_mhr = y_mhr[:n]
    y_mhr_fmm = y_mhr_fmm[:n]
    m_fmm = np.isfinite(x) & np.isfinite(y_fmm) & (x > 0.0) & (y_fmm > 0.0)
    m_dij = np.isfinite(x) & np.isfinite(y_dij) & (x > 0.0) & (y_dij > 0.0)
    m_fsm = np.isfinite(x) & np.isfinite(y_fsm) & (x > 0.0) & (y_fsm > 0.0) if y_fsm.size == n else np.zeros(n, dtype=bool)
    m_mhr = np.isfinite(x) & np.isfinite(y_mhr) & (x > 0.0) & (y_mhr > 0.0)
    m_mhr_fmm = np.isfinite(x) & np.isfinite(y_mhr_fmm) & (x > 0.0) & (y_mhr_fmm > 0.0)
    ax.scatter(x[m_fmm], y_fmm[m_fmm], color=COLORS["fpFMM"], label="$t_{\\text{min}}\\text{(FMM)}$", alpha=0.6, s=12)
    ax.scatter(x[m_dij], y_dij[m_dij], color=COLORS["fpDijkstra"], label="$t_{\\text{min}}\\text{(Dijkstra)}$", alpha=0.6, s=12)
    if np.any(m_fsm):
        ax.scatter(x[m_fsm], y_fsm[m_fsm], color=COLORS["fpFSM"], label="$t_{\\text{min}}\\text{(FSM)}$", alpha=0.6, s=12)
    ax.scatter(x[m_mhr], y_mhr[m_mhr], color=COLORS["lrp"], label=r"$R_{\text{min}}/\mathcal{J}$", alpha=0.6, s=12)
    ax.scatter(x[m_mhr_fmm], y_mhr_fmm[m_mhr_fmm], color=COLORS["lrpFMM"], label=r"$R_{\text{min}}^{\mathrm{FMM}}/\mathcal{J}$", alpha=0.6, s=12)

    # Linear regression fits
    if np.any(m_fmm):
        coeffs_fmm = np.polyfit(x[m_fmm], y_fmm[m_fmm], 1)
        ax.plot(x[m_fmm], np.polyval(coeffs_fmm, x[m_fmm]), color=COLORS["fpFMM"], linestyle=LINESTYLES["fpFMM"], linewidth=LINEWIDTH, path_effects=[pe.Stroke(linewidth=2*LINEWIDTH, foreground="white"), pe.Normal()])
    if np.any(m_dij):
        coeffs_dij = np.polyfit(x[m_dij], y_dij[m_dij], 1)
        ax.plot(x[m_dij], np.polyval(coeffs_dij, x[m_dij]), color=COLORS["fpDijkstra"], linestyle=LINESTYLES["fpDijkstra"], linewidth=LINEWIDTH, path_effects=[pe.Stroke(linewidth=2*LINEWIDTH, foreground="white"), pe.Normal()])
    if np.any(m_fsm):
        coeffs_fsm = np.polyfit(x[m_fsm], y_fsm[m_fsm], 1)
        ax.plot(x[m_fsm], np.polyval(coeffs_fsm, x[m_fsm]), color=COLORS["fpFSM"], linestyle=LINESTYLES["fpFSM"], linewidth=LINEWIDTH, path_effects=[pe.Stroke(linewidth=2*LINEWIDTH, foreground="white"), pe.Normal()])
    if np.any(m_mhr):
        coeffs_mhr = np.polyfit(x[m_mhr], y_mhr[m_mhr], 1)
        ax.plot(x[m_mhr], np.polyval(coeffs_mhr, x[m_mhr]), color=COLORS["lrp"], linestyle=LINESTYLES["lrp"], linewidth=LINEWIDTH, path_effects=[pe.Stroke(linewidth=2*LINEWIDTH, foreground="white"), pe.Normal()])
    if np.any(m_mhr_fmm):
        coeffs_mhr_fmm = np.polyfit(x[m_mhr_fmm], y_mhr_fmm[m_mhr_fmm], 1)
        ax.plot(x[m_mhr_fmm], np.polyval(coeffs_mhr_fmm, x[m_mhr_fmm]), color=COLORS["lrpFMM"], linestyle=LINESTYLES["lrpFMM"], linewidth=LINEWIDTH, path_effects=[pe.Stroke(linewidth=2*LINEWIDTH, foreground="white"), pe.Normal()])

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlim(left=0, right=16)
    ax.set_ylim(bottom=0, top=14)
    ax.set_xlabel("$t_{\\text{min}}\\text{(RK45)}$ (s)")
    ax.set_ylabel("$\\tau$ (s)")
    ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Characteristic Time Scale ($\\sigma_Y^2={variance_value:g}$)")
    ax.legend()
    fig.tight_layout()
    return fig, {"kind": "scatter_tmin_vs_rmin_all_points", "variance": float(variance_value)}


def plot_hausdorff_vs_variance(
    var: np.ndarray,
    stats_hausdorff_lrp: np.ndarray,
    stats_hausdorff_lrp_fmm: np.ndarray,
    stats_hausdorff_fmm: np.ndarray,
    stats_hausdorff_dijkstra: np.ndarray,
    i_y: float,
    stats_hausdorff_fsm: np.ndarray | None = None,
    stats_hausdorff_fm_asr: np.ndarray | None = None,
    stats_hausdorff_fm_asr_iso: np.ndarray | None = None,
    stats_hausdorff_fm_asr_conic: np.ndarray | None = None,
    stats_hausdorff_pypar2: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=[6, 4])
    # fig, ax = plt.subplots(figsize=[5, 4])

    ax.errorbar(var, stats_hausdorff_fmm[:, 0] / i_y, label="FP (FMM)", linestyle="-", color=COLORS["fpFMM"])
    ax.errorbar(var, stats_hausdorff_dijkstra[:, 0] / i_y, label="FP (Dijkstra)", linestyle=":", color=COLORS["fpDijkstra"])
    ax.errorbar(var, stats_hausdorff_lrp_fmm[:, 0] / i_y, label="LRP (FMM)", linestyle="-", color=COLORS["lrpFMM"])
    ax.errorbar(var, stats_hausdorff_lrp[:, 0] / i_y, label="LRP (Dijkstra)", linestyle=":", color=COLORS["lrp"])
    if stats_hausdorff_fm_asr is not None and np.any(np.isfinite(stats_hausdorff_fm_asr[:, 0])):
        ax.errorbar(
            var,
            stats_hausdorff_fm_asr[:, 0] / i_y,
            label=r"FP (FM-ASR)",
            linestyle="-",
            color=COLORS["fpFmAsr"],
        )
    # if stats_hausdorff_fm_asr_iso is not None and np.any(np.isfinite(stats_hausdorff_fm_asr_iso[:, 0])):
        # ax.errorbar(
        #     var,
        #     stats_hausdorff_fm_asr_iso[:, 0] / i_y,
        #     label=r"FP (FM-ASR\, iso)",
        #     linestyle="--",
        #     color=COLORS["fpFmAsrIso"],
        #     marker="x",
        #     markersize=4,
        #     markerfacecolor="None",
        # )
    if stats_hausdorff_fm_asr_conic is not None and np.any(np.isfinite(stats_hausdorff_fm_asr_conic[:, 0])):
        ax.errorbar(
            var,
            stats_hausdorff_fm_asr_conic[:, 0] / i_y,
            label=r"FP (FM-ASR\, conic)",
            linestyle="-",
            color=COLORS["fpFmAsrConic"],
        )
    if stats_hausdorff_pypar2 is not None and np.any(np.isfinite(stats_hausdorff_pypar2[:, 0])):
        ax.errorbar(
            var,
            stats_hausdorff_pypar2[:, 0] / i_y,
            label="FP (PyPar2)",
            linestyle="-",
            color=COLORS["fpPypar2"],
        )
    # ax.errorbar(var, stats_hausdorff_lrp[:, 0] / i_y, yerr=stats_hausdorff_lrp[:, 1] / i_y, label="$d_H$[FP, LRP]", linestyle="-", color=COLORS["lrp"], marker="o", markerfacecolor="None")
    # ax.errorbar(var, stats_hausdorff_lrp_fmm[:, 0] / i_y, yerr=stats_hausdorff_lrp_fmm[:, 1] / i_y, label="$d_H$[FP, LRP(FMM)]", linestyle="-", color=COLORS["lrpFMM"], marker="o", markerfacecolor="None")
    # ax.errorbar(var, stats_hausdorff_fmm[:, 0] / i_y, yerr=stats_hausdorff_fmm[:, 1] / i_y, label="$d_H$[FP, FP(FMM)]", linestyle="-", color=COLORS["fpFMM"], marker="o", markerfacecolor="None")
    # ax.errorbar(var, stats_hausdorff_dijkstra[:, 0] / i_y, yerr=stats_hausdorff_dijkstra[:, 1] / i_y, label="$d_H$[FP, FP(Dijkstra)]", linestyle="-", color=COLORS["fpDijkstra"], marker="o", markerfacecolor="None")
    if stats_hausdorff_fsm is not None and np.any(np.isfinite(stats_hausdorff_fsm[:, 0])):
        pass
        # ax.errorbar(var, stats_hausdorff_fsm[:, 0] / i_y, yerr=stats_hausdorff_fsm[:, 1] / i_y, label="$d_H$[FP, FP(FSM)]", linestyle="-", color=COLORS["fpFSM"], marker="o", markerfacecolor="None")
    # ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathbb{E}\left[ d_{\mathrm{H}} \right] / I_Y $")
    ax.legend(fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "hausdorff_variance"}


def plot_frechet_vs_variance(
    var: np.ndarray,
    stats_frechet_lrp: np.ndarray,
    stats_frechet_lrp_fmm: np.ndarray,
    stats_frechet_fmm: np.ndarray,
    stats_frechet_dijkstra: np.ndarray,
    i_y: float,
    stats_frechet_fsm: np.ndarray | None = None,
    stats_frechet_fm_asr: np.ndarray | None = None,
    stats_frechet_fm_asr_iso: np.ndarray | None = None,
    stats_frechet_fm_asr_conic: np.ndarray | None = None,
    stats_frechet_pypar2: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.errorbar(var, stats_frechet_lrp[:, 0] / i_y, yerr=stats_frechet_lrp[:, 1] / i_y, label="$d_F$[FP, LRP]", linestyle="-", color=COLORS["lrp"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_frechet_lrp_fmm[:, 0] / i_y, yerr=stats_frechet_lrp_fmm[:, 1] / i_y, label="$d_F$[FP, LRP(FMM)]", linestyle="-", color=COLORS["lrpFMM"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_frechet_fmm[:, 0] / i_y, yerr=stats_frechet_fmm[:, 1] / i_y, label="$d_F$[FP, FP(FMM)]", linestyle="-", color=COLORS["fpFMM"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_frechet_dijkstra[:, 0] / i_y, yerr=stats_frechet_dijkstra[:, 1] / i_y, label="$d_F$[FP, FP(Dijkstra)]", linestyle="-", color=COLORS["fpDijkstra"], marker="o", markerfacecolor="None")
    if stats_frechet_fsm is not None and np.any(np.isfinite(stats_frechet_fsm[:, 0])):
        ax.errorbar(var, stats_frechet_fsm[:, 0] / i_y, yerr=stats_frechet_fsm[:, 1] / i_y, label="$d_F$[FP, FP(FSM)]", linestyle="-", color=COLORS["fpFSM"], marker="o", markerfacecolor="None")
    if stats_frechet_fm_asr is not None and np.any(np.isfinite(stats_frechet_fm_asr[:, 0])):
        ax.errorbar(
            var,
            stats_frechet_fm_asr[:, 0] / i_y,
            yerr=stats_frechet_fm_asr[:, 1] / i_y,
            label=r"$d_F$[FP,\,FM-ASR]",
            linestyle="-",
            color=COLORS["fpFmAsr"],
            marker="o",
            markerfacecolor="None",
        )
    if stats_frechet_fm_asr_iso is not None and np.any(np.isfinite(stats_frechet_fm_asr_iso[:, 0])):
        ax.errorbar(
            var,
            stats_frechet_fm_asr_iso[:, 0] / i_y,
            yerr=stats_frechet_fm_asr_iso[:, 1] / i_y,
            label=r"$d_F$[FP,\,FM-ASR\,iso]",
            linestyle="-",
            color=COLORS["fpFmAsrIso"],
            marker="o",
            markerfacecolor="None",
        )
    if stats_frechet_fm_asr_conic is not None and np.any(np.isfinite(stats_frechet_fm_asr_conic[:, 0])):
        ax.errorbar(
            var,
            stats_frechet_fm_asr_conic[:, 0] / i_y,
            yerr=stats_frechet_fm_asr_conic[:, 1] / i_y,
            label=r"$d_F$[FP,\,FM-ASR\,conic]",
            linestyle="-",
            color=COLORS["fpFmAsrConic"],
            marker="o",
            markerfacecolor="None",
        )
    if stats_frechet_pypar2 is not None and np.any(np.isfinite(stats_frechet_pypar2[:, 0])):
        ax.errorbar(
            var,
            stats_frechet_pypar2[:, 0] / i_y,
            yerr=stats_frechet_pypar2[:, 1] / i_y,
            label=r"$d_F$[FP,\,PyPar2]",
            linestyle="-",
            color=COLORS["fpPypar2"],
            marker="o",
            markerfacecolor="None",
        )
    # ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathbb{E}\left[ d_F / I_Y \right]$")
    ax.legend(fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "frechet_variance"}


def plot_endpoint_box_by_variance(
    var: np.ndarray,
    endpoint_samples: dict[str, dict[str, np.ndarray]],
    i_y: float,
):
    fig, ax = plt.subplots(figsize=[6.5, 4])
    pairs = [
        ("fp_lrp", "$d_{EP}$[FP, LRP]", COLORS["lrp"]),
        ("fp_lrpFMM", "$d_{EP}$[FP, LRP(FMM)]", COLORS["lrpFMM"]),
        ("fp_fpFMM", "$d_{EP}$[FP, FP(FMM)]", COLORS["fpFMM"]),
        ("fp_fpDijkstra", "$d_{EP}$[FP, FP(Dijkstra)]", COLORS["fpDijkstra"]),
        ("fp_fpFSM", "$d_{EP}$[FP, FP(FSM)]", COLORS["fpFSM"]),
        ("fp_fm_asr", r"$d_{EP}$[FP,\,FM-ASR]", COLORS["fpFmAsr"]),
        ("fp_fm_asr_iso", r"$d_{EP}$[FP,\,FM-ASR\,iso]", COLORS["fpFmAsrIso"]),
        ("fp_fm_asr_conic", r"$d_{EP}$[FP,\,FM-ASR\,conic]", COLORS["fpFmAsrConic"]),
        ("fp_pypar2", r"$d_{EP}$[FP,\,PyPar2]", COLORS["fpPypar2"]),
    ]
    var_values = [float(v) for v in np.asarray(var, dtype=float).ravel()]
    width = 0.11
    n_pairs = len(pairs)
    half = (n_pairs - 1) / 2.0
    offsets = np.array([idx - half for idx in range(n_pairs)], dtype=float) * width
    centers = np.arange(len(var_values), dtype=float)

    legend_handles = []
    for pair_idx, (pair_key, label, color) in enumerate(pairs):
        series: list[np.ndarray] = []
        positions: list[float] = []
        for idx, variance_value in enumerate(var_values):
            sample_key = _resolve_sample_key(endpoint_samples, variance_value)
            if sample_key is None:
                continue
            arr = np.asarray(endpoint_samples[sample_key].get(pair_key, []), dtype=float).ravel()
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            series.append(finite / i_y)
            positions.append(float(centers[idx] + offsets[pair_idx]))
        if not series:
            continue
        bp = ax.boxplot(
            series,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=True,
            manage_ticks=False,
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.35)
        for median in bp["medians"]:
            median.set(color=color, linewidth=1.5)
        for whisker in bp["whiskers"]:
            whisker.set(color=color, linewidth=1.0)
        for cap in bp["caps"]:
            cap.set(color=color, linewidth=1.0)
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=2.5, markerfacecolor=color, markeredgecolor=color, alpha=0.45)
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=label))

    ax.set_xticks(centers)
    ax.set_xticklabels([f"{v:g}" for v in var_values])
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$d_{EP} / I_Y$")
    ax.legend(handles=legend_handles, fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "endpoint_box_variance"}


def plot_hausdorff_box_by_variance(
    var: np.ndarray,
    hausdorff_samples: dict[str, dict[str, np.ndarray]],
    i_y: float,
):
    fig, ax = plt.subplots(figsize=[6.5, 4])
    pairs = [
        ("fp_lrp", "$d_H$[FP, LRP]", COLORS["lrp"]),
        ("fp_lrpFMM", "$d_H$[FP, LRP(FMM)]", COLORS["lrpFMM"]),
        ("fp_fpFMM", "$d_H$[FP, FP(FMM)]", COLORS["fpFMM"]),
        ("fp_fpDijkstra", "$d_H$[FP, FP(Dijkstra)]", COLORS["fpDijkstra"]),
        ("fp_fpFSM", "$d_H$[FP, FP(FSM)]", COLORS["fpFSM"]),
        ("fp_fm_asr", r"$d_H$[FP,\,FM-ASR]", COLORS["fpFmAsr"]),
        ("fp_fm_asr_iso", r"$d_H$[FP,\,FM-ASR\,iso]", COLORS["fpFmAsrIso"]),
        ("fp_fm_asr_conic", r"$d_H$[FP,\,FM-ASR\,conic]", COLORS["fpFmAsrConic"]),
        ("fp_pypar2", r"$d_H$[FP,\,PyPar2]", COLORS["fpPypar2"]),
    ]
    var_values = [float(v) for v in np.asarray(var, dtype=float).ravel()]
    width = 0.1
    n_pairs = len(pairs)
    half = (n_pairs - 1) / 2.0
    offsets = np.array([idx - half for idx in range(n_pairs)], dtype=float) * width
    centers = np.arange(len(var_values), dtype=float)

    legend_handles = []
    for pair_idx, (pair_key, label, color) in enumerate(pairs):
        series: list[np.ndarray] = []
        positions: list[float] = []
        for idx, variance_value in enumerate(var_values):
            sample_key = _resolve_sample_key(hausdorff_samples, variance_value)
            if sample_key is None:
                continue
            arr = np.asarray(hausdorff_samples[sample_key].get(pair_key, []), dtype=float).ravel()
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            series.append(finite / i_y)
            positions.append(float(centers[idx] + offsets[pair_idx]))
        if not series:
            continue
        bp = ax.boxplot(series, positions=positions, widths=width * 0.85, patch_artist=True, showfliers=True, manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.35)
        for median in bp["medians"]:
            median.set(color=color, linewidth=1.5)
        for whisker in bp["whiskers"]:
            whisker.set(color=color, linewidth=1.0)
        for cap in bp["caps"]:
            cap.set(color=color, linewidth=1.0)
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=2.5, markerfacecolor=color, markeredgecolor=color, alpha=0.45)
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=label))

    ax.set_xticks(centers)
    ax.set_xticklabels([f"{v:g}" for v in var_values])
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$d_H / I_Y$")
    ax.legend(handles=legend_handles, fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "hausdorff_box_variance"}


_HAUSDORFF_CDF_PAIRS = (
    ("fp_lrp", "$d_H$[FP, LRP]", COLORS["lrp"]),
    ("fp_lrpFMM", "$d_H$[FP, LRP(FMM)]", COLORS["lrpFMM"]),
    ("fp_fpDijkstra", "$d_H$[FP, FP(Dijkstra)]", COLORS["fpDijkstra"]),
    ("fp_fpFMM", "$d_H$[FP, FP(FMM)]", COLORS["fpFMM"]),
    ("fp_fpFSM", "$d_H$[FP, FP(FSM)]", COLORS["fpFSM"]),
    ("fp_fm_asr", r"$d_H$[FP,\,FM-ASR]", COLORS["fpFmAsr"]),
    ("fp_fm_asr_iso", r"$d_H$[FP,\,FM-ASR\,iso]", COLORS["fpFmAsrIso"]),
    ("fp_fm_asr_conic", r"$d_H$[FP,\,FM-ASR\,conic]", COLORS["fpFmAsrConic"]),
    ("fp_pypar2", r"$d_H$[FP,\,PyPar2]", COLORS["fpPypar2"]),
)


def plot_hausdorff_cdf_ensemble(
    ensemble_data: dict[str, dict[str, np.ndarray]],
    variance_value: float,
    i_y: float,
):
    fig, ax = plt.subplots(figsize=[5, 4])
    for pair_key, label, color in _HAUSDORFF_CDF_PAIRS:
        entry = ensemble_data.get(pair_key)
        if entry is None:
            continue
        x = entry["x"] / i_y
        mean_cdf = entry["mean_cdf"]
        std_cdf = entry["std_cdf"]
        ax.plot(x, mean_cdf, linestyle="-", color=color, label=label)
        ax.fill_between(
            x,
            np.clip(mean_cdf - std_cdf, 0.0, 1.0),
            np.clip(mean_cdf + std_cdf, 0.0, 1.0),
            color=color,
            alpha=0.2,
        )
    ax.set_xlabel(r"$d / I_Y$")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(fr"Hausdorff Distance CDF ($\sigma_Y^2={variance_value:g}$)")
    ax.legend(fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "hausdorff_cdf_ensemble", "variance": float(variance_value)}


_ENTROPY_HAUSDORFF_PANELS = (
    ("fp_fpFMM", "FP (FMM)"),
    ("fp_fpDijkstra", "FP (Dijkstra)"),
    ("fp_fpFSM", "FP (FSM)"),
    ("fp_fm_asr", "FP (FM-ASR)"),
    ("fp_fm_asr_iso", r"FP (FM-ASR\, iso)"),
    ("fp_fm_asr_conic", r"FP (FM-ASR\, conic)"),
    ("fp_pypar2", "FP (PyPar2)"),
    ("fp_lrpFMM", "LRP (FMM)"),
    ("fp_lrp", "LRP (Dijkstra)"),
)

_HAUSDORFF_STAT_STYLES = {
    "max": {"color": "#D62728", "marker": "^", "label": "max"},
    "mean": {"color": "#1F77B4", "marker": "o", "label": "mean"},
    "median": {"color": "#2CA02C", "marker": "s", "label": "median"},
}


def plot_entropy_vs_hausdorff(
    samples: dict[str, Any],
    variance_value: float,
    i_y: float,
    *,
    sort_by_entropy: bool = False,
    subsample: int = 1,
):
    renyi_s1_list = samples.get("renyi_S1", [])
    renyi_s2_list = samples.get("renyi_S2", [])
    mean_s1 = np.asarray([float(np.nanmean(a)) for a in renyi_s1_list], dtype=float)
    mean_s2 = np.asarray([float(np.nanmean(a)) for a in renyi_s2_list], dtype=float)
    
    def _r_squared(x: np.ndarray, y: np.ndarray) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return float("nan")
        r = float(np.corrcoef(x[mask], y[mask])[0, 1])
        return r * r

    n_rows = len(_ENTROPY_HAUSDORFF_PANELS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 2.8 * n_rows))

    entropy_cols = [
        (mean_s1, r"$\overline{S}_1(x)$"),
        (mean_s2, r"$\overline{S}_2(x)$"),
    ]

    for row_idx, (pair_key, panel_label) in enumerate(_ENTROPY_HAUSDORFF_PANELS):
        for col_idx, (entropy_arr, entropy_label) in enumerate(entropy_cols):
            ax = axes[row_idx, col_idx]
            for stat_key, style in _HAUSDORFF_STAT_STYLES.items():
                haus_data = samples.get(f"hausdorff_{stat_key}", {})
                full_arr = np.asarray(haus_data.get(pair_key, []), dtype=float)
                if full_arr.size == 0:
                    continue
                n = min(len(entropy_arr), len(full_arr))
                x = entropy_arr[:n]
                y = full_arr[:n] / i_y
                r2 = _r_squared(x, y)
                label = f"{style['label']}: $R^2={r2:.2f}$"
                ax.scatter(
                    x, y,
                    s=12, alpha=0.5,
                    marker=style["marker"],
                    color=style["color"],
                    label=label,
                )
            ax.set_ylabel(r"$d_H / I_Y$")
            ax.set_yscale("log")
            ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.05, 1))
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.set_title(entropy_label, fontsize=10)
            if row_idx == n_rows - 1:
                ax.set_xlabel(entropy_label)
            ax.annotate(
                panel_label, xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=8, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
            )

    fig.suptitle(fr"$\sigma_Y^2 = {variance_value:g}$", fontsize=11)
    fig.tight_layout()
    return fig, {"kind": "entropy_hausdorff", "variance": float(variance_value)}


def plot_times_box_by_variance(
    var: np.ndarray,
    time_samples: dict[str, dict[str, np.ndarray]],
    t_ref: float,
    dh: float,
):
    fig, ax = plt.subplots(figsize=[8, 4.5])
    series_defs = [
        ("fpt", "FP", COLORS["fp"], t_ref),
        ("mhr", "LRP", COLORS["lrp"], t_ref * dh),
        ("mhr_fmm", "LRP (FMM)", COLORS["lrpFMM"], t_ref * dh),
        ("fpt_fmm", "FP (FMM)", COLORS["fpFMM"], t_ref),
        ("fpt_dijkstra", "FP (Dijkstra)", COLORS["fpDijkstra"], t_ref),
        ("fpt_fsm", "FP (FSM)", COLORS["fpFSM"], t_ref),
        ("fpt_fm_asr", "FP (FM-ASR)", COLORS["fpFmAsr"], t_ref),
        ("fpt_fm_asr_iso", r"FP (FM-ASR\, iso)", COLORS["fpFmAsrIso"], t_ref),
        ("fpt_fm_asr_conic", r"FP (FM-ASR\, conic)", COLORS["fpFmAsrConic"], t_ref),
        ("fpt_pypar2", "FP (PyPar2)", COLORS["fpPypar2"], t_ref),
    ]
    var_values = [float(v) for v in np.asarray(var, dtype=float).ravel()]
    n_series = len(series_defs)
    width = 0.12
    half = (n_series - 1) / 2.0
    offsets = np.array([i - half for i in range(n_series)], dtype=float) * width
    centers = np.arange(len(var_values), dtype=float)

    legend_handles = []
    for series_idx, (key, label, color, scale) in enumerate(series_defs):
        samples: list[np.ndarray] = []
        positions: list[float] = []
        for idx, variance_value in enumerate(var_values):
            sample_key = _resolve_sample_key(time_samples, variance_value)
            if sample_key is None:
                continue
            arr = np.asarray(time_samples[sample_key].get(key, []), dtype=float).ravel()
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            samples.append(finite / scale)
            positions.append(float(centers[idx] + offsets[series_idx]))
        if not samples:
            continue
        bp = ax.boxplot(samples, positions=positions, widths=width * 0.85, patch_artist=True, showfliers=True, manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.35)
        for median in bp["medians"]:
            median.set(color=color, linewidth=1.5)
        for whisker in bp["whiskers"]:
            whisker.set(color=color, linewidth=1.0)
        for cap in bp["caps"]:
            cap.set(color=color, linewidth=1.0)
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=2.5, markerfacecolor=color, markeredgecolor=color, alpha=0.45)
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=label))

    ax.set_xticks(centers)
    ax.set_xticklabels([f"{v:g}" for v in var_values])
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$tU / I_Y$")
    ax.legend(handles=legend_handles, fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "times_box_variance"}


def plot_comp_time_mean_vs_variance(
    var: np.ndarray,
    s: dict[str, np.ndarray],
):
    fig, ax = plt.subplots(figsize=[6, 4])
    series_defs = [
        ("stats_ct_fp", "FP", COLORS["fp"]),
        ("stats_ct_lrp_dijkstra", "LRP (Dijkstra)", COLORS["lrp"]),
        ("stats_ct_lrp_fmm", "LRP (FMM)", COLORS["lrpFMM"]),
        ("stats_ct_fp_fmm", "FP (FMM)", COLORS["fpFMM"]),
        ("stats_ct_fp_dijkstra", "FP (Dijkstra)", COLORS["fpDijkstra"]),
        ("stats_ct_fp_fsm", "FP (FSM)", COLORS["fpFSM"]),
        ("stats_ct_fp_fm_asr", "FP (FM-ASR)", COLORS["fpFmAsr"]),
        ("stats_ct_fp_fm_asr_iso", r"FP (FM-ASR\, iso)", COLORS["fpFmAsrIso"]),
        ("stats_ct_fp_fm_asr_conic", r"FP (FM-ASR\, conic)", COLORS["fpFmAsrConic"]),
        ("stats_ct_fp_pypar2", "FP (PyPar2)", COLORS["fpPypar2"]),
    ]
    for key, label, color in series_defs:
        arr = s.get(key)
        if arr is None or not np.any(np.isfinite(arr[:, 0])):
            continue
        ax.errorbar(
            var,
            arr[:, 0],
            yerr=arr[:, 1],
            label=label,
            color=color,
            linewidth=LINEWIDTH,
            marker="o",
            markerfacecolor="None",
            capsize=3,
        )
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel("Computation Time (s)")
    ax.set_title("Mean Computation Time per Solver")
    ax.legend(fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "comp_time_mean_variance"}


def plot_comp_time_box_by_variance(
    var: np.ndarray,
    comp_time_samples: dict[str, dict[str, np.ndarray]],
):
    fig, ax = plt.subplots(figsize=[8, 4.5])
    series_defs = [
        ("ct_fp", "FP", COLORS["fp"]),
        ("ct_lrp_dijkstra", "LRP (Dijkstra)", COLORS["lrp"]),
        ("ct_lrp_fmm", "LRP (FMM)", COLORS["lrpFMM"]),
        ("ct_fp_fmm", "FP (FMM)", COLORS["fpFMM"]),
        ("ct_fp_dijkstra", "FP (Dijkstra)", COLORS["fpDijkstra"]),
        ("ct_fp_fsm", "FP (FSM)", COLORS["fpFSM"]),
        ("ct_fp_fm_asr", "FP (FM-ASR)", COLORS["fpFmAsr"]),
        ("ct_fp_fm_asr_iso", r"FP (FM-ASR\, iso)", COLORS["fpFmAsrIso"]),
        ("ct_fp_fm_asr_conic", r"FP (FM-ASR\, conic)", COLORS["fpFmAsrConic"]),
        ("ct_fp_pypar2", "FP (PyPar2)", COLORS["fpPypar2"]),
    ]
    var_values = [float(v) for v in np.asarray(var, dtype=float).ravel()]
    n_series = len(series_defs)
    width = 0.12
    half = (n_series - 1) / 2.0
    offsets = np.array([i - half for i in range(n_series)], dtype=float) * width
    centers = np.arange(len(var_values), dtype=float)

    legend_handles = []
    for series_idx, (key, label, color) in enumerate(series_defs):
        samples: list[np.ndarray] = []
        positions: list[float] = []
        for idx, variance_value in enumerate(var_values):
            sample_key = _resolve_sample_key(comp_time_samples, variance_value)
            if sample_key is None:
                continue
            arr = np.asarray(comp_time_samples[sample_key].get(key, []), dtype=float).ravel()
            finite = arr[np.isfinite(arr) & (arr > 0)]
            if finite.size == 0:
                continue
            samples.append(finite)
            positions.append(float(centers[idx] + offsets[series_idx]))
        if not samples:
            continue
        bp = ax.boxplot(samples, positions=positions, widths=width * 0.85, patch_artist=True, showfliers=True, manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.35)
        for median in bp["medians"]:
            median.set(color=color, linewidth=1.5)
        for whisker in bp["whiskers"]:
            whisker.set(color=color, linewidth=1.0)
        for cap in bp["caps"]:
            cap.set(color=color, linewidth=1.0)
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=2.5, markerfacecolor=color, markeredgecolor=color, alpha=0.45)
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=label))

    ax.set_xticks(centers)
    ax.set_xticklabels([f"{v:g}" for v in var_values])
    ax.set_yscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel("Computation Time (s)")
    ax.legend(handles=legend_handles, fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "comp_time_box_variance"}


def plot_ratio_mean_vs_variance(
    var: np.ndarray,
    ratio_fp: np.ndarray,
    ratio_fmm: np.ndarray,
    ratio_dijkstra: np.ndarray,
    ratio_fsm: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.plot(var, ratio_fp, linestyle="-", color=COLORS["fp"], marker="o", markerfacecolor="None", label="FP")
    ax.plot(var, ratio_fmm, linestyle="-", color=COLORS["fpFMM"], marker="o", markerfacecolor="None", label="FP (FMM)")
    ax.plot(var, ratio_dijkstra, linestyle="-", color=COLORS["fpDijkstra"], marker="o", markerfacecolor="None", label="FP (Dijkstra)")
    if ratio_fsm is not None and np.any(np.isfinite(ratio_fsm)):
        ax.plot(var, ratio_fsm, linestyle="-", color=COLORS["fpFSM"], marker="o", markerfacecolor="None", label="FP (FSM)")
    # ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathbb{E}\left[t_{\mathrm{min}}/(R_{\mathrm{min}}/dh)\right]$")
    ax.set_title("Variation of Mean Ratio with Variance")
    ax.legend(fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "ratio_mean_variance"}


def plot_ratio_var_vs_variance(
    var: np.ndarray,
    ratio_fp: np.ndarray,
    ratio_fmm: np.ndarray,
    ratio_dijkstra: np.ndarray,
    ratio_fsm: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.plot(var, ratio_fp, linestyle="-", color=COLORS["fp"], marker="o", markerfacecolor="None", label="FP")
    ax.plot(var, ratio_fmm, linestyle="-", color=COLORS["fpFMM"], marker="o", markerfacecolor="None", label="FP (FMM)")
    ax.plot(var, ratio_dijkstra, linestyle="-", color=COLORS["fpDijkstra"], marker="o", markerfacecolor="None", label="FP (Dijkstra)")
    if ratio_fsm is not None and np.any(np.isfinite(ratio_fsm)):
        ax.plot(var, ratio_fsm, linestyle="-", color=COLORS["fpFSM"], marker="o", markerfacecolor="None", label="FP (FSM)")
    # ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathrm{Var}\left(t_{\mathrm{min}}/(R_{\mathrm{min}}/dh)\right)$")
    ax.set_title("Variation of Variance Ratio with Variance")
    ax.legend(fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "ratio_var_variance"}


def plot_wasserstein_len_vs_variance(
    var: np.ndarray,
    w_fp_lrp: np.ndarray,
    w_fp_lrp_fmm: np.ndarray,
    w_fp_fmm: np.ndarray,
    w_fp_dijkstra: np.ndarray,
    i_y: float,
    w_fp_fsm: np.ndarray | None = None,
    w_fp_fm_asr: np.ndarray | None = None,
    w_fp_fm_asr_iso: np.ndarray | None = None,
    w_fp_fm_asr_conic: np.ndarray | None = None,
    w_fp_pypar2: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.plot(var, w_fp_lrp / i_y, linestyle="-", color=COLORS["lrp"], marker="o", markerfacecolor="None", label="$W_1$[FP, LRP]")
    ax.plot(var, w_fp_lrp_fmm / i_y, linestyle="-", color=COLORS["lrpFMM"], marker="o", markerfacecolor="None", label="$W_1$[FP, LRP(FMM)]")
    ax.plot(var, w_fp_fmm / i_y, linestyle="-", color=COLORS["fpFMM"], marker="o", markerfacecolor="None", label="$W_1$[FP, FP(FMM)]")
    ax.plot(var, w_fp_dijkstra / i_y, linestyle="-", color=COLORS["fpDijkstra"], marker="o", markerfacecolor="None", label="$W_1$[FP, FP(Dijkstra)]")
    if w_fp_fsm is not None and np.any(np.isfinite(w_fp_fsm)):
        ax.plot(var, w_fp_fsm / i_y, linestyle="-", color=COLORS["fpFSM"], marker="o", markerfacecolor="None", label="$W_1$[FP, FP(FSM)]")
    if w_fp_fm_asr is not None and np.any(np.isfinite(w_fp_fm_asr)):
        ax.plot(var, w_fp_fm_asr / i_y, linestyle="-", color=COLORS["fpFmAsr"], marker="o", markerfacecolor="None", label=r"$W_1$[FP,\,FM-ASR]")
    if w_fp_fm_asr_iso is not None and np.any(np.isfinite(w_fp_fm_asr_iso)):
        ax.plot(var, w_fp_fm_asr_iso / i_y, linestyle="-", color=COLORS["fpFmAsrIso"], marker="o", markerfacecolor="None", label=r"$W_1$[FP,\,FM-ASR\,iso]")
    if w_fp_fm_asr_conic is not None and np.any(np.isfinite(w_fp_fm_asr_conic)):
        ax.plot(var, w_fp_fm_asr_conic / i_y, linestyle="-", color=COLORS["fpFmAsrConic"], marker="o", markerfacecolor="None", label=r"$W_1$[FP,\,FM-ASR\,conic]")
    if w_fp_pypar2 is not None and np.any(np.isfinite(w_fp_pypar2)):
        ax.plot(var, w_fp_pypar2 / i_y, linestyle="-", color=COLORS["fpPypar2"], marker="o", markerfacecolor="None", label=r"$W_1$[FP,\,PyPar2]")
    # ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$W_1(\mathrm{len})/I_Y$")
    ax.set_title("Wasserstein Distance of Path-Length CDFs")
    ax.legend(fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "wasserstein_len_variance"}


def plot_length_cdf_for_variance(*, samples: dict[str, np.ndarray], variance_value: float, i_y: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    for key, label, color in [
        ("fp", "FP", COLORS["fp"]),
        ("lrp", "LRP", COLORS["lrp"]),
        ("lrp_fmm", "LRP (FMM)", COLORS["lrpFMM"]),
        ("fp_fmm", "FP (FMM)", COLORS["fpFMM"]),
        ("fp_dijkstra", "FP (Dijkstra)", COLORS["fpDijkstra"]),
        ("fp_fsm", "FP (FSM)", COLORS["fpFSM"]),
        ("fp_fm_asr", r"FP (FM-ASR)", COLORS["fpFmAsr"]),
        ("fp_fm_asr_iso", r"FP (FM-ASR\, iso)", COLORS["fpFmAsrIso"]),
        ("fp_fm_asr_conic", r"FP (FM-ASR\, conic)", COLORS["fpFmAsrConic"]),
        ("fp_pypar2", "FP (PyPar2)", COLORS["fpPypar2"]),
    ]:
        arr = np.asarray(samples.get(key, []), dtype=float).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        x, y = _empirical_cdf(finite / i_y)
        ax.plot(x, y, linestyle="-", color=color, label=label)
    ax.set_xlabel(r"$\ell / I_Y$")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(fr"Path-Length CDFs ($\sigma_Y^2={variance_value:g}$)")
    ax.legend(fontsize="9", loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "length_cdf", "variance": float(variance_value)}

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _short_hash(obj: dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def save_figure(
    fig,
    *,
    section: str,
    output_dir: Path,
    run_id: int,
    dpi: int,
    save_formats: list[str],
    metadata: dict[str, Any],
    unique_filenames: bool = False,
) -> list[FigureArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp() if unique_filenames else None
    h8 = _short_hash(metadata) if unique_filenames else None
    artifacts: list[FigureArtifact] = []

    for ext in save_formats:
        if unique_filenames:
            filename = f"run{run_id}_{section}_{stamp}_{h8}.{ext}"
        else:
            filename = f"run{run_id}_{section}.{ext}"
        path = output_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        artifacts.append(FigureArtifact(id=f"{section}:{ext}", filename=filename, path=path, section=section, metadata=metadata))

    return artifacts


def write_manifest(output_dir: Path, *, config: dict[str, Any], artifacts: list[FigureArtifact], warnings: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "warnings": warnings,
        "figures": [
            {
                "id": a.id,
                "section": a.section,
                "filename": a.filename,
                "path": str(a.path),
                "metadata": a.metadata,
            }
            for a in artifacts
        ],
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def run_stage4(
    cfg: VisualizationConfig,
    *,
    cache_dir: Path | None = None,
    unique_filenames: bool = False,
    plot_variances: list[float] | None = None,
    scatter_modes: list[str] | None = None,
    hausdorff_cdf: bool = False,
    entropy_hausdorff: bool = False,
) -> list[FigureArtifact]:
    cache_dir = cache_dir or default_cache_dir(cfg)
    payload = load_pickle(cache_dir / "stage3_plot_inputs.pkl")
    prepared = payload["plot_inputs"]
    warning_dict = payload.get("warnings", {})
    all_warnings = warning_dict.get("variance", []) + warning_dict.get("std", []) + list(prepared.get("derived_warnings", []))

    configure_style(cfg)
    artifacts: list[FigureArtifact] = []

    var = prepared["var"]
    s = prepared["stats"]
    i_y = prepared["i_y"]
    t_ref = prepared["t_ref"]
    field_size = float(prepared.get("field_size", cfg.field_size_ref))
    scatter_modes = ["mean"] if scatter_modes is None else scatter_modes
    ratio_mean_fp = ratio_mean_fmm = ratio_mean_dijkstra = ratio_mean_fsm = None
    ratio_var_fp = ratio_var_fmm = ratio_var_dijkstra = ratio_var_fsm = None

    if "times" in cfg.sections:
        u1 = cfg.k_g * cfg.dh
        length = compute_effective_length(field_size)
        analytical_tmin, analytical_warnings = compute_analytical_tmin_curve(
            var,
            length=length,
            velocity=u1,
            i_y=i_y,
            percentile=cfg.analytical_percentile,
        )
        all_warnings.extend(analytical_warnings)
        fig, meta = plot_characteristic_time(
            var,
            s["stats_fpt"],
            s["stats_mhr"],
            s["stats_mhr_fmm"],
            s["stats_fpt_fmm"],
            s["stats_fpt_dijkstra"],
            t_ref,
            cfg.dh,
            analytical_tmin=analytical_tmin,
            analytical_label=r"$\gamma^{\!\star}_{\mathcal{S}}$ (Analytical)", # f" $t_{{\\mathrm{{min}}}}$ (p={cfg.analytical_percentile:g})",
            stats_fpt_fsm=s.get("stats_fpt_fsm"),
            stats_fpt_fm_asr=s.get("stats_fpt_fm_asr"),
            stats_fpt_fm_asr_iso=s.get("stats_fpt_fm_asr_iso"),
            stats_fpt_fm_asr_conic=s.get("stats_fpt_fm_asr_conic"),
            stats_fpt_pypar2=s.get("stats_fpt_pypar2"),
        )
        artifacts.extend(save_figure(fig, section="times", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "times_box" in cfg.sections:
        scatter_samples_by_var = prepared.get("scatter_time_samples")
        if not isinstance(scatter_samples_by_var, dict):
            all_warnings.append("Requested 'times_box' section but Stage 3 cache is missing scatter_time_samples.")
        else:
            fig, meta = plot_times_box_by_variance(var, scatter_samples_by_var, t_ref, cfg.dh)
            artifacts.extend(save_figure(fig, section="times_box", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
            plt.close(fig)

    if "comp_time_mean" in cfg.sections:
        fig, meta = plot_comp_time_mean_vs_variance(var, s)
        artifacts.extend(save_figure(fig, section="comp_time_mean", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "comp_time_box" in cfg.sections:
        ct_samples = prepared.get("comp_time_samples")
        if not isinstance(ct_samples, dict):
            all_warnings.append("Requested 'comp_time_box' section but Stage 3 cache is missing comp_time_samples.")
        else:
            fig, meta = plot_comp_time_box_by_variance(var, ct_samples)
            artifacts.extend(save_figure(fig, section="comp_time_box", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
            plt.close(fig)

    if "scatter" in cfg.sections:
        if "mean" in scatter_modes:
            fig, meta = plot_scatter_tmin_vs_rmin(
                s["stats_fpt"],
                s["stats_fpt_fmm"],
                s["stats_fpt_dijkstra"],
                s["stats_mhr"],
                s["stats_mhr_fmm"],
                t_ref,
                cfg.dh,
                stats_fpt_fsm=s.get("stats_fpt_fsm"),
            )
            artifacts.extend(save_figure(fig, section="scatter", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
            plt.close(fig)
        if "all_points" in scatter_modes:
            scatter_samples_by_var = prepared.get("scatter_time_samples")
            if not isinstance(scatter_samples_by_var, dict):
                all_warnings.append("Requested scatter all-points mode but Stage 3 cache is missing scatter_time_samples.")
            else:
                requested = _resolve_requested_plot_variances(var, plot_variances)
                for variance_value in tqdm(requested, desc="Scatter all-points", unit="var", leave=False):
                    key = _resolve_sample_key(scatter_samples_by_var, variance_value)
                    if key is None:
                        all_warnings.append(f"Missing scatter samples for sigma2={variance_value}.")
                        continue
                    sample_dict = scatter_samples_by_var.get(key, {})
                    required = ("fpt", "fpt_fmm", "fpt_dijkstra", "mhr", "mhr_fmm")
                    if not all(rk in sample_dict for rk in required):
                        all_warnings.append(f"Incomplete scatter samples for sigma2={variance_value}; skipping scatter figure.")
                        continue
                    try:
                        fig, meta = plot_scatter_tmin_vs_rmin_all_points(sample_dict, t_ref=t_ref, dh=cfg.dh, variance_value=variance_value)
                    except ValueError:
                        all_warnings.append(f"Scatter samples invalid for sigma2={variance_value}; skipping scatter figure.")
                        continue
                    section = f"scatter_var{_value_slug(variance_value)}"
                    artifacts.extend(
                        save_figure(
                            fig,
                            section=section,
                            output_dir=cfg.output_dir,
                            run_id=cfg.run_id,
                            dpi=cfg.figure_dpi,
                            save_formats=cfg.save_formats,
                            metadata=meta,
                            unique_filenames=unique_filenames,
                        )
                    )
                    plt.close(fig)

    if "hausdorff" in cfg.sections:
        fig, meta = plot_hausdorff_vs_variance(
            var,
            s["stats_hausdorff_lrp"],
            s["stats_hausdorff_lrp_fmm"],
            s["stats_hausdorff_fmm"],
            s["stats_hausdorff_dijkstra"],
            i_y,
            stats_hausdorff_fsm=s.get("stats_hausdorff_fsm"),
            stats_hausdorff_fm_asr=s.get("stats_hausdorff_fm_asr"),
            stats_hausdorff_fm_asr_iso=s.get("stats_hausdorff_fm_asr_iso"),
            stats_hausdorff_fm_asr_conic=s.get("stats_hausdorff_fm_asr_conic"),
            stats_hausdorff_pypar2=s.get("stats_hausdorff_pypar2"),
        )
        artifacts.extend(save_figure(fig, section="hausdorff", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "hausdorff_box" in cfg.sections:
        hausdorff_samples = prepared.get("hausdorff_box_samples")
        if not isinstance(hausdorff_samples, dict):
            all_warnings.append("Requested 'hausdorff_box' section but Stage 3 cache is missing hausdorff_box_samples.")
        else:
            fig, meta = plot_hausdorff_box_by_variance(var, hausdorff_samples, i_y)
            artifacts.extend(save_figure(fig, section="hausdorff_box", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
            plt.close(fig)

    if "frechet" in cfg.sections:
        fig, meta = plot_frechet_vs_variance(
            var,
            s["stats_frechet_lrp"],
            s["stats_frechet_lrp_fmm"],
            s["stats_frechet_fmm"],
            s["stats_frechet_dijkstra"],
            i_y,
            stats_frechet_fsm=s.get("stats_frechet_fsm"),
            stats_frechet_fm_asr=s.get("stats_frechet_fm_asr"),
            stats_frechet_fm_asr_iso=s.get("stats_frechet_fm_asr_iso"),
            stats_frechet_fm_asr_conic=s.get("stats_frechet_fm_asr_conic"),
            stats_frechet_pypar2=s.get("stats_frechet_pypar2"),
        )
        artifacts.extend(save_figure(fig, section="frechet", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "endpoint" in cfg.sections:
        endpoint_samples = prepared.get("endpoint_box_samples")
        if not isinstance(endpoint_samples, dict):
            all_warnings.append("Requested 'endpoint' section but Stage 3 cache is missing endpoint_box_samples.")
        else:
            fig, meta = plot_endpoint_box_by_variance(var, endpoint_samples, i_y)
            artifacts.extend(save_figure(fig, section="endpoint", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
            plt.close(fig)

    if "ratio_mean" in cfg.sections:
        if ratio_mean_fp is None:
            scatter_samples_by_var = prepared.get("scatter_time_samples")
            (
                ratio_mean_fp,
                ratio_mean_fmm,
                ratio_mean_dijkstra,
                ratio_mean_fsm,
                ratio_var_fp,
                ratio_var_fmm,
                ratio_var_dijkstra,
                ratio_var_fsm,
            ) = _compute_ratio_moments_series(
                var,
                scatter_samples_by_var=scatter_samples_by_var,
                dh=cfg.dh,
                warnings=all_warnings,
            )
        fig, meta = plot_ratio_mean_vs_variance(var, ratio_mean_fp, ratio_mean_fmm, ratio_mean_dijkstra, ratio_mean_fsm)
        artifacts.extend(save_figure(fig, section="ratio_mean", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "ratio_var" in cfg.sections:
        if ratio_var_fp is None:
            scatter_samples_by_var = prepared.get("scatter_time_samples")
            (
                ratio_mean_fp,
                ratio_mean_fmm,
                ratio_mean_dijkstra,
                ratio_mean_fsm,
                ratio_var_fp,
                ratio_var_fmm,
                ratio_var_dijkstra,
                ratio_var_fsm,
            ) = _compute_ratio_moments_series(
                var,
                scatter_samples_by_var=scatter_samples_by_var,
                dh=cfg.dh,
                warnings=all_warnings,
            )
        fig, meta = plot_ratio_var_vs_variance(var, ratio_var_fp, ratio_var_fmm, ratio_var_dijkstra, ratio_var_fsm)
        artifacts.extend(save_figure(fig, section="ratio_var", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
        plt.close(fig)

    if "wasserstein_len" in cfg.sections:
        if "wasserstein_len" not in prepared:
            all_warnings.append("Requested 'wasserstein_len' section but Stage 3 cache is missing wasserstein_len data.")
        else:
            wl = prepared["wasserstein_len"]
            w_fp_lrp = _coerce_series(wl.get("fp_lrp", []), len(var), label="wasserstein_len.fp_lrp", warnings=all_warnings)
            w_fp_lrp_fmm = _coerce_series(wl.get("fp_lrpFMM", []), len(var), label="wasserstein_len.fp_lrpFMM", warnings=all_warnings)
            w_fp_fmm = _coerce_series(wl.get("fp_fpFMM", []), len(var), label="wasserstein_len.fp_fpFMM", warnings=all_warnings)
            w_fp_dijkstra = _coerce_series(wl.get("fp_fpDijkstra", []), len(var), label="wasserstein_len.fp_fpDijkstra", warnings=all_warnings)
            w_fp_fsm = _coerce_series(wl["fp_fpFSM"], len(var), label="wasserstein_len.fp_fpFSM", warnings=all_warnings) if "fp_fpFSM" in wl else None
            w_fp_fm_asr = _coerce_series(wl.get("fp_fm_asr", []), len(var), label="wasserstein_len.fp_fm_asr", warnings=all_warnings)
            w_fp_fm_asr_iso = _coerce_series(wl.get("fp_fm_asr_iso", []), len(var), label="wasserstein_len.fp_fm_asr_iso", warnings=all_warnings)
            w_fp_fm_asr_conic = _coerce_series(wl.get("fp_fm_asr_conic", []), len(var), label="wasserstein_len.fp_fm_asr_conic", warnings=all_warnings)
            w_fp_pypar2 = _coerce_series(wl.get("fp_pypar2", []), len(var), label="wasserstein_len.fp_pypar2", warnings=all_warnings)
            fig, meta = plot_wasserstein_len_vs_variance(
                var,
                w_fp_lrp,
                w_fp_lrp_fmm,
                w_fp_fmm,
                w_fp_dijkstra,
                i_y,
                w_fp_fsm=w_fp_fsm,
                w_fp_fm_asr=w_fp_fm_asr,
                w_fp_fm_asr_iso=w_fp_fm_asr_iso,
                w_fp_fm_asr_conic=w_fp_fm_asr_conic,
                w_fp_pypar2=w_fp_pypar2,
            )
            artifacts.extend(save_figure(fig, section="wasserstein_len", output_dir=cfg.output_dir, run_id=cfg.run_id, dpi=cfg.figure_dpi, save_formats=cfg.save_formats, metadata=meta, unique_filenames=unique_filenames))
            plt.close(fig)

            samples_by_var = prepared.get("length_cdf_samples")
            if not isinstance(samples_by_var, dict):
                all_warnings.append("Requested Wasserstein/CDF plot but Stage 3 cache is missing length_cdf_samples.")
            else:
                requested = _resolve_requested_plot_variances(var, plot_variances)
                for variance_value in tqdm(requested, desc="Length CDF figures", unit="var", leave=False):
                    key = _resolve_sample_key(samples_by_var, variance_value)
                    if key is None:
                        all_warnings.append(f"Missing length CDF samples for sigma2={variance_value}.")
                        continue
                    sample_dict = samples_by_var.get(key, {})
                    required = ("fp", "lrp", "lrp_fmm", "fp_fmm", "fp_dijkstra")
                    if not all(rk in sample_dict for rk in required):
                        all_warnings.append(f"Incomplete length CDF samples for sigma2={variance_value}; skipping CDF figure.")
                        continue
                    fig, meta = plot_length_cdf_for_variance(samples=sample_dict, variance_value=variance_value, i_y=i_y)
                    section = f"cdf_len_var{_value_slug(variance_value)}"
                    artifacts.extend(
                        save_figure(
                            fig,
                            section=section,
                            output_dir=cfg.output_dir,
                            run_id=cfg.run_id,
                            dpi=cfg.figure_dpi,
                            save_formats=cfg.save_formats,
                            metadata=meta,
                            unique_filenames=unique_filenames,
                        )
                    )
                    plt.close(fig)

    if hausdorff_cdf:
        hcdf_data = prepared.get("hausdorff_cdf_ensemble")
        if not isinstance(hcdf_data, dict) or not hcdf_data:
            all_warnings.append("Requested hausdorff CDF ensemble plot but Stage 3 cache is missing hausdorff_cdf_ensemble data.")
        else:
            requested = _resolve_requested_plot_variances(var, plot_variances)
            for variance_value in tqdm(requested, desc="Hausdorff CDF figures", unit="var", leave=False):
                key = _resolve_sample_key(hcdf_data, variance_value)
                if key is None:
                    all_warnings.append(f"Missing hausdorff CDF ensemble data for sigma2={variance_value}.")
                    continue
                fig, meta = plot_hausdorff_cdf_ensemble(hcdf_data[key], variance_value, i_y)
                section = f"hausdorff_cdf_var{_value_slug(variance_value)}"
                artifacts.extend(
                    save_figure(
                        fig,
                        section=section,
                        output_dir=cfg.output_dir,
                        run_id=cfg.run_id,
                        dpi=cfg.figure_dpi,
                        save_formats=cfg.save_formats,
                        metadata=meta,
                        unique_filenames=unique_filenames,
                    )
                )
                plt.close(fig)

    if entropy_hausdorff:
        eh_data = prepared.get("entropy_hausdorff_samples")
        if not isinstance(eh_data, dict) or not eh_data:
            all_warnings.append("Requested entropy-hausdorff plot but Stage 3 cache is missing entropy_hausdorff_samples data.")
        else:
            requested = _resolve_requested_plot_variances(var, plot_variances)
            for variance_value in tqdm(requested, desc="Entropy+Hausdorff figures", unit="var", leave=False):
                key = _resolve_sample_key(eh_data, variance_value)
                if key is None:
                    all_warnings.append(f"Missing entropy+hausdorff samples for sigma2={variance_value}.")
                    continue
                fig, meta = plot_entropy_vs_hausdorff(eh_data[key], variance_value, i_y)
                section = f"entropy_hausdorff_var{_value_slug(variance_value)}"
                artifacts.extend(
                    save_figure(
                        fig,
                        section=section,
                        output_dir=cfg.output_dir,
                        run_id=cfg.run_id,
                        dpi=cfg.figure_dpi,
                        save_formats=cfg.save_formats,
                        metadata=meta,
                        unique_filenames=unique_filenames,
                    )
                )
                plt.close(fig)

    write_manifest(cfg.output_dir, config=asdict(cfg), artifacts=artifacts, warnings=all_warnings)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: render figures from cached plotting inputs")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override for figures/manifest")
    parser.add_argument(
        "--plot-variances",
        type=str,
        default=None,
        help="Comma-separated variance values used by CDF and scatter(all_points) plots (default: first available variance).",
    )
    parser.add_argument(
        "--scatter-mode",
        type=str,
        default="mean",
        help="Comma-separated scatter modes: mean, all_points, or both.",
    )
    parser.add_argument(
        "--unique-filenames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include timestamp/hash in output figure filenames",
    )
    parser.add_argument(
        "--hausdorff-cdf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot ensemble-averaged empirical CDF of Hausdorff nearest-neighbor distances",
    )
    parser.add_argument(
        "--entropy-hausdorff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot per-MC Renyi entropy and Hausdorff summary stats (mean/median/max) vs MC index",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir.expanduser().resolve()
    artifacts = run_stage4(
        cfg,
        cache_dir=args.cache_dir,
        unique_filenames=args.unique_filenames,
        plot_variances=_parse_plot_variances(args.plot_variances),
        scatter_modes=_parse_scatter_modes(args.scatter_mode),
        hausdorff_cdf=args.hausdorff_cdf,
        entropy_hausdorff=args.entropy_hausdorff,
    )
    print(f"Generated {len(artifacts)} figure artifact(s) in {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
