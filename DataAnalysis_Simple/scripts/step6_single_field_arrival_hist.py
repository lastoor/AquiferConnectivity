#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import step1_index as step1
import step4_plot as step4


@dataclass(slots=True)
class ArrivalSimulationResult:
    logk_field: np.ndarray
    arrival_times: np.ndarray
    status_counts: dict[int, int]
    n_particles: int
    success_fraction: float
    bad_fraction: float


def _add_search_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _import_solver_modules() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    algo_src = repo_root / "Algorithm" / "src"
    flowsim_path = Path(os.environ.get("FLOWSIM_PATH", str(repo_root.parent / "FlowSimulation"))).expanduser().resolve()

    if not algo_src.is_dir():
        raise RuntimeError(f"Algorithm src path does not exist: {algo_src}")
    if not flowsim_path.is_dir():
        raise RuntimeError(
            f"FlowSimulation path does not exist: {flowsim_path}. "
            "Set FLOWSIM_PATH to your FlowSimulation repository root."
        )

    os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    _add_search_path(algo_src)
    _add_search_path(flowsim_path)

    try:
        from simulation import grf_generation as sim_grf
        from simulation import hydraulic as sim_hyd
        from simulation import particle_tracking as sim_pt
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Failed to import solver dependencies. "
            "Ensure Algorithm/src and FlowSimulation are available and dependencies are installed."
        ) from exc

    return {
        "sim_grf": sim_grf,
        "sim_hyd": sim_hyd,
        "sim_pt": sim_pt,
    }


def _build_domain(field_size: int) -> dict[str, Any]:
    x0 = field_size // 20 - 1
    x1 = field_size - field_size // 20 - 1
    y0 = field_size // 20 - 1
    y1 = field_size - field_size // 20 - 1
    return {
        "x": np.arange(0, field_size, 1.0),
        "y": np.arange(0, field_size, 1.0),
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }


def _value_slug(value: float) -> str:
    return str(f"{float(value):g}").replace("-", "m").replace(".", "p")


def _compute_std_and_var(*, std: float | None, var: float | None) -> tuple[float, float]:
    if std is None and var is None:
        raise ValueError("Either std or var must be provided.")
    if std is not None and var is not None:
        raise ValueError("Only one of std or var can be provided.")
    if std is not None:
        if std <= 0:
            raise ValueError("std must be > 0.")
        return float(std), float(std**2)
    assert var is not None
    if var <= 0:
        raise ValueError("var must be > 0.")
    return float(np.sqrt(var)), float(var)


def _empirical_cdf(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def _extract_success_arrival_times(pt: np.ndarray, length_arr: np.ndarray, status_arr: np.ndarray) -> np.ndarray:
    pt = np.asarray(pt, dtype=float)
    length_arr = np.asarray(length_arr, dtype=np.int32)
    status_arr = np.asarray(status_arr, dtype=np.int32)
    valid = (status_arr == 0) & (length_arr > 0)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.asarray([], dtype=float)
    arrival_times = pt[idx, length_arr[idx] - 1]
    arrival_times = np.asarray(arrival_times, dtype=float)
    return arrival_times[np.isfinite(arrival_times) & (arrival_times >= 0.0)]


def _analytical_cdf_on_tau_grid(
    tau_grid: np.ndarray,
    *,
    length: float,
    velocity: float,
    i_y: float,
    sigma_y_sq: float,
) -> np.ndarray:
    t = np.asarray(tau_grid, dtype=float) * (i_y / velocity)
    euler_const = float(getattr(step4.constants, "euler_gamma", np.euler_gamma))
    out = np.asarray(
        [
            step4._g_tau(
                float(t_i),
                length=length,
                velocity=velocity,
                i_y=i_y,
                sigma_y_sq=sigma_y_sq,
                euler_const=euler_const,
            )
            for t_i in t
        ],
        dtype=float,
    )
    return np.clip(out, 0.0, 1.0)


def _ks_distance_from_cdfs(
    empirical_x: np.ndarray,
    empirical_y: np.ndarray,
    analytical_x: np.ndarray,
    analytical_y: np.ndarray,
) -> float:
    if empirical_x.size == 0 or analytical_x.size == 0:
        return float("nan")
    analytical_interp = np.interp(empirical_x, analytical_x, analytical_y, left=analytical_y[0], right=analytical_y[-1])
    return float(np.max(np.abs(empirical_y - analytical_interp)))


def plot_arrival_cdf_comparison(
    tau_emp: np.ndarray,
    ecdf: np.ndarray,
    tau_grid: np.ndarray,
    cdf_analytical: np.ndarray,
    *,
    title: str,
    cdf_mc_mean: np.ndarray | None = None,
    cdf_mc_band_low: np.ndarray | None = None,
    cdf_mc_band_high: np.ndarray | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=[5, 4])
    if cdf_mc_band_low is not None and cdf_mc_band_high is not None:
        ax.fill_between(
            tau_grid,
            cdf_mc_band_low,
            cdf_mc_band_high,
            color="#8C8C8C",
            alpha=0.25,
            linewidth=0.0,
            label="MC CDF band",
        )
    if cdf_mc_mean is not None:
        ax.plot(tau_grid, cdf_mc_mean, color="#595959", linewidth=1.4, linestyle="-", label="MC mean CDF")
    ax.step(tau_emp, ecdf, where="post", color="black", linewidth=2.0, label="Empirical CDF (status=0)")
    ax.plot(tau_grid, cdf_analytical, color="#C46552", linewidth=2.0, linestyle="--", label=r"Analytical $\_g_{\tau}$")
    ax.set_xlabel(r"$tU/I_Y$")
    ax.set_ylabel("CDF")
    ax.set_ylim([0.0, 1.01])
    ax.grid(True, which="both")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _ecdf_on_grid(samples: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    x, y = _empirical_cdf(samples)
    if x.size == 0:
        return np.full_like(tau_grid, np.nan, dtype=float)
    return np.interp(tau_grid, x, y, left=0.0, right=1.0)


def _mc_cdf_band(
    tau_by_mc: list[np.ndarray],
    tau_grid: np.ndarray,
    *,
    band_low: float,
    band_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cdf_rows = np.asarray([_ecdf_on_grid(samples, tau_grid) for samples in tau_by_mc], dtype=float)
    cdf_mean = np.nanmean(cdf_rows, axis=0)
    cdf_low = np.nanpercentile(cdf_rows, band_low, axis=0)
    cdf_high = np.nanpercentile(cdf_rows, band_high, axis=0)
    return cdf_mean, cdf_low, cdf_high


def simulate_single_field_arrivals(
    *,
    field_size: int,
    length_scale: float,
    std: float,
    minpadding: int,
    dh: float,
    density_particle: int,
    eps_speed: float,
    ds_min: float,
    ds_max: float,
    tol: float,
    y_policy: str,
    max_points: int,
    max_iters: int,
    rk45_min_success_frac: float,
    rk45_max_bad_frac: float,
    seed: int,
) -> ArrivalSimulationResult:
    modules = _import_solver_modules()
    sim_grf = modules["sim_grf"]
    sim_hyd = modules["sim_hyd"]
    sim_pt = modules["sim_pt"]

    np.random.seed(seed)
    cov = sim_grf.ExpCov(sigma=std, length_scale=length_scale)
    pts = (np.linspace(0, 1, field_size),) * 2
    mean = np.zeros((field_size, field_size))
    grf_gen = sim_grf.GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=minpadding)
    logk_field = np.asarray(grf_gen.sample(), dtype=float)
    k_field = np.exp(logk_field)

    domain = _build_domain(field_size)
    x = domain["x"]
    y = domain["y"]
    x0 = int(domain["x0"])
    x1 = int(domain["x1"])
    y0 = int(domain["y0"])
    y1 = int(domain["y1"])

    h, msg, is_success = sim_hyd.solve_head(k_field, x, y, dh=dh, rtol=1e-10, atol=1e-20, max_it=10000)
    if not is_success:
        raise RuntimeError(f"Head solver failed: {msg}")

    vx, vy = sim_hyd.compute_velocity(k_field, x, y, h, output_location="face")

    n_particles = max(2, int(density_particle * abs(y1 - y0)))
    seeds_x = np.full(n_particles, x0, np.float32)
    seeds_y = np.linspace(y0, y1, n_particles).astype(np.float32)
    track_args = {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "right_x": x1,
        "velocity_location": "face",
        "eps_speed": eps_speed,
        "ds_min": ds_min,
        "ds_max": ds_max,
        "tol": tol,
        "y_policy": y_policy,
        "max_points": max_points,
        "max_iters": max_iters,
        "threads_per_block": 256,
        "dtype": np.float64,
    }
    _px, _py, pt, length_arr, status_arr = sim_pt.particle_tracking_cpu(track_args)
    status_counter = Counter(status_arr.tolist())
    frac_success = status_counter.get(0, 0) / max(1, n_particles)
    frac_bad = (status_counter.get(1, 0) + status_counter.get(4, 0)) / max(1, n_particles)
    if frac_success < rk45_min_success_frac or frac_bad > rk45_max_bad_frac:
        raise RuntimeError(
            "Invalid velocity field for tracking. "
            f"status={dict(status_counter)}, success_frac={frac_success:.3f}, bad_frac={frac_bad:.3f}"
        )

    arrival_times = _extract_success_arrival_times(pt, length_arr, status_arr)
    if arrival_times.size == 0:
        raise RuntimeError("No successful particle arrivals (status=0) with finite non-negative arrival times.")

    return ArrivalSimulationResult(
        logk_field=logk_field,
        arrival_times=arrival_times,
        status_counts={int(k): int(v) for k, v in status_counter.items()},
        n_particles=int(n_particles),
        success_fraction=float(frac_success),
        bad_fraction=float(frac_bad),
    )


def run_step6(
    cfg: step1.VisualizationConfig,
    *,
    field_size: int | None,
    length_scale: float | None,
    std: float | None,
    var: float | None,
    minpadding: int,
    dh: float | None,
    k_g: float | None,
    density_particle: int,
    eps_speed: float,
    ds_min: float,
    ds_max: float,
    tol: float,
    y_policy: str,
    max_points: int,
    max_iters: int,
    rk45_min_success_frac: float,
    rk45_max_bad_frac: float,
    n_mc: int,
    seed_start: int,
    band_low: float,
    band_high: float,
    output_dir: Path | None,
    figure_dpi: int | None,
    save_npz: bool,
    save_json: bool,
) -> tuple[Path, Path | None, Path | None]:
    field_size_eff = int(cfg.field_size_ref if field_size is None else field_size)
    length_scale_eff = float(cfg.length_scale if length_scale is None else length_scale)
    std_eff, var_eff = _compute_std_and_var(std=std, var=var)
    dh_eff = float(cfg.dh if dh is None else dh)
    k_g_eff = float(cfg.k_g if k_g is None else k_g)
    velocity = float(k_g_eff * dh_eff)
    if velocity <= 0.0 or not np.isfinite(velocity):
        raise ValueError(f"Invalid advective velocity k_g*dh={velocity}; expected positive finite.")

    if n_mc <= 0:
        raise ValueError("n_mc must be > 0.")
    if not (0.0 <= band_low <= 100.0 and 0.0 <= band_high <= 100.0 and band_low <= band_high):
        raise ValueError("band_low and band_high must satisfy 0 <= band_low <= band_high <= 100.")

    step4.configure_style(cfg)
    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.figure_dpi if figure_dpi is None else figure_dpi)

    stem = (
        f"run{cfg.run_id}_singlefield_std{_value_slug(std_eff)}_var{_value_slug(var_eff)}_"
        f"size{field_size_eff}_seed{seed_start}_mc{n_mc}"
    )

    i_y = float(length_scale_eff * field_size_eff)
    length = float(field_size_eff - 2 * (field_size_eff // 20))
    seeds_used: list[int] = []
    tau_by_mc: list[np.ndarray] = []
    n_particles_by_mc: list[int] = []
    n_success_by_mc: list[int] = []
    success_fraction_by_mc: list[float] = []
    bad_fraction_by_mc: list[float] = []
    status_counts_by_mc: list[dict[str, int]] = []

    for j in range(n_mc):
        seed = int(seed_start + j)
        simulation = simulate_single_field_arrivals(
            field_size=field_size_eff,
            length_scale=length_scale_eff,
            std=std_eff,
            minpadding=minpadding,
            dh=dh_eff,
            density_particle=density_particle,
            eps_speed=eps_speed,
            ds_min=ds_min,
            ds_max=ds_max,
            tol=tol,
            y_policy=y_policy,
            max_points=max_points,
            max_iters=max_iters,
            rk45_min_success_frac=rk45_min_success_frac,
            rk45_max_bad_frac=rk45_max_bad_frac,
            seed=seed,
        )
        tau_samples_mc = np.asarray(simulation.arrival_times, dtype=float) * velocity / i_y
        tau_samples_mc = tau_samples_mc[np.isfinite(tau_samples_mc) & (tau_samples_mc >= 0.0)]
        if tau_samples_mc.size == 0:
            continue
        seeds_used.append(seed)
        tau_by_mc.append(tau_samples_mc)
        n_particles_by_mc.append(int(simulation.n_particles))
        n_success_by_mc.append(int(len(tau_samples_mc)))
        success_fraction_by_mc.append(float(simulation.success_fraction))
        bad_fraction_by_mc.append(float(simulation.bad_fraction))
        status_counts_by_mc.append({str(k): int(v) for k, v in simulation.status_counts.items()})

    if not tau_by_mc:
        raise RuntimeError("No valid successful particle arrivals across all Monte Carlo realizations.")

    tau_samples = np.concatenate(tau_by_mc)
    if tau_samples.size == 0:
        raise RuntimeError("No finite non-negative dimensionless arrival times available for CDF analysis.")

    tau_emp, ecdf = _empirical_cdf(tau_samples)
    tau_lo = max(0.0, float(tau_emp[0]) * 0.8)
    tau_hi = float(tau_emp[-1]) * 1.2 + 1e-12
    tau_grid = np.linspace(tau_lo, tau_hi, 512)
    analytical_cdf = _analytical_cdf_on_tau_grid(
        tau_grid,
        length=length,
        velocity=velocity,
        i_y=i_y,
        sigma_y_sq=var_eff,
    )
    ks_dist = _ks_distance_from_cdfs(tau_emp, ecdf, tau_grid, analytical_cdf)
    cdf_mc_mean, cdf_mc_band_low, cdf_mc_band_high = _mc_cdf_band(
        tau_by_mc,
        tau_grid,
        band_low=band_low,
        band_high=band_high,
    )

    summary = {
        "run_id": int(cfg.run_id),
        "field_size": int(field_size_eff),
        "length_scale": float(length_scale_eff),
        "std": float(std_eff),
        "var": float(var_eff),
        "dh": float(dh_eff),
        "k_g": float(k_g_eff),
        "velocity": float(velocity),
        "i_y": float(i_y),
        "length": float(length),
        "seed_start": int(seed_start),
        "n_mc_requested": int(n_mc),
        "n_mc_used": int(len(tau_by_mc)),
        "seeds_used": [int(s) for s in seeds_used],
        "n_particles": int(np.sum(np.asarray(n_particles_by_mc, dtype=int))),
        "n_success": int(len(tau_samples)),
        "n_particles_by_mc": n_particles_by_mc,
        "n_success_by_mc": n_success_by_mc,
        "success_fraction_by_mc": success_fraction_by_mc,
        "bad_fraction_by_mc": bad_fraction_by_mc,
        "status_counts_by_mc": status_counts_by_mc,
        "tau_mean": float(np.mean(tau_samples)),
        "tau_median": float(np.median(tau_samples)),
        "tau_min": float(np.min(tau_samples)),
        "tau_max": float(np.max(tau_samples)),
        "ks_distance_empirical_vs_analytical": float(ks_dist),
    }

    title = (
        rf"Arrival CDF, $\sigma_Y^2={var_eff:g}$, $L={length:g}$, "
        rf"$n_{{\mathrm{{success}}}}={len(tau_samples)}$, MC={len(tau_by_mc)}"
    )
    fig = plot_arrival_cdf_comparison(
        tau_emp,
        ecdf,
        tau_grid,
        analytical_cdf,
        title=title,
        cdf_mc_mean=cdf_mc_mean,
        cdf_mc_band_low=cdf_mc_band_low,
        cdf_mc_band_high=cdf_mc_band_high,
    )

    fig_path = out_dir / f"{stem}_arrival_cdf_vs_analytical.png"
    npz_path = out_dir / f"{stem}_arrival_times.npz"
    json_path = out_dir / f"{stem}_arrival_summary.json"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    npz_path_out: Path | None = None
    if save_npz:
        np.savez(
            npz_path,
            tau_samples=tau_samples,
            tau_empirical=tau_emp,
            cdf_empirical=ecdf,
            tau_grid=tau_grid,
            cdf_analytical=analytical_cdf,
            cdf_mc_mean=cdf_mc_mean,
            cdf_mc_band_low=cdf_mc_band_low,
            cdf_mc_band_high=cdf_mc_band_high,
            seeds_used=np.asarray(seeds_used, dtype=np.int32),
            n_success_by_mc=np.asarray(n_success_by_mc, dtype=np.int32),
            tau_samples_by_mc=np.asarray(tau_by_mc, dtype=object),
        )
        npz_path_out = npz_path

    json_path_out: Path | None = None
    if save_json:
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        json_path_out = json_path

    return fig_path, npz_path_out, json_path_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 6: single-field arrival-time distribution of all successful particles vs analytical _g_tau CDF"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--field-size", type=int, default=None, help="Grid size, default: cfg.field_size_ref")
    parser.add_argument("--length-scale", type=float, default=None, help="Normalized length scale, default: cfg.length_scale")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--std", type=float, default=None, help="GRF standard deviation")
    group.add_argument("--var", type=float, default=None, help="GRF variance")
    parser.add_argument("--minpadding", type=int, default=0, help="Minimal padding for GRF generation")
    parser.add_argument("--dh", type=float, default=None, help="Hydraulic head x-gradient; default cfg.dh")
    parser.add_argument("--k-g", type=float, default=None, help="Reference conductivity scale; default cfg.k_g")
    parser.add_argument("--density-particle", type=int, default=4)
    parser.add_argument("--n-mc", type=int, default=10, help="Number of Monte Carlo realizations")
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed for MC sequence (seed_start + i)")
    parser.add_argument("--band-low", type=float, default=5.0, help="Lower percentile for MC CDF band")
    parser.add_argument("--band-high", type=float, default=95.0, help="Upper percentile for MC CDF band")
    parser.add_argument("--eps-speed", type=float, default=1e-30)
    parser.add_argument("--ds-min", type=float, default=1e-20)
    parser.add_argument("--ds-max", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--y-policy", choices=["stop", "reflect", "clip"], default="stop")
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--max-iters", type=int, default=300000)
    parser.add_argument("--rk45-min-success-frac", type=float, default=0.5)
    parser.add_argument("--rk45-max-bad-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None, help="Alias for --seed-start (for compatibility)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-dpi", type=int, default=None)
    parser.add_argument(
        "--save-npz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save NPZ data artifact (default: disabled).",
    )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save JSON summary artifact (default: disabled).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = step1.load_config(args.config)
    seed_start = int(args.seed_start)
    if args.seed is not None:
        if args.seed_start != 0 and int(args.seed) != int(args.seed_start):
            raise ValueError(f"Conflicting --seed ({args.seed}) and --seed-start ({args.seed_start}).")
        seed_start = int(args.seed)
    fig_path, npz_path, json_path = run_step6(
        cfg,
        field_size=args.field_size,
        length_scale=args.length_scale,
        std=args.std,
        var=args.var,
        minpadding=args.minpadding,
        dh=args.dh,
        k_g=args.k_g,
        density_particle=args.density_particle,
        eps_speed=args.eps_speed,
        ds_min=args.ds_min,
        ds_max=args.ds_max,
        tol=args.tol,
        y_policy=args.y_policy,
        max_points=args.max_points,
        max_iters=args.max_iters,
        rk45_min_success_frac=args.rk45_min_success_frac,
        rk45_max_bad_frac=args.rk45_max_bad_frac,
        n_mc=args.n_mc,
        seed_start=seed_start,
        band_low=args.band_low,
        band_high=args.band_high,
        output_dir=args.output_dir,
        figure_dpi=args.figure_dpi,
        save_npz=args.save_npz,
        save_json=args.save_json,
    )
    print(f"Saved figure: {fig_path}")
    if npz_path is not None:
        print(f"Saved data: {npz_path}")
    else:
        print("NPZ data save is disabled.")
    if json_path is not None:
        print(f"Saved summary: {json_path}")
    else:
        print("JSON summary save is disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
