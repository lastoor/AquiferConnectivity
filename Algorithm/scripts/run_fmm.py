import argparse
import cProfile
import gc
import multiprocessing as mp
import os
import pstats
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Ensure local package imports work when running as: python scripts/run_fmm.py
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_ROOT))

from fmm_core import compute_fmm_path
from fmm_core.fsm import compute_fsm_path as compute_fsm_path_velocity
from fmm_core.metrics import distance_distribution
from fmm_core.speed import prepare_fmm_speed
from simulation import grf_generation as sim_grf
from simulation import hydraulic as sim_hyd
from simulation import particle_tracking as sim_pt
from simulation import path_finding as sim_path
from simulation import utils as sim_utils

# Add libgfortran.so.5 and libstdc++.so.6 in /apps/generic/gcc/13.3.0/lib64 as an environment variable
os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

global_grf_gen = None


MC_SCHEMA_VERSION = 10

# Flat output schema v7: deterministic and explicit for analysis pipelines.
MC_FIELDS = [
    "schema_version",
    "mc_index",
    "trial_used",
    "status_code",
    "status_message",
    "len_lrp",
    "len_lrp_fmm",
    "len_fp",
    "len_fp_dijkstra",
    "len_fp_fmm",
    "len_fp_fsm",
    "travel_time_fp",
    "travel_time_dijkstra",
    "travel_time_fmm",
    "travel_time_fsm",
    "resistance_lrp",
    "resistance_lrp_fmm",
    "endpoint_dy_fp_vs_lrp",
    "endpoint_dy_fp_vs_lrp_fmm",
    "endpoint_dy_fp_vs_dijkstra",
    "endpoint_dy_fp_vs_fmm",
    "endpoint_dy_fp_vs_fsm",
    "endpoint_method",
    "endpoint_x_ref",
    "x_end_fp",
    "x_end_lrp",
    "x_end_lrp_fmm",
    "x_end_fp_dijkstra",
    "x_end_fp_fmm",
    "x_end_fp_fsm",
    "y_end_fp",
    "y_end_lrp",
    "y_end_lrp_fmm",
    "y_end_fp_dijkstra",
    "y_end_fp_fmm",
    "y_end_fp_fsm",
    "pt_solver",
    "pt_backend",
    "pt_num_particles",
    "pt_velocity_location",
    "pt_align_with_modflow_velocity",
    "fmm_velocity_layout",
    "fmm_face_output",
    "fmm_face_mapping",
    "fmm_solver",
    "fmm_backtracking",
    "fmm_zero_policy",
    "fmm_min_speed",
    "fmm_zero_count",
    "fmm_nontraversable_count",
    "seed_base",
    "seed_effective_run",
    "seed_mode",
    "variance_seed_policy",
    "variance_rank",
    "variance_seed_override",
    "seed_mc",
    "seed_trial",
]

OBJECT_FIELDS = {
    "status_message",
    "endpoint_method",
    "pt_solver",
    "pt_backend",
    "pt_velocity_location",
    "fmm_velocity_layout",
    "fmm_face_output",
    "fmm_face_mapping",
    "fmm_solver",
    "fmm_backtracking",
    "fmm_zero_policy",
    "seed_mode",
    "variance_seed_policy",
}

INT_FIELDS = {
    "schema_version",
    "mc_index",
    "trial_used",
    "status_code",
    "pt_num_particles",
}

BOOL_FIELDS = {
    "pt_align_with_modflow_velocity",
}


def init_worker(grf_gen, seed_mode: str = "strict_mc", effective_run_seed: int | None = None):
    """Initializer for simulation pool workers."""
    global global_grf_gen
    global_grf_gen = grf_gen
    if effective_run_seed is None:
        return
    if seed_mode in {"global_stream", "best_effort"}:
        # Scheduling-dependent mode: worker seed depends on worker PID.
        worker_seed = derive_seed(int(effective_run_seed), int(os.getpid()))
        np.random.seed(int(np.uint32(worker_seed & 0xFFFFFFFF)))


def build_base_record(mc_index: int, trial_used: int | None = None) -> dict[str, Any]:
    record = {
        "schema_version": MC_SCHEMA_VERSION,
        "mc_index": int(mc_index),
        "trial_used": -1 if trial_used is None else int(trial_used),
        "status_code": 9,
        "status_message": "failed",
    }
    for key in MC_FIELDS:
        if key not in record:
            record[key] = np.nan
    return record


def _hausdorff_detail(D: np.ndarray) -> dict[str, Any]:
    return {
        "distances": D,
        "max": float(np.max(D)),
        "mean": float(np.mean(D)),
        "std": float(np.std(D)),
        "median": float(np.median(D)),
    }


def _renyi_entropy_vx_profile(vx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Renyi entropy profiles S_1(x) and S_2(x) from |v_x(y)| at each x-column.

    For each column i, the probability distribution is p_j = |vx[j,i]| / sum_j |vx[j,i]|.
    S_1 (order 1, Shannon) = -sum_j p_j * log(p_j).
    S_2 (order 2)          = -log(sum_j p_j^2).

    Returns arrays of shape (n_columns,).  Columns where sum |vx| <= 0 yield NaN.
    """
    abs_vx = np.abs(vx)
    col_sums = abs_vx.sum(axis=0)
    valid = col_sums > 0

    n_cols = vx.shape[1]
    s1 = np.full(n_cols, np.nan, dtype=np.float64)
    s2 = np.full(n_cols, np.nan, dtype=np.float64)

    if not np.any(valid):
        return s1, s2

    p = abs_vx[:, valid] / col_sums[valid]
    mask = p > 0

    log_p = np.zeros_like(p)
    log_p[mask] = np.log(p[mask])

    s1[valid] = -np.sum(p * log_p, axis=0)
    s2[valid] = -np.log(np.sum(p ** 2, axis=0))

    return s1, s2


def validate_covariance_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.covariance == "exp":
        if args.length_scale is None:
            parser.error("--length_scale is required when --covariance exp")
        if args.fbm_alpha is not None or args.fbm_c is not None:
            parser.error("--fbm_alpha/--fbm_c are only valid when --covariance fbm")
        return

    if args.fbm_alpha is None:
        parser.error("--fbm_alpha is required when --covariance fbm")
    if args.fbm_c is None:
        parser.error("--fbm_c is required when --covariance fbm")
    if args.length_scale is not None:
        parser.error("--length_scale is only valid when --covariance exp")


def build_grf_arg(args: argparse.Namespace) -> dict[str, Any]:
    grf_arg: dict[str, Any] = {
        "covariance": args.covariance,
        "minpadding": int(args.minpadding),
    }
    if args.covariance == "exp":
        grf_arg.update(
            {
                "length_scale": float(args.length_scale),
                "std": float(args.std if args.std is not None else np.sqrt(args.var)),
            }
        )
    else:
        grf_arg.update(
            {
                "fbm_alpha": float(args.fbm_alpha),
                "fbm_c": float(args.fbm_c),
            }
        )
    return grf_arg


def make_grf_generator(grf_arg: dict[str, Any], field_size_: int) -> Any:
    covariance = str(grf_arg["covariance"])
    if covariance == "exp":
        cov = sim_grf.ExpCov(sigma=grf_arg["std"], length_scale=grf_arg["length_scale"])
        pts = (np.linspace(0, 1, field_size_),) * 2
        mean = np.zeros((field_size_, field_size_))
        return sim_grf.GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=grf_arg["minpadding"])
    if covariance == "fbm":
        return sim_grf.fBmGenerator(
            c=grf_arg["fbm_c"],
            alpha=grf_arg["fbm_alpha"],
            num_pt=field_size_,
            dim=2,
            minpadding=grf_arg["minpadding"],
        )
    raise ValueError(f"Unsupported covariance type: {covariance}")


def grf_path_tag(grf_arg: dict[str, Any]) -> str:
    covariance = str(grf_arg["covariance"])
    if covariance == "exp":
        return f"Covexp_LengthScale{grf_arg['length_scale']}"
    if covariance == "fbm":
        return f"Covfbm_Alpha{grf_arg['fbm_alpha']}_C{grf_arg['fbm_c']}"
    raise ValueError(f"Unsupported covariance type: {covariance}")


def save_mc_consolidated(
    config_mc: dict[str, Any],
    mc_index: int,
    consolidated: dict[str, Any],
) -> None:
    np.save(
        str(config_mc["path_size"] / f"size{config_mc['field_size_']}_mc{mc_index}.npy"),
        consolidated,
        allow_pickle=True,
    )


def _uint64_to_u32_pair(value: int) -> tuple[int, int]:
    v = int(np.uint64(value))
    return int(v & 0xFFFFFFFF), int((v >> 32) & 0xFFFFFFFF)


def derive_seed(base_seed: int, *components: int) -> int:
    entropy: list[int] = []
    entropy.extend(_uint64_to_u32_pair(base_seed))
    for comp in components:
        entropy.extend(_uint64_to_u32_pair(comp))
    ss = np.random.SeedSequence(entropy=entropy)
    state = ss.generate_state(2, dtype=np.uint32)
    return int((np.uint64(state[0]) << np.uint64(32)) | np.uint64(state[1]))


def none_to_nan(value: Any) -> Any:
    return np.nan if value is None else value


def resolve_effective_run_seed(args: argparse.Namespace) -> int | None:
    if args.seed is None:
        return None
    if args.variance_seed_policy == "same":
        return int(args.seed)
    if args.variance_seed_policy == "offset":
        return int(args.seed) + int(args.variance_rank) * int(args.variance_seed_offset)
    if args.variance_seed_policy == "explicit":
        if args.variance_seed is None:
            raise ValueError("--variance_seed is required when --variance_seed_policy explicit")
        return int(args.variance_seed)
    raise ValueError(f"Unsupported variance seed policy: {args.variance_seed_policy}")


def generate_logk_field(seed_trial: int | None = None) -> tuple[np.ndarray | None, str]:
    """Generate the logk field using the worker's GRF generator."""
    start_time_grf = time.time()
    if seed_trial is not None:
        # GRF generator in FlowSimulation uses global np.random APIs.
        np.random.seed(int(np.uint32(seed_trial & 0xFFFFFFFF)))
    try:
        logk_field = global_grf_gen.sample()
    except Exception as exc:
        elapsed_time_grf = time.time() - start_time_grf
        msg = f"GRF field generation failed after {np.round(elapsed_time_grf, 4)} seconds: {exc}"
        return None, msg

    elapsed_time_grf = time.time() - start_time_grf
    msg = f"GRF field generated successfully after {np.round(elapsed_time_grf, 4)} seconds."
    return logk_field, msg


def endpoint_y_at_x(path_xy: np.ndarray, x_ref: float) -> float:
    """
    Interpolate path y at a common x-reference.
    Fallback to nearest-x sample when there is no crossing segment.
    """
    p = np.asarray(path_xy, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] == 0:
        return float("nan")
    if p.shape[0] == 1:
        return float(p[0, 1])

    x = p[:, 0]
    y = p[:, 1]
    for i in range(p.shape[0] - 1):
        x0 = x[i]
        x1 = x[i + 1]
        if (x0 <= x_ref <= x1) or (x1 <= x_ref <= x0):
            t = (x_ref - x0) / (x1 - x0 + 1e-30)
            t = min(max(t, 0.0), 1.0)
            return float(y[i] + t * (y[i + 1] - y[i]))

    idx = int(np.argmin(np.abs(x - x_ref)))
    return float(y[idx])


def build_face_coordinates_from_centers(coord_centers: np.ndarray) -> np.ndarray:
    """Construct face coordinates from 1D cell centers by midpoint extrapolation."""
    coord_centers = np.asarray(coord_centers, dtype=float).ravel()
    if coord_centers.size < 2:
        raise ValueError("coord_centers must contain at least 2 values")
    faces = np.empty(coord_centers.size + 1, dtype=float)
    faces[1:-1] = 0.5 * (coord_centers[:-1] + coord_centers[1:])
    faces[0] = coord_centers[0] - 0.5 * (coord_centers[1] - coord_centers[0])
    faces[-1] = coord_centers[-1] + 0.5 * (coord_centers[-1] - coord_centers[-2])
    return faces


def resolve_particle_count(config_mc: dict[str, Any], source_rows: np.ndarray) -> int:
    pt_cfg = config_mc["particleTracking_arg"]
    num_particles = pt_cfg.get("num_particles")
    if num_particles is not None:
        return max(2, int(num_particles))
    return max(2, int(pt_cfg["density_particle"] * max(1, source_rows.size)))


def run_particle_tracking(
    *,
    config_mc: dict[str, Any],
    x0: int,
    x1: int,
    source_rows: np.ndarray,
    k_field: np.ndarray,
    h_field: np.ndarray,
    logger: Any,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    n_particles = resolve_particle_count(config_mc, source_rows)
    seeds_x = np.full(n_particles, x0, np.float32)
    seeds_y = np.linspace(float(source_rows[0]), float(source_rows[-1]), n_particles, endpoint=True).astype(np.float32)

    pt_cfg = config_mc["particleTracking_arg"]
    pt_solver = str(pt_cfg["solver"])
    requested_backend = str(pt_cfg.get("backend", "auto"))
    velocity_location = str(pt_cfg["velocity_location"])
    align_with_modflow = bool(pt_cfg["align_with_modflow_velocity"])

    summary_params: dict[str, Any] = {
        "backend": "cpu" if requested_backend == "auto" and pt_solver == "rk45" else requested_backend,
        "method": pt_solver,
        "tracking_mode": "arrival",
        "stop_mode": "first",
        "arrival_fraction": 1.0,
        "save_trajectories": False,
        "return_fastest": True,
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "right_x": x1,
        "max_points": pt_cfg["max_points"],
        "max_iters": pt_cfg["max_iters"],
        "threads_per_block": pt_cfg["threads_per_block"],
        "dtype": np.float64,
    }

    if pt_solver == "pollock":
        x_faces = build_face_coordinates_from_centers(config_mc["x"]).astype(np.float32)
        y_faces = build_face_coordinates_from_centers(config_mc["y"]).astype(np.float32)
        vx_pt, vy_pt = sim_hyd.compute_velocity(
            k_field,
            config_mc["x"],
            config_mc["y"],
            h_field,
            output_location="face",
            align_with_modflow=align_with_modflow,
            face_output="full",
        )
        summary_params.update(
            {
                "x_faces": x_faces,
                "y_faces": y_faces,
                "vx": vx_pt,
                "vy": vy_pt,
                "max_steps": pt_cfg["max_iters"],
            }
        )
    elif pt_solver == "rk45":
        vx_pt, vy_pt = sim_hyd.compute_velocity(
            k_field,
            config_mc["x"],
            config_mc["y"],
            h_field,
            output_location=velocity_location,
            align_with_modflow=align_with_modflow,
        )
        summary_params.update(
            {
                "x": config_mc["x"],
                "y": config_mc["y"],
                "vx": vx_pt,
                "vy": vy_pt,
                "velocity_location": velocity_location,
                "eps_speed": pt_cfg["eps_speed"],
                "ds_min": pt_cfg["ds_min"],
                "ds_max": pt_cfg["ds_max"],
                "tol": pt_cfg["tol"],
                "y_policy": pt_cfg["y_policy"],
            }
        )
    else:
        raise ValueError(f"Unsupported particle-tracking solver: {pt_solver}")

    backend_attempts = [summary_params["backend"]]
    if pt_solver == "pollock" and requested_backend == "auto":
        backend_attempts = ["gpu", "cpu"]

    summary: dict[str, Any] | None = None
    errors: list[str] = []
    for backend in backend_attempts:
        summary_params["backend"] = backend
        try:
            summary = sim_pt.particle_tracking_summary(summary_params)
            break
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            logger.warning(
                f"[PT] method={pt_solver}, requested_backend={requested_backend}, backend={backend} failed: {exc}"
            )
    if summary is None:
        raise ValueError(
            "Particle tracking summary failed for all backends: " + "; ".join(errors)
        )

    st = np.asarray(summary["status"])
    count = Counter(st.tolist())
    frac_success = count.get(0, 0) / max(1, n_particles)
    frac_bad = (count.get(1, 0) + count.get(4, 0)) / max(1, n_particles)
    logger.info(
        f"[PT] solver={pt_solver}, requested_backend={requested_backend}, "
        f"used_backend={summary.get('backend', summary_params['backend'])}, "
        f"velocity_location={velocity_location}, align_with_modflow={align_with_modflow}, "
        f"status_counts={dict(count)}"
    )
    if frac_success < 0.5 or frac_bad > 0.1:
        raise ValueError(f"Invalid velocity field for tracking: {count}")

    res = summary.get("fastest")
    if res is None:
        raise ValueError("Particle tracking summary did not return the fastest trajectory.")
    fp = np.vstack((res["x"], res["y"])).T
    res = dict(res)
    res["backend"] = summary.get("backend", summary_params["backend"])
    return fp, res, {"vx": np.asarray(vx_pt), "vy": np.asarray(vy_pt)}


def compute_fmm_path_regions(
    field: np.ndarray,
    *,
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    x0: int,
    x1: int,
    grid_spacing: float,
    solver: str,
    backtracking: str,
    traversable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float | None, str | None]:
    n_y, n_x = field.shape
    source_mask = np.zeros((n_y, n_x), dtype=bool)
    source_mask[source_rows, x0] = True
    target_mask = np.zeros((n_y, n_x), dtype=bool)
    target_mask[target_rows, x1] = True

    if traversable_mask is None:
        traversable_mask = np.isfinite(field) & (field > 0.0)
    source_mask = source_mask & traversable_mask
    target_mask = target_mask & traversable_mask
    if not np.any(source_mask) or not np.any(target_mask):
        return None, None, "No traversable source/target after masking"

    try:
        path_xy, travel_value, _chosen_target = compute_fmm_path(
            field,
            source_mask,
            target_mask,
            h=grid_spacing,
            traversable_mask=traversable_mask,
            solver=str(solver),
            backtracking=str(backtracking),
        )
    except Exception as exc:
        return None, None, str(exc)
    return path_xy, travel_value, None


def make_config_for_size(config_base: dict[str, Any], idx_size: int) -> dict[str, Any]:
    field_size_ = int(config_base["field_size"][idx_size])
    config = config_base.copy()
    config.update(
        {
            "idx_size": idx_size,
            "field_size_": field_size_,
            "lx": field_size_,
            "ly": field_size_,
            "lz": 1,
        }
    )
    config.update(
        {
            "source_vertex": [
                [config["lx"] // 20 - 1, config["ly"] // 20 - 1, 0],
                [config["lx"] // 20 - 1, config["ly"] - config["ly"] // 20 - 1, 1],
            ],
            "target_vertex": [
                [config["lx"] - config["lx"] // 20 - 1, 0, 0],
                [config["lx"] - config["lx"] // 20 - 1, config["ly"] - 1, 1],
            ],
        }
    )
    config["x_cp"] = config["target_vertex"][0][0]
    config["x"] = np.arange(0, field_size_, config["dx"], dtype=float)
    config["y"] = np.arange(0, field_size_, config["dy"], dtype=float)

    config["path_size"] = config["path_simulation"] / f"{grf_path_tag(config['grf_arg'])}_Size{field_size_}"

    if config["path_size"].exists() and config["path_size"].is_dir():
        import shutil

        shutil.rmtree(config["path_size"])

    config["path_size"].mkdir(parents=True, exist_ok=True)
    return config


def run_unit_simulation(config: dict[str, Any], mc_index: int, num_trial: int = 200) -> dict[str, Any]:
    """Run one Monte Carlo simulation with retry logic."""
    logger = sim_utils.get_logger(path_logfile=str(config["path_mainlogfile"]))
    logger.info(f"[MC {mc_index}] Starting simulation (up to {num_trial} trials).")

    config_mc = config.copy()
    x0 = config_mc["source_vertex"][0][0]
    x1 = config_mc["target_vertex"][1][0]
    y0 = config_mc["source_vertex"][0][1]
    y1 = config_mc["source_vertex"][1][1]
    domain_box = (x0, x1, y0, y1)
    source_rows = np.arange(y0, y1 + 1, dtype=int)
    target_rows = np.arange(0, config_mc["ly"], dtype=int)
    x_ref = float(x1)
    run_seed = config_mc.get("seed_effective_run")
    seed_mode = str(config_mc.get("seed_mode", "strict_mc"))
    seed_mc: int | None = None
    if run_seed is not None and seed_mode == "strict_mc":
        seed_mc = derive_seed(int(run_seed), int(config_mc["field_size_"]), int(mc_index))

    for trial in range(num_trial):
        record = build_base_record(mc_index=mc_index, trial_used=trial)
        seed_trial: int | None = None
        if run_seed is not None and seed_mode == "strict_mc" and seed_mc is not None:
            seed_trial = derive_seed(seed_mc, int(trial))
        record.update(
            {
                "seed_base": none_to_nan(config_mc.get("seed_base", np.nan)),
                "seed_effective_run": none_to_nan(config_mc.get("seed_effective_run", np.nan)),
                "seed_mode": seed_mode,
                "variance_seed_policy": config_mc.get("variance_seed_policy", "same"),
                "variance_rank": none_to_nan(config_mc.get("variance_rank", np.nan)),
                "variance_seed_override": none_to_nan(config_mc.get("variance_seed_override", np.nan)),
                "fmm_solver": config_mc["fmm_arg"]["solver"],
                "fmm_backtracking": config_mc["fmm_arg"]["backtracking"],
                "fmm_face_output": config_mc["fmm_arg"]["face_output"],
                "pt_solver": config_mc["particleTracking_arg"]["solver"],
                "pt_backend": config_mc["particleTracking_arg"].get("backend", "auto"),
                "pt_num_particles": int(resolve_particle_count(config_mc, source_rows)),
                "pt_velocity_location": config_mc["particleTracking_arg"]["velocity_location"],
                "pt_align_with_modflow_velocity": bool(
                    config_mc["particleTracking_arg"]["align_with_modflow_velocity"]
                ),
                "seed_mc": seed_mc if seed_mc is not None else np.nan,
                "seed_trial": seed_trial if seed_trial is not None else np.nan,
            }
        )
        logger.info(
            f"[MC {mc_index}] Seed context: seed_mc={record['seed_mc']}, "
            f"seed_trial={record['seed_trial']}, mode={seed_mode}"
        )
        logger.info(f"[MC {mc_index}] Trial {trial + 1}/{num_trial} starting.")

        try:
            # 1) GRF generation
            logk_field, grf_msg = generate_logk_field(seed_trial=seed_trial)
            if logk_field is None:
                record["status_code"] = 1
                record["status_message"] = grf_msg
                logger.warning(f"[MC {mc_index}] {grf_msg}")
                continue
            logger.info(f"[MC {mc_index}] {grf_msg}")
            k_field = np.exp(logk_field)

            # 2) Least-resistance path
            source = [(int(y), x0) for y in source_rows]
            target = [(int(y), x1) for y in target_rows]
            lrp, total_R, _s_node, _e_node, _Rmap = sim_path.dijkstra_2d_least_resistance(
                k_field, source, target, average="harmonic"
            )
            if lrp.size == 0 or not sim_path.is_path_valid(lrp, domain_box, len_min=np.abs(x1 - x0)):
                record["status_code"] = 2
                record["status_message"] = "Invalid LRP geometry"
                logger.warning(f"[MC {mc_index}] Invalid LRP, retrying next trial.")
                continue

            k_traversable_mask = np.isfinite(k_field) & (k_field > 0.0)
            lrp_fmm, total_R_fmm, lrp_fmm_error = compute_fmm_path_regions(
                k_field,
                source_rows=source_rows,
                target_rows=target_rows,
                x0=x0,
                x1=x1,
                grid_spacing=config_mc["dx"],
                solver=config_mc["fmm_arg"]["solver"],
                backtracking=config_mc["fmm_arg"]["backtracking"],
                traversable_mask=k_traversable_mask,
            )
            if lrp_fmm is None:
                record["status_code"] = 10
                record["status_message"] = lrp_fmm_error or "LRP(FMM) path extraction failed"
                logger.warning(f"[MC {mc_index}] Invalid LRP (FMM), retrying next trial.")
                continue
            if not sim_path.is_path_valid(lrp_fmm, domain_box, len_min=np.abs(x1 - x0)):
                record["status_code"] = 10
                record["status_message"] = "Invalid LRP(FMM) geometry"
                logger.warning(f"[MC {mc_index}] Invalid LRP (FMM), retrying next trial.")
                continue

            # 3) Hydraulic head and velocity
            h, msg, is_successful = sim_hyd.solve_head(
                k_field,
                config_mc["x"],
                config_mc["y"],
                dh=config_mc["headSolver_arg"]["dh"],
                rtol=config_mc["headSolver_arg"]["rtol"],
                atol=config_mc["headSolver_arg"]["atol"],
                max_it=config_mc["headSolver_arg"]["max_it"],
            )
            if not is_successful:
                record["status_code"] = 3
                record["status_message"] = f"Head solver failed: {msg}"
                logger.warning(f"[MC {mc_index}] Head not converged, retrying next trial. {msg}")
                continue
            if config_mc["fmm_arg"]["velocity_layout"] == "face":
                vx_fmm, vy_fmm = sim_hyd.compute_velocity(
                    k_field,
                    config_mc["x"],
                    config_mc["y"],
                    h,
                    output_location="face",
                    face_output=config_mc["fmm_arg"]["face_output"],
                )
            else:
                vx_fmm, vy_fmm = sim_hyd.compute_velocity(
                    k_field,
                    config_mc["x"],
                    config_mc["y"],
                    h,
                    output_location="cell",
                )

            speed_fmm, traversable_mask, speed_diag = prepare_fmm_speed(
                vx=vx_fmm,
                vy=vy_fmm,
                layout=config_mc["fmm_arg"]["velocity_layout"],
                face_output=config_mc["fmm_arg"]["face_output"],
                face_mapping=config_mc["fmm_arg"]["face_mapping"],
                zero_policy=config_mc["fmm_arg"]["zero_policy"],
                min_speed=config_mc["fmm_arg"]["min_speed"],
            )
            nontraversable_count = int((~traversable_mask).sum())
            logger.info(
                f"[MC {mc_index}] FMM speed prep: layout={speed_diag['layout']}, "
                f"face_output={speed_diag['face_output']}, face_mapping={speed_diag['face_mapping']}, "
                f"zero_policy={speed_diag['zero_policy']}, "
                f"zero_or_bad={speed_diag['zero_or_bad_count']}, nontraversable={nontraversable_count}"
            )

            # 3.1) Dijkstra on speed
            fp_dijkstra, total_t_dijkstra, _sd, _ed, _Rmap_v = sim_path.dijkstra_2d_least_resistance(
                speed_fmm, source, target, average="arithmetic"
            )
            if fp_dijkstra.size == 0 or not sim_path.is_path_valid(fp_dijkstra, domain_box, len_min=np.abs(x1 - x0)):
                record["status_code"] = 4
                record["status_message"] = "Invalid FP(Dijkstra) geometry"
                logger.warning(f"[MC {mc_index}] Invalid FP (Dijkstra), retrying next trial.")
                continue

            # 3.2) FMM on speed
            fp_fmm, total_t_fmm, fp_fmm_error = compute_fmm_path_regions(
                speed_fmm,
                source_rows=source_rows,
                target_rows=target_rows,
                x0=x0,
                x1=x1,
                grid_spacing=config_mc["dx"],
                solver=config_mc["fmm_arg"]["solver"],
                backtracking=config_mc["fmm_arg"]["backtracking"],
                traversable_mask=traversable_mask,
            )
            if fp_fmm is None:
                record["status_code"] = 5
                record["status_message"] = fp_fmm_error or "FMM path extraction failed"
                logger.warning(f"[MC {mc_index}] Invalid FP (FMM), retrying next trial.")
                continue
            if not sim_path.is_path_valid(fp_fmm, domain_box, len_min=np.abs(x1 - x0)):
                record["status_code"] = 5
                record["status_message"] = "Invalid FP(FMM) geometry"
                logger.warning(f"[MC {mc_index}] Invalid FP (FMM), retrying next trial.")
                continue

            # 3.3) FSM path (cell-centred Darcy velocity, non-fatal)
            vx_cell, vy_cell = sim_hyd.compute_velocity(
                k_field,
                config_mc["x"],
                config_mc["y"],
                h,
                output_location="cell",
            )
            _n_y_fsm, _n_x_fsm = vx_cell.shape
            fsm_source_mask = np.zeros((_n_y_fsm, _n_x_fsm), dtype=bool)
            fsm_source_mask[source_rows, x0] = True
            fsm_target_mask = np.zeros((_n_y_fsm, _n_x_fsm), dtype=bool)
            fsm_target_mask[target_rows, x1] = True

            fp_fsm_cfg = config_mc.get("fsm_arg", {})
            fp_fsm: np.ndarray | None = None
            arrival_fsm: float | None = None
            fp_fsm_error: str | None = None
            try:
                fp_fsm, _t_field_fsm, arrival_fsm = compute_fsm_path_velocity(
                    vx_cell,
                    vy_cell,
                    fsm_source_mask,
                    fsm_target_mask,
                    config_mc["x"],
                    config_mc["y"],
                    max_sweeps=int(fp_fsm_cfg.get("max_sweeps", 100)),
                    tol=float(fp_fsm_cfg.get("tol", 1e-6)),
                    dt=float(fp_fsm_cfg.get("dt", 0.5)),
                    max_steps=int(fp_fsm_cfg.get("max_steps", 10_000)),
                )
            except Exception as exc:
                fp_fsm_error = str(exc)
                logger.warning(f"[MC {mc_index}] FSM path failed (non-fatal): {exc}")

            # 4) Particle tracking
            try:
                fp, pt_res, _pt_velocity = run_particle_tracking(
                    config_mc=config_mc,
                    x0=x0,
                    x1=x1,
                    source_rows=source_rows,
                    k_field=k_field,
                    h_field=h,
                    logger=logger,
                )
            except ValueError as exc:
                record["status_code"] = 6
                record["status_message"] = str(exc)
                logger.warning(f"[MC {mc_index}] Invalid velocity field, retrying next trial. {exc}")
                continue

            if fp.size == 0 or not sim_path.is_path_valid(fp, domain_box, len_min=np.abs(x1 - x0)):
                record["status_code"] = 7
                record["status_message"] = f"Invalid FP({config_mc['particleTracking_arg']['solver']}) geometry"
                logger.warning(
                    f"[MC {mc_index}] Invalid FP ({config_mc['particleTracking_arg']['solver']}), retrying next trial."
                )
                continue

            # 5) Metrics and persist
            D_fp_lrp, _, _ = distance_distribution(fp, lrp, ds=1.0)
            D_fp_lrp_fmm, _, _ = distance_distribution(fp, lrp_fmm, ds=1.0)
            D_fp_dij, _, _ = distance_distribution(fp, fp_dijkstra, ds=1.0)
            D_fp_fmm, _, _ = distance_distribution(fp, fp_fmm, ds=1.0)
            D_fp_fsm: np.ndarray | None = None
            if fp_fsm is not None:
                D_fp_fsm, _, _ = distance_distribution(fp, fp_fsm, ds=1.0)

            vx_pt = np.asarray(_pt_velocity["vx"])
            renyi_s1, renyi_s2 = _renyi_entropy_vx_profile(vx_pt)

            y_end_fp = endpoint_y_at_x(fp, x_ref=x_ref)
            y_end_lrp = endpoint_y_at_x(lrp, x_ref=x_ref)
            y_end_lrp_fmm = endpoint_y_at_x(lrp_fmm, x_ref=x_ref)
            y_end_fp_dij = endpoint_y_at_x(fp_dijkstra, x_ref=x_ref)
            y_end_fp_fmm = endpoint_y_at_x(fp_fmm, x_ref=x_ref)
            y_end_fp_fsm = endpoint_y_at_x(fp_fsm, x_ref=x_ref) if fp_fsm is not None else np.nan

            seeds_dict = {
                "base": none_to_nan(config_mc.get("seed_base", np.nan)),
                "effective_run": none_to_nan(config_mc.get("seed_effective_run", np.nan)),
                "mode": seed_mode,
                "mc": seed_mc if seed_mc is not None else np.nan,
                "trial": seed_trial if seed_trial is not None else np.nan,
                "variance": {
                    "policy": config_mc.get("variance_seed_policy", "same"),
                    "rank": none_to_nan(config_mc.get("variance_rank", np.nan)),
                    "override": none_to_nan(config_mc.get("variance_seed_override", np.nan)),
                },
            }
            config_dict = {
                "particle_tracking": {
                    "solver": config_mc["particleTracking_arg"]["solver"],
                    "backend": str(pt_res["backend"]),
                    "num_particles": int(resolve_particle_count(config_mc, source_rows)),
                    "velocity_location": config_mc["particleTracking_arg"]["velocity_location"],
                    "align_with_modflow_velocity": bool(
                        config_mc["particleTracking_arg"]["align_with_modflow_velocity"]
                    ),
                },
                "fmm": {
                    "velocity_layout": config_mc["fmm_arg"]["velocity_layout"],
                    "face_output": config_mc["fmm_arg"]["face_output"],
                    "face_mapping": config_mc["fmm_arg"]["face_mapping"],
                    "solver": config_mc["fmm_arg"]["solver"],
                    "backtracking": config_mc["fmm_arg"]["backtracking"],
                    "zero_policy": config_mc["fmm_arg"]["zero_policy"],
                    "min_speed": float(config_mc["fmm_arg"]["min_speed"]),
                    "zero_count": int(speed_diag["zero_or_bad_count"]),
                    "nontraversable_count": int(nontraversable_count),
                },
            }

            consolidated = {
                "meta": {
                    "schema_version": MC_SCHEMA_VERSION,
                    "mc_index": int(mc_index),
                    "trial_used": int(trial),
                    "status_code": 0,
                    "status_message": "ok",
                },
                "seeds": seeds_dict,
                "config": config_dict,
                "fields": {
                    "logk": logk_field,
                },
                "paths": {
                    "fp": {
                        "coords": fp,
                        "length": float(sim_utils.euclidean_length(fp)),
                        "travel_time": float(pt_res["travel_time"]),
                    },
                    "fp_dijkstra": {
                        "coords": fp_dijkstra,
                        "length": float(sim_utils.euclidean_length(fp_dijkstra)),
                        "travel_time": float(total_t_dijkstra),
                    },
                    "fp_fmm": {
                        "coords": fp_fmm,
                        "length": float(sim_utils.euclidean_length(fp_fmm)),
                        "travel_time": float(total_t_fmm),
                    },
                    "lrp": {
                        "coords": lrp,
                        "length": float(sim_utils.euclidean_length(lrp)),
                        "resistance": float(total_R),
                    },
                    "lrp_fmm": {
                        "coords": lrp_fmm,
                        "length": float(sim_utils.euclidean_length(lrp_fmm)),
                        "resistance": float(total_R_fmm),
                    },
                    "fp_fsm": {
                        "coords": fp_fsm,
                        "length": float(sim_utils.euclidean_length(fp_fsm)) if fp_fsm is not None else np.nan,
                        "travel_time": float(arrival_fsm) if arrival_fsm is not None else np.nan,
                        "error": fp_fsm_error,
                    },
                },
                "endpoint": {
                    "method": "interp_at_xref",
                    "x_ref": float(x_ref),
                    "fp": {"x": float(fp[-1, 0]), "y": float(y_end_fp)},
                    "fp_dijkstra": {"x": float(fp_dijkstra[-1, 0]), "y": float(y_end_fp_dij)},
                    "fp_fmm": {"x": float(fp_fmm[-1, 0]), "y": float(y_end_fp_fmm)},
                    "fp_fsm": {
                        "x": float(fp_fsm[-1, 0]) if fp_fsm is not None else np.nan,
                        "y": float(y_end_fp_fsm),
                    },
                    "lrp": {"x": float(lrp[-1, 0]), "y": float(y_end_lrp)},
                    "lrp_fmm": {"x": float(lrp_fmm[-1, 0]), "y": float(y_end_lrp_fmm)},
                    "dy": {
                        "fp_vs_lrp": float(y_end_fp - y_end_lrp),
                        "fp_vs_lrp_fmm": float(y_end_fp - y_end_lrp_fmm),
                        "fp_vs_dijkstra": float(y_end_fp - y_end_fp_dij),
                        "fp_vs_fmm": float(y_end_fp - y_end_fp_fmm),
                        "fp_vs_fsm": float(y_end_fp - y_end_fp_fsm) if fp_fsm is not None else np.nan,
                    },
                },
                "hausdorff": {
                    "fp_vs_lrp": _hausdorff_detail(D_fp_lrp),
                    "fp_vs_lrp_fmm": _hausdorff_detail(D_fp_lrp_fmm),
                    "fp_vs_dijkstra": _hausdorff_detail(D_fp_dij),
                    "fp_vs_fmm": _hausdorff_detail(D_fp_fmm),
                    "fp_vs_fsm": _hausdorff_detail(D_fp_fsm) if D_fp_fsm is not None else None,
                },
                "renyi_entropy": {
                    "S1": renyi_s1,
                    "S2": renyi_s2,
                },
            }

            save_mc_consolidated(config_mc=config_mc, mc_index=mc_index, consolidated=consolidated)

            record.update(
                {
                    "status_code": 0,
                    "status_message": "ok",
                    "len_lrp": consolidated["paths"]["lrp"]["length"],
                    "len_lrp_fmm": consolidated["paths"]["lrp_fmm"]["length"],
                    "len_fp": consolidated["paths"]["fp"]["length"],
                    "len_fp_dijkstra": consolidated["paths"]["fp_dijkstra"]["length"],
                    "len_fp_fmm": consolidated["paths"]["fp_fmm"]["length"],
                    "len_fp_fsm": consolidated["paths"]["fp_fsm"]["length"],
                    "travel_time_fp": consolidated["paths"]["fp"]["travel_time"],
                    "travel_time_dijkstra": consolidated["paths"]["fp_dijkstra"]["travel_time"],
                    "travel_time_fmm": consolidated["paths"]["fp_fmm"]["travel_time"],
                    "travel_time_fsm": consolidated["paths"]["fp_fsm"]["travel_time"],
                    "resistance_lrp": consolidated["paths"]["lrp"]["resistance"],
                    "resistance_lrp_fmm": consolidated["paths"]["lrp_fmm"]["resistance"],
                    "endpoint_method": "interp_at_xref",
                    "endpoint_x_ref": float(x_ref),
                    "x_end_fp": consolidated["endpoint"]["fp"]["x"],
                    "x_end_lrp": consolidated["endpoint"]["lrp"]["x"],
                    "x_end_lrp_fmm": consolidated["endpoint"]["lrp_fmm"]["x"],
                    "x_end_fp_dijkstra": consolidated["endpoint"]["fp_dijkstra"]["x"],
                    "x_end_fp_fmm": consolidated["endpoint"]["fp_fmm"]["x"],
                    "x_end_fp_fsm": consolidated["endpoint"]["fp_fsm"]["x"],
                    "y_end_fp": consolidated["endpoint"]["fp"]["y"],
                    "y_end_lrp": consolidated["endpoint"]["lrp"]["y"],
                    "y_end_lrp_fmm": consolidated["endpoint"]["lrp_fmm"]["y"],
                    "y_end_fp_dijkstra": consolidated["endpoint"]["fp_dijkstra"]["y"],
                    "y_end_fp_fmm": consolidated["endpoint"]["fp_fmm"]["y"],
                    "y_end_fp_fsm": consolidated["endpoint"]["fp_fsm"]["y"],
                    "endpoint_dy_fp_vs_lrp": consolidated["endpoint"]["dy"]["fp_vs_lrp"],
                    "endpoint_dy_fp_vs_lrp_fmm": consolidated["endpoint"]["dy"]["fp_vs_lrp_fmm"],
                    "endpoint_dy_fp_vs_dijkstra": consolidated["endpoint"]["dy"]["fp_vs_dijkstra"],
                    "endpoint_dy_fp_vs_fmm": consolidated["endpoint"]["dy"]["fp_vs_fmm"],
                    "endpoint_dy_fp_vs_fsm": consolidated["endpoint"]["dy"]["fp_vs_fsm"],
                    "pt_solver": config_dict["particle_tracking"]["solver"],
                    "pt_backend": config_dict["particle_tracking"]["backend"],
                    "pt_num_particles": config_dict["particle_tracking"]["num_particles"],
                    "pt_velocity_location": config_dict["particle_tracking"]["velocity_location"],
                    "pt_align_with_modflow_velocity": config_dict["particle_tracking"]["align_with_modflow_velocity"],
                    "fmm_velocity_layout": config_dict["fmm"]["velocity_layout"],
                    "fmm_face_output": config_dict["fmm"]["face_output"],
                    "fmm_face_mapping": config_dict["fmm"]["face_mapping"],
                    "fmm_solver": config_dict["fmm"]["solver"],
                    "fmm_backtracking": config_dict["fmm"]["backtracking"],
                    "fmm_zero_policy": config_dict["fmm"]["zero_policy"],
                    "fmm_min_speed": config_dict["fmm"]["min_speed"],
                    "fmm_zero_count": config_dict["fmm"]["zero_count"],
                    "fmm_nontraversable_count": config_dict["fmm"]["nontraversable_count"],
                }
            )

            logger.info(f"[MC {mc_index}] Trial {trial + 1} successful and outputs saved.")
            return record

        except Exception as exc:
            record["status_code"] = 8
            record["status_message"] = f"Unexpected error: {exc}"
            logger.exception(f"[MC {mc_index}] Unexpected error on trial {trial + 1}: {exc}")
            continue

    failed = build_base_record(mc_index=mc_index, trial_used=None)
    failed.update(
        {
            "seed_base": none_to_nan(config_mc.get("seed_base", np.nan)),
            "seed_effective_run": none_to_nan(config_mc.get("seed_effective_run", np.nan)),
            "seed_mode": str(config_mc.get("seed_mode", "strict_mc")),
            "variance_seed_policy": config_mc.get("variance_seed_policy", "same"),
            "variance_rank": none_to_nan(config_mc.get("variance_rank", np.nan)),
            "variance_seed_override": none_to_nan(config_mc.get("variance_seed_override", np.nan)),
            "fmm_solver": config_mc["fmm_arg"]["solver"],
            "fmm_backtracking": config_mc["fmm_arg"]["backtracking"],
            "fmm_face_output": config_mc["fmm_arg"]["face_output"],
            "pt_solver": config_mc["particleTracking_arg"]["solver"],
            "pt_backend": config_mc["particleTracking_arg"].get("backend", "auto"),
            "pt_num_particles": int(resolve_particle_count(config_mc, source_rows)),
            "pt_velocity_location": config_mc["particleTracking_arg"]["velocity_location"],
            "pt_align_with_modflow_velocity": bool(config_mc["particleTracking_arg"]["align_with_modflow_velocity"]),
            "seed_mc": seed_mc if seed_mc is not None else np.nan,
            "seed_trial": np.nan,
        }
    )
    failed["status_code"] = 9
    failed["status_message"] = f"All {num_trial} trials failed"

    failed_consolidated = {
        "meta": {
            "schema_version": MC_SCHEMA_VERSION,
            "mc_index": int(mc_index),
            "trial_used": -1,
            "status_code": 9,
            "status_message": f"All {num_trial} trials failed",
        },
        "seeds": {
            "base": none_to_nan(config_mc.get("seed_base", np.nan)),
            "effective_run": none_to_nan(config_mc.get("seed_effective_run", np.nan)),
            "mode": str(config_mc.get("seed_mode", "strict_mc")),
            "mc": seed_mc if seed_mc is not None else np.nan,
            "trial": np.nan,
            "variance": {
                "policy": config_mc.get("variance_seed_policy", "same"),
                "rank": none_to_nan(config_mc.get("variance_rank", np.nan)),
                "override": none_to_nan(config_mc.get("variance_seed_override", np.nan)),
            },
        },
        "config": {
            "particle_tracking": {
                "solver": config_mc["particleTracking_arg"]["solver"],
                "backend": config_mc["particleTracking_arg"].get("backend", "auto"),
                "num_particles": int(resolve_particle_count(config_mc, source_rows)),
                "velocity_location": config_mc["particleTracking_arg"]["velocity_location"],
                "align_with_modflow_velocity": bool(
                    config_mc["particleTracking_arg"]["align_with_modflow_velocity"]
                ),
            },
            "fmm": {
                "velocity_layout": config_mc["fmm_arg"]["velocity_layout"],
                "face_output": config_mc["fmm_arg"]["face_output"],
                "face_mapping": config_mc["fmm_arg"]["face_mapping"],
                "solver": config_mc["fmm_arg"]["solver"],
                "backtracking": config_mc["fmm_arg"]["backtracking"],
                "zero_policy": config_mc["fmm_arg"]["zero_policy"],
                "min_speed": float(config_mc["fmm_arg"]["min_speed"]),
                "zero_count": np.nan,
                "nontraversable_count": np.nan,
            },
        },
    }
    save_mc_consolidated(config_mc=config_mc, mc_index=mc_index, consolidated=failed_consolidated)

    logger.error(f"[MC {mc_index}] All {num_trial} trials failed to produce valid outputs.")
    return failed


def run_field_size_simulation(idx_size: int, config_base: dict[str, Any]) -> dict[str, Any]:
    """Run all Monte Carlo simulations for one field size."""
    logger = sim_utils.get_logger(path_logfile=str(config_base["path_mainlogfile"]))
    config = make_config_for_size(config_base=config_base, idx_size=idx_size)
    field_size_ = config["field_size_"]
    logger.info(f"[SIZE {field_size_}] Starting simulation.")

    grf_gen = make_grf_generator(config["grf_arg"], field_size_)

    num_mc_ = int(config["num_mc"][idx_size])
    num_process = min(num_mc_, len(os.sched_getaffinity(0)))
    with mp.Pool(
        processes=num_process,
        initializer=init_worker,
        initargs=(grf_gen, config.get("seed_mode", "strict_mc"), config.get("seed_effective_run")),
    ) as pool:
        records = pool.starmap(run_unit_simulation, [(config, i) for i in range(num_mc_)])

    agg = {
        "schema_version": np.array([MC_SCHEMA_VERSION], dtype=np.int32),
        "field_size": np.array([field_size_], dtype=np.int32),
    }
    for key in MC_FIELDS:
        values = [rec.get(key, np.nan) for rec in records]
        if key in OBJECT_FIELDS:
            agg[key] = np.asarray(values, dtype=object)
        elif key in INT_FIELDS:
            agg[key] = np.asarray(values, dtype=np.int32)
        elif key in BOOL_FIELDS:
            agg[key] = np.asarray(values, dtype=bool)
        else:
            agg[key] = np.asarray(values, dtype=float)

    np.savez(str(config["path_size"] / f"size{field_size_}.npz"), **agg)
    logger.info(f"[SIZE {field_size_}] Main outputs saved.")
    logger.info(f"[SIZE {field_size_}] Completed simulation.")
    return {"idx_size": idx_size, "field_size": field_size_, "records": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--path_workdir", required=True, type=str, help="Path to output working directory.")
    parser.add_argument("--num_mc", required=True, nargs="+", type=int, help="Monte Carlo count list.")
    parser.add_argument("--field_size", required=True, nargs="+", type=int, help="Field size list.")
    parser.add_argument("--covariance", type=str, default="exp", choices=["exp", "fbm"], help="Covariance model for log-k field generation.")
    parser.add_argument("--length_scale", type=float, default=None, help="Normalized length scale in (0, 1) for exp covariance.")
    parser.add_argument("--fbm_alpha", type=float, default=None, help="Alpha parameter for fbm covariance.")
    parser.add_argument("--fbm_c", type=float, default=None, help="Scale parameter c for fbm covariance.")
    parser.add_argument("--minpadding", type=int, default=0, help="Minimal padding for GRF generation.")
    parser.add_argument("--dh", type=float, default=0.01, help="Hydraulic head x-gradient.")
    parser.add_argument("--output_schema_version", type=int, default=MC_SCHEMA_VERSION, help="Output schema version.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for deterministic Monte Carlo runs.")
    parser.add_argument(
        "--seed_mode",
        type=str,
        default="strict_mc",
        choices=["strict_mc", "global_stream", "best_effort"],
        help="Random seeding mode. strict_mc gives deterministic per-MC/per-trial seeds.",
    )
    parser.add_argument(
        "--variance_seed_policy",
        type=str,
        default="same",
        choices=["same", "offset", "explicit"],
        help="How to derive run seed across variance sweeps.",
    )
    parser.add_argument(
        "--variance_seed_offset",
        type=int,
        default=1000003,
        help="Offset multiplier used by variance_seed_policy=offset.",
    )
    parser.add_argument(
        "--variance_seed",
        type=int,
        default=None,
        help="Explicit run seed used when variance_seed_policy=explicit.",
    )
    parser.add_argument(
        "--variance_rank",
        type=int,
        default=0,
        help="Variance index used when variance_seed_policy=offset.",
    )
    parser.add_argument(
        "--fmm_velocity_layout",
        type=str,
        default="face",
        choices=["face", "cell"],
        help="Velocity layout for FMM speed preparation.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="pollock",
        choices=["pollock", "rk45"],
        help="Particle-tracking solver used for fastest-path extraction.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Particle-tracking backend. GPU runs are handled by scripts/run_fmm_gpu_worker.py.",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=None,
        help="Exact particle count override for fastest-path tracking. If unset, density-based seeding is used.",
    )
    parser.add_argument(
        "--velocity-location",
        dest="velocity_location",
        type=str,
        default="face",
        choices=["face", "cell"],
        help="Velocity layout used for RK45 particle tracking. Pollock requires face velocities.",
    )
    parser.add_argument(
        "--align-with-modflow-velocity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether particle-tracking velocity output should match MODFLOW-style zeroed outer right/top faces.",
    )
    parser.add_argument(
        "--fmm_face_mapping",
        type=str,
        default="to_cell_avg",
        choices=["to_cell_avg", "direct"],
        help="Mapping from face-centered flux to FMM speed field.",
    )
    parser.add_argument(
        "--fmm_face_output",
        type=str,
        default="full",
        choices=["full", "trimmed"],
        help="Face-field contract for FMM when --fmm_velocity_layout face.",
    )
    parser.add_argument(
        "--fmm_solver",
        type=str,
        default="scikit",
        choices=["custom", "scikit"],
        help="Travel-time solver used for all FMM-based path extraction.",
    )
    parser.add_argument(
        "--fmm_backtracking",
        type=str,
        default="continuous",
        choices=["discrete", "continuous"],
        help="Backtracking method used for all FMM-based path extraction.",
    )
    parser.add_argument(
        "--fmm_zero_policy",
        type=str,
        default="obstacle",
        choices=["obstacle", "clip", "error"],
        help="How to handle zero/non-finite speeds before FMM.",
    )
    parser.add_argument(
        "--fmm_min_speed",
        type=float,
        default=1e-12,
        help="Minimum speed floor used by clip/obstacle policies.",
    )
    group.add_argument("--std", type=float, default=None, help="Standard deviation for GRF generation.")
    group.add_argument("--var", type=float, default=None, help="Variance for GRF generation.")
    args = parser.parse_args()
    if args.solver == "pollock" and args.velocity_location != "face":
        parser.error("--solver pollock requires --velocity-location face")
    if args.backend == "gpu":
        parser.error("--backend gpu is supported via scripts/run_fmm_gpu_worker.py; use that entrypoint instead")
    if args.fmm_velocity_layout == "face" and args.fmm_face_output == "full" and args.fmm_face_mapping == "direct":
        parser.error("--fmm_face_output full is not supported with --fmm_face_mapping direct")
    validate_covariance_args(args, parser)
    return args


def main() -> None:
    args = parse_args()
    if args.output_schema_version != MC_SCHEMA_VERSION:
        raise ValueError(f"Only schema version {MC_SCHEMA_VERSION} is supported.")
    if len(args.num_mc) != len(args.field_size):
        raise ValueError("--num_mc and --field_size must have the same length.")
    if args.seed is None and args.seed_mode != "strict_mc":
        raise ValueError("--seed_mode global_stream/best_effort requires --seed.")
    if args.variance_seed_policy == "explicit" and args.variance_seed is None and args.seed is not None:
        raise ValueError("--variance_seed is required when --variance_seed_policy explicit with --seed.")

    effective_run_seed = resolve_effective_run_seed(args)

    print("\nAll input parameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print(f"effective_run_seed: {effective_run_seed}")
    print("\n")

    mp.set_start_method("spawn", force=True)

    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    path_workdir = Path(args.path_workdir).expanduser().resolve()

    config_base: dict[str, Any] = {
        "num_mc": args.num_mc,
        "dx": 1,
        "dy": 1,
        "dz": 1,
        "field_size": args.field_size,
        "grf_arg": build_grf_arg(args),
        "path_simulation": path_workdir,
        "seed_base": args.seed,
        "seed_effective_run": effective_run_seed,
        "seed_mode": args.seed_mode,
        "variance_seed_policy": args.variance_seed_policy,
        "variance_rank": int(args.variance_rank) if args.seed is not None else np.nan,
        "variance_seed_override": float(args.variance_seed) if args.variance_seed is not None else np.nan,
        "headSolver_arg": {
            "dh": args.dh,
            "rtol": 1e-10,
            "atol": 1e-20,
            "max_it": 10000,
        },
        "fmm_arg": {
            "velocity_layout": args.fmm_velocity_layout,
            "face_output": args.fmm_face_output,
            "face_mapping": args.fmm_face_mapping,
            "solver": args.fmm_solver,
            "backtracking": args.fmm_backtracking,
            "zero_policy": args.fmm_zero_policy,
            "min_speed": args.fmm_min_speed,
        },
        "particleTracking_arg": {
            "solver": args.solver,
            "backend": args.backend,
            "num_particles": args.num_particles,
            "velocity_location": args.velocity_location,
            "align_with_modflow_velocity": bool(args.align_with_modflow_velocity),
            "density_particle": 20,
            "eps_speed": 1e-30,
            "ds_min": 1e-20,
            "ds_max": 0.5,
            "tol": 1e-4,
            "y_policy": "stop",
            "max_points": 100000,
            "max_iters": 1000000,
            "threads_per_block": 256,
        },
    }

    config_base["path_log"] = config_base["path_simulation"] / "log"
    config_base["path_mainlogfile"] = config_base["path_log"] / f"simulation_{timestamp}.log"
    config_base["path_gpulogfile"] = config_base["path_log"] / f"gpu_profile_{timestamp}.log"
    config_base["path_cpulogfile"] = config_base["path_log"] / f"cpu_profile_{timestamp}.log"

    config_base["path_simulation"].mkdir(parents=True, exist_ok=True)
    config_base["path_log"].mkdir(parents=True, exist_ok=True)

    logger = sim_utils.get_logger(path_logfile=str(config_base["path_mainlogfile"]))
    logger.info("[MAIN] Starting simulation.")
    logger.info(
        "[MAIN] Seed config: "
        f"seed_base={config_base['seed_base']}, "
        f"seed_effective_run={config_base['seed_effective_run']}, "
        f"seed_mode={config_base['seed_mode']}, "
        f"variance_seed_policy={config_base['variance_seed_policy']}, "
        f"variance_rank={config_base['variance_rank']}, "
        f"variance_seed_override={config_base['variance_seed_override']}, "
        f"fmm_solver={config_base['fmm_arg']['solver']}, "
        f"fmm_backtracking={config_base['fmm_arg']['backtracking']}, "
        f"fmm_face_output={config_base['fmm_arg']['face_output']}, "
        f"pt_solver={config_base['particleTracking_arg']['solver']}, "
        f"pt_backend={config_base['particleTracking_arg']['backend']}, "
        f"pt_velocity_location={config_base['particleTracking_arg']['velocity_location']}, "
        "pt_align_with_modflow_velocity="
        f"{config_base['particleTracking_arg']['align_with_modflow_velocity']}"
    )

    event_stop_gpuprofiler = threading.Event()
    thread_gpuprofiler = threading.Thread(
        target=sim_utils.log_gpu_usage,
        kwargs={
            "path_logfile": str(config_base["path_gpulogfile"]),
            "interval": 1.0,
            "event_stop": event_stop_gpuprofiler,
        },
    )
    thread_gpuprofiler.start()

    cpu_profiler = cProfile.Profile()
    cpu_profiler.enable()

    results = []
    for idx_size in range(len(config_base["field_size"])):
        results.append(run_field_size_simulation(idx_size=idx_size, config_base=config_base))

    end_time = datetime.now()
    elapsed_hours = round((end_time - start_time).total_seconds() / 3600, 2)

    cpu_profiler.disable()
    with open(str(config_base["path_cpulogfile"]), "w", encoding="utf-8") as handle:
        pstats.Stats(cpu_profiler, stream=handle).sort_stats("time").print_stats()

    event_stop_gpuprofiler.set()
    thread_gpuprofiler.join()

    logger.info(
        f"[MAIN] Completed simulation. {elapsed_hours} hours elapsed "
        f"from {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}."
    )

    # reduce chance of leaked large arrays in long-lived interpreters
    del results
    gc.collect()


if __name__ == "__main__":
    main()
