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

from fmm_core.fmm import extract_path_source_to_target, fast_marching_2d_regions
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


MC_SCHEMA_VERSION = 2

# Flat output schema v2: deterministic and explicit for analysis pipelines.
MC_FIELDS = [
    "schema_version",
    "mc_index",
    "trial_used",
    "status_code",
    "status_message",
    "len_lrp",
    "len_fp_rk45",
    "len_fp_dijkstra",
    "len_fp_fmm",
    "travel_time_rk45",
    "travel_time_dijkstra",
    "travel_time_fmm",
    "resistance_lrp",
    "hausdorff_fp_vs_lrp",
    "hausdorff_fp_vs_dijkstra",
    "hausdorff_fp_vs_fmm",
    "endpoint_dy_fp_vs_lrp",
    "endpoint_dy_fp_vs_dijkstra",
    "endpoint_dy_fp_vs_fmm",
    "fmm_velocity_layout",
    "fmm_face_mapping",
    "fmm_zero_policy",
    "fmm_min_speed",
    "fmm_zero_count",
    "fmm_nontraversable_count",
]


def init_worker(grf_gen):
    """Initializer for simulation pool workers."""
    global global_grf_gen
    global_grf_gen = grf_gen


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


def save_mc_outputs(config_mc: dict[str, Any], mc_index: int, outputs: dict[str, Any]) -> None:
    np.savez(
        str(config_mc["path_size_mainOutput"] / f"mainOutput_size{config_mc['field_size_']}_mc{mc_index}.npz"),
        **outputs,
    )


def save_path_outputs(config_mc: dict[str, Any], mc_index: int, lrp: np.ndarray, fp_dijkstra: np.ndarray, fp_fmm: np.ndarray, fp_rk45: np.ndarray) -> None:
    np.save(config_mc["path_size_path"] / f"lrp_size{config_mc['field_size_']}_mc{mc_index}.npy", lrp)
    np.save(config_mc["path_size_path"] / f"fp_dijkstra_size{config_mc['field_size_']}_mc{mc_index}.npy", fp_dijkstra)
    np.save(config_mc["path_size_path"] / f"fp_fmm_size{config_mc['field_size_']}_mc{mc_index}.npy", fp_fmm)
    np.save(config_mc["path_size_path"] / f"fp_size{config_mc['field_size_']}_mc{mc_index}.npy", fp_rk45)


def generate_logk_field() -> tuple[np.ndarray | None, str]:
    """Generate the logk field using the worker's GRF generator."""
    start_time_grf = time.time()
    try:
        logk_field = global_grf_gen.sample()
    except Exception as exc:
        elapsed_time_grf = time.time() - start_time_grf
        msg = f"GRF field generation failed after {np.round(elapsed_time_grf, 4)} seconds: {exc}"
        return None, msg

    elapsed_time_grf = time.time() - start_time_grf
    msg = f"GRF field generated successfully after {np.round(elapsed_time_grf, 4)} seconds."
    return logk_field, msg


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

    config["path_size"] = config["path_simulation"] / f"lengthScale{config['grf_arg']['length_scale']}_size{field_size_}"
    config["path_size_mainOutput"] = config["path_size"] / "main_output"
    config["path_size_path"] = config["path_size"] / "path"
    config["path_size_logk"] = config["path_size"] / "logk_field"

    if config["path_size"].exists() and config["path_size"].is_dir():
        import shutil

        shutil.rmtree(config["path_size"])

    config["path_size"].mkdir(parents=True, exist_ok=True)
    config["path_size_mainOutput"].mkdir(parents=True, exist_ok=True)
    config["path_size_path"].mkdir(parents=True, exist_ok=True)
    config["path_size_logk"].mkdir(parents=True, exist_ok=True)
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

    for trial in range(num_trial):
        record = build_base_record(mc_index=mc_index, trial_used=trial)
        logger.info(f"[MC {mc_index}] Trial {trial + 1}/{num_trial} starting.")

        try:
            # 1) GRF generation
            logk_field, grf_msg = generate_logk_field()
            if logk_field is None:
                record["status_code"] = 1
                record["status_message"] = grf_msg
                logger.warning(f"[MC {mc_index}] {grf_msg}")
                continue
            np.save(
                config_mc["path_size_logk"] / f"logKField_size{config_mc['field_size_']}_mc{mc_index}.npy",
                logk_field,
            )
            logger.info(f"[MC {mc_index}] {grf_msg}")
            k_field = np.exp(logk_field)

            # 2) Least-resistance path
            source = [(y, x0) for y in range(y0, y1)]
            target = [(y, x1) for y in range(0, config_mc["ly"] - 1)]
            lrp, total_R, _s_node, _e_node, _Rmap = sim_path.dijkstra_2d_least_resistance(
                k_field, source, target, average="harmonic"
            )
            if lrp.size == 0 or not sim_path.is_path_valid(lrp, domain_box, len_min=np.abs(x1 - x0)):
                record["status_code"] = 2
                record["status_message"] = "Invalid LRP geometry"
                logger.warning(f"[MC {mc_index}] Invalid LRP, retrying next trial.")
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
            vx, vy = sim_hyd.compute_velocity(
                k_field,
                config_mc["x"],
                config_mc["y"],
                h,
                output_location=config_mc["fmm_arg"]["velocity_layout"],
            )

            speed_fmm, traversable_mask, speed_diag = prepare_fmm_speed(
                vx=vx,
                vy=vy,
                layout=config_mc["fmm_arg"]["velocity_layout"],
                face_mapping=config_mc["fmm_arg"]["face_mapping"],
                zero_policy=config_mc["fmm_arg"]["zero_policy"],
                min_speed=config_mc["fmm_arg"]["min_speed"],
            )
            nontraversable_count = int((~traversable_mask).sum())
            logger.info(
                f"[MC {mc_index}] FMM speed prep: layout={speed_diag['layout']}, "
                f"face_mapping={speed_diag['face_mapping']}, zero_policy={speed_diag['zero_policy']}, "
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
            n_y, n_x = speed_fmm.shape
            source_mask = np.zeros((n_y, n_x), dtype=bool)
            source_mask[y0:y1, x0] = True
            target_mask = np.zeros((n_y, n_x), dtype=bool)
            target_mask[:, x1] = True

            source_mask = source_mask & traversable_mask
            target_mask = target_mask & traversable_mask
            if not np.any(source_mask) or not np.any(target_mask):
                record["status_code"] = 5
                record["status_message"] = "No traversable source/target after masking"
                logger.warning(f"[MC {mc_index}] No traversable source/target for FMM, retrying.")
                continue

            T, _state = fast_marching_2d_regions(
                speed_fmm,
                source_mask,
                h=config_mc["dx"],
                traversable_mask=traversable_mask,
            )
            path_ij, chosen_target = extract_path_source_to_target(
                T,
                source_mask,
                target_mask,
                traversable_mask=traversable_mask,
            )
            if len(path_ij) < 2:
                record["status_code"] = 5
                record["status_message"] = "FMM path extraction failed"
                logger.warning(f"[MC {mc_index}] Invalid FP (FMM), retrying next trial.")
                continue

            path_ij = np.asarray(path_ij, dtype=float)
            # convert (i, j) -> (x, y), source -> target orientation
            fp_fmm = path_ij[:, [1, 0]][::-1]
            if not sim_path.is_path_valid(fp_fmm, domain_box, len_min=np.abs(x1 - x0)):
                record["status_code"] = 5
                record["status_message"] = "Invalid FP(FMM) geometry"
                logger.warning(f"[MC {mc_index}] Invalid FP (FMM), retrying next trial.")
                continue

            # 4) Particle tracking RK45 CPU
            nP = int(config_mc["particleTracking_arg"]["density_particle"] * np.abs(y1 - y0))
            seeds_x = np.full(nP, x0, np.float32)
            seeds_y = np.linspace(y0, y1, nP).astype(np.float32)
            args_particle_tracking = {
                "x": config_mc["x"],
                "y": config_mc["y"],
                "vx": vx,
                "vy": vy,
                "seeds_x": seeds_x,
                "seeds_y": seeds_y,
                "right_x": x1,
                "velocity_location": "face",
                "eps_speed": config_mc["particleTracking_arg"]["eps_speed"],
                "ds_min": config_mc["particleTracking_arg"]["ds_min"],
                "ds_max": config_mc["particleTracking_arg"]["ds_max"],
                "tol": config_mc["particleTracking_arg"]["tol"],
                "y_policy": config_mc["particleTracking_arg"]["y_policy"],
                "max_points": config_mc["particleTracking_arg"]["max_points"],
                "max_iters": config_mc["particleTracking_arg"]["max_iters"],
                "threads_per_block": config_mc["particleTracking_arg"]["threads_per_block"],
                "dtype": np.float64,
            }

            px, py, pt, L, st = sim_pt.particle_tracking_cpu(args_particle_tracking)
            count = Counter(st.tolist())
            frac_success = count.get(0, 0) / max(1, nP)
            frac_bad = (count.get(1, 0) + count.get(4, 0)) / max(1, nP)
            if frac_success < 0.5 or frac_bad > 0.1:
                record["status_code"] = 6
                record["status_message"] = f"Invalid velocity field for tracking: {count}"
                logger.warning(f"[MC {mc_index}] Invalid velocity field, retrying next trial. {count}")
                continue

            res = sim_pt.fastest_trajectory(px, py, pt, L, st)
            fp_rk45 = np.vstack((res["x"], res["y"])).T
            fp_rk45 = fp_rk45[fp_rk45[:, 0] <= config_mc["x_cp"]]
            if fp_rk45.size == 0 or not sim_path.is_path_valid(fp_rk45, domain_box, len_min=np.abs(x1 - x0)):
                record["status_code"] = 7
                record["status_message"] = "Invalid FP(RK45) geometry"
                logger.warning(f"[MC {mc_index}] Invalid FP (RK45), retrying next trial.")
                continue

            # 5) Metrics and persist
            hist_fp_lrp, _, _ = distance_distribution(fp_rk45, lrp, ds=1.0)
            hist_fp_dij, _, _ = distance_distribution(fp_rk45, fp_dijkstra, ds=1.0)
            hist_fp_fmm, _, _ = distance_distribution(fp_rk45, fp_fmm, ds=1.0)

            record.update(
                {
                    "status_code": 0,
                    "status_message": "ok",
                    "len_lrp": float(sim_utils.euclidean_length(lrp)),
                    "len_fp_rk45": float(sim_utils.euclidean_length(fp_rk45)),
                    "len_fp_dijkstra": float(sim_utils.euclidean_length(fp_dijkstra)),
                    "len_fp_fmm": float(sim_utils.euclidean_length(fp_fmm)),
                    "travel_time_rk45": float(res["travel_time"]),
                    "travel_time_dijkstra": float(total_t_dijkstra),
                    "travel_time_fmm": float(T[chosen_target[0], chosen_target[1]]),
                    "resistance_lrp": float(total_R),
                    "hausdorff_fp_vs_lrp": float(np.max(hist_fp_lrp)),
                    "hausdorff_fp_vs_dijkstra": float(np.max(hist_fp_dij)),
                    "hausdorff_fp_vs_fmm": float(np.max(hist_fp_fmm)),
                    "endpoint_dy_fp_vs_lrp": float(fp_rk45[-1, 1] - lrp[-1, 1]),
                    "endpoint_dy_fp_vs_dijkstra": float(fp_rk45[-1, 1] - fp_dijkstra[-1, 1]),
                    "endpoint_dy_fp_vs_fmm": float(fp_rk45[-1, 1] - fp_fmm[-1, 1]),
                    "fmm_velocity_layout": config_mc["fmm_arg"]["velocity_layout"],
                    "fmm_face_mapping": config_mc["fmm_arg"]["face_mapping"],
                    "fmm_zero_policy": config_mc["fmm_arg"]["zero_policy"],
                    "fmm_min_speed": float(config_mc["fmm_arg"]["min_speed"]),
                    "fmm_zero_count": int(speed_diag["zero_or_bad_count"]),
                    "fmm_nontraversable_count": int(nontraversable_count),
                }
            )

            save_mc_outputs(config_mc=config_mc, mc_index=mc_index, outputs=record)
            save_path_outputs(
                config_mc=config_mc,
                mc_index=mc_index,
                lrp=lrp,
                fp_dijkstra=fp_dijkstra,
                fp_fmm=fp_fmm,
                fp_rk45=fp_rk45,
            )
            logger.info(f"[MC {mc_index}] Trial {trial + 1} successful and outputs saved.")
            return record

        except Exception as exc:
            record["status_code"] = 8
            record["status_message"] = f"Unexpected error: {exc}"
            logger.exception(f"[MC {mc_index}] Unexpected error on trial {trial + 1}: {exc}")
            continue

    failed = build_base_record(mc_index=mc_index, trial_used=None)
    failed["status_code"] = 9
    failed["status_message"] = f"All {num_trial} trials failed"
    save_mc_outputs(config_mc=config_mc, mc_index=mc_index, outputs=failed)
    logger.error(f"[MC {mc_index}] All {num_trial} trials failed to produce valid outputs.")
    return failed


def run_field_size_simulation(idx_size: int, config_base: dict[str, Any]) -> dict[str, Any]:
    """Run all Monte Carlo simulations for one field size."""
    logger = sim_utils.get_logger(path_logfile=str(config_base["path_mainlogfile"]))
    config = make_config_for_size(config_base=config_base, idx_size=idx_size)
    field_size_ = config["field_size_"]
    logger.info(f"[SIZE {field_size_}] Starting simulation.")

    cov = sim_grf.ExpCov(sigma=config["grf_arg"]["std"], length_scale=config["grf_arg"]["length_scale"])
    pts = (np.linspace(0, 1, field_size_),) * 2
    mean = np.zeros((field_size_, field_size_))
    grf_gen = sim_grf.GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=config["grf_arg"]["minpadding"])

    num_mc_ = int(config["num_mc"][idx_size])
    num_process = min(num_mc_, len(os.sched_getaffinity(0)))
    with mp.Pool(processes=num_process, initializer=init_worker, initargs=(grf_gen,)) as pool:
        records = pool.starmap(run_unit_simulation, [(config, i) for i in range(num_mc_)])

    agg = {
        "schema_version": np.array([MC_SCHEMA_VERSION], dtype=np.int32),
        "field_size": np.array([field_size_], dtype=np.int32),
    }
    for key in MC_FIELDS:
        values = [rec.get(key, np.nan) for rec in records]
        if key == "status_message":
            agg[key] = np.asarray(values, dtype=object)
        elif key in {"schema_version", "mc_index", "trial_used", "status_code"}:
            agg[key] = np.asarray(values, dtype=np.int32)
        else:
            agg[key] = np.asarray(values, dtype=object if key in {"fmm_velocity_layout", "fmm_face_mapping", "fmm_zero_policy"} else float)

    np.savez(str(config["path_size_mainOutput"] / f"mainOutput_size{field_size_}.npz"), **agg)
    logger.info(f"[SIZE {field_size_}] Main outputs saved.")
    logger.info(f"[SIZE {field_size_}] Completed simulation.")
    return {"idx_size": idx_size, "field_size": field_size_, "records": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--path_workdir", required=True, type=str, help="Path to output working directory.")
    parser.add_argument("--num_mc", required=True, nargs="+", type=int, help="Monte Carlo count list.")
    parser.add_argument("--field_size", required=True, nargs="+", type=int, help="Field size list.")
    parser.add_argument("--length_scale", required=True, type=float, help="Normalized length scale in (0, 1).")
    parser.add_argument("--minpadding", type=int, default=0, help="Minimal padding for GRF generation.")
    parser.add_argument("--dh", type=float, default=0.01, help="Hydraulic head x-gradient.")
    parser.add_argument("--output_schema_version", type=int, default=MC_SCHEMA_VERSION, help="Output schema version.")
    parser.add_argument(
        "--fmm_velocity_layout",
        type=str,
        default="face",
        choices=["face", "cell"],
        help="Velocity layout for FMM speed preparation.",
    )
    parser.add_argument(
        "--fmm_face_mapping",
        type=str,
        default="to_cell_avg",
        choices=["to_cell_avg", "direct"],
        help="Mapping from face-centered flux to FMM speed field.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_schema_version != MC_SCHEMA_VERSION:
        raise ValueError(f"Only schema version {MC_SCHEMA_VERSION} is supported.")
    if len(args.num_mc) != len(args.field_size):
        raise ValueError("--num_mc and --field_size must have the same length.")

    print("\nAll input parameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
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
        "grf_arg": {
            "length_scale": args.length_scale,
            "std": args.std if args.std is not None else float(np.sqrt(args.var)),
            "minpadding": args.minpadding,
        },
        "path_simulation": path_workdir,
        "headSolver_arg": {
            "dh": args.dh,
            "rtol": 1e-10,
            "atol": 1e-20,
            "max_it": 10000,
        },
        "fmm_arg": {
            "velocity_layout": args.fmm_velocity_layout,
            "face_mapping": args.fmm_face_mapping,
            "zero_policy": args.fmm_zero_policy,
            "min_speed": args.fmm_min_speed,
        },
        "particleTracking_arg": {
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
