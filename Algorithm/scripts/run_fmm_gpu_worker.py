import argparse
import cProfile
import gc
import multiprocessing as mp
import os
import pstats
import queue
import subprocess
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

import run_fmm as base
from simulation.particle_tracking import pollock_gpu as sim_pt_pollock_gpu
from simulation.particle_tracking import utils as sim_pt_utils


MAX_BATCH_SIZE = 32
MAX_BATCH_WAIT_MS = 5
GPU_QUEUE_MAXSIZE = 128
GPU_SUBMIT_TIMEOUT_SEC = 30.0
GPU_RESULT_POLL_SEC = 0.001

global_queue_gpu = None
global_result_dict = None
global_pollock_tracker = None
global_pollock_tracker_signature = None


def parse_visible_gpu_tokens() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env.strip():
        tokens = [token.strip() for token in env.split(",") if token.strip() and token.strip() != "-1"]
        if tokens:
            return tokens

    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("GPU ")]
    return [str(idx) for idx in range(len(lines))]


def init_worker_gpu(
    grf_gen: Any,
    queue_gpu: Any,
    result_dict: Any,
    seed_mode: str = "strict_mc",
    effective_run_seed: int | None = None,
) -> None:
    global global_queue_gpu
    global global_result_dict
    global_queue_gpu = queue_gpu
    global_result_dict = result_dict
    base.init_worker(grf_gen, seed_mode=seed_mode, effective_run_seed=effective_run_seed)


def _payload_batch_key(func_name: str, payload: dict[str, Any]) -> tuple[Any, ...]:
    vx_shape, vx_dtype = _array_signature(payload, "vx")
    vy_shape, vy_dtype = _array_signature(payload, "vy")
    x_faces_shape, _x_faces_dtype = _array_signature(payload, "x_faces")
    y_faces_shape, _y_faces_dtype = _array_signature(payload, "y_faces")
    return (
        func_name,
        str(payload.get("method", "")),
        str(payload.get("backend", "")),
        vx_shape,
        vy_shape,
        x_faces_shape,
        y_faces_shape,
        vx_dtype,
        vy_dtype,
        float(payload.get("right_x", _array_last_value(payload, "x_faces"))),
        int(payload.get("max_points", 0)),
        int(payload.get("max_steps", payload.get("max_iters", 0))),
        str(payload.get("tracking_mode", "")),
        str(payload.get("stop_mode", "")),
    )


def _array_signature(payload: dict[str, Any], key: str) -> tuple[tuple[int, ...], str]:
    shm_key = f"{key}_shm"
    if shm_key in payload:
        meta = dict(payload[shm_key])
        return tuple(meta["shape"]), str(meta["dtype"])
    arr = np.asarray(payload[key])
    return tuple(arr.shape), str(arr.dtype)


def _array_last_value(payload: dict[str, Any], key: str) -> float:
    shm_key = f"{key}_shm"
    if shm_key in payload:
        meta = dict(payload[shm_key])
        return float(meta["last_value"])
    arr = np.asarray(payload[key])
    return float(arr.ravel()[-1])


def _build_seed_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "seed_spec" in payload:
        spec = dict(payload["seed_spec"])
        n_particles = max(2, int(spec["n_particles"]))
        x0 = float(spec["x0"])
        y0 = float(spec["y0"])
        y1 = float(spec["y1"])
        seeds_x = np.full(n_particles, x0, np.float32)
        seeds_y = np.linspace(y0, y1, n_particles, endpoint=True).astype(np.float32)
        return seeds_x, seeds_y
    return (
        np.asarray(payload["seeds_x"], dtype=np.float32).ravel(),
        np.asarray(payload["seeds_y"], dtype=np.float32).ravel(),
    )


def _share_array(array: np.ndarray) -> tuple[dict[str, Any], shared_memory.SharedMemory]:
    arr = np.ascontiguousarray(array)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_view[...] = arr
    descriptor = {
        "name": shm.name,
        "shape": tuple(int(v) for v in arr.shape),
        "dtype": str(arr.dtype),
        "nbytes": int(arr.nbytes),
        "last_value": float(arr.ravel()[-1]) if arr.size else np.nan,
    }
    return descriptor, shm


def _prepare_payload_for_ipc(payload: dict[str, Any]) -> tuple[dict[str, Any], list[shared_memory.SharedMemory]]:
    payload_ipc = dict(payload)
    shm_handles: list[shared_memory.SharedMemory] = []
    for key in ("vx", "vy", "x_faces", "y_faces"):
        descriptor, shm = _share_array(np.asarray(payload_ipc.pop(key)))
        payload_ipc[f"{key}_shm"] = descriptor
        shm_handles.append(shm)
    return payload_ipc, shm_handles


def _cleanup_shared_memory(handles: list[shared_memory.SharedMemory]) -> None:
    for shm in handles:
        try:
            shm.close()
        except FileNotFoundError:
            pass
        except BufferError:
            pass
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def _materialize_payload_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    payload_local = dict(payload)
    attached: list[shared_memory.SharedMemory] = []
    try:
        for key in ("vx", "vy", "x_faces", "y_faces"):
            shm_key = f"{key}_shm"
            if shm_key not in payload_local:
                continue
            meta = dict(payload_local.pop(shm_key))
            shm = shared_memory.SharedMemory(name=meta["name"])
            attached.append(shm)
            arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf).copy()
            payload_local[key] = arr
    finally:
        for shm in attached:
            try:
                shm.close()
            except FileNotFoundError:
                pass
    return payload_local


def gpu_call(func_name: str, payload: dict[str, Any], timeout_sec: float = 7200.0) -> dict[str, Any]:
    if global_queue_gpu is None or global_result_dict is None:
        raise RuntimeError("GPU queue/result globals are not initialized in this worker.")

    payload_ipc, shm_handles = _prepare_payload_for_ipc(payload)
    task_id = uuid.uuid4().hex
    task = {
        "id": task_id,
        "func": func_name,
        "payload": payload_ipc,
        "batch_key": _payload_batch_key(func_name, payload_ipc),
        "submitted_at": time.monotonic(),
    }
    try:
        global_queue_gpu.put(task, timeout=GPU_SUBMIT_TIMEOUT_SEC)
    except queue.Full as exc:
        _cleanup_shared_memory(shm_handles)
        raise TimeoutError(f"GPU task queue is full after waiting {GPU_SUBMIT_TIMEOUT_SEC} seconds.") from exc

    t0 = time.monotonic()
    try:
        while True:
            result = global_result_dict.pop(task_id, None)
            if result is not None:
                if not result.get("ok", False):
                    raise RuntimeError(result.get("error", "unknown GPU worker error"))
                return result["result"]
            if (time.monotonic() - t0) > timeout_sec:
                raise TimeoutError(f"GPU task timeout for {func_name} after {timeout_sec} seconds")
            time.sleep(GPU_RESULT_POLL_SEC)
    finally:
        _cleanup_shared_memory(shm_handles)


def _run_single_gpu_payload(func_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if func_name != "particle_tracking_summary":
        raise ValueError(f"Unsupported GPU function: {func_name}")
    payload_local = _materialize_payload_arrays(payload)
    if "seed_spec" in payload_local and "seeds_x" not in payload_local and "seeds_y" not in payload_local:
        seeds_x, seeds_y = _build_seed_arrays(payload_local)
        payload_local["seeds_x"] = seeds_x
        payload_local["seeds_y"] = seeds_y
    return base.sim_pt.particle_tracking_summary(payload_local)


def _pollock_tracker_signature(
    x_faces: np.ndarray,
    y_faces: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    n_particles: int,
) -> tuple[Any, ...]:
    return (
        tuple(x_faces.shape),
        tuple(y_faces.shape),
        tuple(vx.shape),
        tuple(vy.shape),
        str(x_faces.dtype),
        str(y_faces.dtype),
        str(vx.dtype),
        str(vy.dtype),
        int(n_particles),
    )


def _run_pollock_gpu_summary_single(payload: dict[str, Any]) -> dict[str, Any]:
    global global_pollock_tracker
    global global_pollock_tracker_signature

    payload_local = _materialize_payload_arrays(payload)
    if str(payload_local.get("tracking_mode", "arrival")) != "arrival":
        raise ValueError("Persistent GPU Pollock currently supports tracking_mode='arrival' only.")
    if bool(payload_local.get("save_trajectories", False)):
        raise ValueError("Persistent GPU Pollock currently requires save_trajectories=False.")
    if str(payload_local.get("method", "")) != "pollock" or str(payload_local.get("backend", "")) != "gpu":
        raise ValueError("Persistent GPU execution only supports method='pollock', backend='gpu'.")

    x_faces = np.asarray(payload_local["x_faces"], dtype=np.float32)
    y_faces = np.asarray(payload_local["y_faces"], dtype=np.float32)
    vx = np.asarray(payload_local["vx"], dtype=np.float32)
    vy = np.asarray(payload_local["vy"], dtype=np.float32)
    seeds_x, seeds_y = _build_seed_arrays(payload_local)
    right_x = np.float32(payload_local.get("right_x", x_faces[-1]))
    max_points = int(payload_local.get("max_points", 2000))
    max_steps = int(payload_local.get("max_steps", payload_local.get("max_iters", max_points)))

    tracker_sig = _pollock_tracker_signature(x_faces, y_faces, vx, vy, int(seeds_x.size))
    sim_pt_pollock_gpu._ensure_ti_init()
    if global_pollock_tracker is None:
        PollockTracking2D = sim_pt_pollock_gpu._make_pollock_tracker()
        global_pollock_tracker = PollockTracking2D(
            x_faces,
            y_faces,
            vx,
            vy,
            seeds_x,
            seeds_y,
            right_x,
            max_steps,
        )
        global_pollock_tracker_signature = tracker_sig
    elif global_pollock_tracker_signature != tracker_sig:
        raise RuntimeError(
            "Persistent GPU Pollock worker already materialized a tracker for a different "
            "grid/particle shape. Use one field size and exact particle count per worker process."
        )
    else:
        tracker = global_pollock_tracker
        tracker.x_faces.from_numpy(x_faces)
        tracker.y_faces.from_numpy(y_faces)
        tracker.vx_faces.from_numpy(vx)
        tracker.vy_faces.from_numpy(vy)
        tracker.seed_x.from_numpy(seeds_x)
        tracker.seed_y.from_numpy(seeds_y)
        tracker._seeds_x = seeds_x.copy()
        tracker._seeds_y = seeds_y.copy()
        tracker._x_faces_np = x_faces
        tracker._y_faces_np = y_faces
        tracker._vx_faces_np = vx
        tracker._vy_faces_np = vy
        tracker.max_steps = int(max_steps)
        tracker.right_x = np.float32(right_x)

    tracker = global_pollock_tracker
    run_result = tracker.run()
    if len(run_result) < 2:
        raise RuntimeError(
            f"PollockTracking2D.run() returned {len(run_result)} values; expected at least travel_times and status."
        )
    raw_travel_times, status = run_result[:2]
    travel_times = np.where(status == sim_pt_pollock_gpu.STATUS_SUCCESS, raw_travel_times, np.nan).astype(np.float32)
    stop_mode = str(payload_local.get("stop_mode", "first"))
    arrival_fraction = float(payload_local.get("arrival_fraction", 1.0))
    return_fastest = bool(payload_local.get("return_fastest", stop_mode == "first"))
    summary = sim_pt_utils.summarize_arrivals(
        travel_times,
        status,
        stop_mode=stop_mode,
        arrival_fraction=arrival_fraction,
    )
    summary.update(
        {
            "backend": "gpu",
            "method": "pollock",
            "tracking_mode": "arrival",
            "stop_mode": stop_mode,
            "save_trajectories": False,
            "return_fastest": return_fastest,
            "status": np.asarray(status, dtype=np.int32),
            "lengths": None,
        }
    )
    if return_fastest and summary["n_arrived"] > 0:
        fastest_id = sim_pt_utils.fastest_particle_id(travel_times)
        traj = tracker.replay_selected(np.array([fastest_id], dtype=np.int32), right_x, max_points, max_steps)[0]
        traj["particle_id"] = int(fastest_id)
        summary["fastest"] = sim_pt_utils.build_fastest_result_from_trajectory(
            fastest_id,
            float(travel_times[fastest_id]),
            traj,
        )
    return summary


def run_gpu_batch(func_name: str, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for payload in payloads:
        if (
            func_name == "particle_tracking_summary"
            and str(payload.get("method", "")) == "pollock"
            and str(payload.get("backend", "")) == "gpu"
        ):
            outputs.append(_run_pollock_gpu_summary_single(payload))
        else:
            outputs.append(_run_single_gpu_payload(func_name, payload))
    return outputs


def _store_error_results(result_dict: Any, tasks: list[dict[str, Any]], message: str) -> None:
    for task in tasks:
        result_dict[task["id"]] = {"ok": False, "result": None, "error": message}


def _execute_batch(logger: Any, batch: list[dict[str, Any]], result_dict: Any) -> None:
    func_name = str(batch[0]["func"])
    payloads = [task["payload"] for task in batch]
    t0 = time.monotonic()
    try:
        outputs = run_gpu_batch(func_name, payloads)
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"Batched GPU function returned {len(outputs)} outputs for {len(batch)} inputs."
            )
    except Exception as exc:
        _store_error_results(result_dict, batch, str(exc))
        return

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    logger.info(f"[GPU-Worker] Executed batch size={len(batch)} func={func_name} elapsed_ms={elapsed_ms:.3f}")
    for task, output in zip(batch, outputs):
        result_dict[task["id"]] = {"ok": True, "result": output, "error": ""}


def gpu_worker(config_base: dict[str, Any], gpu_token: str, queue_gpu: Any, result_dict: Any) -> None:
    logger = base.sim_utils.get_logger(path_logfile=str(config_base["path_mainlogfile"]))
    logger.info(f"[GPU-Worker {gpu_token}] Starting")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_token

    pending: deque[dict[str, Any]] = deque()
    shutdown_requested = False

    while True:
        if pending:
            first_task = pending.popleft()
        else:
            try:
                first_task = queue_gpu.get(timeout=0.5)
            except queue.Empty:
                if shutdown_requested:
                    break
                continue

        if first_task is None:
            shutdown_requested = True
            if not pending:
                break
            continue

        batch = [first_task]
        batch_key = first_task["batch_key"]
        deadline = time.monotonic() + (MAX_BATCH_WAIT_MS / 1000.0)

        while len(batch) < MAX_BATCH_SIZE:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            if pending:
                candidate = pending.popleft()
            else:
                try:
                    candidate = queue_gpu.get(timeout=timeout)
                except queue.Empty:
                    break

            if candidate is None:
                shutdown_requested = True
                continue
            if candidate["batch_key"] == batch_key:
                batch.append(candidate)
            else:
                pending.append(candidate)

        _execute_batch(logger, batch, result_dict)
        if shutdown_requested and not pending:
            break

    while pending:
        task = pending.popleft()
        if task is not None:
            _store_error_results(result_dict, [task], "GPU worker shut down before task execution.")

    logger.info(f"[GPU-Worker {gpu_token}] Stopping")


def run_particle_tracking_gpu_worker(
    *,
    config_mc: dict[str, Any],
    x0: int,
    x1: int,
    source_rows: np.ndarray,
    k_field: np.ndarray,
    h_field: np.ndarray,
    logger: Any,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    n_particles = base.resolve_particle_count(config_mc, source_rows)

    pt_cfg = config_mc["particleTracking_arg"]
    pt_solver = str(pt_cfg["solver"])
    requested_backend = str(pt_cfg.get("backend", "gpu"))
    velocity_location = str(pt_cfg["velocity_location"])
    align_with_modflow = bool(pt_cfg["align_with_modflow_velocity"])

    if requested_backend != "gpu":
        raise ValueError("GPU-worker runner only supports particle-tracking backend='gpu'.")
    if pt_solver != "pollock":
        raise ValueError("GPU-worker runner currently supports solver='pollock' only.")

    x_faces = base.build_face_coordinates_from_centers(config_mc["x"]).astype(np.float32)
    y_faces = base.build_face_coordinates_from_centers(config_mc["y"]).astype(np.float32)
    vx_pt, vy_pt = base.sim_hyd.compute_velocity(
        k_field,
        config_mc["x"],
        config_mc["y"],
        h_field,
        output_location="face",
        align_with_modflow=align_with_modflow,
        face_output="full",
    )
    summary_params: dict[str, Any] = {
        "backend": "gpu",
        "method": "pollock",
        "tracking_mode": "arrival",
        "stop_mode": "first",
        "arrival_fraction": 1.0,
        "save_trajectories": False,
        "return_fastest": True,
        "x_faces": x_faces,
        "y_faces": y_faces,
        "vx": vx_pt,
        "vy": vy_pt,
        "seed_spec": {
            "n_particles": int(n_particles),
            "x0": float(x0),
            "y0": float(source_rows[0]),
            "y1": float(source_rows[-1]),
        },
        "right_x": x1,
        "max_points": pt_cfg["max_points"],
        "max_steps": pt_cfg["max_iters"],
    }
    summary = gpu_call("particle_tracking_summary", summary_params)

    st = np.asarray(summary["status"])
    count = Counter(st.tolist())
    frac_success = count.get(0, 0) / max(1, n_particles)
    frac_bad = (count.get(1, 0) + count.get(4, 0)) / max(1, n_particles)
    logger.info(
        f"[PT] solver={pt_solver}, requested_backend={requested_backend}, "
        f"used_backend={summary.get('backend', 'gpu')}, velocity_location={velocity_location}, "
        f"align_with_modflow={align_with_modflow}, status_counts={dict(count)}"
    )
    if frac_success < 0.5 or frac_bad > 0.1:
        raise ValueError(f"Invalid velocity field for tracking: {count}")

    res = summary.get("fastest")
    if res is None:
        raise ValueError("Particle tracking summary did not return the fastest trajectory.")
    fp = np.vstack((res["x"], res["y"])).T
    res = dict(res)
    res["backend"] = str(summary.get("backend", "gpu"))
    return fp, res, {"vx": np.asarray(vx_pt), "vy": np.asarray(vy_pt)}


def run_pypar2_tracking_gpu_worker(
    *,
    config_mc: dict[str, Any],
    x0: int,
    x1: int,
    source_rows: np.ndarray,
    vx_face: np.ndarray,
    vy_face: np.ndarray,
    seed_trial: int | None,
    logger: Any,
) -> tuple[np.ndarray | None, float | None, str | None, dict]:
    """GPU-offloaded variant of `base.run_pypar2_tracking`.

    Builds the same `particle_tracking_summary` payload (method='pypar2') and
    dispatches it through the persistent GPU worker subprocess via `gpu_call`,
    which routes pypar2 through the generic single-payload path in
    `_run_single_gpu_payload` (FlowSimulation api.py handles method='pypar2').

    Returns ``(fp_xy, travel_time, error_message, replay_meta)``; the pilot
    pass scales with the Pollock-style particle count because it is
    trajectory-free, while pass-2 trajectory storage uses only the small
    top-K * particles-per-center ensemble.
    """
    cfg = config_mc.get("pypar2_arg", {})
    n_particles_override = cfg.get("num_particles", None)
    if n_particles_override is None:
        n_particles = base.resolve_particle_count(config_mc, source_rows)
    else:
        n_particles = int(n_particles_override)
    n_particles = max(2, n_particles)
    x_faces = base.build_face_coordinates_from_centers(config_mc["x"]).astype(np.float32)
    y_faces = base.build_face_coordinates_from_centers(config_mc["y"]).astype(np.float32)
    seeds_x = np.full(n_particles, x0, dtype=np.float32)
    seeds_y = np.linspace(
        float(source_rows[0]), float(source_rows[-1]), n_particles, endpoint=True
    ).astype(np.float32)

    params: dict[str, Any] = {
        "method": "pypar2",
        "backend": str(cfg.get("taichi_arch", "gpu")),
        "tracking_mode": "arrival",
        "stop_mode": "first",
        "save_trajectories": True,
        "return_fastest": True,
        "x_faces": x_faces,
        "y_faces": y_faces,
        "vx": np.asarray(vx_face, dtype=np.float32),
        "vy": np.asarray(vy_face, dtype=np.float32),
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "right_x": float(x1),
        "dt": float(cfg.get("dt", 0.01)),
        "steps": int(cfg.get("steps", 200_000)),
        "molecular_diffusion": float(cfg.get("molecular_diffusion", 0.0)),
        "longitudinal_dispersivity": float(cfg.get("alpha_l", 0.0)),
        "transverse_dispersivity": float(cfg.get("alpha_t", 0.0)),
        "max_points": int(cfg.get("max_points", 2_000)),
        "trajectory_memory_limit": int(cfg.get("trajectory_memory_limit", 5_000_000)),
        "replay_top_k": int(cfg.get("replay_top_k", 5)),
        "replay_particles_per_center": int(cfg.get("replay_particles_per_center", 40)),
        "replay_radius_cells": float(cfg.get("replay_radius_cells", 0.5)),
        "replay_tolerance": float(cfg.get("replay_tolerance", 1.05)),
        "replay_max_redos": int(cfg.get("replay_max_redos", 3)),
        "seed": int(seed_trial) if seed_trial is not None else 0,
    }
    if cfg.get("velocity_floor") is not None:
        params["velocity_floor"] = float(cfg["velocity_floor"])

    try:
        summary = gpu_call("particle_tracking_summary", params)
    except Exception as exc:
        logger.warning(f"[PT-PyPar2 GPU] gpu_call failed: {exc}")
        return None, None, f"PyPar2 GPU dispatch failed: {exc}", {}

    res = summary.get("fastest")
    replay_meta = summary.get("replay", {}) or {}
    if res is None:
        return None, None, "PyPar2: no particle reached right_x", replay_meta
    fp = np.vstack((np.asarray(res["x"]), np.asarray(res["y"]))).T
    travel_time = float(res["travel_time"])
    logger.info(
        f"[PT-PyPar2 GPU] taichi_arch={summary.get('taichi_arch', params['backend'])}, "
        f"n_particles={n_particles}, n_arrived={summary.get('n_arrived', 'NA')}, "
        f"travel_time={travel_time:.6g}, "
        f"replay_status={replay_meta.get('status', 'NA')}, "
        f"t_pilot={replay_meta.get('t_pilot', 'NA')}, "
        f"t_replay_best={replay_meta.get('t_replay_best', 'NA')}, "
        f"attempts={len(replay_meta.get('attempts', []))}"
    )
    return fp, travel_time, None, replay_meta


def run_unit_simulation_gpu(config: dict[str, Any], mc_index: int, num_trial: int = 200) -> dict[str, Any]:
    original_pt = base.run_particle_tracking
    original_pypar2 = base.run_pypar2_tracking
    base.run_particle_tracking = run_particle_tracking_gpu_worker
    base.run_pypar2_tracking = run_pypar2_tracking_gpu_worker
    try:
        return base.run_unit_simulation(config, mc_index, num_trial=num_trial)
    finally:
        base.run_particle_tracking = original_pt
        base.run_pypar2_tracking = original_pypar2


def run_field_size_simulation_gpu(
    idx_size: int,
    config_base: dict[str, Any],
    queue_gpu: Any,
    result_dict: Any,
) -> dict[str, Any]:
    logger = base.sim_utils.get_logger(path_logfile=str(config_base["path_mainlogfile"]))
    config = base.make_config_for_size(config_base=config_base, idx_size=idx_size)
    field_size_ = config["field_size_"]
    logger.info(f"[SIZE {field_size_}] Starting simulation.")

    grf_gen = base.make_grf_generator(config["grf_arg"], field_size_)

    num_mc_ = int(config["num_mc"][idx_size])
    cpu_workers = int(config.get("cpu_workers") or min(num_mc_, base.available_cpu_count()))
    cpu_workers = max(1, min(num_mc_, cpu_workers))
    with mp.Pool(
        processes=cpu_workers,
        initializer=init_worker_gpu,
        initargs=(
            grf_gen,
            queue_gpu,
            result_dict,
            config.get("seed_mode", "strict_mc"),
            config.get("seed_effective_run"),
        ),
    ) as pool:
        records = pool.starmap(run_unit_simulation_gpu, [(config, i) for i in range(num_mc_)])

    agg = {
        "schema_version": np.array([base.MC_SCHEMA_VERSION], dtype=np.int32),
        "field_size": np.array([field_size_], dtype=np.int32),
    }
    for key in base.MC_FIELDS:
        values = [rec.get(key, np.nan) for rec in records]
        if key in base.OBJECT_FIELDS:
            agg[key] = np.asarray(values, dtype=object)
        elif key in base.INT_FIELDS:
            agg[key] = np.asarray(values, dtype=np.int32)
        elif key in base.BOOL_FIELDS:
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
    parser.add_argument("--covariance", type=str, default="exp", choices=["exp", "fbm"])
    parser.add_argument("--length_scale", type=float, default=None, help="Normalized length scale in (0, 1) for exp covariance.")
    parser.add_argument("--fbm_alpha", type=float, default=None, help="Alpha parameter for fbm covariance.")
    parser.add_argument("--fbm_c", type=float, default=None, help="Scale parameter c for fbm covariance.")
    parser.add_argument("--minpadding", type=int, default=0, help="Minimal padding for GRF generation.")
    parser.add_argument("--dh", type=float, default=0.01, help="Hydraulic head x-gradient.")
    parser.add_argument("--output_schema_version", type=int, default=base.MC_SCHEMA_VERSION, help="Output schema version.")
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
    parser.add_argument("--variance_seed_offset", type=int, default=1000003)
    parser.add_argument("--variance_seed", type=int, default=None)
    parser.add_argument("--variance_rank", type=int, default=0)
    parser.add_argument("--fmm_velocity_layout", type=str, default="face", choices=["face", "cell"])
    parser.add_argument("--solver", type=str, default="pollock", choices=["pollock", "rk45"])
    parser.add_argument("--backend", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument(
        "--num_particles",
        type=int,
        default=None,
        help="Exact particle count override for fastest-path tracking. If unset, density-based seeding is used.",
    )
    parser.add_argument("--velocity-location", dest="velocity_location", type=str, default="face", choices=["face", "cell"])
    parser.add_argument("--align-with-modflow-velocity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fmm_face_mapping", type=str, default="to_cell_avg", choices=["to_cell_avg", "direct"])
    parser.add_argument("--fmm_face_output", type=str, default="full", choices=["full", "trimmed"])
    parser.add_argument("--fmm_solver", type=str, default="scikit", choices=["custom", "scikit"])
    parser.add_argument("--fmm_backtracking", type=str, default="discrete", choices=["discrete", "continuous"])
    parser.add_argument("--fmm_zero_policy", type=str, default="obstacle", choices=["obstacle", "clip", "error"])
    parser.add_argument("--fmm_min_speed", type=float, default=1e-12)
    parser.add_argument("--fsm_solver", type=str, default="fipy", choices=["fipy", "sweep", "legacy", "new", "fd"])
    parser.add_argument("--fsm_max_sweeps", type=int, default=100)
    parser.add_argument("--fsm_tol", type=float, default=1e-6)
    parser.add_argument("--fsm_mse_tol", type=float, default=None)
    parser.add_argument(
        "--fsm_convergence_mode",
        type=str,
        default=None,
        choices=["update", "mse", "update_or_mse", "update_and_mse"],
    )
    parser.add_argument("--fsm_mse_check_every", type=int, default=5)
    parser.add_argument("--fsm_mse_activation_factor", type=float, default=10.0)
    parser.add_argument("--fsm_consecutive_passes", type=int, default=1)
    parser.add_argument("--fsm_dt", type=float, default=0.5)
    parser.add_argument("--fsm_max_steps", type=int, default=10_000)
    parser.add_argument("--fm_asr", "--fm-asr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fm_asr_perp_ratio", type=float, default=0.5)
    parser.add_argument("--fm_asr_back_factor", type=float, default=2.0)
    parser.add_argument("--fm_asr_min_speed", type=float, default=None)
    parser.add_argument("--fm_asr_velocity_multiplier", type=float, default=1.0)
    parser.add_argument("--fm_asr_max_steps", type=int, default=200_000)
    parser.add_argument(
        "--pypar2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the Taichi PAR2 random-walk particle-tracking comparison path (fp_pypar2).",
    )
    parser.add_argument("--pypar2_dt", type=float, default=0.01, help="PyPar2 time step.")
    parser.add_argument("--pypar2_steps", type=int, default=200_000, help="PyPar2 max number of time steps.")
    parser.add_argument(
        "--pypar2_dm", "--pypar2_molecular_diffusion",
        dest="pypar2_dm", type=float, default=1e-9,
        help="PyPar2 molecular diffusion coefficient (Dm).",
    )
    parser.add_argument("--pypar2_alpha_l", type=float, default=0.1, help="PyPar2 longitudinal dispersivity (alphaL).")
    parser.add_argument("--pypar2_alpha_t", type=float, default=0.01, help="PyPar2 transverse dispersivity (alphaT).")
    parser.add_argument(
        "--pypar2_velocity_floor",
        type=float,
        default=None,
        help="PyPar2 velocity-floor regularization (default: auto from Dm/alpha_l).",
    )
    parser.add_argument(
        "--pypar2_taichi_arch",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Taichi backend for PyPar2 (forwarded as taichi_arch through the dispatcher).",
    )
    parser.add_argument(
        "--pypar2_max_points",
        type=int,
        default=2_000,
        help="PyPar2 trajectory max stored points per particle (kept small so trajectory storage stays enabled).",
    )
    parser.add_argument(
        "--pypar2_num_particles",
        type=int,
        default=None,
        help=(
            "PyPar2 pilot-pass particle count. If unset, reuses --num_particles "
            "(pilot pass is trajectory-free)."
        ),
    )
    parser.add_argument(
        "--pypar2_trajectory_memory_limit",
        type=int,
        default=5_000_000,
        help="PyPar2 trajectory_memory_limit forwarded to the dispatcher (pass-2 only).",
    )
    parser.add_argument(
        "--pypar2_replay_top_k",
        type=int,
        default=5,
        help="PyPar2 replay: top-K fastest pilot particles whose release points seed pass-2.",
    )
    parser.add_argument(
        "--pypar2_replay_particles_per_center",
        type=int,
        default=40,
        help="PyPar2 replay: particles per top-K release point in pass-2.",
    )
    parser.add_argument(
        "--pypar2_replay_radius_cells",
        type=float,
        default=0.5,
        help="PyPar2 replay: pass-2 disk radius in units of dx (0.0 = exact resample).",
    )
    parser.add_argument(
        "--pypar2_replay_tolerance",
        type=float,
        default=1.05,
        help="PyPar2 replay: accept when t_best <= t_pilot * tolerance.",
    )
    parser.add_argument(
        "--pypar2_replay_max_redos",
        type=int,
        default=3,
        help="PyPar2 replay: max additional retries (total attempts = this + 1).",
    )
    parser.add_argument("--cpu_workers", type=int, default=None, help="CPU worker count for preprocessing/finalization.")
    parser.add_argument("--gpu_device", type=str, default=None, help="GPU token to pin the single GPU worker to.")
    group.add_argument("--std", type=float, default=None)
    group.add_argument("--var", type=float, default=None)
    args = parser.parse_args()

    if args.backend != "gpu":
        parser.error("--backend cpu should use scripts/run_fmm.py; this entrypoint is for GPU-worker runs.")
    if args.solver != "pollock":
        parser.error("scripts/run_fmm_gpu_worker.py currently supports --solver pollock only.")
    if args.solver == "pollock" and args.velocity_location != "face":
        parser.error("--solver pollock requires --velocity-location face")
    if args.fmm_velocity_layout == "face" and args.fmm_face_output == "full" and args.fmm_face_mapping == "direct":
        parser.error("--fmm_face_output full is not supported with --fmm_face_mapping direct")
    base.validate_covariance_args(args, parser)
    return args


def build_config_base(args: argparse.Namespace, effective_run_seed: int | None, timestamp: str) -> dict[str, Any]:
    path_workdir = Path(args.path_workdir).expanduser().resolve()
    config_base: dict[str, Any] = {
        "num_mc": args.num_mc,
        "dx": 1,
        "dy": 1,
        "dz": 1,
        "field_size": args.field_size,
        "grf_arg": base.build_grf_arg(args),
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
        "fsm_arg": {
            "solver": args.fsm_solver,
            "max_sweeps": args.fsm_max_sweeps,
            "tol": args.fsm_tol,
            "mse_tol": args.fsm_mse_tol,
            "convergence_mode": args.fsm_convergence_mode,
            "mse_check_every": args.fsm_mse_check_every,
            "mse_activation_factor": args.fsm_mse_activation_factor,
            "consecutive_passes": args.fsm_consecutive_passes,
            "dt": args.fsm_dt,
            "max_steps": args.fsm_max_steps,
        },
        "fm_asr_arg": {
            "enabled": bool(args.fm_asr),
            "perp_ratio": float(args.fm_asr_perp_ratio),
            "back_factor": float(args.fm_asr_back_factor),
            "min_speed": float(args.fmm_min_speed if args.fm_asr_min_speed is None else args.fm_asr_min_speed),
            "velocity_multiplier": float(args.fm_asr_velocity_multiplier),
            "max_steps": int(args.fm_asr_max_steps),
        },
        "pypar2_arg": {
            "enabled": bool(args.pypar2),
            "dt": float(args.pypar2_dt),
            "steps": int(args.pypar2_steps),
            "molecular_diffusion": float(args.pypar2_dm),
            "alpha_l": float(args.pypar2_alpha_l),
            "alpha_t": float(args.pypar2_alpha_t),
            "velocity_floor": (
                None if args.pypar2_velocity_floor is None else float(args.pypar2_velocity_floor)
            ),
            "taichi_arch": str(args.pypar2_taichi_arch),
            "max_points": int(args.pypar2_max_points),
            "num_particles": (
                None if args.pypar2_num_particles is None else int(args.pypar2_num_particles)
            ),
            "trajectory_memory_limit": int(args.pypar2_trajectory_memory_limit),
            "replay_top_k": int(args.pypar2_replay_top_k),
            "replay_particles_per_center": int(args.pypar2_replay_particles_per_center),
            "replay_radius_cells": float(args.pypar2_replay_radius_cells),
            "replay_tolerance": float(args.pypar2_replay_tolerance),
            "replay_max_redos": int(args.pypar2_replay_max_redos),
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
        "cpu_workers": args.cpu_workers,
    }

    config_base["path_log"] = config_base["path_simulation"] / "log"
    config_base["path_mainlogfile"] = config_base["path_log"] / f"simulation_{timestamp}.log"
    config_base["path_gpulogfile"] = config_base["path_log"] / f"gpu_profile_{timestamp}.log"
    config_base["path_cpulogfile"] = config_base["path_log"] / f"cpu_profile_{timestamp}.log"
    config_base["path_simulation"].mkdir(parents=True, exist_ok=True)
    config_base["path_log"].mkdir(parents=True, exist_ok=True)
    return config_base


def main() -> None:
    args = parse_args()
    if args.output_schema_version != base.MC_SCHEMA_VERSION:
        raise ValueError(f"Only schema version {base.MC_SCHEMA_VERSION} is supported.")
    if len(args.num_mc) != len(args.field_size):
        raise ValueError("--num_mc and --field_size must have the same length.")
    if args.seed is None and args.seed_mode != "strict_mc":
        raise ValueError("--seed_mode global_stream/best_effort requires --seed.")
    if args.variance_seed_policy == "explicit" and args.variance_seed is None and args.seed is not None:
        raise ValueError("--variance_seed is required when --variance_seed_policy explicit with --seed.")

    effective_run_seed = base.resolve_effective_run_seed(args)
    mp.set_start_method("spawn", force=True)

    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    config_base = build_config_base(args, effective_run_seed, timestamp)

    logger = base.sim_utils.get_logger(path_logfile=str(config_base["path_mainlogfile"]))
    logger.info("[MAIN] Starting GPU-worker simulation.")
    logger.info(
        "[MAIN] Seed config: "
        f"seed_base={config_base['seed_base']}, "
        f"seed_effective_run={config_base['seed_effective_run']}, "
        f"seed_mode={config_base['seed_mode']}, "
        f"variance_seed_policy={config_base['variance_seed_policy']}, "
        f"variance_rank={config_base['variance_rank']}, "
        f"variance_seed_override={config_base['variance_seed_override']}, "
        f"pt_solver={config_base['particleTracking_arg']['solver']}, "
        f"pt_backend={config_base['particleTracking_arg']['backend']}, "
        f"cpu_workers={config_base['cpu_workers']}"
    )

    gpu_tokens = [args.gpu_device] if args.gpu_device else parse_visible_gpu_tokens()
    if not gpu_tokens:
        raise RuntimeError("No visible GPU found for scripts/run_fmm_gpu_worker.py.")
    gpu_token = gpu_tokens[0]

    event_stop_gpuprofiler = threading.Event()
    thread_gpuprofiler = threading.Thread(
        target=base.sim_utils.log_gpu_usage,
        kwargs={
            "path_logfile": str(config_base["path_gpulogfile"]),
            "interval": 1.0,
            "event_stop": event_stop_gpuprofiler,
        },
    )
    thread_gpuprofiler.start()

    cpu_profiler = cProfile.Profile()
    cpu_profiler.enable()

    manager = mp.Manager()
    result_dict = manager.dict()
    queue_gpu = mp.Queue(maxsize=GPU_QUEUE_MAXSIZE)
    process_gpu_worker = mp.Process(
        target=gpu_worker,
        args=(config_base, gpu_token, queue_gpu, result_dict),
        name=f"GPU-Worker-{gpu_token}",
    )
    process_gpu_worker.start()

    results = []
    try:
        for idx_size in range(len(config_base["field_size"])):
            results.append(
                run_field_size_simulation_gpu(
                    idx_size=idx_size,
                    config_base=config_base,
                    queue_gpu=queue_gpu,
                    result_dict=result_dict,
                )
            )
    finally:
        queue_gpu.put(None)
        process_gpu_worker.join(timeout=30)
        if process_gpu_worker.is_alive():
            process_gpu_worker.terminate()
            process_gpu_worker.join(timeout=10)
        manager.shutdown()

        cpu_profiler.disable()
        with open(str(config_base["path_cpulogfile"]), "w", encoding="utf-8") as handle:
            pstats.Stats(cpu_profiler, stream=handle).sort_stats("time").print_stats()

        event_stop_gpuprofiler.set()
        thread_gpuprofiler.join(timeout=10)

    end_time = datetime.now()
    elapsed_hours = round((end_time - start_time).total_seconds() / 3600, 2)
    logger.info(
        f"[MAIN] Completed GPU-worker simulation. {elapsed_hours} hours elapsed "
        f"from {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}."
    )

    del results
    gc.collect()


if __name__ == "__main__":
    main()
