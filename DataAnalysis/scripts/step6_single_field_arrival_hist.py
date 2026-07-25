#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
import sys
import uuid
from collections import Counter
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

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
    t_fmm_fp: float
    t_fmm_lrp: float
    fmm_fp_target_times: np.ndarray
    fmm_lrp_target_times: np.ndarray


@dataclass(slots=True)
class ArrivalSavedData:
    tau_samples: np.ndarray
    tau_empirical: np.ndarray
    cdf_empirical: np.ndarray
    tau_grid: np.ndarray
    cdf_analytical: np.ndarray
    cdf_lognormal: np.ndarray | None
    cdf_mc_mean: np.ndarray | None
    cdf_mc_band_low: np.ndarray | None
    cdf_mc_band_high: np.ndarray | None
    pt_solver: str | None
    pt_backend: str | None
    pt_velocity_location: str | None
    pt_align_with_modflow_velocity: bool | None
    pt_num_particles: int | None
    source_x: int | None
    target_x: int | None
    source_y_start: int | None
    source_y_end: int | None
    target_y_start: int | None
    target_y_end: int | None
    seeds_used: np.ndarray | None
    n_success_by_mc: np.ndarray | None
    tau_samples_by_mc: list[np.ndarray] | None
    length: float | None
    velocity: float | None
    i_y: float | None
    var: float | None
    band_low: float | None
    band_high: float | None
    pdf_hist_bins: int | None
    pdf_hist_range_low: float | None
    pdf_hist_range_high: float | None
    pdf_hist_bin_edges: np.ndarray | None
    tau_pt_first_arrival_empirical: np.ndarray | None
    cdf_pt_first_arrival: np.ndarray | None
    tau_fmm_fp_empirical: np.ndarray | None
    cdf_fmm_fp: np.ndarray | None
    tau_fmm_lrp_empirical: np.ndarray | None
    cdf_fmm_lrp: np.ndarray | None
    tau_fmm_fp_target_empirical: np.ndarray | None
    cdf_fmm_fp_target: np.ndarray | None
    tau_fmm_lrp_target_empirical: np.ndarray | None
    cdf_fmm_lrp_target: np.ndarray | None


@dataclass(slots=True)
class ReloadPlotData:
    tau_empirical: np.ndarray
    cdf_empirical: np.ndarray
    tau_grid: np.ndarray
    cdf_analytical: np.ndarray
    cdf_lognormal: np.ndarray | None
    cdf_mc_mean: np.ndarray | None
    cdf_mc_band_low: np.ndarray | None
    cdf_mc_band_high: np.ndarray | None
    title: str
    note: str | None
    tau_pt_first_arrival_empirical: np.ndarray | None
    cdf_pt_first_arrival: np.ndarray | None
    tau_fmm_fp_empirical: np.ndarray | None
    cdf_fmm_fp: np.ndarray | None
    tau_fmm_lrp_empirical: np.ndarray | None
    cdf_fmm_lrp: np.ndarray | None
    tau_fmm_fp_target_empirical: np.ndarray | None
    cdf_fmm_fp_target: np.ndarray | None
    tau_fmm_lrp_target_empirical: np.ndarray | None
    cdf_fmm_lrp_target: np.ndarray | None


@dataclass(slots=True)
class ReloadPdfData:
    tau_centers: np.ndarray
    empirical_pdf: np.ndarray
    analytical_pdf: np.ndarray
    lognormal_pdf: np.ndarray | None
    mc_pdf_mean: np.ndarray | None
    mc_pdf_band_low: np.ndarray | None
    mc_pdf_band_high: np.ndarray | None
    title: str
    note: str | None


@dataclass(slots=True)
class ArrivalGeometry:
    x: np.ndarray
    y: np.ndarray
    source_x: int
    target_x: int
    source_y_start: int
    source_y_end: int
    target_y_start: int
    target_y_end: int

    @property
    def source_y_start_inclusive(self) -> int:
        return int(self.source_y_start)

    @property
    def source_y_end_exclusive(self) -> int:
        return int(self.source_y_end)

    @property
    def source_span(self) -> int:
        return int(self.source_y_end - self.source_y_start)

    @property
    def transport_length(self) -> int:
        return int(self.target_x - self.source_x)


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


def _default_arrival_geometry(field_size: int) -> ArrivalGeometry:
    source_x = field_size // 10 - 1
    target_x = field_size - field_size // 10 - 1
    source_y_start = field_size // 10 - 1
    source_y_end = field_size - field_size // 10 - 1
    target_y_start = 0
    target_y_end = field_size - 1
    return ArrivalGeometry(
        x=np.arange(0, field_size, 1.0),
        y=np.arange(0, field_size, 1.0),
        source_x=int(source_x),
        target_x=int(target_x),
        source_y_start=int(source_y_start),
        source_y_end=int(source_y_end),
        target_y_start=int(target_y_start),
        target_y_end=int(target_y_end),
    )


def _validate_line_span(*, field_size: int, start: int, end: int, label: str) -> None:
    if start < 0 or start >= field_size:
        raise ValueError(f"{label}_start must be within [0, {field_size - 1}], got {start}.")
    if end <= 0 or end > field_size:
        raise ValueError(f"{label}_end must be within [1, {field_size}], got {end}.")
    if start >= end:
        raise ValueError(f"{label}_start must be < {label}_end, got {start} >= {end}.")


def _validate_axis_index(*, field_size: int, value: int, label: str) -> None:
    if value < 0 or value >= field_size:
        raise ValueError(f"{label} must be within [0, {field_size - 1}], got {value}.")


def _resolve_arrival_geometry(
    *,
    field_size: int,
    source_x: int | None = None,
    target_x: int | None = None,
    source_y_start: int | None = None,
    source_y_end: int | None = None,
    target_y_start: int | None = None,
    target_y_end: int | None = None,
) -> ArrivalGeometry:
    geometry = _default_arrival_geometry(field_size)
    if source_x is not None:
        geometry.source_x = int(source_x)
    if target_x is not None:
        geometry.target_x = int(target_x)
    if source_y_start is not None:
        geometry.source_y_start = int(source_y_start)
    if source_y_end is not None:
        geometry.source_y_end = int(source_y_end)
    if target_y_start is not None:
        geometry.target_y_start = int(target_y_start)
    if target_y_end is not None:
        geometry.target_y_end = int(target_y_end)

    _validate_axis_index(field_size=field_size, value=geometry.source_x, label="source_x")
    _validate_axis_index(field_size=field_size, value=geometry.target_x, label="target_x")
    _validate_line_span(
        field_size=field_size,
        start=geometry.source_y_start,
        end=geometry.source_y_end,
        label="source_y",
    )
    _validate_line_span(
        field_size=field_size,
        start=geometry.target_y_start,
        end=geometry.target_y_end,
        label="target_y",
    )
    if geometry.target_x <= geometry.source_x:
        raise ValueError(
            f"target_x must be greater than source_x for forward transport, got "
            f"source_x={geometry.source_x}, target_x={geometry.target_x}."
        )
    return geometry


def _build_face_coordinates_from_centers(coord_centers: np.ndarray) -> np.ndarray:
    coord_centers = np.asarray(coord_centers, dtype=float).ravel()
    if coord_centers.size < 2:
        raise ValueError("coord_centers must contain at least 2 values")
    faces = np.empty(coord_centers.size + 1, dtype=float)
    faces[1:-1] = 0.5 * (coord_centers[:-1] + coord_centers[1:])
    faces[0] = coord_centers[0] - 0.5 * (coord_centers[1] - coord_centers[0])
    faces[-1] = coord_centers[-1] + 0.5 * (coord_centers[-1] - coord_centers[-2])
    return faces


MAX_BATCH_SIZE = 32
MAX_BATCH_WAIT_MS = 5
GPU_SUBMIT_TIMEOUT_SEC = 60.0
GPU_RESULT_POLL_SEC = 0.001

global_queue_gpu: Any = None
global_result_dict: Any = None
global_pollock_tracker: Any = None
global_pollock_tracker_signature: tuple[Any, ...] | None = None
sim_pt_pollock_gpu: Any = None
sim_pt_utils: Any = None


def _resolve_particle_count(
    *,
    num_particles: int | None,
    density_particle: int,
    source_y_start: int,
    source_y_end: int,
) -> int:
    if num_particles is not None:
        return max(2, int(num_particles))
    return max(2, int(density_particle * max(1, source_y_end - source_y_start)))


def _build_source_seed_coordinates(
    *,
    source_x: float,
    source_y_start: float,
    source_y_end: float,
    n_particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_particles = max(2, int(n_particles))
    seeds_x = np.full(n_particles, source_x, dtype=np.float32)
    y_stop = np.nextafter(np.float32(source_y_end), np.float32(source_y_start))
    seeds_y = np.linspace(source_y_start, y_stop, n_particles, endpoint=True).astype(np.float32)
    return seeds_x, seeds_y


def _value_slug(value: float) -> str:
    return str(f"{float(value):g}").replace("-", "m").replace(".", "p")


def _prepare_payload_for_ipc(payload: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    # Step 6 has a single main process feeding a single GPU worker, so queueing
    # arrays directly is simpler and avoids shared-memory lifecycle leaks.
    return dict(payload), []


def _cleanup_shared_memory(handles: list[Any]) -> None:
    del handles


def _materialize_payload_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload)


def _build_seed_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    spec = dict(payload["seed_spec"])
    n_particles = max(2, int(spec["n_particles"]))
    source_x = float(spec["source_x"])
    source_y_start = float(spec["source_y_start"])
    source_y_end = float(spec["source_y_end"])
    return _build_source_seed_coordinates(
        source_x=source_x,
        source_y_start=source_y_start,
        source_y_end=source_y_end,
        n_particles=n_particles,
    )


def _payload_batch_key(func_name: str, payload: dict[str, Any]) -> tuple[Any, ...]:
    def _shape_for(key: str) -> tuple[int, ...]:
        if key in payload:
            return tuple(np.asarray(payload[key]).shape)
        return ()

    return (
        str(func_name),
        str(payload.get("method", "")),
        str(payload.get("backend", "")),
        str(payload.get("tracking_mode", "")),
        str(payload.get("stop_mode", "")),
        _shape_for("x_faces"),
        _shape_for("y_faces"),
        _shape_for("vx"),
        _shape_for("vy"),
        int(payload.get("seed_spec", {}).get("n_particles", 0)),
        float(payload.get("right_x", 0.0)),
        int(payload.get("max_points", 0)),
        int(payload.get("max_steps", payload.get("max_iters", 0))),
    )


def gpu_call(func_name: str, payload: dict[str, Any], timeout_sec: float = 7200.0) -> dict[str, Any]:
    if global_queue_gpu is None or global_result_dict is None:
        raise RuntimeError("GPU queue/result globals are not initialized.")

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
    global sim_pt_pollock_gpu
    global sim_pt_utils

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
    # FlowSimulation's Pollock GPU tracker may return either
    # (travel_times, status) or (travel_times, status, snap_x, snap_y).
    run_result = tracker.run()
    if not isinstance(run_result, tuple):
        raise RuntimeError(f"Unexpected Pollock GPU run() return type: {type(run_result).__name__}")
    if len(run_result) == 2:
        raw_travel_times, status = run_result
    elif len(run_result) == 4:
        raw_travel_times, status, _snap_x, _snap_y = run_result
    else:
        raise RuntimeError(f"Unexpected Pollock GPU run() return arity: {len(run_result)}")
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


def _run_single_gpu_payload(func_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if func_name != "particle_tracking_summary":
        raise ValueError(f"Unsupported GPU function: {func_name}")
    return _run_pollock_gpu_summary_single(payload)


def _store_error_results(result_dict: Any, tasks: list[dict[str, Any]], message: str) -> None:
    for task in tasks:
        result_dict[task["id"]] = {"ok": False, "result": None, "error": message}


def _execute_batch(batch: list[dict[str, Any]], result_dict: Any) -> None:
    outputs: list[dict[str, Any]] = []
    try:
        for task in batch:
            outputs.append(_run_single_gpu_payload(str(task["func"]), task["payload"]))
    except Exception as exc:
        _store_error_results(result_dict, batch, str(exc))
        return
    for task, output in zip(batch, outputs):
        result_dict[task["id"]] = {"ok": True, "result": output, "error": ""}


def gpu_worker(queue_gpu: Any, result_dict: Any) -> None:
    global sim_pt_pollock_gpu
    global sim_pt_utils

    _import_solver_modules()
    from simulation.particle_tracking import pollock_gpu as _sim_pt_pollock_gpu
    from simulation.particle_tracking import utils as _sim_pt_utils

    sim_pt_pollock_gpu = _sim_pt_pollock_gpu
    sim_pt_utils = _sim_pt_utils

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

        _execute_batch(batch, result_dict)
        if shutdown_requested and not pending:
            break

    while pending:
        task = pending.popleft()
        if task is not None:
            _store_error_results(result_dict, [task], "GPU worker shut down before task execution.")


class PollockGpuWorker:
    def __init__(self) -> None:
        self._manager = mp.Manager()
        self._queue = self._manager.Queue(maxsize=MAX_BATCH_SIZE)
        self._result_dict = self._manager.dict()
        self._process = mp.Process(target=gpu_worker, args=(self._queue, self._result_dict), daemon=True)

    def __enter__(self) -> "PollockGpuWorker":
        global global_queue_gpu
        global global_result_dict
        self._process.start()
        global_queue_gpu = self._queue
        global_result_dict = self._result_dict
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        global global_queue_gpu
        global global_result_dict
        try:
            self._queue.put(None, timeout=1.0)
        except Exception:
            pass
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        self._manager.shutdown()
        global_queue_gpu = None
        global_result_dict = None


def _validate_particle_tracking_options(*, solver: str, velocity_location: str, backend: str = "cpu") -> None:
    if solver not in {"pollock", "rk45"}:
        raise ValueError(f"Unsupported particle-tracking solver: {solver}")
    if velocity_location not in {"face", "cell"}:
        raise ValueError(f"Unsupported velocity_location={velocity_location}; expected 'face' or 'cell'.")
    if backend not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported backend={backend}; expected 'cpu' or 'gpu'.")
    if solver == "pollock" and velocity_location != "face":
        raise ValueError("Pollock solver requires velocity_location='face'.")
    if solver != "pollock" and backend == "gpu":
        raise ValueError("GPU backend is currently supported only with solver='pollock'.")


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


def _extract_success_arrival_times(travel_times: np.ndarray, status_arr: np.ndarray) -> np.ndarray:
    travel_times = np.asarray(travel_times, dtype=float)
    status_arr = np.asarray(status_arr, dtype=np.int32)
    valid = (status_arr == 0) & np.isfinite(travel_times) & (travel_times >= 0.0)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.asarray([], dtype=float)
    return travel_times[idx]


def _compute_fmm_target_arrivals(
    velocity_field: np.ndarray,
    *,
    fast_marching_2d_regions: Any,
    source_x: int,
    target_x: int,
    source_rows: np.ndarray,
    target_rows: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run FMM and return all finite target-line arrivals and their minimum."""
    n_y, n_x = velocity_field.shape
    traversable = np.isfinite(velocity_field) & (velocity_field > 0.0)
    source_mask = np.zeros((n_y, n_x), dtype=bool)
    target_mask = np.zeros((n_y, n_x), dtype=bool)
    source_mask[source_rows, source_x] = True
    target_mask[target_rows, target_x] = True
    source_mask &= traversable
    target_mask &= traversable
    t_field, _ = fast_marching_2d_regions(velocity_field, source_mask, h=1.0, traversable_mask=traversable)
    finite = t_field[target_mask]
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([], dtype=float), float("nan")
    finite = np.asarray(finite, dtype=float)
    return finite, float(np.min(finite))


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


def _lognormal_cdf_on_tau_grid(
    tau_grid: np.ndarray,
    *,
    length: float,
    velocity: float,
    i_y: float,
    sigma_y_sq: float,
) -> tuple[np.ndarray, dict[str, float]]:
    tau = np.asarray(tau_grid, dtype=float)
    if length <= 0.0 or not np.isfinite(length):
        raise ValueError(f"Invalid length={length}; expected positive finite.")
    if velocity <= 0.0 or not np.isfinite(velocity):
        raise ValueError(f"Invalid velocity={velocity}; expected positive finite.")
    if i_y <= 0.0 or not np.isfinite(i_y):
        raise ValueError(f"Invalid i_y={i_y}; expected positive finite.")
    if sigma_y_sq < 0.0 or not np.isfinite(sigma_y_sq):
        raise ValueError(f"Invalid sigma_y_sq={sigma_y_sq}; expected non-negative finite.")
    euler_const = float(getattr(step4.constants, "euler_gamma", np.euler_gamma))
    transit_time = float(length / velocity)
    mean_tau = float(length / i_y)
    var_tau = float(
        step4._compute_x11(
            transit_time,
            velocity=velocity,
            i_y=i_y,
            sigma_y_sq=sigma_y_sq,
            euler_const=euler_const,
        )
        / (i_y**2)
    )
    if mean_tau <= 0.0 or not np.isfinite(mean_tau):
        raise ValueError(f"Invalid lognormal mean tau={mean_tau}; expected positive finite.")
    if var_tau < 0.0 or not np.isfinite(var_tau):
        raise ValueError(f"Invalid lognormal variance tau={var_tau}; expected non-negative finite.")

    sigma_ln_sq = float(np.log1p(var_tau / (mean_tau * mean_tau)))
    sigma_ln = float(np.sqrt(max(sigma_ln_sq, 0.0)))
    mu_ln = float(np.log(mean_tau) - 0.5 * sigma_ln_sq)
    out = np.zeros_like(tau, dtype=float)
    positive = tau > 0.0
    if sigma_ln > 0.0:
        z = (np.log(tau[positive]) - mu_ln) / (np.sqrt(2.0) * sigma_ln)
        out[positive] = 0.5 * (1.0 + erf(z))
    else:
        out[positive] = np.where(tau[positive] >= mean_tau, 1.0, 0.0)
    out = np.clip(out, 0.0, 1.0)
    return out, {
        "lognormal_mean_tau": mean_tau,
        "lognormal_var_tau": var_tau,
        "lognormal_mu": mu_ln,
        "lognormal_sigma_sq": sigma_ln_sq,
    }


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


def _build_tau_grid(
    tau_lo: float,
    tau_hi: float,
    *,
    refine_tau_grid: bool,
    tau_split: float,
    tau_grid_fine: int,
    tau_grid_coarse: int,
) -> np.ndarray:
    if not np.isfinite(tau_lo) or not np.isfinite(tau_hi):
        raise ValueError("tau_lo and tau_hi must be finite.")
    if tau_hi < tau_lo:
        raise ValueError(f"Expected tau_hi >= tau_lo, got tau_lo={tau_lo}, tau_hi={tau_hi}.")
    if tau_grid_fine < 2 or tau_grid_coarse < 2:
        raise ValueError("tau_grid_fine and tau_grid_coarse must be >= 2.")
    if not refine_tau_grid:
        return np.linspace(tau_lo, tau_hi, tau_grid_fine + tau_grid_coarse - 1)

    tau_split_eff = min(float(tau_split), float(tau_hi))
    if tau_split_eff <= tau_lo:
        return np.linspace(tau_lo, tau_hi, tau_grid_coarse)

    s = np.linspace(0.0, 1.0, tau_grid_fine)
    tau_fine = tau_lo + (tau_split_eff - tau_lo) * s**2
    tau_coarse = np.linspace(tau_split_eff, tau_hi, tau_grid_coarse, endpoint=True)
    return np.unique(np.concatenate([tau_fine, tau_coarse]))


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
    cdf_lognormal: np.ndarray | None = None,
    tau_pt_first_arrival_emp: np.ndarray | None = None,
    ecdf_pt_first_arrival: np.ndarray | None = None,
    tau_fmm_fp_emp: np.ndarray | None = None,
    ecdf_fmm_fp: np.ndarray | None = None,
    tau_fmm_lrp_emp: np.ndarray | None = None,
    ecdf_fmm_lrp: np.ndarray | None = None,
    tau_fmm_fp_target_emp: np.ndarray | None = None,
    ecdf_fmm_fp_target: np.ndarray | None = None,
    tau_fmm_lrp_target_emp: np.ndarray | None = None,
    ecdf_fmm_lrp_target: np.ndarray | None = None,
    y_exp: float = 1.0,
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
        ax.plot(tau_grid, cdf_mc_mean, color="#634949", linewidth=1.4, linestyle="-", label="MC mean CDF", zorder=10)
    ax.step(tau_emp, ecdf, where="post", color="black", linewidth=2.0, label="Empirical CDF")
    ax.plot(tau_grid, cdf_analytical, color="#C46552", linewidth=2.0, linestyle="--", label=r"Analytical $G_{\tau}$")
    ax.axhline(y=0.01, color="#C46552", linewidth=1.0, linestyle=":", label="1% quantile")
    # if cdf_lognormal is not None:
    #     ax.plot(tau_grid, cdf_lognormal, color="#2A9D8F", linewidth=1.8, linestyle="-.", label="Lognormal CDF")
    # if tau_pt_first_arrival_emp is not None and ecdf_pt_first_arrival is not None and tau_pt_first_arrival_emp.size > 0:
    #     ax.step(
    #         tau_pt_first_arrival_emp,
    #         ecdf_pt_first_arrival,
    #         where="post",
    #         color="#7A3E9D",
    #         linewidth=1.8,
    #         linestyle="-",
    #         label="PT first-arrival CDF",
    #     )
    # if tau_fmm_fp_emp is not None and ecdf_fmm_fp is not None and tau_fmm_fp_emp.size > 0:
    #     ax.step(tau_fmm_fp_emp, ecdf_fmm_fp, where="post", color="#3F73B8", linewidth=1.8, linestyle="--",
    #             label=r"FMM FP CDF $\gamma^*_{\mathcal{E}}$")
    # if tau_fmm_lrp_emp is not None and ecdf_fmm_lrp is not None and tau_fmm_lrp_emp.size > 0:
    #     ax.step(tau_fmm_lrp_emp, ecdf_fmm_lrp, where="post", color="#F9B43F", linewidth=1.8, linestyle="-.",
    #             label=r"FMM LRP CDF $\gamma^*_{\mathcal{R}}$")
    # if tau_fmm_fp_target_emp is not None and ecdf_fmm_fp_target is not None and tau_fmm_fp_target_emp.size > 0:
    #     ax.step(
    #         tau_fmm_fp_target_emp,
    #         ecdf_fmm_fp_target,
    #         where="post",
    #         color="#0F6D8C",
    #         linewidth=1.6,
    #         linestyle=":",
    #         label="FMM FP target-line CDF",
    #     )
    # if tau_fmm_lrp_target_emp is not None and ecdf_fmm_lrp_target is not None and tau_fmm_lrp_target_emp.size > 0:
    #     ax.step(
    #         tau_fmm_lrp_target_emp,
    #         ecdf_fmm_lrp_target,
    #         where="post",
    #         color="#C77D2B",
    #         linewidth=1.6,
    #         linestyle=(0, (5, 2)),
    #         label="FMM LRP target-line CDF",
    #     )
    ax.set_xlabel(r"$tU/I_Y$")
    ax.set_ylabel("CDF")
    ax.set_xscale("log")
    ax.set_ylim([0.0, 1.01])
    if y_exp <= 0.0 or not np.isfinite(y_exp):
        raise ValueError(f"y_exp must be positive and finite, got {y_exp}.")
    if abs(y_exp - 1.0) > 1e-12:
        ax.set_yscale("function", functions=(lambda y: y**y_exp, lambda y: y**(1.0 / y_exp)))
    ax.grid(True, which="both")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _build_pdf_hist_bin_edges(
    tau_samples: np.ndarray,
    *,
    pdf_hist_bins: int,
) -> np.ndarray:
    samples = np.asarray(tau_samples, dtype=float)
    samples = samples[np.isfinite(samples) & (samples >= 0.0)]
    if pdf_hist_bins < 2:
        raise ValueError("pdf_hist_bins must be >= 2.")
    if samples.size == 0:
        raise ValueError("Need at least one finite non-negative tau sample to build PDF histogram bins.")
    tau_lo = max(0.0, float(np.min(samples)) * 0.8)
    tau_hi = float(np.max(samples)) * 1.2 + 1e-12
    if tau_hi <= tau_lo:
        tau_hi = tau_lo + 1e-6
    return np.linspace(tau_lo, tau_hi, pdf_hist_bins + 1)


def _histogram_pdf(samples: np.ndarray, bin_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values) & (values >= 0.0)]
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be a 1D array with at least 2 entries.")
    counts, _ = np.histogram(values, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, np.asarray(counts, dtype=float)


def _mc_pdf_band(
    tau_by_mc: list[np.ndarray],
    bin_edges: np.ndarray,
    *,
    band_low: float,
    band_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers, _ = _histogram_pdf(tau_by_mc[0], bin_edges)
    pdf_rows = np.asarray([_histogram_pdf(samples, bin_edges)[1] for samples in tau_by_mc], dtype=float)
    return (
        centers,
        np.nanmean(pdf_rows, axis=0),
        np.nanpercentile(pdf_rows, band_low, axis=0),
        np.nanpercentile(pdf_rows, band_high, axis=0),
    )


def _analytical_pdf_on_tau_centers(
    tau_centers: np.ndarray,
    *,
    length: float,
    velocity: float,
    i_y: float,
    sigma_y_sq: float,
) -> np.ndarray:
    tau = np.asarray(tau_centers, dtype=float)
    if np.any(tau < 0.0):
        raise ValueError("tau_centers must be non-negative.")
    dtau = np.maximum(1e-6, 1e-3 * np.maximum(tau, 1.0))
    cdf_plus = _analytical_cdf_on_tau_grid(
        tau + dtau,
        length=length,
        velocity=velocity,
        i_y=i_y,
        sigma_y_sq=sigma_y_sq,
    )
    cdf_minus = _analytical_cdf_on_tau_grid(
        np.maximum(0.0, tau - dtau),
        length=length,
        velocity=velocity,
        i_y=i_y,
        sigma_y_sq=sigma_y_sq,
    )
    pdf = (cdf_plus - cdf_minus) / (np.maximum(0.0, tau + dtau) - np.maximum(0.0, tau - dtau))
    return np.clip(pdf, 0.0, None)


def _lognormal_pdf_on_tau_centers(
    tau_centers: np.ndarray,
    *,
    length: float,
    velocity: float,
    i_y: float,
    sigma_y_sq: float,
) -> np.ndarray:
    tau = np.asarray(tau_centers, dtype=float)
    _cdf, summary = _lognormal_cdf_on_tau_grid(
        np.asarray([max(float(np.min(tau[tau > 0.0], initial=1e-12)), 1e-12), max(float(np.max(tau)), 1e-12)]),
        length=length,
        velocity=velocity,
        i_y=i_y,
        sigma_y_sq=sigma_y_sq,
    )
    sigma_ln_sq = float(summary["lognormal_sigma_sq"])
    sigma_ln = float(np.sqrt(max(sigma_ln_sq, 0.0)))
    mean_tau = float(summary["lognormal_mean_tau"])
    mu_ln = float(summary["lognormal_mu"])
    out = np.zeros_like(tau, dtype=float)
    positive = tau > 0.0
    if sigma_ln > 0.0:
        out[positive] = (
            np.exp(-((np.log(tau[positive]) - mu_ln) ** 2) / (2.0 * sigma_ln_sq))
            / (tau[positive] * sigma_ln * np.sqrt(2.0 * np.pi))
        )
    else:
        out[positive] = 0.0
        closest = int(np.argmin(np.abs(tau - mean_tau)))
        out[closest] = np.inf
    return out


def plot_arrival_pdf_comparison(
    tau_centers: np.ndarray,
    empirical_pdf: np.ndarray,
    analytical_pdf: np.ndarray,
    *,
    title: str,
    bin_edges: np.ndarray,
    mc_pdf_mean: np.ndarray | None = None,
    mc_pdf_band_low: np.ndarray | None = None,
    mc_pdf_band_high: np.ndarray | None = None,
    lognormal_pdf: np.ndarray | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=[5, 4])
    widths = np.diff(np.asarray(bin_edges, dtype=float))
    ax.bar(
        tau_centers,
        empirical_pdf,
        width=widths,
        align="center",
        color="#8C8C8C",
        alpha=0.35,
        edgecolor="none",
        label="Empirical PDF",
    )
    ax.plot(tau_centers, analytical_pdf, color="#C46552", linewidth=2.0, linestyle="--", label=r"Analytical $g_{\tau}$")
    if lognormal_pdf is not None:
        finite = np.isfinite(lognormal_pdf)
        ax.plot(tau_centers[finite], lognormal_pdf[finite], color="#2A9D8F", linewidth=1.8, linestyle="-.", label="Lognormal PDF")
    if mc_pdf_band_low is not None and mc_pdf_band_high is not None:
        ax.fill_between(
            tau_centers,
            mc_pdf_band_low,
            mc_pdf_band_high,
            color="#8C8C8C",
            alpha=0.25,
            linewidth=0.0,
            label="MC PDF band",
        )
    if mc_pdf_mean is not None:
        ax.plot(tau_centers, mc_pdf_mean, color="#634949", linewidth=1.4, linestyle="-", label="MC mean PDF", zorder=10)
    ax.set_xlabel(r"$tU/I_Y$")
    ax.set_ylabel("PDF")
    ax.set_xscale("log")
    ax.grid(True, which="both")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _optional_np_array(payload: Any, key: str) -> np.ndarray | None:
    if key not in payload.files:
        return None
    return np.asarray(payload[key])


def _optional_scalar_string(payload: Any, key: str) -> str | None:
    if key not in payload.files:
        return None
    value = payload[key].tolist()
    if value is None:
        return None
    return str(value)


def _optional_scalar_bool(payload: Any, key: str) -> bool | None:
    if key not in payload.files:
        return None
    value = payload[key].tolist()
    if value is None:
        return None
    return bool(value)


def _optional_scalar_float(payload: Any, key: str) -> float | None:
    if key not in payload.files:
        return None
    value = payload[key].tolist()
    if value is None:
        return None
    return float(value)


def _optional_scalar_int(payload: Any, key: str) -> int | None:
    if key not in payload.files:
        return None
    value = payload[key].tolist()
    if value is None:
        return None
    return int(value)


def load_arrival_npz(npz_path: Path) -> ArrivalSavedData:
    path = Path(npz_path)
    with np.load(path, allow_pickle=True) as payload:
        tau_samples_by_mc: list[np.ndarray] | None = None
        if "tau_samples_by_mc" in payload.files:
            tau_samples_by_mc = [np.asarray(row, dtype=float) for row in payload["tau_samples_by_mc"].tolist()]
        return ArrivalSavedData(
            tau_samples=np.asarray(payload["tau_samples"], dtype=float),
            tau_empirical=np.asarray(payload["tau_empirical"], dtype=float),
            cdf_empirical=np.asarray(payload["cdf_empirical"], dtype=float),
            tau_grid=np.asarray(payload["tau_grid"], dtype=float),
            cdf_analytical=np.asarray(payload["cdf_analytical"], dtype=float),
            cdf_lognormal=_optional_np_array(payload, "cdf_lognormal"),
            cdf_mc_mean=_optional_np_array(payload, "cdf_mc_mean"),
            cdf_mc_band_low=_optional_np_array(payload, "cdf_mc_band_low"),
            cdf_mc_band_high=_optional_np_array(payload, "cdf_mc_band_high"),
            pt_solver=_optional_scalar_string(payload, "pt_solver"),
            pt_backend=_optional_scalar_string(payload, "pt_backend"),
            pt_velocity_location=_optional_scalar_string(payload, "pt_velocity_location"),
            pt_align_with_modflow_velocity=_optional_scalar_bool(payload, "pt_align_with_modflow_velocity"),
            pt_num_particles=_optional_scalar_int(payload, "pt_num_particles"),
            source_x=_optional_scalar_int(payload, "source_x"),
            target_x=_optional_scalar_int(payload, "target_x"),
            source_y_start=_optional_scalar_int(payload, "source_y_start"),
            source_y_end=_optional_scalar_int(payload, "source_y_end"),
            target_y_start=_optional_scalar_int(payload, "target_y_start"),
            target_y_end=_optional_scalar_int(payload, "target_y_end"),
            seeds_used=_optional_np_array(payload, "seeds_used"),
            n_success_by_mc=_optional_np_array(payload, "n_success_by_mc"),
            tau_samples_by_mc=tau_samples_by_mc,
            length=_optional_scalar_float(payload, "length"),
            velocity=_optional_scalar_float(payload, "velocity"),
            i_y=_optional_scalar_float(payload, "i_y"),
            var=_optional_scalar_float(payload, "var"),
            band_low=_optional_scalar_float(payload, "band_low"),
            band_high=_optional_scalar_float(payload, "band_high"),
            pdf_hist_bins=_optional_scalar_int(payload, "pdf_hist_bins"),
            pdf_hist_range_low=_optional_scalar_float(payload, "pdf_hist_range_low"),
            pdf_hist_range_high=_optional_scalar_float(payload, "pdf_hist_range_high"),
            pdf_hist_bin_edges=_optional_np_array(payload, "pdf_hist_bin_edges"),
            tau_pt_first_arrival_empirical=_optional_np_array(payload, "tau_pt_first_arrival_empirical"),
            cdf_pt_first_arrival=_optional_np_array(payload, "cdf_pt_first_arrival"),
            tau_fmm_fp_empirical=_optional_np_array(payload, "tau_fmm_fp_empirical"),
            cdf_fmm_fp=_optional_np_array(payload, "cdf_fmm_fp"),
            tau_fmm_lrp_empirical=_optional_np_array(payload, "tau_fmm_lrp_empirical"),
            cdf_fmm_lrp=_optional_np_array(payload, "cdf_fmm_lrp"),
            tau_fmm_fp_target_empirical=_optional_np_array(payload, "tau_fmm_fp_target_empirical"),
            cdf_fmm_fp_target=_optional_np_array(payload, "cdf_fmm_fp_target"),
            tau_fmm_lrp_target_empirical=_optional_np_array(payload, "tau_fmm_lrp_target_empirical"),
            cdf_fmm_lrp_target=_optional_np_array(payload, "cdf_fmm_lrp_target"),
        )


def _title_from_saved_data(saved: ArrivalSavedData) -> str:
    n_success = int(saved.tau_samples.size)
    mc_count = 1
    if saved.tau_samples_by_mc is not None:
        mc_count = len(saved.tau_samples_by_mc)
    elif saved.n_success_by_mc is not None:
        mc_count = int(np.asarray(saved.n_success_by_mc).size)
    parts = [f"Arrival CDF (reloaded), $n_{{\\mathrm{{success}}}}={n_success}$, MC={mc_count}"]
    if saved.pt_solver is not None:
        parts.append(f"solver={saved.pt_solver}")
    if saved.pt_backend is not None:
        parts.append(f"backend={saved.pt_backend}")
    if saved.pt_velocity_location is not None:
        parts.append(f"velocity={saved.pt_velocity_location}")
    if saved.pt_num_particles is not None:
        parts.append(f"particles={saved.pt_num_particles}")
    if saved.source_x is not None and saved.target_x is not None:
        parts.append(f"source_x={saved.source_x}")
        parts.append(f"target_x={saved.target_x}")
    return ", ".join(parts)


def _summary_json_path_for_npz(npz_path: Path) -> Path:
    path = Path(npz_path)
    suffix = "_arrival_times.npz"
    if path.name.endswith(suffix):
        return path.with_name(path.name[: -len(suffix)] + "_arrival_summary.json")
    return path.with_suffix(".json")


def _load_summary_json(npz_path: Path) -> dict[str, Any] | None:
    json_path = _summary_json_path_for_npz(npz_path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _resolve_reload_metadata(saved: ArrivalSavedData, npz_path: Path) -> dict[str, float] | None:
    data: dict[str, float | None] = {
        "length": saved.length,
        "velocity": saved.velocity,
        "i_y": saved.i_y,
        "var": saved.var,
        "band_low": saved.band_low,
        "band_high": saved.band_high,
    }
    if any(value is None for value in data.values()):
        summary = _load_summary_json(npz_path)
        if summary is not None:
            for key in data:
                if data[key] is None and key in summary and summary[key] is not None:
                    data[key] = float(summary[key])
    if any(value is None for value in data.values()):
        return None
    return {key: float(value) for key, value in data.items() if value is not None}


def _resolve_reload_pdf_bin_edges(saved: ArrivalSavedData) -> np.ndarray | None:
    if saved.pdf_hist_bin_edges is None:
        return None
    edges = np.asarray(saved.pdf_hist_bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        return None
    return edges


def _prepare_reload_plot_data(
    npz_path: Path,
    *,
    refine_tau_grid: bool,
    tau_split: float,
    tau_grid_fine: int,
    tau_grid_coarse: int,
) -> ReloadPlotData:
    saved = load_arrival_npz(npz_path)
    title = _title_from_saved_data(saved)
    metadata = _resolve_reload_metadata(saved, npz_path)
    if metadata is None or saved.tau_samples_by_mc is None:
        note = "Reload used saved curves because recompute metadata was unavailable."
        return ReloadPlotData(
            tau_empirical=saved.tau_empirical,
            cdf_empirical=saved.cdf_empirical,
            tau_grid=saved.tau_grid,
            cdf_analytical=saved.cdf_analytical,
            cdf_lognormal=saved.cdf_lognormal,
            cdf_mc_mean=saved.cdf_mc_mean,
            cdf_mc_band_low=saved.cdf_mc_band_low,
            cdf_mc_band_high=saved.cdf_mc_band_high,
            title=title,
            note=note,
            tau_pt_first_arrival_empirical=saved.tau_pt_first_arrival_empirical,
            cdf_pt_first_arrival=saved.cdf_pt_first_arrival,
            tau_fmm_fp_empirical=saved.tau_fmm_fp_empirical,
            cdf_fmm_fp=saved.cdf_fmm_fp,
            tau_fmm_lrp_empirical=saved.tau_fmm_lrp_empirical,
            cdf_fmm_lrp=saved.cdf_fmm_lrp,
            tau_fmm_fp_target_empirical=saved.tau_fmm_fp_target_empirical,
            cdf_fmm_fp_target=saved.cdf_fmm_fp_target,
            tau_fmm_lrp_target_empirical=saved.tau_fmm_lrp_target_empirical,
            cdf_fmm_lrp_target=saved.cdf_fmm_lrp_target,
        )

    tau_emp = np.asarray(saved.tau_empirical, dtype=float)
    ecdf = np.asarray(saved.cdf_empirical, dtype=float)
    if tau_emp.size == 0 or ecdf.size == 0:
        tau_emp, ecdf = _empirical_cdf(saved.tau_samples)
    tau_lo = max(0.0, float(tau_emp[0]) * 0.8)
    tau_hi = float(tau_emp[-1]) * 1.2 + 1e-12
    tau_grid = _build_tau_grid(
        tau_lo,
        tau_hi,
        refine_tau_grid=refine_tau_grid,
        tau_split=tau_split,
        tau_grid_fine=tau_grid_fine,
        tau_grid_coarse=tau_grid_coarse,
    )
    analytical_cdf = _analytical_cdf_on_tau_grid(
        tau_grid,
        length=metadata["length"],
        velocity=metadata["velocity"],
        i_y=metadata["i_y"],
        sigma_y_sq=metadata["var"],
    )
    cdf_lognormal, _summary = _lognormal_cdf_on_tau_grid(
        tau_grid,
        length=metadata["length"],
        velocity=metadata["velocity"],
        i_y=metadata["i_y"],
        sigma_y_sq=metadata["var"],
    )
    cdf_mc_mean, cdf_mc_band_low, cdf_mc_band_high = _mc_cdf_band(
        saved.tau_samples_by_mc,
        tau_grid,
        band_low=metadata["band_low"],
        band_high=metadata["band_high"],
    )
    return ReloadPlotData(
        tau_empirical=tau_emp,
        cdf_empirical=ecdf,
        tau_grid=tau_grid,
        cdf_analytical=analytical_cdf,
        cdf_lognormal=cdf_lognormal,
        cdf_mc_mean=cdf_mc_mean,
        cdf_mc_band_low=cdf_mc_band_low,
        cdf_mc_band_high=cdf_mc_band_high,
        title=title,
        note=None,
        tau_pt_first_arrival_empirical=saved.tau_pt_first_arrival_empirical,
        cdf_pt_first_arrival=saved.cdf_pt_first_arrival,
        tau_fmm_fp_empirical=saved.tau_fmm_fp_empirical,
        cdf_fmm_fp=saved.cdf_fmm_fp,
        tau_fmm_lrp_empirical=saved.tau_fmm_lrp_empirical,
        cdf_fmm_lrp=saved.cdf_fmm_lrp,
        tau_fmm_fp_target_empirical=saved.tau_fmm_fp_target_empirical,
        cdf_fmm_fp_target=saved.cdf_fmm_fp_target,
        tau_fmm_lrp_target_empirical=saved.tau_fmm_lrp_target_empirical,
        cdf_fmm_lrp_target=saved.cdf_fmm_lrp_target,
    )


def _prepare_reload_pdf_data(npz_path: Path) -> ReloadPdfData:
    saved = load_arrival_npz(npz_path)
    title = _title_from_saved_data(saved).replace("CDF", "PDF")
    metadata = _resolve_reload_metadata(saved, npz_path)
    bin_edges = _resolve_reload_pdf_bin_edges(saved)
    if metadata is None or saved.tau_samples_by_mc is None or bin_edges is None:
        return ReloadPdfData(
            tau_centers=np.asarray([], dtype=float),
            empirical_pdf=np.asarray([], dtype=float),
            analytical_pdf=np.asarray([], dtype=float),
            lognormal_pdf=None,
            mc_pdf_mean=None,
            mc_pdf_band_low=None,
            mc_pdf_band_high=None,
            title=title,
            note="Reload skipped PDF recalculation because saved PDF metadata was unavailable.",
        )
    tau_centers, empirical_pdf = _histogram_pdf(saved.tau_samples, bin_edges)
    tau_centers_mc, mc_pdf_mean, mc_pdf_band_low, mc_pdf_band_high = _mc_pdf_band(
        saved.tau_samples_by_mc,
        bin_edges,
        band_low=metadata["band_low"],
        band_high=metadata["band_high"],
    )
    if not np.allclose(tau_centers, tau_centers_mc):
        raise ValueError("Empirical and MC PDF centers do not match.")
    analytical_pdf = _analytical_pdf_on_tau_centers(
        tau_centers,
        length=metadata["length"],
        velocity=metadata["velocity"],
        i_y=metadata["i_y"],
        sigma_y_sq=metadata["var"],
    )
    lognormal_pdf = _lognormal_pdf_on_tau_centers(
        tau_centers,
        length=metadata["length"],
        velocity=metadata["velocity"],
        i_y=metadata["i_y"],
        sigma_y_sq=metadata["var"],
    )
    return ReloadPdfData(
        tau_centers=tau_centers,
        empirical_pdf=empirical_pdf,
        analytical_pdf=analytical_pdf,
        lognormal_pdf=lognormal_pdf,
        mc_pdf_mean=mc_pdf_mean,
        mc_pdf_band_low=mc_pdf_band_low,
        mc_pdf_band_high=mc_pdf_band_high,
        title=title,
        note=None,
    )


def visualize_saved_arrival_npz(
    npz_path: Path,
    *,
    output_dir: Path | None,
    figure_dpi: int | None,
    refine_tau_grid: bool,
    tau_split: float,
    tau_grid_fine: int,
    tau_grid_coarse: int,
    y_exp: float,
    save_pdf_plot: bool,
) -> tuple[Path, Path | None]:
    plot_data = _prepare_reload_plot_data(
        npz_path,
        refine_tau_grid=refine_tau_grid,
        tau_split=tau_split,
        tau_grid_fine=tau_grid_fine,
        tau_grid_coarse=tau_grid_coarse,
    )
    fig = plot_arrival_cdf_comparison(
        plot_data.tau_empirical,
        plot_data.cdf_empirical,
        plot_data.tau_grid,
        plot_data.cdf_analytical,
        title=plot_data.title,
        cdf_mc_mean=plot_data.cdf_mc_mean,
        cdf_mc_band_low=plot_data.cdf_mc_band_low,
        cdf_mc_band_high=plot_data.cdf_mc_band_high,
        cdf_lognormal=plot_data.cdf_lognormal,
        tau_pt_first_arrival_emp=plot_data.tau_pt_first_arrival_empirical,
        ecdf_pt_first_arrival=plot_data.cdf_pt_first_arrival,
        tau_fmm_fp_emp=plot_data.tau_fmm_fp_empirical,
        ecdf_fmm_fp=plot_data.cdf_fmm_fp,
        tau_fmm_lrp_emp=plot_data.tau_fmm_lrp_empirical,
        ecdf_fmm_lrp=plot_data.cdf_fmm_lrp,
        tau_fmm_fp_target_emp=plot_data.tau_fmm_fp_target_empirical,
        ecdf_fmm_fp_target=plot_data.cdf_fmm_fp_target,
        tau_fmm_lrp_target_emp=plot_data.tau_fmm_lrp_target_empirical,
        ecdf_fmm_lrp_target=plot_data.cdf_fmm_lrp_target,
        y_exp=y_exp,
    )
    out_dir = Path(output_dir) if output_dir is not None else Path(npz_path).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{Path(npz_path).stem}_reloaded.png"
    save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if figure_dpi is not None:
        save_kwargs["dpi"] = int(figure_dpi)
    fig.savefig(fig_path, **save_kwargs)
    plt.close(fig)
    pdf_fig_path: Path | None = None
    if save_pdf_plot:
        pdf_plot_data = _prepare_reload_pdf_data(npz_path)
        if pdf_plot_data.note is not None:
            print(pdf_plot_data.note)
        else:
            pdf_fig = plot_arrival_pdf_comparison(
                pdf_plot_data.tau_centers,
                pdf_plot_data.empirical_pdf,
                pdf_plot_data.analytical_pdf,
                title=pdf_plot_data.title,
                bin_edges=_resolve_reload_pdf_bin_edges(load_arrival_npz(npz_path)),
                mc_pdf_mean=pdf_plot_data.mc_pdf_mean,
                mc_pdf_band_low=pdf_plot_data.mc_pdf_band_low,
                mc_pdf_band_high=pdf_plot_data.mc_pdf_band_high,
                lognormal_pdf=pdf_plot_data.lognormal_pdf,
            )
            pdf_fig_path = out_dir / f"{Path(npz_path).stem}_reloaded_pdf.png"
            pdf_fig.savefig(pdf_fig_path, **save_kwargs)
            plt.close(pdf_fig)
    if plot_data.note is not None:
        print(plot_data.note)
    return fig_path, pdf_fig_path


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
    solver: str,
    backend: str = "cpu",
    velocity_location: str,
    align_with_modflow_velocity: bool,
    density_particle: int,
    num_particles: int | None = None,
    geometry: ArrivalGeometry,
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
    gpu_worker_ctx: PollockGpuWorker | None = None,
    fast_marching_2d_regions: Any = None,
) -> ArrivalSimulationResult:
    _validate_particle_tracking_options(solver=solver, velocity_location=velocity_location, backend=backend)
    modules = _import_solver_modules()
    sim_grf = modules["sim_grf"]
    sim_hyd = modules["sim_hyd"]
    sim_pt = modules["sim_pt"]
    if fast_marching_2d_regions is None:
        try:
            from fmm_core.fmm import fast_marching_2d_regions as _fmm_fn
        except Exception as exc:
            raise RuntimeError("Failed to import fmm_core.fmm. Ensure Algorithm/src is on sys.path.") from exc
        fast_marching_2d_regions = _fmm_fn

    np.random.seed(seed)
    cov = sim_grf.ExpCov(sigma=std, length_scale=length_scale)
    pts = (np.linspace(0, 1, field_size),) * 2
    mean = np.zeros((field_size, field_size))
    grf_gen = sim_grf.GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=minpadding)
    logk_field = np.asarray(grf_gen.sample(), dtype=float)
    k_field = np.exp(logk_field)

    x = geometry.x
    y = geometry.y

    h, msg, is_success = sim_hyd.solve_head(k_field, x, y, dh=dh, rtol=1e-10, atol=1e-20, max_it=10000)
    if not is_success:
        raise RuntimeError(f"Head solver failed: {msg}")

    # FMM arrival times
    grad_h_y_arr, grad_h_x_arr = np.gradient(np.asarray(h, dtype=float), np.asarray(y, dtype=float), np.asarray(x, dtype=float), edge_order=2)
    grad_h_mag = np.hypot(grad_h_x_arr, grad_h_y_arr)
    velocity_fp = np.maximum(k_field * grad_h_mag, 1e-12)
    velocity_lrp = np.maximum(k_field * abs(float(dh)), 1e-12)
    fmm_source_rows = np.arange(geometry.source_y_start, geometry.source_y_end, dtype=int)
    fmm_target_rows = np.arange(geometry.target_y_start, geometry.target_y_end, dtype=int)
    fmm_fp_target_times, t_fmm_fp = _compute_fmm_target_arrivals(
        velocity_fp,
        fast_marching_2d_regions=fast_marching_2d_regions,
        source_x=geometry.source_x,
        target_x=geometry.target_x,
        source_rows=fmm_source_rows,
        target_rows=fmm_target_rows,
    )
    fmm_lrp_target_times, t_fmm_lrp = _compute_fmm_target_arrivals(
        velocity_lrp,
        fast_marching_2d_regions=fast_marching_2d_regions,
        source_x=geometry.source_x,
        target_x=geometry.target_x,
        source_rows=fmm_source_rows,
        target_rows=fmm_target_rows,
    )

    n_particles = _resolve_particle_count(
        num_particles=num_particles,
        density_particle=density_particle,
        source_y_start=geometry.source_y_start,
        source_y_end=geometry.source_y_end,
    )
    seeds_x, seeds_y = _build_source_seed_coordinates(
        source_x=float(geometry.source_x),
        source_y_start=float(geometry.source_y_start),
        source_y_end=float(geometry.source_y_end),
        n_particles=n_particles,
    )
    summary_params: dict[str, Any] = {
        "backend": backend,
        "method": solver,
        "tracking_mode": "arrival",
        "stop_mode": "full_btc",
        "arrival_fraction": 1.0,
        "save_trajectories": False,
        "return_fastest": False,
        "right_x": geometry.target_x,
        "max_points": max_points,
        "max_iters": max_iters,
        "threads_per_block": 256,
        "dtype": np.float64,
    }
    if solver == "pollock":
        vx, vy = sim_hyd.compute_velocity(
            k_field,
            x,
            y,
            h,
            output_location="face",
            align_with_modflow=align_with_modflow_velocity,
            face_output="full",
        )
        summary_params.update(
            {
                "x_faces": _build_face_coordinates_from_centers(x).astype(np.float32),
                "y_faces": _build_face_coordinates_from_centers(y).astype(np.float32),
                "vx": vx,
                "vy": vy,
                "max_steps": max_iters,
            }
        )
        if backend == "gpu":
            if gpu_worker_ctx is None:
                raise RuntimeError("GPU Pollock step6 requires an active PollockGpuWorker.")
            summary_params["seed_spec"] = {
                "n_particles": int(n_particles),
                "source_x": float(geometry.source_x),
                "source_y_start": float(geometry.source_y_start),
                "source_y_end": float(geometry.source_y_end),
            }
            summary = gpu_call("particle_tracking_summary", summary_params)
        else:
            summary_params["seeds_x"] = seeds_x
            summary_params["seeds_y"] = seeds_y
            summary = sim_pt.particle_tracking_summary(summary_params)
    else:
        vx, vy = sim_hyd.compute_velocity(
            k_field,
            x,
            y,
            h,
            output_location=velocity_location,
            align_with_modflow=align_with_modflow_velocity,
        )
        summary_params.update(
            {
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "seeds_x": seeds_x,
                "seeds_y": seeds_y,
                "velocity_location": velocity_location,
                "eps_speed": eps_speed,
                "ds_min": ds_min,
                "ds_max": ds_max,
                "tol": tol,
                "y_policy": y_policy,
            }
        )
        summary = sim_pt.particle_tracking_summary(summary_params)
    status_arr = np.asarray(summary["status"], dtype=np.int32)
    status_counter = Counter(status_arr.tolist())
    frac_success = status_counter.get(0, 0) / max(1, n_particles)
    frac_bad = (status_counter.get(1, 0) + status_counter.get(4, 0)) / max(1, n_particles)
    if frac_success < rk45_min_success_frac or frac_bad > rk45_max_bad_frac:
        raise RuntimeError(
            "Invalid velocity field for tracking. "
            f"status={dict(status_counter)}, success_frac={frac_success:.3f}, bad_frac={frac_bad:.3f}"
        )

    arrival_times = _extract_success_arrival_times(np.asarray(summary["travel_times"], dtype=float), status_arr)
    if arrival_times.size == 0:
        raise RuntimeError("No successful particle arrivals (status=0) with finite non-negative arrival times.")

    return ArrivalSimulationResult(
        logk_field=logk_field,
        arrival_times=arrival_times,
        status_counts={int(k): int(v) for k, v in status_counter.items()},
        n_particles=int(n_particles),
        success_fraction=float(frac_success),
        bad_fraction=float(frac_bad),
        t_fmm_fp=t_fmm_fp,
        t_fmm_lrp=t_fmm_lrp,
        fmm_fp_target_times=np.asarray(fmm_fp_target_times, dtype=float),
        fmm_lrp_target_times=np.asarray(fmm_lrp_target_times, dtype=float),
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
    solver: str,
    backend: str = "cpu",
    velocity_location: str,
    align_with_modflow_velocity: bool,
    density_particle: int,
    num_particles: int | None = None,
    source_x: int | None,
    target_x: int | None,
    source_y_start: int | None,
    source_y_end: int | None,
    target_y_start: int | None,
    target_y_end: int | None,
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
    refine_tau_grid: bool,
    tau_split: float,
    tau_grid_fine: int,
    tau_grid_coarse: int,
    y_exp: float,
    pdf_hist_bins: int,
    output_dir: Path | None,
    figure_dpi: int | None,
    save_pdf_plot: bool,
    save_npz: bool,
    save_json: bool,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    _validate_particle_tracking_options(solver=solver, velocity_location=velocity_location, backend=backend)
    field_size_eff = int(cfg.field_size_ref if field_size is None else field_size)
    length_scale_eff = float(cfg.length_scale if length_scale is None else length_scale)
    geometry = _resolve_arrival_geometry(
        field_size=field_size_eff,
        source_x=source_x,
        target_x=target_x,
        source_y_start=source_y_start,
        source_y_end=source_y_end,
        target_y_start=target_y_start,
        target_y_end=target_y_end,
    )
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
    if tau_split <= 0.0 or not np.isfinite(tau_split):
        raise ValueError("tau_split must be positive and finite.")
    if tau_grid_fine < 2 or tau_grid_coarse < 2:
        raise ValueError("tau_grid_fine and tau_grid_coarse must be >= 2.")
    if y_exp <= 0.0 or not np.isfinite(y_exp):
        raise ValueError("y_exp must be positive and finite.")
    if pdf_hist_bins < 2:
        raise ValueError("pdf_hist_bins must be >= 2.")

    step4.configure_style(cfg)
    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.figure_dpi if figure_dpi is None else figure_dpi)

    stem = (
        f"run{cfg.run_id}_singlefield_std{_value_slug(std_eff)}_var{_value_slug(var_eff)}_"
        f"size{field_size_eff}_seed{seed_start}_mc{n_mc}"
    )

    i_y = float(length_scale_eff * field_size_eff)
    length = float(geometry.transport_length)
    seeds_used: list[int] = []
    tau_by_mc: list[np.ndarray] = []
    n_particles_by_mc: list[int] = []
    n_success_by_mc: list[int] = []
    success_fraction_by_mc: list[float] = []
    bad_fraction_by_mc: list[float] = []
    status_counts_by_mc: list[dict[str, int]] = []
    tau_pt_first_arrival_by_mc: list[float] = []
    t_fmm_fp_by_mc: list[float] = []
    t_fmm_lrp_by_mc: list[float] = []
    tau_fmm_fp_target_by_mc: list[np.ndarray] = []
    tau_fmm_lrp_target_by_mc: list[np.ndarray] = []

    worker_cm: Any = PollockGpuWorker() if solver == "pollock" and backend == "gpu" else None
    if worker_cm is None:
        worker_ctx = None
        manager = None
    else:
        manager = worker_cm
        worker_ctx = manager.__enter__()
    try:
        for j in range(n_mc):
            seed = int(seed_start + j)
            simulation = simulate_single_field_arrivals(
                field_size=field_size_eff,
                length_scale=length_scale_eff,
                std=std_eff,
                minpadding=minpadding,
                dh=dh_eff,
                solver=solver,
                backend=backend,
                velocity_location=velocity_location,
                align_with_modflow_velocity=align_with_modflow_velocity,
                density_particle=density_particle,
                num_particles=num_particles,
                geometry=geometry,
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
                gpu_worker_ctx=worker_ctx,
            )
            tau_samples_mc = np.asarray(simulation.arrival_times, dtype=float) * velocity / i_y
            tau_samples_mc = tau_samples_mc[np.isfinite(tau_samples_mc) & (tau_samples_mc >= 0.0)]
            if tau_samples_mc.size == 0:
                continue
            seeds_used.append(seed)
            tau_by_mc.append(tau_samples_mc)
            tau_pt_first_arrival_by_mc.append(float(np.min(tau_samples_mc)))
            n_particles_by_mc.append(int(simulation.n_particles))
            n_success_by_mc.append(int(len(tau_samples_mc)))
            success_fraction_by_mc.append(float(simulation.success_fraction))
            bad_fraction_by_mc.append(float(simulation.bad_fraction))
            status_counts_by_mc.append({str(k): int(v) for k, v in simulation.status_counts.items()})
            if np.isfinite(simulation.t_fmm_fp):
                t_fmm_fp_by_mc.append(float(simulation.t_fmm_fp))
            if np.isfinite(simulation.t_fmm_lrp):
                t_fmm_lrp_by_mc.append(float(simulation.t_fmm_lrp))
            tau_fmm_fp_target_mc = np.asarray(simulation.fmm_fp_target_times, dtype=float) * velocity / i_y
            tau_fmm_fp_target_mc = tau_fmm_fp_target_mc[np.isfinite(tau_fmm_fp_target_mc) & (tau_fmm_fp_target_mc >= 0.0)]
            if tau_fmm_fp_target_mc.size > 0:
                tau_fmm_fp_target_by_mc.append(tau_fmm_fp_target_mc)
            tau_fmm_lrp_target_mc = np.asarray(simulation.fmm_lrp_target_times, dtype=float) * velocity / i_y
            tau_fmm_lrp_target_mc = tau_fmm_lrp_target_mc[np.isfinite(tau_fmm_lrp_target_mc) & (tau_fmm_lrp_target_mc >= 0.0)]
            if tau_fmm_lrp_target_mc.size > 0:
                tau_fmm_lrp_target_by_mc.append(tau_fmm_lrp_target_mc)
    finally:
        if manager is not None:
            manager.__exit__(None, None, None)

    if not tau_by_mc:
        raise RuntimeError("No valid successful particle arrivals across all Monte Carlo realizations.")

    tau_samples = np.concatenate(tau_by_mc)
    if tau_samples.size == 0:
        raise RuntimeError("No finite non-negative dimensionless arrival times available for CDF analysis.")

    tau_emp, ecdf = _empirical_cdf(tau_samples)
    tau_pt_first_arr = np.asarray(tau_pt_first_arrival_by_mc, dtype=float)
    tau_pt_first_emp, ecdf_pt_first = _empirical_cdf(tau_pt_first_arr)
    tau_fmm_fp_arr = np.array(t_fmm_fp_by_mc, dtype=float) * velocity / i_y
    tau_fmm_lrp_arr = np.array(t_fmm_lrp_by_mc, dtype=float) * velocity / i_y
    tau_fmm_fp_emp, ecdf_fmm_fp = _empirical_cdf(tau_fmm_fp_arr)
    tau_fmm_lrp_emp, ecdf_fmm_lrp = _empirical_cdf(tau_fmm_lrp_arr)
    tau_fmm_fp_target = (
        np.concatenate(tau_fmm_fp_target_by_mc) if tau_fmm_fp_target_by_mc else np.asarray([], dtype=float)
    )
    tau_fmm_lrp_target = (
        np.concatenate(tau_fmm_lrp_target_by_mc) if tau_fmm_lrp_target_by_mc else np.asarray([], dtype=float)
    )
    tau_fmm_fp_target_emp, ecdf_fmm_fp_target = _empirical_cdf(tau_fmm_fp_target)
    tau_fmm_lrp_target_emp, ecdf_fmm_lrp_target = _empirical_cdf(tau_fmm_lrp_target)
    tau_lo = max(0.0, float(tau_emp[0]) * 0.8)
    tau_hi = float(tau_emp[-1]) * 1.2 + 1e-12
    tau_grid = _build_tau_grid(
        tau_lo,
        tau_hi,
        refine_tau_grid=refine_tau_grid,
        tau_split=tau_split,
        tau_grid_fine=tau_grid_fine,
        tau_grid_coarse=tau_grid_coarse,
    )
    analytical_cdf = _analytical_cdf_on_tau_grid(
        tau_grid,
        length=length,
        velocity=velocity,
        i_y=i_y,
        sigma_y_sq=var_eff,
    )
    cdf_lognormal, lognormal_summary = _lognormal_cdf_on_tau_grid(
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
    pdf_bin_edges = _build_pdf_hist_bin_edges(tau_samples, pdf_hist_bins=pdf_hist_bins)
    tau_pdf_centers, empirical_pdf = _histogram_pdf(tau_samples, pdf_bin_edges)
    _tau_mc_centers, mc_pdf_mean, mc_pdf_band_low, mc_pdf_band_high = _mc_pdf_band(
        tau_by_mc,
        pdf_bin_edges,
        band_low=band_low,
        band_high=band_high,
    )
    analytical_pdf = _analytical_pdf_on_tau_centers(
        tau_pdf_centers,
        length=length,
        velocity=velocity,
        i_y=i_y,
        sigma_y_sq=var_eff,
    )
    lognormal_pdf = _lognormal_pdf_on_tau_centers(
        tau_pdf_centers,
        length=length,
        velocity=velocity,
        i_y=i_y,
        sigma_y_sq=var_eff,
    )

    summary = {
        "run_id": cfg.run_id,
        "field_size": int(field_size_eff),
        "length_scale": float(length_scale_eff),
        "std": float(std_eff),
        "var": float(var_eff),
        "dh": float(dh_eff),
        "k_g": float(k_g_eff),
        "velocity": float(velocity),
        "i_y": float(i_y),
        "length": float(length),
        "band_low": float(band_low),
        "band_high": float(band_high),
        "pdf_hist_bins": int(pdf_hist_bins),
        "pdf_hist_range_low": float(pdf_bin_edges[0]),
        "pdf_hist_range_high": float(pdf_bin_edges[-1]),
        "pt_solver": str(solver),
        "pt_backend": str(backend),
        "pt_velocity_location": str(velocity_location),
        "pt_align_with_modflow_velocity": bool(align_with_modflow_velocity),
        "pt_num_particles": int(
            _resolve_particle_count(
                num_particles=num_particles,
                density_particle=density_particle,
                source_y_start=geometry.source_y_start,
                source_y_end=geometry.source_y_end,
            )
        ),
        "source_x": int(geometry.source_x),
        "target_x": int(geometry.target_x),
        "source_y_start": int(geometry.source_y_start),
        "source_y_end": int(geometry.source_y_end),
        "target_y_start": int(geometry.target_y_start),
        "target_y_end": int(geometry.target_y_end),
        "seed_start": int(seed_start),
        "n_mc_requested": int(n_mc),
        "n_mc_used": int(len(tau_by_mc)),
        "seeds_used": [int(s) for s in seeds_used],
        "n_particles": int(np.sum(np.asarray(n_particles_by_mc, dtype=int))),
        "n_success": int(len(tau_samples)),
        "n_particles_by_mc": n_particles_by_mc,
        "n_success_by_mc": n_success_by_mc,
        "n_mc_pt_first_arrival": int(tau_pt_first_arr.size),
        "n_fmm_fp_target_samples": int(tau_fmm_fp_target.size),
        "n_fmm_lrp_target_samples": int(tau_fmm_lrp_target.size),
        "success_fraction_by_mc": success_fraction_by_mc,
        "bad_fraction_by_mc": bad_fraction_by_mc,
        "status_counts_by_mc": status_counts_by_mc,
        "tau_mean": float(np.mean(tau_samples)),
        "tau_median": float(np.median(tau_samples)),
        "tau_min": float(np.min(tau_samples)),
        "tau_max": float(np.max(tau_samples)),
        "tau_pt_first_arrival_mean": float(np.mean(tau_pt_first_arr)),
        "tau_pt_first_arrival_median": float(np.median(tau_pt_first_arr)),
        "ks_distance_empirical_vs_analytical": float(ks_dist),
        **lognormal_summary,
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
        cdf_lognormal=cdf_lognormal,
        tau_pt_first_arrival_emp=tau_pt_first_emp if tau_pt_first_emp.size > 0 else None,
        ecdf_pt_first_arrival=ecdf_pt_first if ecdf_pt_first.size > 0 else None,
        tau_fmm_fp_emp=tau_fmm_fp_emp if tau_fmm_fp_emp.size > 0 else None,
        ecdf_fmm_fp=ecdf_fmm_fp if ecdf_fmm_fp.size > 0 else None,
        tau_fmm_lrp_emp=tau_fmm_lrp_emp if tau_fmm_lrp_emp.size > 0 else None,
        ecdf_fmm_lrp=ecdf_fmm_lrp if ecdf_fmm_lrp.size > 0 else None,
        tau_fmm_fp_target_emp=tau_fmm_fp_target_emp if tau_fmm_fp_target_emp.size > 0 else None,
        ecdf_fmm_fp_target=ecdf_fmm_fp_target if ecdf_fmm_fp_target.size > 0 else None,
        tau_fmm_lrp_target_emp=tau_fmm_lrp_target_emp if tau_fmm_lrp_target_emp.size > 0 else None,
        ecdf_fmm_lrp_target=ecdf_fmm_lrp_target if ecdf_fmm_lrp_target.size > 0 else None,
        y_exp=y_exp,
    )

    fig_path = out_dir / f"{stem}_arrival_cdf_vs_analytical.png"
    pdf_fig_path = out_dir / f"{stem}_arrival_pdf_vs_analytical.png"
    npz_path = out_dir / f"{stem}_arrival_times.npz"
    json_path = out_dir / f"{stem}_arrival_summary.json"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    pdf_fig_path_out: Path | None = None
    if save_pdf_plot:
        pdf_fig = plot_arrival_pdf_comparison(
            tau_pdf_centers,
            empirical_pdf,
            analytical_pdf,
            title=title.replace("CDF", "PDF"),
            bin_edges=pdf_bin_edges,
            mc_pdf_mean=mc_pdf_mean,
            mc_pdf_band_low=mc_pdf_band_low,
            mc_pdf_band_high=mc_pdf_band_high,
            lognormal_pdf=lognormal_pdf,
        )
        pdf_fig.savefig(pdf_fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(pdf_fig)
        pdf_fig_path_out = pdf_fig_path

    npz_path_out: Path | None = None
    if save_npz:
        np.savez(
            npz_path,
            tau_samples=tau_samples,
            tau_empirical=tau_emp,
            cdf_empirical=ecdf,
            tau_grid=tau_grid,
            cdf_analytical=analytical_cdf,
            cdf_lognormal=cdf_lognormal,
            cdf_mc_mean=cdf_mc_mean,
            cdf_mc_band_low=cdf_mc_band_low,
            cdf_mc_band_high=cdf_mc_band_high,
            length=np.asarray(float(length)),
            velocity=np.asarray(float(velocity)),
            i_y=np.asarray(float(i_y)),
            var=np.asarray(float(var_eff)),
            band_low=np.asarray(float(band_low)),
            band_high=np.asarray(float(band_high)),
            pdf_hist_bins=np.asarray(int(pdf_hist_bins)),
            pdf_hist_range_low=np.asarray(float(pdf_bin_edges[0])),
            pdf_hist_range_high=np.asarray(float(pdf_bin_edges[-1])),
            pdf_hist_bin_edges=np.asarray(pdf_bin_edges, dtype=float),
            pt_solver=np.asarray(str(solver)),
            pt_backend=np.asarray(str(backend)),
            pt_velocity_location=np.asarray(str(velocity_location)),
            pt_align_with_modflow_velocity=np.asarray(bool(align_with_modflow_velocity)),
            pt_num_particles=np.asarray(
                int(
                    _resolve_particle_count(
                        num_particles=num_particles,
                        density_particle=density_particle,
                        source_y_start=geometry.source_y_start,
                        source_y_end=geometry.source_y_end,
                    )
                )
            ),
            source_x=np.asarray(int(geometry.source_x)),
            target_x=np.asarray(int(geometry.target_x)),
            source_y_start=np.asarray(int(geometry.source_y_start)),
            source_y_end=np.asarray(int(geometry.source_y_end)),
            target_y_start=np.asarray(int(geometry.target_y_start)),
            target_y_end=np.asarray(int(geometry.target_y_end)),
            seeds_used=np.asarray(seeds_used, dtype=np.int32),
            n_success_by_mc=np.asarray(n_success_by_mc, dtype=np.int32),
            tau_samples_by_mc=np.asarray(tau_by_mc, dtype=object),
            tau_pt_first_arrival_empirical=np.asarray(tau_pt_first_emp, dtype=float),
            cdf_pt_first_arrival=np.asarray(ecdf_pt_first, dtype=float),
            tau_fmm_fp_empirical=np.asarray(tau_fmm_fp_emp, dtype=float),
            cdf_fmm_fp=np.asarray(ecdf_fmm_fp, dtype=float),
            tau_fmm_lrp_empirical=np.asarray(tau_fmm_lrp_emp, dtype=float),
            cdf_fmm_lrp=np.asarray(ecdf_fmm_lrp, dtype=float),
            tau_fmm_fp_target_empirical=np.asarray(tau_fmm_fp_target_emp, dtype=float),
            cdf_fmm_fp_target=np.asarray(ecdf_fmm_fp_target, dtype=float),
            tau_fmm_lrp_target_empirical=np.asarray(tau_fmm_lrp_target_emp, dtype=float),
            cdf_fmm_lrp_target=np.asarray(ecdf_fmm_lrp_target, dtype=float),
        )
        npz_path_out = npz_path

    json_path_out: Path | None = None
    if save_json:
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        json_path_out = json_path

    return fig_path, pdf_fig_path_out, npz_path_out, json_path_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 6: single-field arrival-time distribution of all successful particles vs analytical _g_tau CDF"
    )
    parser.add_argument("--config", required=False, type=Path)
    parser.add_argument("--load-npz", type=Path, default=None, help="Reload a saved NPZ artifact and regenerate the figure.")
    parser.add_argument("--field-size", type=int, default=None, help="Grid size, default: cfg.field_size_ref")
    parser.add_argument("--length-scale", type=float, default=None, help="Normalized length scale, default: cfg.length_scale")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--std", type=float, default=None, help="GRF standard deviation")
    group.add_argument("--var", type=float, default=None, help="GRF variance")
    parser.add_argument("--minpadding", type=int, default=0, help="Minimal padding for GRF generation")
    parser.add_argument("--dh", type=float, default=None, help="Hydraulic head x-gradient; default cfg.dh")
    parser.add_argument("--k-g", type=float, default=None, help="Reference conductivity scale; default cfg.k_g")
    parser.add_argument("--solver", choices=["pollock", "rk45"], default="pollock")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--velocity-location", choices=["face", "cell"], default="face")
    parser.add_argument(
        "--align-with-modflow-velocity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to align face velocities with MODFLOW convention for particle tracking.",
    )
    parser.add_argument("--density-particle", type=int, default=20)
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--source-x", type=int, default=None, help="Source-line x index; default uses the current inset location.")
    parser.add_argument("--target-x", type=int, default=None, help="Target-line x index; default uses the current inset location.")
    parser.add_argument("--source-y-start", type=int, default=None, help="Inclusive source-line start y index; default uses the current trimmed band.")
    parser.add_argument("--source-y-end", type=int, default=None, help="Exclusive source-line end y index; default uses the current trimmed band.")
    parser.add_argument("--target-y-start", type=int, default=None, help="Inclusive target-line start y index; saved as metadata only; default uses the current outlet span.")
    parser.add_argument("--target-y-end", type=int, default=None, help="Exclusive target-line end y index; saved as metadata only; default uses the current outlet span.")
    parser.add_argument("--n-mc", type=int, default=10, help="Number of Monte Carlo realizations")
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed for MC sequence (seed_start + i)")
    parser.add_argument("--band-low", type=float, default=5.0, help="Lower percentile for MC CDF band")
    parser.add_argument("--band-high", type=float, default=95.0, help="Upper percentile for MC CDF band")
    parser.add_argument(
        "--refine-tau-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use denser tau-grid sampling below --tau-split.",
    )
    parser.add_argument("--tau-split", type=float, default=10.0, help="Use denser tau-grid sampling below this value.")
    parser.add_argument("--tau-grid-fine", type=int, default=1024, help="Number of tau-grid points used below --tau-split.")
    parser.add_argument("--tau-grid-coarse", type=int, default=256, help="Number of tau-grid points used at and above --tau-split.")
    parser.add_argument("--y-exp", type=float, default=1.0, help="Exponent for optional y-axis function scaling; 1.0 keeps a linear y-axis.")
    parser.add_argument("--pdf-hist-bins", type=int, default=48, help="Number of bins used when estimating PDF histograms.")
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
        "--save-pdf-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save an extra probability-density figure (default: disabled).",
    )
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
    args = parser.parse_args()
    if args.load_npz is None:
        if args.config is None:
            parser.error("--config is required unless --load-npz is used.")
        if args.std is None and args.var is None:
            parser.error("One of --std or --var is required unless --load-npz is used.")
        _validate_particle_tracking_options(
            solver=args.solver,
            velocity_location=args.velocity_location,
            backend=args.backend,
        )
    return args


def main() -> int:
    args = parse_args()
    if args.load_npz is not None:
        fig_path, pdf_fig_path = visualize_saved_arrival_npz(
            args.load_npz,
            output_dir=args.output_dir,
            figure_dpi=args.figure_dpi,
            refine_tau_grid=args.refine_tau_grid,
            tau_split=args.tau_split,
            tau_grid_fine=args.tau_grid_fine,
            tau_grid_coarse=args.tau_grid_coarse,
            y_exp=args.y_exp,
            save_pdf_plot=args.save_pdf_plot,
        )
        print(f"Loaded data: {args.load_npz}")
        print(f"Saved figure: {fig_path}")
        if pdf_fig_path is not None:
            print(f"Saved PDF plot: {pdf_fig_path}")
        return 0

    cfg = step1.load_config(args.config)
    seed_start = int(args.seed_start)
    if args.seed is not None:
        if args.seed_start != 0 and int(args.seed) != int(args.seed_start):
            raise ValueError(f"Conflicting --seed ({args.seed}) and --seed-start ({args.seed_start}).")
        seed_start = int(args.seed)
    fig_path, pdf_fig_path, npz_path, json_path = run_step6(
        cfg,
        field_size=args.field_size,
        length_scale=args.length_scale,
        std=args.std,
        var=args.var,
        minpadding=args.minpadding,
        dh=args.dh,
        k_g=args.k_g,
        solver=args.solver,
        backend=args.backend,
        velocity_location=args.velocity_location,
        align_with_modflow_velocity=args.align_with_modflow_velocity,
        density_particle=args.density_particle,
        num_particles=args.num_particles,
        source_x=args.source_x,
        target_x=args.target_x,
        source_y_start=args.source_y_start,
        source_y_end=args.source_y_end,
        target_y_start=args.target_y_start,
        target_y_end=args.target_y_end,
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
        refine_tau_grid=args.refine_tau_grid,
        tau_split=args.tau_split,
        tau_grid_fine=args.tau_grid_fine,
        tau_grid_coarse=args.tau_grid_coarse,
        y_exp=args.y_exp,
        pdf_hist_bins=args.pdf_hist_bins,
        output_dir=args.output_dir,
        figure_dpi=args.figure_dpi,
        save_pdf_plot=args.save_pdf_plot,
        save_npz=args.save_npz,
        save_json=args.save_json,
    )
    print(f"Saved figure: {fig_path}")
    if pdf_fig_path is not None:
        print(f"Saved PDF plot: {pdf_fig_path}")
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
