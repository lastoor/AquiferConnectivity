#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from step1_index import VisualizationConfig, default_cache_dir, load_config, save_pickle, write_stage_metadata


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except AttributeError as exc:
            # Legacy compatibility: stage2 may have been produced by running
            # `python scripts/step2_metrics.py`, which stores dataclass types
            # under __main__.* in the pickle stream.
            if "__main__" not in str(exc):
                raise
            import __main__
            import step2_metrics as step2

            setattr(__main__, "VarianceResults", step2.VarianceResults)
            setattr(__main__, "StdResults", step2.StdResults)
            setattr(__main__, "SummaryStats", step2.SummaryStats)
            f.seek(0)
            return pickle.load(f)


def _pick(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _mean_std_pair(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        if a.shape[0] == 0:
            return np.asarray([np.nan, np.nan], dtype=float)
        return a[0, :]
    if a.ndim == 1:
        if a.size >= 2:
            return a[:2]
        return np.asarray([np.nan, np.nan], dtype=float)
    return np.asarray([np.nan, np.nan], dtype=float)


def _available_var_values(stats: dict[str, dict[str, Any]], configured: list[float]) -> list[float]:
    out: list[float] = []
    for var_ in configured:
        if f"var_{var_}" in stats:
            out.append(var_)
    return out


def _infer_std_count(stats_std: dict[str, np.ndarray] | None) -> int | None:
    if stats_std is None:
        return None
    for metric_data in stats_std.values():
        for arr in metric_data.values():
            a = np.asarray(arr)
            if a.ndim >= 1:
                return int(a.shape[0])
    return 0


def collect_primary_stats(stats: dict[str, dict[str, Any]], var_values: list[float]) -> dict[str, np.ndarray]:
    stats_lrp = []
    stats_fp = []
    stats_fp_dijkstra = []
    stats_fp_fmm = []
    stats_mhr = []
    stats_fpt = []
    stats_fpt_dijkstra = []
    stats_fpt_fmm = []
    stats_hausdorff_lrp = []
    stats_hausdorff_dijkstra = []
    stats_hausdorff_fmm = []
    stats_endpoint_lrp = []
    stats_endpoint_dijkstra = []
    stats_endpoint_fmm = []

    for var_ in var_values:
        key = f"var_{var_}"
        entry = stats[key]
        stats_lrp.append(_mean_std_pair(entry["lengths"]["lrp"]))
        stats_fp.append(_mean_std_pair(entry["lengths"]["fp"]))
        stats_fp_dijkstra.append(_mean_std_pair(entry["lengths"]["fp_dijkstra"]))
        stats_fp_fmm.append(_mean_std_pair(entry["lengths"]["fp_fmm"]))
        stats_mhr.append(_mean_std_pair(entry["times"]["mhr"]))
        stats_fpt.append(_mean_std_pair(entry["times"]["fpt"]))
        stats_fpt_dijkstra.append(_mean_std_pair(entry["times"]["fpt_dijkstra"]))
        stats_fpt_fmm.append(_mean_std_pair(entry["times"]["fpt_fmm"]))
        stats_hausdorff_lrp.append(_mean_std_pair(entry["distances"]["hausdorff"]["fp_lrp"]))
        stats_hausdorff_dijkstra.append(_mean_std_pair(entry["distances"]["hausdorff"]["fp_fpDijkstra"]))
        stats_hausdorff_fmm.append(_mean_std_pair(entry["distances"]["hausdorff"]["fp_fpFMM"]))
        stats_endpoint_lrp.append(_mean_std_pair(entry["distances"]["end_point"]["fp_lrp"]))
        stats_endpoint_dijkstra.append(_mean_std_pair(entry["distances"]["end_point"]["fp_fpDijkstra"]))
        stats_endpoint_fmm.append(_mean_std_pair(entry["distances"]["end_point"]["fp_fpFMM"]))

    return {
        "stats_lrp": np.asarray(stats_lrp, dtype=float),
        "stats_fp": np.asarray(stats_fp, dtype=float),
        "stats_fp_dijkstra": np.asarray(stats_fp_dijkstra, dtype=float),
        "stats_fp_fmm": np.asarray(stats_fp_fmm, dtype=float),
        "stats_mhr": np.asarray(stats_mhr, dtype=float),
        "stats_fpt": np.asarray(stats_fpt, dtype=float),
        "stats_fpt_dijkstra": np.asarray(stats_fpt_dijkstra, dtype=float),
        "stats_fpt_fmm": np.asarray(stats_fpt_fmm, dtype=float),
        "stats_hausdorff_lrp": np.asarray(stats_hausdorff_lrp, dtype=float),
        "stats_hausdorff_dijkstra": np.asarray(stats_hausdorff_dijkstra, dtype=float),
        "stats_hausdorff_fmm": np.asarray(stats_hausdorff_fmm, dtype=float),
        "stats_endpoint_lrp": np.asarray(stats_endpoint_lrp, dtype=float),
        "stats_endpoint_dijkstra": np.asarray(stats_endpoint_dijkstra, dtype=float),
        "stats_endpoint_fmm": np.asarray(stats_endpoint_fmm, dtype=float),
    }


def collect_integral_stats(line_integrals: dict[str, dict[str, Any]], var_values: list[float]) -> dict[str, np.ndarray]:
    out = {
        "int_fp": [],
        "int2_fp": [],
        "int_lrp": [],
        "int2_lrp": [],
        "int_fp_dijkstra": [],
        "int2_fp_dijkstra": [],
        "int_fp_fmm": [],
        "int2_fp_fmm": [],
    }
    for var_ in var_values:
        li = line_integrals[f"var_{var_}"]
        out["int_fp"].append([li["int_logk_fp"]["mean"], li["int_logk_fp"]["std"]])
        out["int2_fp"].append([li["int_logk2_fp"]["mean"], li["int_logk2_fp"]["std"]])
        out["int_lrp"].append([li["int_logk_lrp"]["mean"], li["int_logk_lrp"]["std"]])
        out["int2_lrp"].append([li["int_logk2_lrp"]["mean"], li["int_logk2_lrp"]["std"]])
        out["int_fp_dijkstra"].append([li["int_logk_fp_dijkstra"]["mean"], li["int_logk_fp_dijkstra"]["std"]])
        out["int2_fp_dijkstra"].append([li["int_logk2_fp_dijkstra"]["mean"], li["int_logk2_fp_dijkstra"]["std"]])
        out["int_fp_fmm"].append([li["int_logk_fp_fmm"]["mean"], li["int_logk_fp_fmm"]["std"]])
        out["int2_fp_fmm"].append([li["int_logk2_fp_fmm"]["mean"], li["int_logk2_fp_fmm"]["std"]])

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def compute_t_ref(k_g: float, dh: float, field_size: float, length_scale: float) -> float:
    i_y = length_scale * field_size
    q_ref = k_g * dh
    return i_y / q_ref


def compute_i_y(field_size: float, length_scale: float) -> float:
    return length_scale * field_size


def compute_taylor_times(
    stats_fp: np.ndarray,
    stats_lrp: np.ndarray,
    stats_fp_dijkstra: np.ndarray,
    stats_fp_fmm: np.ndarray,
    int_fp: np.ndarray,
    int_lrp: np.ndarray,
    int_fp_dijkstra: np.ndarray,
    int_fp_fmm: np.ndarray,
    int2_fp: np.ndarray,
    int2_lrp: np.ndarray,
    int2_fp_dijkstra: np.ndarray,
    int2_fp_fmm: np.ndarray,
    dh: float,
) -> dict[str, np.ndarray]:
    taylor_lrp = stats_lrp[:, 0] - int_lrp[:, 0] + int2_lrp[:, 0] / 2.0
    taylor_fp = (stats_fp[:, 0] - int_fp[:, 0] + int2_fp[:, 0] / 2.0) / dh
    taylor_fp_dijkstra = (stats_fp_dijkstra[:, 0] - int_fp_dijkstra[:, 0] + int2_fp_dijkstra[:, 0] / 2.0) / dh
    taylor_fp_fmm = (stats_fp_fmm[:, 0] - int_fp_fmm[:, 0] + int2_fp_fmm[:, 0] / 2.0) / dh
    return {
        "lrp": taylor_lrp,
        "fp": taylor_fp,
        "fp_dijkstra": taylor_fp_dijkstra,
        "fp_fmm": taylor_fp_fmm,
    }


def prepare_plot_inputs(cfg: VisualizationConfig, var_results: Any, std_results: Any | None) -> dict[str, Any]:
    var_stats = _pick(var_results, "stats")
    var_line_integrals = _pick(var_results, "line_integrals")
    var_field_size = _pick(var_results, "field_size")

    var_values = _available_var_values(var_stats, cfg.variance_values)
    var = np.asarray(var_values, dtype=float)
    s = collect_primary_stats(var_stats, var_values)
    ints = collect_integral_stats(var_line_integrals, var_values)

    field_size_arr = np.asarray(var_field_size, dtype=float)
    field_size = cfg.field_size_ref if field_size_arr.size == 0 else float(field_size_arr[0])
    i_y = compute_i_y(field_size, cfg.length_scale)
    t_ref = compute_t_ref(cfg.k_g, cfg.dh, field_size, cfg.length_scale)
    taylor = compute_taylor_times(
        s["stats_fp"],
        s["stats_lrp"],
        s["stats_fp_dijkstra"],
        s["stats_fp_fmm"],
        ints["int_fp"],
        ints["int_lrp"],
        ints["int_fp_dijkstra"],
        ints["int_fp_fmm"],
        ints["int2_fp"],
        ints["int2_lrp"],
        ints["int2_fp_dijkstra"],
        ints["int2_fp_fmm"],
        cfg.dh,
    )

    stats_std = None if std_results is None else _pick(std_results, "stats_std")
    std_count = _infer_std_count(stats_std)
    std_values = cfg.std_values if std_count is None else cfg.std_values[:std_count]

    return {
        "var": var,
        "field_size": field_size,
        "std_x": np.asarray(std_values, dtype=float) ** 2,
        "stats": s,
        "integrals": ints,
        "taylor": taylor,
        "i_y": i_y,
        "t_ref": t_ref,
        "stats_std": stats_std,
    }


def run_stage3(cfg: VisualizationConfig, *, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or default_cache_dir(cfg)
    metrics = load_pickle(cache_dir / "stage2_metrics.pkl")
    var_results = metrics["var_results"]
    std_results = metrics["std_results"]

    plot_inputs = prepare_plot_inputs(cfg, var_results, std_results)
    payload = {
        "plot_inputs": plot_inputs,
        "warnings": metrics.get("warnings", {}),
    }
    cache_file = cache_dir / "stage3_plot_inputs.pkl"
    save_pickle(cache_file, payload)

    notes = {
        "cache_file": str(cache_file),
        "has_std_inputs": plot_inputs["stats_std"] is not None,
    }
    write_stage_metadata(cache_dir / "stage3_plot_inputs.meta.json", stage="stage3_plot_inputs", config=cfg, notes=notes)
    return cache_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: prepare plotting inputs from metric caches")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cache_file = run_stage3(cfg, cache_dir=args.cache_dir)
    print(f"Wrote plot-input cache: {cache_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
