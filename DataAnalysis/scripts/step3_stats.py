#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **kwargs):  # type: ignore[override]
        return iterable

from step1_index import RunPaths, VisualizationConfig, default_cache_dir, load_config, save_pickle, write_stage_metadata
from step2_metrics import deserialize_discovered, load_mc_consolidated


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
    stats_lrp_fmm = []
    stats_fp = []
    stats_fp_dijkstra = []
    stats_fp_fmm = []
    stats_fp_fsm = []
    stats_fp_fm_asr = []
    stats_fp_fm_asr_iso = []
    stats_mhr = []
    stats_mhr_fmm = []
    stats_fpt = []
    stats_fpt_dijkstra = []
    stats_fpt_fmm = []
    stats_fpt_fsm = []
    stats_fpt_fm_asr = []
    stats_fpt_fm_asr_iso = []
    stats_ct_lrp_dijkstra = []
    stats_ct_lrp_fmm = []
    stats_ct_fp_dijkstra = []
    stats_ct_fp_fmm = []
    stats_ct_fp_fsm = []
    stats_ct_fp = []
    stats_ct_fp_fm_asr = []
    stats_ct_fp_fm_asr_iso = []
    stats_hausdorff_lrp = []
    stats_hausdorff_lrp_fmm = []
    stats_hausdorff_dijkstra = []
    stats_hausdorff_fmm = []
    stats_hausdorff_fsm = []
    stats_hausdorff_fm_asr = []
    stats_hausdorff_fm_asr_iso = []
    stats_frechet_lrp = []
    stats_frechet_lrp_fmm = []
    stats_frechet_dijkstra = []
    stats_frechet_fmm = []
    stats_frechet_fsm = []
    stats_frechet_fm_asr = []
    stats_frechet_fm_asr_iso = []
    stats_endpoint_lrp = []
    stats_endpoint_lrp_fmm = []
    stats_endpoint_dijkstra = []
    stats_endpoint_fmm = []
    stats_endpoint_fsm = []
    stats_endpoint_fm_asr = []
    stats_endpoint_fm_asr_iso = []

    for var_ in var_values:
        key = f"var_{var_}"
        entry = stats[key]
        stats_lrp.append(_mean_std_pair(entry["lengths"]["lrp"]))
        stats_lrp_fmm.append(_mean_std_pair(entry["lengths"]["lrp_fmm"]))
        stats_fp.append(_mean_std_pair(entry["lengths"]["fp"]))
        stats_fp_dijkstra.append(_mean_std_pair(entry["lengths"]["fp_dijkstra"]))
        stats_fp_fmm.append(_mean_std_pair(entry["lengths"]["fp_fmm"]))
        stats_fp_fsm.append(_mean_std_pair(entry["lengths"].get("fp_fsm", [np.nan, np.nan])))
        stats_fp_fm_asr.append(_mean_std_pair(entry["lengths"].get("fp_fm_asr", [np.nan, np.nan])))
        stats_fp_fm_asr_iso.append(_mean_std_pair(entry["lengths"].get("fp_fm_asr_iso", [np.nan, np.nan])))
        stats_mhr.append(_mean_std_pair(entry["times"]["mhr"]))
        stats_mhr_fmm.append(_mean_std_pair(entry["times"]["mhr_fmm"]))
        stats_fpt.append(_mean_std_pair(entry["times"]["fpt"]))
        stats_fpt_dijkstra.append(_mean_std_pair(entry["times"]["fpt_dijkstra"]))
        stats_fpt_fmm.append(_mean_std_pair(entry["times"]["fpt_fmm"]))
        stats_fpt_fsm.append(_mean_std_pair(entry["times"].get("fpt_fsm", [np.nan, np.nan])))
        stats_fpt_fm_asr.append(_mean_std_pair(entry["times"].get("fpt_fm_asr", [np.nan, np.nan])))
        stats_fpt_fm_asr_iso.append(_mean_std_pair(entry["times"].get("fpt_fm_asr_iso", [np.nan, np.nan])))
        ct = entry.get("comp_times", {})
        stats_ct_lrp_dijkstra.append(_mean_std_pair(ct.get("ct_lrp_dijkstra", [np.nan, np.nan])))
        stats_ct_lrp_fmm.append(_mean_std_pair(ct.get("ct_lrp_fmm", [np.nan, np.nan])))
        stats_ct_fp_dijkstra.append(_mean_std_pair(ct.get("ct_fp_dijkstra", [np.nan, np.nan])))
        stats_ct_fp_fmm.append(_mean_std_pair(ct.get("ct_fp_fmm", [np.nan, np.nan])))
        stats_ct_fp_fsm.append(_mean_std_pair(ct.get("ct_fp_fsm", [np.nan, np.nan])))
        stats_ct_fp.append(_mean_std_pair(ct.get("ct_fp", [np.nan, np.nan])))
        stats_ct_fp_fm_asr.append(_mean_std_pair(ct.get("ct_fp_fm_asr", [np.nan, np.nan])))
        stats_ct_fp_fm_asr_iso.append(_mean_std_pair(ct.get("ct_fp_fm_asr_iso", [np.nan, np.nan])))
        stats_hausdorff_lrp.append(_mean_std_pair(entry["distances"]["hausdorff"]["fp_lrp"]))
        stats_hausdorff_lrp_fmm.append(_mean_std_pair(entry["distances"]["hausdorff"]["fp_lrpFMM"]))
        stats_hausdorff_dijkstra.append(_mean_std_pair(entry["distances"]["hausdorff"]["fp_fpDijkstra"]))
        stats_hausdorff_fmm.append(_mean_std_pair(entry["distances"]["hausdorff"]["fp_fpFMM"]))
        stats_hausdorff_fsm.append(_mean_std_pair(entry["distances"]["hausdorff"].get("fp_fpFSM", [np.nan, np.nan])))
        stats_hausdorff_fm_asr.append(_mean_std_pair(entry["distances"]["hausdorff"].get("fp_fm_asr", [np.nan, np.nan])))
        stats_hausdorff_fm_asr_iso.append(_mean_std_pair(entry["distances"]["hausdorff"].get("fp_fm_asr_iso", [np.nan, np.nan])))
        stats_frechet_lrp.append(_mean_std_pair(entry["distances"]["frechet"]["fp_lrp"]))
        stats_frechet_lrp_fmm.append(_mean_std_pair(entry["distances"]["frechet"]["fp_lrpFMM"]))
        stats_frechet_dijkstra.append(_mean_std_pair(entry["distances"]["frechet"]["fp_fpDijkstra"]))
        stats_frechet_fmm.append(_mean_std_pair(entry["distances"]["frechet"]["fp_fpFMM"]))
        stats_frechet_fsm.append(_mean_std_pair(entry["distances"]["frechet"].get("fp_fpFSM", [np.nan, np.nan])))
        stats_frechet_fm_asr.append(_mean_std_pair(entry["distances"]["frechet"].get("fp_fm_asr", [np.nan, np.nan])))
        stats_frechet_fm_asr_iso.append(_mean_std_pair(entry["distances"]["frechet"].get("fp_fm_asr_iso", [np.nan, np.nan])))
        stats_endpoint_lrp.append(_mean_std_pair(entry["distances"]["end_point"]["fp_lrp"]))
        stats_endpoint_lrp_fmm.append(_mean_std_pair(entry["distances"]["end_point"]["fp_lrpFMM"]))
        stats_endpoint_dijkstra.append(_mean_std_pair(entry["distances"]["end_point"]["fp_fpDijkstra"]))
        stats_endpoint_fmm.append(_mean_std_pair(entry["distances"]["end_point"]["fp_fpFMM"]))
        stats_endpoint_fsm.append(_mean_std_pair(entry["distances"]["end_point"].get("fp_fpFSM", [np.nan, np.nan])))
        stats_endpoint_fm_asr.append(_mean_std_pair(entry["distances"]["end_point"].get("fp_fm_asr", [np.nan, np.nan])))
        stats_endpoint_fm_asr_iso.append(_mean_std_pair(entry["distances"]["end_point"].get("fp_fm_asr_iso", [np.nan, np.nan])))

    return {
        "stats_lrp": np.asarray(stats_lrp, dtype=float),
        "stats_lrp_fmm": np.asarray(stats_lrp_fmm, dtype=float),
        "stats_fp": np.asarray(stats_fp, dtype=float),
        "stats_fp_dijkstra": np.asarray(stats_fp_dijkstra, dtype=float),
        "stats_fp_fmm": np.asarray(stats_fp_fmm, dtype=float),
        "stats_fp_fsm": np.asarray(stats_fp_fsm, dtype=float),
        "stats_fp_fm_asr": np.asarray(stats_fp_fm_asr, dtype=float),
        "stats_fp_fm_asr_iso": np.asarray(stats_fp_fm_asr_iso, dtype=float),
        "stats_mhr": np.asarray(stats_mhr, dtype=float),
        "stats_mhr_fmm": np.asarray(stats_mhr_fmm, dtype=float),
        "stats_fpt": np.asarray(stats_fpt, dtype=float),
        "stats_fpt_dijkstra": np.asarray(stats_fpt_dijkstra, dtype=float),
        "stats_fpt_fmm": np.asarray(stats_fpt_fmm, dtype=float),
        "stats_fpt_fsm": np.asarray(stats_fpt_fsm, dtype=float),
        "stats_fpt_fm_asr": np.asarray(stats_fpt_fm_asr, dtype=float),
        "stats_fpt_fm_asr_iso": np.asarray(stats_fpt_fm_asr_iso, dtype=float),
        "stats_hausdorff_lrp": np.asarray(stats_hausdorff_lrp, dtype=float),
        "stats_hausdorff_lrp_fmm": np.asarray(stats_hausdorff_lrp_fmm, dtype=float),
        "stats_hausdorff_dijkstra": np.asarray(stats_hausdorff_dijkstra, dtype=float),
        "stats_hausdorff_fmm": np.asarray(stats_hausdorff_fmm, dtype=float),
        "stats_hausdorff_fsm": np.asarray(stats_hausdorff_fsm, dtype=float),
        "stats_hausdorff_fm_asr": np.asarray(stats_hausdorff_fm_asr, dtype=float),
        "stats_hausdorff_fm_asr_iso": np.asarray(stats_hausdorff_fm_asr_iso, dtype=float),
        "stats_frechet_lrp": np.asarray(stats_frechet_lrp, dtype=float),
        "stats_frechet_lrp_fmm": np.asarray(stats_frechet_lrp_fmm, dtype=float),
        "stats_frechet_dijkstra": np.asarray(stats_frechet_dijkstra, dtype=float),
        "stats_frechet_fmm": np.asarray(stats_frechet_fmm, dtype=float),
        "stats_frechet_fsm": np.asarray(stats_frechet_fsm, dtype=float),
        "stats_frechet_fm_asr": np.asarray(stats_frechet_fm_asr, dtype=float),
        "stats_frechet_fm_asr_iso": np.asarray(stats_frechet_fm_asr_iso, dtype=float),
        "stats_endpoint_lrp": np.asarray(stats_endpoint_lrp, dtype=float),
        "stats_endpoint_lrp_fmm": np.asarray(stats_endpoint_lrp_fmm, dtype=float),
        "stats_endpoint_dijkstra": np.asarray(stats_endpoint_dijkstra, dtype=float),
        "stats_endpoint_fmm": np.asarray(stats_endpoint_fmm, dtype=float),
        "stats_endpoint_fsm": np.asarray(stats_endpoint_fsm, dtype=float),
        "stats_endpoint_fm_asr": np.asarray(stats_endpoint_fm_asr, dtype=float),
        "stats_endpoint_fm_asr_iso": np.asarray(stats_endpoint_fm_asr_iso, dtype=float),
        "stats_ct_lrp_dijkstra": np.asarray(stats_ct_lrp_dijkstra, dtype=float),
        "stats_ct_lrp_fmm": np.asarray(stats_ct_lrp_fmm, dtype=float),
        "stats_ct_fp_dijkstra": np.asarray(stats_ct_fp_dijkstra, dtype=float),
        "stats_ct_fp_fmm": np.asarray(stats_ct_fp_fmm, dtype=float),
        "stats_ct_fp_fsm": np.asarray(stats_ct_fp_fsm, dtype=float),
        "stats_ct_fp": np.asarray(stats_ct_fp, dtype=float),
        "stats_ct_fp_fm_asr": np.asarray(stats_ct_fp_fm_asr, dtype=float),
        "stats_ct_fp_fm_asr_iso": np.asarray(stats_ct_fp_fm_asr_iso, dtype=float),
    }


def collect_integral_stats(line_integrals: dict[str, dict[str, Any]], var_values: list[float]) -> dict[str, np.ndarray]:
    out = {
        "int_fp": [],
        "int2_fp": [],
        "int_lrp": [],
        "int2_lrp": [],
        "int_lrp_fmm": [],
        "int2_lrp_fmm": [],
        "int_fp_dijkstra": [],
        "int2_fp_dijkstra": [],
        "int_fp_fmm": [],
        "int2_fp_fmm": [],
        "int_fp_fsm": [],
        "int2_fp_fsm": [],
        "int_fp_fm_asr": [],
        "int2_fp_fm_asr": [],
        "int_fp_fm_asr_iso": [],
        "int2_fp_fm_asr_iso": [],
    }
    for var_ in var_values:
        li = line_integrals[f"var_{var_}"]
        out["int_fp"].append([li["int_logk_fp"]["mean"], li["int_logk_fp"]["std"]])
        out["int2_fp"].append([li["int_logk2_fp"]["mean"], li["int_logk2_fp"]["std"]])
        out["int_lrp"].append([li["int_logk_lrp"]["mean"], li["int_logk_lrp"]["std"]])
        out["int2_lrp"].append([li["int_logk2_lrp"]["mean"], li["int_logk2_lrp"]["std"]])
        out["int_lrp_fmm"].append([li["int_logk_lrp_fmm"]["mean"], li["int_logk_lrp_fmm"]["std"]])
        out["int2_lrp_fmm"].append([li["int_logk2_lrp_fmm"]["mean"], li["int_logk2_lrp_fmm"]["std"]])
        out["int_fp_dijkstra"].append([li["int_logk_fp_dijkstra"]["mean"], li["int_logk_fp_dijkstra"]["std"]])
        out["int2_fp_dijkstra"].append([li["int_logk2_fp_dijkstra"]["mean"], li["int_logk2_fp_dijkstra"]["std"]])
        out["int_fp_fmm"].append([li["int_logk_fp_fmm"]["mean"], li["int_logk_fp_fmm"]["std"]])
        out["int2_fp_fmm"].append([li["int_logk2_fp_fmm"]["mean"], li["int_logk2_fp_fmm"]["std"]])
        _fsm_int = li.get("int_logk_fp_fsm", {"mean": np.nan, "std": np.nan})
        _fsm_int2 = li.get("int_logk2_fp_fsm", {"mean": np.nan, "std": np.nan})
        out["int_fp_fsm"].append([_fsm_int["mean"], _fsm_int["std"]])
        out["int2_fp_fsm"].append([_fsm_int2["mean"], _fsm_int2["std"]])
        _asr = li.get("int_logk_fp_fm_asr", {"mean": np.nan, "std": np.nan})
        _asr2 = li.get("int_logk2_fp_fm_asr", {"mean": np.nan, "std": np.nan})
        out["int_fp_fm_asr"].append([_asr["mean"], _asr["std"]])
        out["int2_fp_fm_asr"].append([_asr2["mean"], _asr2["std"]])
        _asri = li.get("int_logk_fp_fm_asr_iso", {"mean": np.nan, "std": np.nan})
        _asri2 = li.get("int_logk2_fp_fm_asr_iso", {"mean": np.nan, "std": np.nan})
        out["int_fp_fm_asr_iso"].append([_asri["mean"], _asri["std"]])
        out["int2_fp_fm_asr_iso"].append([_asri2["mean"], _asri2["std"]])

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _safe_first_size_samples(lengths: dict[str, list[np.ndarray]], key: str) -> np.ndarray | None:
    values = lengths.get(key, [])
    if not values:
        return None
    arr = np.asarray(values[0], dtype=float).ravel()
    if arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return finite


def collect_wasserstein_length_distance(var_data: dict[str, dict[str, Any]], var_values: list[float]) -> tuple[dict[str, np.ndarray], list[str]]:
    out = {
        "fp_lrp": [],
        "fp_lrpFMM": [],
        "fp_fpDijkstra": [],
        "fp_fpFMM": [],
        "fp_fpFSM": [],
        "fp_fm_asr": [],
        "fp_fm_asr_iso": [],
    }
    warnings: list[str] = []

    for var_ in var_values:
        entry = var_data.get(f"var_{var_}")
        if entry is None:
            out["fp_lrp"].append(np.nan)
            out["fp_lrpFMM"].append(np.nan)
            out["fp_fpDijkstra"].append(np.nan)
            out["fp_fpFMM"].append(np.nan)
            out["fp_fpFSM"].append(np.nan)
            out["fp_fm_asr"].append(np.nan)
            out["fp_fm_asr_iso"].append(np.nan)
            warnings.append(f"Missing variance entry for Wasserstein length metric at sigma2={var_}.")
            continue

        lengths = entry.get("lengths", {})
        fp = _safe_first_size_samples(lengths, "fp")
        lrp = _safe_first_size_samples(lengths, "lrp")
        lrp_fmm = _safe_first_size_samples(lengths, "lrp_fmm")
        fp_dijkstra = _safe_first_size_samples(lengths, "fp_dijkstra")
        fp_fmm = _safe_first_size_samples(lengths, "fp_fmm")
        fp_fsm = _safe_first_size_samples(lengths, "fp_fsm")
        fp_fm_asr = _safe_first_size_samples(lengths, "fp_fm_asr")
        fp_fm_asr_iso = _safe_first_size_samples(lengths, "fp_fm_asr_iso")

        pair_data = {
            "fp_lrp": lrp,
            "fp_lrpFMM": lrp_fmm,
            "fp_fpDijkstra": fp_dijkstra,
            "fp_fpFMM": fp_fmm,
            "fp_fpFSM": fp_fsm,
            "fp_fm_asr": fp_fm_asr,
            "fp_fm_asr_iso": fp_fm_asr_iso,
        }
        for pair_key, other in pair_data.items():
            if fp is None or other is None:
                out[pair_key].append(np.nan)
                warnings.append(f"Wasserstein length metric skipped for {pair_key} at sigma2={var_} due to missing finite samples.")
                continue
            out[pair_key].append(float(wasserstein_distance(fp, other)))

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}, warnings


def collect_length_cdf_samples(var_data: dict[str, dict[str, Any]], var_values: list[float]) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    warnings: list[str] = []

    for var_ in var_values:
        key = f"var_{var_}"
        entry = var_data.get(key)
        if entry is None:
            warnings.append(f"Missing variance entry for length CDF samples at sigma2={var_}.")
            continue

        lengths = entry.get("lengths", {})
        samples = {
            "fp": _safe_first_size_samples(lengths, "fp"),
            "lrp": _safe_first_size_samples(lengths, "lrp"),
            "lrp_fmm": _safe_first_size_samples(lengths, "lrp_fmm"),
            "fp_fmm": _safe_first_size_samples(lengths, "fp_fmm"),
            "fp_dijkstra": _safe_first_size_samples(lengths, "fp_dijkstra"),
            "fp_fsm": _safe_first_size_samples(lengths, "fp_fsm"),
            "fp_fm_asr": _safe_first_size_samples(lengths, "fp_fm_asr"),
            "fp_fm_asr_iso": _safe_first_size_samples(lengths, "fp_fm_asr_iso"),
        }
        required_keys = ("fp", "lrp", "lrp_fmm", "fp_fmm", "fp_dijkstra")
        if any(samples[k] is None for k in required_keys):
            warnings.append(f"Length CDF samples incomplete at sigma2={var_}; skipping this variance.")
            continue

        out[key] = {
            "fp": np.asarray(samples["fp"], dtype=float),
            "lrp": np.asarray(samples["lrp"], dtype=float),
            "lrp_fmm": np.asarray(samples["lrp_fmm"], dtype=float),
            "fp_fmm": np.asarray(samples["fp_fmm"], dtype=float),
            "fp_dijkstra": np.asarray(samples["fp_dijkstra"], dtype=float),
        }
        if samples["fp_fsm"] is not None:
            out[key]["fp_fsm"] = np.asarray(samples["fp_fsm"], dtype=float)
        fm_asr_s = _safe_first_size_samples(lengths, "fp_fm_asr")
        fm_iso_s = _safe_first_size_samples(lengths, "fp_fm_asr_iso")
        if fm_asr_s is not None:
            out[key]["fp_fm_asr"] = np.asarray(fm_asr_s, dtype=float)
        if fm_iso_s is not None:
            out[key]["fp_fm_asr_iso"] = np.asarray(fm_iso_s, dtype=float)

    return out, warnings


def collect_scatter_time_samples(var_data: dict[str, dict[str, Any]], var_values: list[float]) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    warnings: list[str] = []

    required_keys = ("fpt", "fpt_fmm", "fpt_dijkstra", "mhr", "mhr_fmm")
    optional_keys = ("fpt_fsm", "fpt_fm_asr", "fpt_fm_asr_iso")
    keys = required_keys + optional_keys
    for var_ in var_values:
        var_key = f"var_{var_}"
        entry = var_data.get(var_key)
        if entry is None:
            warnings.append(f"Missing variance entry for scatter time samples at sigma2={var_}.")
            continue
        times = entry.get("times", {})
        series = {k: times.get(k, []) for k in keys}
        n_sizes = min(len(series[k]) for k in required_keys)
        if n_sizes == 0:
            warnings.append(f"No scatter time samples found at sigma2={var_}.")
            continue

        aligned: dict[str, list[np.ndarray]] = {k: [] for k in keys}
        for size_idx in range(n_sizes):
            arr = {k: np.asarray(series[k][size_idx], dtype=float).ravel() if size_idx < len(series[k]) else np.array([]) for k in keys}
            n = min(arr[k].size for k in required_keys)
            if n == 0:
                continue
            mask = np.ones(n, dtype=bool)
            for k in required_keys:
                mask &= np.isfinite(arr[k][:n])
            if not np.any(mask):
                continue
            for k in required_keys:
                aligned[k].append(arr[k][:n][mask])
            for k in optional_keys:
                if arr[k].size >= n:
                    aligned[k].append(arr[k][:n][mask])

        if not all(aligned[k] for k in required_keys):
            warnings.append(f"Scatter time samples are incomplete at sigma2={var_}; skipping this variance.")
            continue
        result = {k: np.concatenate(aligned[k]).astype(float) for k in required_keys}
        for k in optional_keys:
            if aligned[k]:
                result[k] = np.concatenate(aligned[k]).astype(float)
        out[var_key] = result

    return out, warnings


def collect_comp_time_samples(var_data: dict[str, dict[str, Any]], var_values: list[float]) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    warnings: list[str] = []
    all_keys = (
        "ct_lrp_dijkstra",
        "ct_lrp_fmm",
        "ct_fp_dijkstra",
        "ct_fp_fmm",
        "ct_fp_fsm",
        "ct_fp",
        "ct_fp_fm_asr",
        "ct_fp_fm_asr_iso",
    )

    for var_ in var_values:
        var_key = f"var_{var_}"
        entry = var_data.get(var_key)
        if entry is None:
            warnings.append(f"Missing variance entry for comp_time samples at sigma2={var_}.")
            continue
        ct = entry.get("comp_times", {})
        if not ct:
            warnings.append(f"No comp_times data found at sigma2={var_}.")
            continue

        result: dict[str, np.ndarray] = {}
        for k in all_keys:
            merged: list[np.ndarray] = []
            for arr in ct.get(k, []):
                flat = np.asarray(arr, dtype=float).ravel()
                finite = flat[np.isfinite(flat) & (flat > 0)]
                if finite.size > 0:
                    merged.append(finite)
            if merged:
                result[k] = np.concatenate(merged).astype(float)
        if result:
            out[var_key] = result

    return out, warnings


def collect_endpoint_box_samples(var_data: dict[str, dict[str, Any]], var_values: list[float]) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    warnings: list[str] = []
    required_pair_keys = ("fp_lrp", "fp_lrpFMM", "fp_fpFMM", "fp_fpDijkstra")
    optional_pair_keys = ("fp_fpFSM", "fp_fm_asr", "fp_fm_asr_iso")

    for var_ in var_values:
        key = f"var_{var_}"
        entry = var_data.get(key)
        if entry is None:
            warnings.append(f"Missing variance entry for endpoint box samples at sigma2={var_}.")
            continue

        distances = entry.get("distances", {}).get("end_point", {})
        sample_dict: dict[str, np.ndarray] = {}
        missing = False
        for pair_key in required_pair_keys:
            arr_list = distances.get(pair_key, [])
            merged: list[np.ndarray] = []
            for arr in arr_list:
                flat = np.asarray(arr, dtype=float).ravel()
                finite = flat[np.isfinite(flat)]
                if finite.size > 0:
                    merged.append(finite)
            if not merged:
                warnings.append(f"Endpoint box samples missing for {pair_key} at sigma2={var_}.")
                missing = True
                continue
            sample_dict[pair_key] = np.concatenate(merged).astype(float)
        for pair_key in optional_pair_keys:
            arr_list = distances.get(pair_key, [])
            merged = []
            for arr in arr_list:
                flat = np.asarray(arr, dtype=float).ravel()
                finite = flat[np.isfinite(flat)]
                if finite.size > 0:
                    merged.append(finite)
            if merged:
                sample_dict[pair_key] = np.concatenate(merged).astype(float)

        if missing:
            continue
        out[key] = sample_dict

    return out, warnings


def collect_hausdorff_box_samples(var_data: dict[str, dict[str, Any]], var_values: list[float]) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    warnings: list[str] = []
    required_pair_keys = ("fp_lrp", "fp_lrpFMM", "fp_fpFMM", "fp_fpDijkstra")
    optional_pair_keys = ("fp_fpFSM", "fp_fm_asr", "fp_fm_asr_iso")

    for var_ in var_values:
        key = f"var_{var_}"
        entry = var_data.get(key)
        if entry is None:
            warnings.append(f"Missing variance entry for Hausdorff box samples at sigma2={var_}.")
            continue

        distances = entry.get("distances", {}).get("hausdorff", {})
        sample_dict: dict[str, np.ndarray] = {}
        missing = False
        for pair_key in required_pair_keys:
            arr_list = distances.get(pair_key, [])
            merged: list[np.ndarray] = []
            for arr in arr_list:
                flat = np.asarray(arr, dtype=float).ravel()
                finite = flat[np.isfinite(flat)]
                if finite.size > 0:
                    merged.append(finite)
            if not merged:
                warnings.append(f"Hausdorff box samples missing for {pair_key} at sigma2={var_}.")
                missing = True
                continue
            sample_dict[pair_key] = np.concatenate(merged).astype(float)
        for pair_key in optional_pair_keys:
            arr_list = distances.get(pair_key, [])
            merged = []
            for arr in arr_list:
                flat = np.asarray(arr, dtype=float).ravel()
                finite = flat[np.isfinite(flat)]
                if finite.size > 0:
                    merged.append(finite)
            if merged:
                sample_dict[pair_key] = np.concatenate(merged).astype(float)

        if missing:
            continue
        out[key] = sample_dict

    return out, warnings


_MC_HAUSDORFF_PAIRS = (
    ("fp_vs_lrp", "fp_lrp"),
    ("fp_vs_lrp_fmm", "fp_lrpFMM"),
    ("fp_vs_dijkstra", "fp_fpDijkstra"),
    ("fp_vs_fmm", "fp_fpFMM"),
    ("fp_vs_fsm", "fp_fpFSM"),
    ("fp_vs_fm_asr", "fp_fm_asr"),
    ("fp_vs_fm_asr_iso", "fp_fm_asr_iso"),
)
_CDF_GRID_SIZE = 500


def _ecdf_on_grid(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    xs = np.sort(samples)
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return np.interp(grid, xs, ys, left=0.0, right=1.0)


def collect_hausdorff_cdf_ensemble(
    discovered_var: dict[float, list[RunPaths]],
    var_values: list[float],
    cfg: VisualizationConfig,
    *,
    show_progress: bool = False,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[str]]:
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    warnings: list[str] = []

    target_size = int(cfg.field_size_ref)

    var_iter = tqdm(var_values, desc="Hausdorff CDF variance", unit="var", leave=False) if show_progress else var_values
    for var_ in var_iter:
        var_key = f"var_{var_}"
        run_paths_list = discovered_var.get(var_, [])
        target_rp = next((rp for rp in run_paths_list if int(rp.size_value) == target_size), None)
        if target_rp is None or not target_rp.mc_files:
            warnings.append(f"Hausdorff CDF ensemble: no MC files for sigma2={var_}, size={target_size}.")
            continue

        raw_dists: dict[str, list[np.ndarray]] = {display: [] for _, display in _MC_HAUSDORFF_PAIRS}

        mc_iter = tqdm(target_rp.mc_files, desc=f"var={var_} MC files", unit="mc", leave=False) if show_progress else target_rp.mc_files
        for mc_path in mc_iter:
            try:
                mc = load_mc_consolidated(mc_path)
            except Exception:
                continue
            hausdorff_block = mc.get("hausdorff") or {}
            if not isinstance(hausdorff_block, dict):
                continue
            for mc_key, display_key in _MC_HAUSDORFF_PAIRS:
                pair_raw = hausdorff_block.get(mc_key)
                pair_data = pair_raw if isinstance(pair_raw, dict) else {}
                d = pair_data.get("distances")
                if d is None:
                    continue
                arr = np.asarray(d, dtype=float).ravel()
                finite = arr[np.isfinite(arr)]
                if finite.size > 0:
                    raw_dists[display_key].append(finite)

        pair_result: dict[str, dict[str, np.ndarray]] = {}
        for _, display_key in _MC_HAUSDORFF_PAIRS:
            dists_list = raw_dists[display_key]
            if len(dists_list) < 2:
                warnings.append(
                    f"Hausdorff CDF ensemble: fewer than 2 MC realizations with data for {display_key} at sigma2={var_}."
                )
                continue
            x_max = max(float(np.max(a)) for a in dists_list)
            if x_max <= 0:
                continue
            grid = np.linspace(0.0, x_max, _CDF_GRID_SIZE)
            cdf_matrix = np.array([_ecdf_on_grid(a, grid) for a in dists_list])
            pair_result[display_key] = {
                "x": grid,
                "mean_cdf": np.mean(cdf_matrix, axis=0),
                "std_cdf": np.std(cdf_matrix, axis=0),
            }

        if pair_result:
            out[var_key] = pair_result

    return out, warnings


_HAUSDORFF_STAT_KEYS = ("max", "mean", "median")


def collect_entropy_hausdorff_samples(
    discovered_var: dict[float, list[RunPaths]],
    var_values: list[float],
    cfg: VisualizationConfig,
    *,
    show_progress: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Collect per-MC Renyi entropy profiles and Hausdorff summary stats from consolidated files."""
    out: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    target_size = int(cfg.field_size_ref)

    var_iter = tqdm(var_values, desc="Entropy+Hausdorff samples", unit="var", leave=False) if show_progress else var_values
    for var_ in var_iter:
        var_key = f"var_{var_}"
        run_paths_list = discovered_var.get(var_, [])
        target_rp = next((rp for rp in run_paths_list if int(rp.size_value) == target_size), None)
        if target_rp is None or not target_rp.mc_files:
            warnings.append(f"Entropy+Hausdorff samples: no MC files for sigma2={var_}, size={target_size}.")
            continue

        renyi_s1_list: list[np.ndarray] = []
        renyi_s2_list: list[np.ndarray] = []
        haus: dict[str, dict[str, list[float]]] = {
            stat: {display: [] for _, display in _MC_HAUSDORFF_PAIRS}
            for stat in _HAUSDORFF_STAT_KEYS
        }

        mc_iter = tqdm(target_rp.mc_files, desc=f"var={var_} MC", unit="mc", leave=False) if show_progress else target_rp.mc_files
        for mc_path in mc_iter:
            try:
                mc = load_mc_consolidated(mc_path)
            except Exception:
                continue
            renyi = mc.get("renyi_entropy", {})
            renyi_s1_list.append(np.asarray(renyi.get("S1", [np.nan]), dtype=float))
            renyi_s2_list.append(np.asarray(renyi.get("S2", [np.nan]), dtype=float))

            hausdorff_block = mc.get("hausdorff") or {}
            if not isinstance(hausdorff_block, dict):
                continue
            for mc_key, display_key in _MC_HAUSDORFF_PAIRS:
                pair_raw = hausdorff_block.get(mc_key)
                pair_data = pair_raw if isinstance(pair_raw, dict) else {}
                for stat in _HAUSDORFF_STAT_KEYS:
                    haus[stat][display_key].append(float(pair_data.get(stat, np.nan)))

        result: dict[str, Any] = {
            "renyi_S1": renyi_s1_list,
            "renyi_S2": renyi_s2_list,
        }
        for stat in _HAUSDORFF_STAT_KEYS:
            result[f"hausdorff_{stat}"] = {
                display: np.asarray(haus[stat][display], dtype=float)
                for _, display in _MC_HAUSDORFF_PAIRS
            }
        out[var_key] = result

    return out, warnings


def compute_t_ref(k_g: float, dh: float, field_size: float, length_scale: float) -> float:
    i_y = length_scale * field_size
    q_ref = k_g * dh
    return i_y / q_ref


def compute_i_y(field_size: float, length_scale: float) -> float:
    return length_scale * field_size


def compute_taylor_times(
    stats_fp: np.ndarray,
    stats_lrp: np.ndarray,
    stats_lrp_fmm: np.ndarray,
    stats_fp_dijkstra: np.ndarray,
    stats_fp_fmm: np.ndarray,
    int_fp: np.ndarray,
    int_lrp: np.ndarray,
    int_lrp_fmm: np.ndarray,
    int_fp_dijkstra: np.ndarray,
    int_fp_fmm: np.ndarray,
    int2_fp: np.ndarray,
    int2_lrp: np.ndarray,
    int2_lrp_fmm: np.ndarray,
    int2_fp_dijkstra: np.ndarray,
    int2_fp_fmm: np.ndarray,
    dh: float,
) -> dict[str, np.ndarray]:
    taylor_lrp = stats_lrp[:, 0] - int_lrp[:, 0] + int2_lrp[:, 0] / 2.0
    taylor_lrp_fmm = stats_lrp_fmm[:, 0] - int_lrp_fmm[:, 0] + int2_lrp_fmm[:, 0] / 2.0
    taylor_fp = (stats_fp[:, 0] - int_fp[:, 0] + int2_fp[:, 0] / 2.0) / dh
    taylor_fp_dijkstra = (stats_fp_dijkstra[:, 0] - int_fp_dijkstra[:, 0] + int2_fp_dijkstra[:, 0] / 2.0) / dh
    taylor_fp_fmm = (stats_fp_fmm[:, 0] - int_fp_fmm[:, 0] + int2_fp_fmm[:, 0] / 2.0) / dh
    return {
        "lrp": taylor_lrp,
        "lrp_fmm": taylor_lrp_fmm,
        "fp": taylor_fp,
        "fp_dijkstra": taylor_fp_dijkstra,
        "fp_fmm": taylor_fp_fmm,
    }


def prepare_plot_inputs(
    cfg: VisualizationConfig,
    var_results: Any,
    std_results: Any | None,
    discovered_var: dict[float, list[RunPaths]] | None = None,
    *,
    show_progress: bool = False,
) -> dict[str, Any]:
    var_stats = _pick(var_results, "stats")
    var_data = _pick(var_results, "data")
    var_line_integrals = _pick(var_results, "line_integrals")
    var_field_size = _pick(var_results, "field_size")

    var_values = _available_var_values(var_stats, cfg.variance_values)
    var = np.asarray(var_values, dtype=float)
    s = collect_primary_stats(var_stats, var_values)
    ints = collect_integral_stats(var_line_integrals, var_values)
    wasserstein_len, derived_warnings = collect_wasserstein_length_distance(var_data, var_values)
    length_cdf_samples, cdf_warnings = collect_length_cdf_samples(var_data, var_values)
    scatter_time_samples, scatter_warnings = collect_scatter_time_samples(var_data, var_values)
    endpoint_box_samples, endpoint_box_warnings = collect_endpoint_box_samples(var_data, var_values)
    hausdorff_box_samples, hausdorff_box_warnings = collect_hausdorff_box_samples(var_data, var_values)
    comp_time_samples, comp_time_warnings = collect_comp_time_samples(var_data, var_values)
    derived_warnings.extend(cdf_warnings)
    derived_warnings.extend(scatter_warnings)
    derived_warnings.extend(endpoint_box_warnings)
    derived_warnings.extend(hausdorff_box_warnings)
    derived_warnings.extend(comp_time_warnings)

    hausdorff_cdf_ensemble: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    entropy_hausdorff_samples: dict[str, dict[str, Any]] = {}
    if discovered_var is not None:
        hausdorff_cdf_ensemble, hcdf_warnings = collect_hausdorff_cdf_ensemble(discovered_var, var_values, cfg, show_progress=show_progress)
        derived_warnings.extend(hcdf_warnings)
        entropy_hausdorff_samples, eh_warnings = collect_entropy_hausdorff_samples(discovered_var, var_values, cfg, show_progress=show_progress)
        derived_warnings.extend(eh_warnings)

    field_size_arr = np.asarray(var_field_size, dtype=float)
    field_size = cfg.field_size_ref if field_size_arr.size == 0 else float(field_size_arr[0])
    i_y = compute_i_y(field_size, cfg.length_scale)
    t_ref = compute_t_ref(cfg.k_g, cfg.dh, field_size, cfg.length_scale)
    taylor = compute_taylor_times(
        s["stats_fp"],
        s["stats_lrp"],
        s["stats_lrp_fmm"],
        s["stats_fp_dijkstra"],
        s["stats_fp_fmm"],
        ints["int_fp"],
        ints["int_lrp"],
        ints["int_lrp_fmm"],
        ints["int_fp_dijkstra"],
        ints["int_fp_fmm"],
        ints["int2_fp"],
        ints["int2_lrp"],
        ints["int2_lrp_fmm"],
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
        "wasserstein_len": wasserstein_len,
        "length_cdf_samples": length_cdf_samples,
        "scatter_time_samples": scatter_time_samples,
        "endpoint_box_samples": endpoint_box_samples,
        "hausdorff_box_samples": hausdorff_box_samples,
        "hausdorff_cdf_ensemble": hausdorff_cdf_ensemble,
        "entropy_hausdorff_samples": entropy_hausdorff_samples,
        "comp_time_samples": comp_time_samples,
        "derived_warnings": derived_warnings,
    }


def run_stage3(cfg: VisualizationConfig, *, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or default_cache_dir(cfg)
    metrics = load_pickle(cache_dir / "stage2_metrics.pkl")
    var_results = metrics["var_results"]
    std_results = metrics["std_results"]

    discovered_var: dict[float, list[RunPaths]] | None = None
    index_path = cache_dir / "stage1_index.pkl"
    if index_path.exists():
        index = load_pickle(index_path)
        discovered_var = deserialize_discovered(index["variance"])

    plot_inputs = prepare_plot_inputs(cfg, var_results, std_results, discovered_var=discovered_var, show_progress=True)
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
