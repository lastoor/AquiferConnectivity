from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import step1_index as step1
import step2_metrics as step2


def test_step2_metrics_respects_limits_and_writes_cache(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    step1.run_stage1(cfg, cache_dir=pipeline_fixture["cache_dir"])

    cache_file = step2.run_stage2(
        cfg,
        cache_dir=pipeline_fixture["cache_dir"],
        hausdorff_compute_method="custom",
        frechet_compute_method="custom",
        max_var_values=1,
        max_std_values=1,
        max_sizes_per_value=1,
        max_mc_per_size=1,
    )

    assert cache_file.exists()

    with cache_file.open("rb") as f:
        payload = pickle.load(f)
    assert "var_results" in payload
    assert "std_results" in payload
    assert "warnings" in payload

    meta = json.loads((pipeline_fixture["cache_dir"] / "stage2_metrics.meta.json").read_text(encoding="utf-8"))
    assert meta["notes"]["max_var_values"] == 1
    assert meta["notes"]["max_std_values"] == 1
    assert meta["notes"]["max_sizes_per_value"] == 1
    assert meta["notes"]["max_mc_per_size"] == 1
    assert meta["notes"]["hausdorff_compute_method"] == "custom"
    assert meta["notes"]["frechet_compute_method"] == "custom"


def _rewrite_main_output_schema(data_root: Path, run_id: int, *, legacy: bool) -> None:
    run_dir = data_root / f"run{run_id}"
    for path in run_dir.glob("**/main_output/mainOutput_size*_mc*.npz"):
        with np.load(path) as raw_data:
            raw = {k: raw_data[k].item() if np.asarray(raw_data[k]).size == 1 else raw_data[k] for k in raw_data.files}

        def _pick(*keys: str) -> float:
            for key in keys:
                if key in raw:
                    return float(raw[key])
            raise KeyError(f"None of the keys exist: {keys}")

        if legacy:
            payload = {
                "len_fp_m": _pick("len_fp_m", "len_fp_rk45"),
                "len_fp_dijkstra_m": _pick("len_fp_dijkstra_m", "len_fp_dijkstra"),
                "len_fp_fmm_m": _pick("len_fp_fmm_m", "len_fp_fmm"),
                "len_lrp_m": _pick("len_lrp_m", "len_lrp"),
                "fpt_m": _pick("fpt_m", "travel_time_rk45"),
                "fpt_dijkstra_m": _pick("fpt_dijkstra_m", "travel_time_dijkstra"),
                "fpt_fmm_m": _pick("fpt_fmm_m", "travel_time_fmm"),
                "mhr_m": _pick("mhr_m", "resistance_lrp"),
                "endPointDist_fp_lrp_m": _pick("endPointDist_fp_lrp_m", "endpoint_dy_fp_vs_lrp"),
                "endPointDist_fp_fpDijkstra_m": _pick("endPointDist_fp_fpDijkstra_m", "endpoint_dy_fp_vs_dijkstra"),
                "endPointDist_fp_fpFMM_m": _pick("endPointDist_fp_fpFMM_m", "endpoint_dy_fp_vs_fmm"),
                "hausdorffDist_fp_lrp_m": _pick("hausdorffDist_fp_lrp_m", "hausdorff_fp_vs_lrp"),
                "hausdorffDist_fp_fpDijkstra_m": _pick("hausdorffDist_fp_fpDijkstra_m", "hausdorff_fp_vs_dijkstra"),
                "hausdorffDist_fp_fpFMM_m": _pick("hausdorffDist_fp_fpFMM_m", "hausdorff_fp_vs_fmm"),
            }
        else:
            payload = {
                "len_fp_rk45": _pick("len_fp_rk45", "len_fp_m"),
                "len_fp_dijkstra": _pick("len_fp_dijkstra", "len_fp_dijkstra_m"),
                "len_fp_fmm": _pick("len_fp_fmm", "len_fp_fmm_m"),
                "len_lrp": _pick("len_lrp", "len_lrp_m"),
                "travel_time_rk45": _pick("travel_time_rk45", "fpt_m"),
                "travel_time_dijkstra": _pick("travel_time_dijkstra", "fpt_dijkstra_m"),
                "travel_time_fmm": _pick("travel_time_fmm", "fpt_fmm_m"),
                "resistance_lrp": _pick("resistance_lrp", "mhr_m"),
                "endpoint_dy_fp_vs_lrp": _pick("endpoint_dy_fp_vs_lrp", "endPointDist_fp_lrp_m"),
                "endpoint_dy_fp_vs_dijkstra": _pick("endpoint_dy_fp_vs_dijkstra", "endPointDist_fp_fpDijkstra_m"),
                "endpoint_dy_fp_vs_fmm": _pick("endpoint_dy_fp_vs_fmm", "endPointDist_fp_fpFMM_m"),
                "hausdorff_fp_vs_lrp": _pick("hausdorff_fp_vs_lrp", "hausdorffDist_fp_lrp_m"),
                "hausdorff_fp_vs_dijkstra": _pick("hausdorff_fp_vs_dijkstra", "hausdorffDist_fp_fpDijkstra_m"),
                "hausdorff_fp_vs_fmm": _pick("hausdorff_fp_vs_fmm", "hausdorffDist_fp_fpFMM_m"),
            }
        np.savez(path, **payload)


def _collect_warnings(payload: dict[str, object]) -> list[str]:
    warnings_by_stage = payload["warnings"]
    assert isinstance(warnings_by_stage, dict)
    combined: list[str] = []
    for stage_name in ("variance", "std"):
        stage_warnings = warnings_by_stage[stage_name]
        assert isinstance(stage_warnings, list)
        combined.extend(str(msg) for msg in stage_warnings)
    return combined


def test_step2_metrics_accepts_legacy_files_with_new_config_keys(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    _rewrite_main_output_schema(cfg.data_root, cfg.run_id, legacy=True)
    cfg.distance_mode["hausdorff"] = "load_from_main_output"
    step1.run_stage1(cfg, cache_dir=pipeline_fixture["cache_dir"])

    cache_file = step2.run_stage2(
        cfg,
        cache_dir=pipeline_fixture["cache_dir"],
        max_var_values=1,
        max_std_values=1,
        max_sizes_per_value=1,
        max_mc_per_size=1,
    )
    with cache_file.open("rb") as f:
        payload = pickle.load(f)
    warnings = _collect_warnings(payload)
    assert all("Missing mainOutput keys" not in msg for msg in warnings)
    fpt_mean = payload["var_results"]["stats"]["var_0.1"]["times"]["fpt"][0][0]
    assert np.isfinite(float(fpt_mean))


def test_step2_metrics_accepts_new_files_with_legacy_config_keys(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    cfg.distance_main_output_keys["end_point"] = {
        "fp_lrp": "endPointDist_fp_lrp_m",
        "fp_fpDijkstra": "endPointDist_fp_fpDijkstra_m",
        "fp_fpFMM": "endPointDist_fp_fpFMM_m",
    }
    cfg.distance_main_output_keys["hausdorff"] = {
        "fp_lrp": "hausdorffDist_fp_lrp_m",
        "fp_fpDijkstra": "hausdorffDist_fp_fpDijkstra_m",
        "fp_fpFMM": "hausdorffDist_fp_fpFMM_m",
    }
    cfg.distance_mode["hausdorff"] = "load_from_main_output"
    step1.run_stage1(cfg, cache_dir=pipeline_fixture["cache_dir"])

    cache_file = step2.run_stage2(
        cfg,
        cache_dir=pipeline_fixture["cache_dir"],
        max_var_values=1,
        max_std_values=1,
        max_sizes_per_value=1,
        max_mc_per_size=1,
    )
    with cache_file.open("rb") as f:
        payload = pickle.load(f)
    warnings = _collect_warnings(payload)
    assert all("Missing mainOutput keys" not in msg for msg in warnings)
