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


def _warning_mentions_metric(msg: str, metric_name: str) -> bool:
    return msg.endswith(f"for {metric_name}")


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
    assert not any(_warning_mentions_metric(msg, "len_fp") for msg in warnings)
    assert not any(_warning_mentions_metric(msg, "travel_time_fp") for msg in warnings)
    fpt_mean = payload["var_results"]["stats"]["var_0.1"]["times"]["fpt"][0][0]
    assert np.isfinite(float(fpt_mean))


def test_step2_metrics_accepts_new_files_with_legacy_config_keys(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    cfg.distance_main_output_keys["end_point"] = {
        "fp_lrp": "endPointDist_fp_lrp_m",
        "fp_lrpFMM": "endPointDist_fp_lrpFMM_m",
        "fp_fpDijkstra": "endPointDist_fp_fpDijkstra_m",
        "fp_fpFMM": "endPointDist_fp_fpFMM_m",
        "fp_fpFSM": "endPointDist_fp_fpFSM_m",
    }
    cfg.distance_main_output_keys["hausdorff"] = {
        "fp_lrp": "hausdorffDist_fp_lrp_m",
        "fp_lrpFMM": "hausdorffDist_fp_lrpFMM_m",
        "fp_fpDijkstra": "hausdorffDist_fp_fpDijkstra_m",
        "fp_fpFMM": "hausdorffDist_fp_fpFMM_m",
        "fp_fpFSM": "hausdorffDist_fp_fpFSM_m",
    }
    cfg.distance_main_output_keys["frechet"] = {
        "fp_lrp": "frechetDist_fp_lrp_m",
        "fp_lrpFMM": "frechetDist_fp_lrpFMM_m",
        "fp_fpDijkstra": "frechetDist_fp_fpDijkstra_m",
        "fp_fpFMM": "frechetDist_fp_fpFMM_m",
        "fp_fpFSM": "frechetDist_fp_fpFSM_m",
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
    assert not any(_warning_mentions_metric(msg, "len_fp") for msg in warnings)
    assert not any(_warning_mentions_metric(msg, "travel_time_fp") for msg in warnings)
    fpt_mean = payload["var_results"]["stats"]["var_0.1"]["times"]["fpt"][0][0]
    assert np.isfinite(float(fpt_mean))


def test_consolidated_to_output_data_extracts_hausdorff_max_and_aliases() -> None:
    mc = {
        "paths": {},
        "endpoint": {"dy": {}},
        "hausdorff": {
            "fp_vs_lrp": {"max": 1.2},
            "fp_vs_lrp_fmm": {"max": 1.1},
            "fp_vs_dijkstra": {"max": 1.3},
            "fp_vs_fmm": {"max": 1.05},
            "fp_vs_fsm": None,
        },
        "renyi_entropy": {},
        "computation_time": {},
    }

    output_data = step2._consolidated_to_output_data(mc)

    assert output_data["hausdorff_fp_vs_lrp"] == 1.2
    assert output_data["hausdorffDist_fp_lrp_m"] == 1.2
    assert output_data["hausdorff_fp_vs_lrp_fmm"] == 1.1
    assert output_data["hausdorffDist_fp_lrpFMM_m"] == 1.1
    assert output_data["hausdorff_fp_vs_dijkstra"] == 1.3
    assert output_data["hausdorffDist_fp_fpDijkstra_m"] == 1.3
    assert output_data["hausdorff_fp_vs_fmm"] == 1.05
    assert output_data["hausdorffDist_fp_fpFMM_m"] == 1.05
    assert np.isnan(output_data["hausdorff_fp_vs_fsm"])
    assert np.isnan(output_data["hausdorffDist_fp_fpFSM_m"])


def test_step2_metrics_loads_hausdorff_from_consolidated_outputs(consolidated_pipeline_fixture) -> None:
    cfg = step1.load_config(consolidated_pipeline_fixture["cfg_path"])
    step1.run_stage1(cfg, cache_dir=consolidated_pipeline_fixture["cache_dir"])

    cache_file = step2.run_stage2(
        cfg,
        cache_dir=consolidated_pipeline_fixture["cache_dir"],
        max_var_values=1,
        max_std_values=1,
        max_sizes_per_value=1,
        max_mc_per_size=1,
    )

    with cache_file.open("rb") as f:
        payload = pickle.load(f)

    warnings = _collect_warnings(payload)
    assert all("for hausdorff:" not in msg for msg in warnings)
    haus_mean = payload["var_results"]["stats"]["var_0.1"]["distances"]["hausdorff"]["fp_lrp"][0][0]
    assert np.isfinite(float(haus_mean))
    assert haus_mean == 1.2


def test_load_main_output_value_prefers_solver_neutral_fp_keys() -> None:
    output_data = {
        "len_fp": 12.0,
        "len_fp_rk45": 99.0,
        "travel_time_fp": 7.0,
        "travel_time_rk45": 77.0,
    }

    len_val, len_key = step2._load_main_output_value(output_data, step2.VARIANCE_METRIC_KEY_CANDIDATES["len_fp"])
    time_val, time_key = step2._load_main_output_value(output_data, step2.VARIANCE_METRIC_KEY_CANDIDATES["travel_time_fp"])

    assert len_val == 12.0
    assert len_key == "len_fp"
    assert time_val == 7.0
    assert time_key == "travel_time_fp"


def test_load_main_output_value_falls_back_to_legacy_fp_keys() -> None:
    output_data = {
        "len_fp_rk45": 12.0,
        "travel_time_rk45": 7.0,
    }

    len_val, len_key = step2._load_main_output_value(output_data, step2.VARIANCE_METRIC_KEY_CANDIDATES["len_fp"])
    time_val, time_key = step2._load_main_output_value(output_data, step2.VARIANCE_METRIC_KEY_CANDIDATES["travel_time_fp"])

    assert len_val == 12.0
    assert len_key == "len_fp_rk45"
    assert time_val == 7.0
    assert time_key == "travel_time_rk45"


def test_load_main_output_value_returns_nan_when_fp_keys_missing() -> None:
    output_data = {
        "travel_time_dijkstra": 5.0,
    }

    len_val, len_key = step2._load_main_output_value(output_data, step2.VARIANCE_METRIC_KEY_CANDIDATES["len_fp"])
    time_val, time_key = step2._load_main_output_value(output_data, step2.VARIANCE_METRIC_KEY_CANDIDATES["travel_time_fp"])

    assert np.isnan(len_val)
    assert len_key is None
    assert np.isnan(time_val)
    assert time_key is None
