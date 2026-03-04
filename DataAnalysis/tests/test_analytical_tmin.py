from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module(module_name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_effective_length_uses_integer_floor_margin() -> None:
    analytical = _load_module("analytical_tmin", "visualization/stats/analytical_tmin.py")
    assert analytical.compute_effective_length(200.0) == 180.0
    assert analytical.compute_effective_length(201.0) == 181.0


def test_compute_analytical_tmin_curve_typical_variances() -> None:
    analytical = _load_module("analytical_tmin", "visualization/stats/analytical_tmin.py")
    var = np.asarray([0.1, 0.5, 1.0, 2.0, 3.0], dtype=float)
    values, warnings = analytical.compute_analytical_tmin_curve(
        var,
        length=180.0,
        velocity=0.1,
        i_y=10.0,
        percentile=0.01,
    )
    assert values.shape == var.shape
    assert np.all(np.isfinite(values))
    assert warnings == []


def test_compute_analytical_tmin_curve_invalid_params_returns_nan() -> None:
    analytical = _load_module("analytical_tmin", "visualization/stats/analytical_tmin.py")
    var = np.asarray([0.1, 0.5], dtype=float)
    values, warnings = analytical.compute_analytical_tmin_curve(
        var,
        length=180.0,
        velocity=0.0,
        i_y=10.0,
        percentile=0.01,
    )
    assert np.all(np.isnan(values))
    assert warnings


def test_config_validates_analytical_percentile(tmp_path: Path) -> None:
    cfg_mod = _load_module("viz_config", "visualization/config.py")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data_root: /tmp/nonexistent",
                "run_id: 3",
                "length_scale: 0.05",
                "dh: 0.1",
                "field_size_ref: 200",
                "variance_values: [0.1, 0.5]",
                "std_values: [1.0]",
                "output_dir: /tmp/out",
                "save_formats: [png]",
                "sections: [times]",
                "analytical_percentile: 1.2",
            ]
        ),
        encoding="utf-8",
    )

    try:
        cfg_mod.load_config(cfg_path)
    except cfg_mod.ConfigError as exc:
        assert "analytical_percentile" in str(exc)
    else:
        raise AssertionError("Expected ConfigError for invalid analytical_percentile")
