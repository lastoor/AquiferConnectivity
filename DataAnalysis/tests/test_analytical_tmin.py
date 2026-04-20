from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import step1_index as step1
import step4_plot as step4


def test_compute_effective_length_uses_integer_floor_margin() -> None:
    assert step4.compute_effective_length(200.0) == 180.0
    assert step4.compute_effective_length(201.0) == 181.0


def test_compute_analytical_tmin_curve_typical_variances() -> None:
    var = np.asarray([0.1, 0.5, 1.0, 2.0, 3.0], dtype=float)
    values, warnings = step4.compute_analytical_tmin_curve(
        var,
        length=180.0,
        velocity=0.1,
        i_y=10.0,
        percentile=0.01,
    )
    assert values.shape == var.shape
    assert np.all(np.isfinite(values))
    assert warnings == []


def test_g_tau_tiny_time_is_finite_and_bounded() -> None:
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        value = step4._g_tau(
            1e-300,
            length=180.0,
            velocity=0.1,
            i_y=10.0,
            sigma_y_sq=1.0,
            euler_const=float(getattr(step4.constants, "euler_gamma", np.euler_gamma)),
        )
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_compute_analytical_tmin_curve_invalid_params_returns_nan() -> None:
    var = np.asarray([0.1, 0.5], dtype=float)
    values, warnings = step4.compute_analytical_tmin_curve(
        var,
        length=180.0,
        velocity=0.0,
        i_y=10.0,
        percentile=0.01,
    )
    assert np.all(np.isnan(values))
    assert warnings


def test_compute_analytical_tmin_curve_stress_params_do_not_crash() -> None:
    var = np.asarray([1e-12, 1e-6, 1e-3], dtype=float)
    values, warnings = step4.compute_analytical_tmin_curve(
        var,
        length=180.0,
        velocity=1e-12,
        i_y=10.0,
        percentile=0.01,
    )
    assert values.shape == var.shape
    # Values may be NaN for difficult regimes, but function should return safely with warnings.
    assert isinstance(warnings, list)


def test_plot_characteristic_time_includes_analytical_label() -> None:
    var = np.asarray([0.1, 0.5, 1.0], dtype=float)
    stats = np.asarray([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]], dtype=float)
    analytical = np.asarray([0.8, 1.7, 2.6], dtype=float)

    fig, _ = step4.plot_characteristic_time(
        var,
        stats,
        stats,
        stats,
        stats,
        t_ref=1.0,
        analytical_tmin=analytical,
        analytical_label="Analytical Test",
    )
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert "Analytical Test" in labels


def test_config_validates_analytical_percentile(tmp_path: Path) -> None:
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
        step1.load_config(cfg_path)
    except step1.ConfigError as exc:
        assert "analytical_percentile" in str(exc)
    else:
        raise AssertionError("Expected ConfigError for invalid analytical_percentile")
