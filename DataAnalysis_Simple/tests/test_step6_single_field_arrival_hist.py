from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import step1_index as step1
import step6_single_field_arrival_hist as step6


def _mock_simulation_result() -> step6.ArrivalSimulationResult:
    return step6.ArrivalSimulationResult(
        logk_field=np.zeros((8, 8), dtype=float),
        arrival_times=np.array([1.0, 1.5, 2.0, 2.5], dtype=float),
        status_counts={0: 4, 1: 1},
        n_particles=5,
        success_fraction=0.8,
        bad_fraction=0.2,
    )


def test_empirical_cdf_sorts_and_normalizes() -> None:
    x, y = step6._empirical_cdf(np.array([3.0, 1.0, 2.0], dtype=float))
    assert np.allclose(x, np.array([1.0, 2.0, 3.0], dtype=float))
    assert np.allclose(y, np.array([1 / 3, 2 / 3, 1.0], dtype=float))


def test_extract_success_arrival_times_uses_terminal_time_per_particle() -> None:
    pt = np.asarray(
        [
            [0.0, 0.4, 1.2, 9.9],
            [0.0, 0.5, 2.0, 8.8],
            [0.0, 0.6, 3.3, 7.7],
        ],
        dtype=float,
    )
    length_arr = np.asarray([3, 2, 4], dtype=np.int32)
    status_arr = np.asarray([0, 1, 0], dtype=np.int32)
    arr = step6._extract_success_arrival_times(pt, length_arr, status_arr)
    # success particles are 0 and 2; terminal times are pt[0,2]=1.2 and pt[2,3]=7.7
    assert np.allclose(arr, np.asarray([1.2, 7.7], dtype=float))


def test_plot_arrival_cdf_comparison_returns_figure() -> None:
    tau_emp = np.array([0.5, 1.0, 1.5], dtype=float)
    ecdf = np.array([1 / 3, 2 / 3, 1.0], dtype=float)
    tau_grid = np.linspace(0.0, 2.0, 10)
    cdf_analytical = np.linspace(0.0, 1.0, 10)
    fig = step6.plot_arrival_cdf_comparison(
        tau_emp,
        ecdf,
        tau_grid,
        cdf_analytical,
        title="test",
    )
    assert fig.axes
    assert fig.axes[0].get_xlabel() == r"$tU/I_Y$"
    assert fig.axes[0].get_ylabel() == "CDF"
    plt.close(fig)


def test_run_step6_figure_only_with_mocked_simulation(pipeline_fixture, monkeypatch, tmp_path: Path) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    monkeypatch.setattr(step6, "simulate_single_field_arrivals", lambda **kwargs: _mock_simulation_result())

    fig_path, npz_path, json_path = step6.run_step6(
        cfg,
        field_size=200,
        length_scale=0.05,
        std=1.0,
        var=None,
        minpadding=0,
        dh=0.1,
        k_g=1.0,
        density_particle=20,
        eps_speed=1e-30,
        ds_min=1e-20,
        ds_max=0.5,
        tol=1e-4,
        y_policy="stop",
        max_points=100000,
        max_iters=1000000,
        rk45_min_success_frac=0.5,
        rk45_max_bad_frac=0.1,
        n_mc=3,
        seed_start=0,
        band_low=5.0,
        band_high=95.0,
        output_dir=tmp_path,
        figure_dpi=120,
        save_npz=False,
        save_json=False,
    )

    assert fig_path.exists()
    assert npz_path is None
    assert json_path is None
    assert fig_path.name.endswith("_arrival_cdf_vs_analytical.png")
    assert "_mc3_" in fig_path.name
    assert not list(tmp_path.glob("*_arrival_times.npz"))
    assert not list(tmp_path.glob("*_arrival_summary.json"))

def test_run_step6_saves_npz_and_json_when_enabled(pipeline_fixture, monkeypatch, tmp_path: Path) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    monkeypatch.setattr(step6, "simulate_single_field_arrivals", lambda **kwargs: _mock_simulation_result())

    fig_path, npz_path, json_path = step6.run_step6(
        cfg,
        field_size=200,
        length_scale=0.05,
        std=1.0,
        var=None,
        minpadding=0,
        dh=0.1,
        k_g=1.0,
        density_particle=20,
        eps_speed=1e-30,
        ds_min=1e-20,
        ds_max=0.5,
        tol=1e-4,
        y_policy="stop",
        max_points=100000,
        max_iters=1000000,
        rk45_min_success_frac=0.5,
        rk45_max_bad_frac=0.1,
        n_mc=2,
        seed_start=7,
        band_low=5.0,
        band_high=95.0,
        output_dir=tmp_path,
        figure_dpi=120,
        save_npz=True,
        save_json=True,
    )

    assert fig_path.exists()
    assert npz_path is not None and npz_path.exists()
    assert json_path is not None and json_path.exists()
    assert npz_path.name.endswith("_arrival_times.npz")
    assert json_path.name.endswith("_arrival_summary.json")

    with np.load(npz_path) as payload:
        assert "tau_samples" in payload
        assert "cdf_empirical" in payload
        assert "cdf_analytical" in payload
        assert "cdf_mc_mean" in payload
        assert "cdf_mc_band_low" in payload
        assert "cdf_mc_band_high" in payload
        assert "seeds_used" in payload
        assert np.all(payload["seeds_used"] == np.asarray([7, 8], dtype=np.int32))

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["n_mc_requested"] == 2
    assert summary["n_mc_used"] == 2
    assert summary["seeds_used"] == [7, 8]
    assert summary["n_particles"] == 10
    assert summary["n_success"] == 8
    assert summary["std"] == 1.0


def test_run_step6_raises_when_all_mc_have_no_valid_arrivals(pipeline_fixture, monkeypatch, tmp_path: Path) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])

    def _mock_empty(**kwargs) -> step6.ArrivalSimulationResult:
        return step6.ArrivalSimulationResult(
            logk_field=np.zeros((8, 8), dtype=float),
            arrival_times=np.asarray([], dtype=float),
            status_counts={1: 5},
            n_particles=5,
            success_fraction=0.0,
            bad_fraction=1.0,
        )

    monkeypatch.setattr(step6, "simulate_single_field_arrivals", _mock_empty)

    try:
        step6.run_step6(
            cfg,
            field_size=200,
            length_scale=0.05,
            std=1.0,
            var=None,
            minpadding=0,
            dh=0.1,
            k_g=1.0,
            density_particle=20,
            n_mc=2,
            seed_start=0,
            band_low=5.0,
            band_high=95.0,
            eps_speed=1e-30,
            ds_min=1e-20,
            ds_max=0.5,
            tol=1e-4,
            y_policy="stop",
            max_points=100000,
            max_iters=1000000,
            rk45_min_success_frac=0.5,
            rk45_max_bad_frac=0.1,
            output_dir=tmp_path,
            figure_dpi=120,
            save_npz=False,
            save_json=False,
        )
    except RuntimeError as exc:
        assert "No valid successful particle arrivals across all Monte Carlo realizations" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for all-empty MC arrivals")
