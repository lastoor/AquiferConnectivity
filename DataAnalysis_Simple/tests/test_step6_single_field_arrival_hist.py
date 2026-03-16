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


def test_extract_success_arrival_times_uses_successful_travel_times() -> None:
    travel_times = np.asarray([1.2, 2.0, 7.7], dtype=float)
    status_arr = np.asarray([0, 1, 0], dtype=np.int32)
    arr = step6._extract_success_arrival_times(travel_times, status_arr)
    assert np.allclose(arr, np.asarray([1.2, 7.7], dtype=float))


def test_validate_particle_tracking_options_rejects_rk45_gpu_mode() -> None:
    try:
        step6._validate_particle_tracking_options(solver="rk45", velocity_location="cell", backend="gpu")
    except ValueError as exc:
        assert "GPU backend" in str(exc)
    else:
        raise AssertionError("Expected RK45 validation to reject GPU backend")


def test_plot_arrival_cdf_comparison_returns_figure() -> None:
    tau_emp = np.array([0.5, 1.0, 1.5], dtype=float)
    ecdf = np.array([1 / 3, 2 / 3, 1.0], dtype=float)
    tau_grid = np.linspace(0.0, 2.0, 10)
    cdf_analytical = np.linspace(0.0, 1.0, 10)
    cdf_lognormal = np.linspace(0.1, 0.9, 10)
    fig = step6.plot_arrival_cdf_comparison(
        tau_emp,
        ecdf,
        tau_grid,
        cdf_analytical,
        title="test",
        cdf_lognormal=cdf_lognormal,
    )
    assert fig.axes
    assert fig.axes[0].get_xlabel() == r"$tU/I_Y$"
    assert fig.axes[0].get_ylabel() == "CDF"
    plt.close(fig)


def test_histogram_pdf_is_nonnegative_and_normalized() -> None:
    tau_samples = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
    bin_edges = step6._build_pdf_hist_bin_edges(tau_samples, pdf_hist_bins=4)
    centers, pdf = step6._histogram_pdf(tau_samples, bin_edges)
    assert centers.shape == pdf.shape
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0.0)
    assert np.isclose(np.sum(pdf * np.diff(bin_edges)), 1.0)


def test_build_tau_grid_is_finer_before_tau_split() -> None:
    tau_grid = step6._build_tau_grid(
        0.8,
        20.0,
        refine_tau_grid=True,
        tau_split=10.0,
        tau_grid_fine=1024,
        tau_grid_coarse=256,
    )
    assert np.all(np.diff(tau_grid) > 0.0)
    below = tau_grid[tau_grid < 10.0]
    above = tau_grid[tau_grid >= 10.0]
    assert below.size > above.size
    assert (below[1] - below[0]) < (above[-1] - above[-2])


def test_build_tau_grid_without_refinement_is_uniform() -> None:
    tau_grid = step6._build_tau_grid(
        0.8,
        20.0,
        refine_tau_grid=False,
        tau_split=10.0,
        tau_grid_fine=8,
        tau_grid_coarse=4,
    )
    diffs = np.diff(tau_grid)
    assert tau_grid.size == 11
    assert np.allclose(diffs, diffs[0])


def test_plot_arrival_cdf_comparison_raises_for_invalid_y_exp() -> None:
    tau_emp = np.array([0.5, 1.0, 1.5], dtype=float)
    ecdf = np.array([1 / 3, 2 / 3, 1.0], dtype=float)
    tau_grid = np.linspace(0.5, 2.0, 10)
    cdf_analytical = np.linspace(0.0, 1.0, 10)
    try:
        step6.plot_arrival_cdf_comparison(
            tau_emp,
            ecdf,
            tau_grid,
            cdf_analytical,
            title="test",
            y_exp=0.0,
        )
    except ValueError as exc:
        assert "y_exp" in str(exc)
    else:
        raise AssertionError("Expected invalid y_exp to fail")


def test_prepare_reload_plot_data_recomputes_tau_grid() -> None:
    tau_samples = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
    tau_emp, ecdf = step6._empirical_cdf(tau_samples)
    tau_grid_old = np.linspace(0.8, 3.0, 8)
    npz_path = Path("/tmp/test_prepare_reload_plot_data_recomputes_tau_grid.npz")
    np.savez(
        npz_path,
        tau_samples=tau_samples,
        tau_empirical=tau_emp,
        cdf_empirical=ecdf,
        tau_grid=tau_grid_old,
        cdf_analytical=np.linspace(0.0, 1.0, tau_grid_old.size),
        cdf_lognormal=np.linspace(0.1, 0.9, tau_grid_old.size),
        cdf_mc_mean=np.linspace(0.0, 1.0, tau_grid_old.size),
        cdf_mc_band_low=np.linspace(0.0, 0.8, tau_grid_old.size),
        cdf_mc_band_high=np.linspace(0.2, 1.0, tau_grid_old.size),
        tau_samples_by_mc=np.asarray([tau_samples[:2], tau_samples[2:]], dtype=object),
        length=np.asarray(180.0),
        velocity=np.asarray(0.1),
        i_y=np.asarray(10.0),
        var=np.asarray(1.0),
        band_low=np.asarray(5.0),
        band_high=np.asarray(95.0),
    )
    try:
        plot_data = step6._prepare_reload_plot_data(
            npz_path,
            refine_tau_grid=True,
            tau_split=1.5,
            tau_grid_fine=32,
            tau_grid_coarse=8,
        )
    finally:
        npz_path.unlink(missing_ok=True)
    assert plot_data.note is None
    assert plot_data.tau_grid.size != tau_grid_old.size
    assert np.all(np.diff(plot_data.tau_grid) > 0.0)


def test_prepare_reload_plot_data_uses_summary_json_for_old_npz(tmp_path: Path) -> None:
    tau_samples = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
    tau_emp, ecdf = step6._empirical_cdf(tau_samples)
    tau_grid_old = np.linspace(0.8, 3.0, 8)
    npz_path = tmp_path / "demo_arrival_times.npz"
    json_path = tmp_path / "demo_arrival_summary.json"
    np.savez(
        npz_path,
        tau_samples=tau_samples,
        tau_empirical=tau_emp,
        cdf_empirical=ecdf,
        tau_grid=tau_grid_old,
        cdf_analytical=np.linspace(0.0, 1.0, tau_grid_old.size),
        cdf_lognormal=np.linspace(0.1, 0.9, tau_grid_old.size),
        cdf_mc_mean=np.linspace(0.0, 1.0, tau_grid_old.size),
        cdf_mc_band_low=np.linspace(0.0, 0.8, tau_grid_old.size),
        cdf_mc_band_high=np.linspace(0.2, 1.0, tau_grid_old.size),
        tau_samples_by_mc=np.asarray([tau_samples[:2], tau_samples[2:]], dtype=object),
    )
    json_path.write_text(
        json.dumps(
            {
                "length": 180.0,
                "velocity": 0.1,
                "i_y": 10.0,
                "var": 1.0,
                "band_low": 5.0,
                "band_high": 95.0,
            }
        ),
        encoding="utf-8",
    )
    plot_data = step6._prepare_reload_plot_data(
        npz_path,
        refine_tau_grid=True,
        tau_split=1.5,
        tau_grid_fine=32,
        tau_grid_coarse=8,
    )
    assert plot_data.note is None
    assert plot_data.tau_grid.size != tau_grid_old.size


def test_prepare_reload_plot_data_falls_back_without_metadata(tmp_path: Path) -> None:
    tau_samples = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
    tau_emp, ecdf = step6._empirical_cdf(tau_samples)
    tau_grid_old = np.linspace(0.8, 3.0, 8)
    npz_path = tmp_path / "legacy_arrival_times.npz"
    np.savez(
        npz_path,
        tau_samples=tau_samples,
        tau_empirical=tau_emp,
        cdf_empirical=ecdf,
        tau_grid=tau_grid_old,
        cdf_analytical=np.linspace(0.0, 1.0, tau_grid_old.size),
        cdf_lognormal=np.linspace(0.1, 0.9, tau_grid_old.size),
        cdf_mc_mean=np.linspace(0.0, 1.0, tau_grid_old.size),
        cdf_mc_band_low=np.linspace(0.0, 0.8, tau_grid_old.size),
        cdf_mc_band_high=np.linspace(0.2, 1.0, tau_grid_old.size),
    )
    plot_data = step6._prepare_reload_plot_data(
        npz_path,
        refine_tau_grid=True,
        tau_split=1.5,
        tau_grid_fine=32,
        tau_grid_coarse=8,
    )
    assert plot_data.note is not None
    assert np.allclose(plot_data.tau_grid, tau_grid_old)


def test_prepare_reload_pdf_data_recomputes_from_saved_samples(tmp_path: Path) -> None:
    tau_samples = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
    tau_emp, ecdf = step6._empirical_cdf(tau_samples)
    bin_edges = step6._build_pdf_hist_bin_edges(tau_samples, pdf_hist_bins=6)
    npz_path = tmp_path / "demo_arrival_times.npz"
    np.savez(
        npz_path,
        tau_samples=tau_samples,
        tau_empirical=tau_emp,
        cdf_empirical=ecdf,
        tau_grid=np.linspace(0.8, 3.0, 8),
        cdf_analytical=np.linspace(0.0, 1.0, 8),
        cdf_lognormal=np.linspace(0.1, 0.9, 8),
        cdf_mc_mean=np.linspace(0.0, 1.0, 8),
        cdf_mc_band_low=np.linspace(0.0, 0.8, 8),
        cdf_mc_band_high=np.linspace(0.2, 1.0, 8),
        tau_samples_by_mc=np.asarray([tau_samples[:2], tau_samples[2:]], dtype=object),
        length=np.asarray(180.0),
        velocity=np.asarray(0.1),
        i_y=np.asarray(10.0),
        var=np.asarray(1.0),
        band_low=np.asarray(5.0),
        band_high=np.asarray(95.0),
        pdf_hist_bins=np.asarray(6),
        pdf_hist_range_low=np.asarray(float(bin_edges[0])),
        pdf_hist_range_high=np.asarray(float(bin_edges[-1])),
        pdf_hist_bin_edges=np.asarray(bin_edges),
    )
    pdf_data = step6._prepare_reload_pdf_data(npz_path)
    assert pdf_data.note is None
    assert pdf_data.tau_centers.size == bin_edges.size - 1
    assert np.all(np.isfinite(pdf_data.empirical_pdf))
    assert np.all(pdf_data.empirical_pdf >= 0.0)


def test_prepare_reload_pdf_data_skips_without_saved_pdf_metadata(tmp_path: Path) -> None:
    tau_samples = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
    tau_emp, ecdf = step6._empirical_cdf(tau_samples)
    npz_path = tmp_path / "legacy_arrival_times.npz"
    np.savez(
        npz_path,
        tau_samples=tau_samples,
        tau_empirical=tau_emp,
        cdf_empirical=ecdf,
        tau_grid=np.linspace(0.8, 3.0, 8),
        cdf_analytical=np.linspace(0.0, 1.0, 8),
        cdf_lognormal=np.linspace(0.1, 0.9, 8),
        cdf_mc_mean=np.linspace(0.0, 1.0, 8),
        cdf_mc_band_low=np.linspace(0.0, 0.8, 8),
        cdf_mc_band_high=np.linspace(0.2, 1.0, 8),
        tau_samples_by_mc=np.asarray([tau_samples[:2], tau_samples[2:]], dtype=object),
        length=np.asarray(180.0),
        velocity=np.asarray(0.1),
        i_y=np.asarray(10.0),
        var=np.asarray(1.0),
        band_low=np.asarray(5.0),
        band_high=np.asarray(95.0),
    )
    pdf_data = step6._prepare_reload_pdf_data(npz_path)
    assert pdf_data.note is not None


def test_lognormal_cdf_on_tau_grid_is_monotone_and_bounded() -> None:
    tau_grid = np.linspace(0.0, 3.0, 64)
    cdf_lognormal, summary = step6._lognormal_cdf_on_tau_grid(
        tau_grid,
        length=180.0,
        velocity=0.1,
        i_y=10.0,
        sigma_y_sq=1.0,
    )
    assert cdf_lognormal.shape == tau_grid.shape
    assert np.all(np.isfinite(cdf_lognormal))
    assert np.all(cdf_lognormal >= 0.0)
    assert np.all(cdf_lognormal <= 1.0)
    assert np.all(np.diff(cdf_lognormal) >= -1e-12)
    assert summary["lognormal_mean_tau"] > 0.0
    assert summary["lognormal_var_tau"] >= 0.0


def test_lognormal_cdf_on_tau_grid_raises_for_invalid_velocity() -> None:
    tau_grid = np.linspace(0.0, 1.0, 8)
    try:
        step6._lognormal_cdf_on_tau_grid(
            tau_grid,
            length=1.0,
            velocity=0.0,
            i_y=1.0,
            sigma_y_sq=1.0,
        )
    except ValueError as exc:
        assert "Invalid velocity" in str(exc)
    else:
        raise AssertionError("Expected invalid velocity to fail")


def test_validate_particle_tracking_options_rejects_pollock_cell_mode() -> None:
    try:
        step6._validate_particle_tracking_options(solver="pollock", velocity_location="cell")
    except ValueError as exc:
        assert "Pollock solver requires" in str(exc)
    else:
        raise AssertionError("Expected Pollock validation to reject cell velocity mode")


def test_run_step6_figure_only_with_mocked_simulation(pipeline_fixture, monkeypatch, tmp_path: Path) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    monkeypatch.setattr(step6, "simulate_single_field_arrivals", lambda **kwargs: _mock_simulation_result())

    fig_path, pdf_fig_path, npz_path, json_path = step6.run_step6(
        cfg,
        field_size=200,
        length_scale=0.05,
        std=1.0,
        var=None,
        minpadding=0,
        dh=0.1,
        k_g=1.0,
        solver="pollock",
        velocity_location="face",
        align_with_modflow_velocity=False,
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
        refine_tau_grid=True,
        tau_split=10.0,
        tau_grid_fine=1024,
        tau_grid_coarse=256,
        y_exp=1.0,
        output_dir=tmp_path,
        figure_dpi=120,
        pdf_hist_bins=48,
        save_pdf_plot=False,
        save_npz=False,
        save_json=False,
    )

    assert fig_path.exists()
    assert pdf_fig_path is None
    assert npz_path is None
    assert json_path is None
    assert fig_path.name.endswith("_arrival_cdf_vs_analytical.png")
    assert "_mc3_" in fig_path.name
    assert not list(tmp_path.glob("*_arrival_times.npz"))
    assert not list(tmp_path.glob("*_arrival_summary.json"))

def test_run_step6_saves_npz_and_json_when_enabled(pipeline_fixture, monkeypatch, tmp_path: Path) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    monkeypatch.setattr(step6, "simulate_single_field_arrivals", lambda **kwargs: _mock_simulation_result())

    fig_path, pdf_fig_path, npz_path, json_path = step6.run_step6(
        cfg,
        field_size=200,
        length_scale=0.05,
        std=1.0,
        var=None,
        minpadding=0,
        dh=0.1,
        k_g=1.0,
        solver="pollock",
        velocity_location="face",
        align_with_modflow_velocity=False,
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
        refine_tau_grid=True,
        tau_split=10.0,
        tau_grid_fine=1024,
        tau_grid_coarse=256,
        y_exp=1.0,
        output_dir=tmp_path,
        figure_dpi=120,
        pdf_hist_bins=48,
        save_pdf_plot=True,
        save_npz=True,
        save_json=True,
    )

    assert fig_path.exists()
    assert pdf_fig_path is not None and pdf_fig_path.exists()
    assert npz_path is not None and npz_path.exists()
    assert json_path is not None and json_path.exists()
    assert pdf_fig_path.name.endswith("_arrival_pdf_vs_analytical.png")
    assert npz_path.name.endswith("_arrival_times.npz")
    assert json_path.name.endswith("_arrival_summary.json")

    with np.load(npz_path) as payload:
        assert "tau_samples" in payload
        assert "cdf_empirical" in payload
        assert "cdf_analytical" in payload
        assert "cdf_lognormal" in payload
        assert "cdf_mc_mean" in payload
        assert "cdf_mc_band_low" in payload
        assert "cdf_mc_band_high" in payload
        assert "pt_solver" in payload
        assert "pt_velocity_location" in payload
        assert "pt_align_with_modflow_velocity" in payload
        assert "length" in payload
        assert "velocity" in payload
        assert "i_y" in payload
        assert "var" in payload
        assert "band_low" in payload
        assert "band_high" in payload
        assert "pdf_hist_bins" in payload
        assert "pdf_hist_range_low" in payload
        assert "pdf_hist_range_high" in payload
        assert "pdf_hist_bin_edges" in payload
        assert "seeds_used" in payload
        assert np.all(payload["seeds_used"] == np.asarray([7, 8], dtype=np.int32))
        assert payload["pt_solver"].tolist() == "pollock"
        assert payload["pt_velocity_location"].tolist() == "face"
        assert payload["pt_align_with_modflow_velocity"].tolist() is False
        assert payload["band_low"].tolist() == 5.0
        assert payload["band_high"].tolist() == 95.0
        assert payload["pdf_hist_bins"].tolist() == 48

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["n_mc_requested"] == 2
    assert summary["n_mc_used"] == 2
    assert summary["seeds_used"] == [7, 8]
    assert summary["n_particles"] == 10
    assert summary["n_success"] == 8
    assert summary["std"] == 1.0
    assert summary["pt_solver"] == "pollock"
    assert summary["pt_velocity_location"] == "face"
    assert summary["pt_align_with_modflow_velocity"] is False
    assert summary["band_low"] == 5.0
    assert summary["band_high"] == 95.0
    assert summary["pdf_hist_bins"] == 48
    assert "lognormal_mean_tau" in summary
    assert "lognormal_var_tau" in summary
    assert "lognormal_mu" in summary
    assert "lognormal_sigma_sq" in summary


def test_load_arrival_npz_and_visualize_round_trip(pipeline_fixture, monkeypatch, tmp_path: Path) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    monkeypatch.setattr(step6, "simulate_single_field_arrivals", lambda **kwargs: _mock_simulation_result())

    _fig_path, _pdf_fig_path, npz_path, _json_path = step6.run_step6(
        cfg,
        field_size=200,
        length_scale=0.05,
        std=1.0,
        var=None,
        minpadding=0,
        dh=0.1,
        k_g=1.0,
        solver="pollock",
        velocity_location="face",
        align_with_modflow_velocity=False,
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
        refine_tau_grid=True,
        tau_split=10.0,
        tau_grid_fine=1024,
        tau_grid_coarse=256,
        y_exp=0.3,
        output_dir=tmp_path,
        figure_dpi=120,
        pdf_hist_bins=48,
        save_pdf_plot=False,
        save_npz=True,
        save_json=False,
    )

    assert npz_path is not None
    saved = step6.load_arrival_npz(npz_path)
    assert saved.pt_solver == "pollock"
    assert saved.pt_velocity_location == "face"
    assert saved.pt_align_with_modflow_velocity is False
    assert saved.seeds_used is not None
    assert np.all(saved.seeds_used == np.asarray([7, 8], dtype=np.int32))
    assert saved.tau_samples_by_mc is not None
    assert len(saved.tau_samples_by_mc) == 2
    assert saved.length is not None
    assert saved.velocity is not None
    assert saved.i_y is not None
    assert saved.var == 1.0
    assert saved.band_low == 5.0
    assert saved.band_high == 95.0
    assert saved.pdf_hist_bins == 48
    assert saved.pdf_hist_bin_edges is not None

    fig_path, pdf_fig_path = step6.visualize_saved_arrival_npz(
        npz_path,
        output_dir=tmp_path,
        figure_dpi=144,
        refine_tau_grid=False,
        tau_split=10.0,
        tau_grid_fine=16,
        tau_grid_coarse=8,
        y_exp=0.3,
        save_pdf_plot=True,
    )
    assert fig_path.exists()
    assert fig_path.name.endswith("_reloaded.png")
    assert pdf_fig_path is not None and pdf_fig_path.exists()
    assert pdf_fig_path.name.endswith("_reloaded_pdf.png")


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
            solver="pollock",
            velocity_location="face",
            align_with_modflow_velocity=False,
            density_particle=20,
            n_mc=2,
            seed_start=0,
            band_low=5.0,
            band_high=95.0,
            refine_tau_grid=True,
            tau_split=10.0,
            tau_grid_fine=1024,
            tau_grid_coarse=256,
            y_exp=1.0,
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
            pdf_hist_bins=48,
            save_pdf_plot=False,
            save_npz=False,
            save_json=False,
        )
    except RuntimeError as exc:
        assert "No valid successful particle arrivals across all Monte Carlo realizations" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for all-empty MC arrivals")
