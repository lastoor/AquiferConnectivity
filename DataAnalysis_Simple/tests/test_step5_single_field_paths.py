from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import step1_index as step1
import step5_single_field_paths as step5


def _mock_result() -> step5.PathSolveResult:
    fp = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 3.0]], dtype=float)
    fp_fmm = np.array([[0.0, 1.0], [2.0, 2.2], [4.0, 3.2]], dtype=float)
    fp_dijkstra = np.array([[0.0, 0.8], [2.0, 1.8], [4.0, 2.8]], dtype=float)
    lrp = np.array([[0.0, 0.2], [2.0, 1.7], [4.0, 2.7]], dtype=float)
    lrp_fmm = np.array([[0.0, 0.4], [2.0, 1.9], [4.0, 2.9]], dtype=float)
    t_min = np.ones((8, 8), dtype=float)
    x = np.arange(0, 8, 1.0)
    y = np.arange(0, 8, 1.0)
    vx = np.ones((8, 8), dtype=float)
    vy = np.zeros((8, 8), dtype=float)
    source_xy = np.column_stack([np.zeros(6), np.arange(1, 7)])
    target_xy = np.column_stack([np.full(7, 7.0), np.arange(0, 7)])
    return step5.PathSolveResult(
        fp=fp,
        fp_fmm=fp_fmm,
        fp_dijkstra=fp_dijkstra,
        lrp=lrp,
        lrp_fmm=lrp_fmm,
        t_min=t_min,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        source_xy=source_xy,
        target_xy=target_xy,
    )


def test_select_run_input_and_load_logk(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    selected = step5.select_run_input(cfg, sweep="variance", value=0.1, field_size=200, mc_index=1)

    assert selected.run_paths.size_str == "200"
    assert selected.mc_label == 1
    field = step5.load_selected_logk(selected)
    assert field.shape == (8, 8)


def test_run_step5_saves_three_figures_with_mock_solver(pipeline_fixture, monkeypatch, tmp_path: Path) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])

    monkeypatch.setattr(step5, "solve_single_field", lambda *args, **kwargs: _mock_result())

    logk_path, tmin_path, stream_path = step5.run_step5(
        cfg,
        sweep="variance",
        value=0.1,
        field_size=200,
        mc_index=1,
        output_dir=tmp_path,
        dh=None,
        solver="pollock",
        velocity_location="face",
        align_with_modflow_velocity=False,
        fmm_velocity_layout="face",
        fmm_face_mapping="to_cell_avg",
        fmm_zero_policy="obstacle",
        fmm_min_speed=1e-12,
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
    )

    assert logk_path.exists()
    assert tmin_path.exists()
    assert stream_path.exists()
    assert logk_path.name.endswith("_logk_paths.png")
    assert tmin_path.name.endswith("_arrival_paths.png")
    assert stream_path.name.endswith("_speed_streamlines.png")


def test_validate_particle_tracking_options_rejects_pollock_cell_mode() -> None:
    try:
        step5._validate_particle_tracking_options(solver="pollock", velocity_location="cell")
    except ValueError as exc:
        assert "Pollock solver requires" in str(exc)
    else:
        raise AssertionError("Expected Pollock validation to reject cell velocity mode")


def test_plot_paths_over_field_returns_figure() -> None:
    fig = step5.plot_paths_over_field(
        np.zeros((8, 8), dtype=float),
        cmap="viridis",
        colorbar_label="field",
        result=_mock_result(),
        i_y=1.0,
    )
    assert fig.axes
    ax = fig.axes[0]
    assert ax.get_title() == ""
    assert ax.get_xlabel() == r"$x/I_Y$"
    assert ax.get_ylabel() == r"$y/I_Y$"
    labels = ax.get_legend_handles_labels()[1]
    assert "Source" in labels
    assert "Target" in labels
    plt.close(fig)


def test_dimensionless_velocity_and_stream_colorbar_label() -> None:
    vx = np.array([[2.0, 4.0]], dtype=float)
    vy = np.array([[0.0, 6.0]], dtype=float)
    vx_star, vy_star = step5._dimensionless_velocity(vx, vy, q_ref=2.0)
    assert np.allclose(vx_star, np.array([[1.0, 2.0]], dtype=float))
    assert np.allclose(vy_star, np.array([[0.0, 3.0]], dtype=float))

    result = _mock_result()
    vx_plot = np.ones((len(result.y), len(result.x)), dtype=float)
    vy_plot = np.zeros((len(result.y), len(result.x)), dtype=float)
    speed_plot = np.hypot(vx_plot, vy_plot)
    fig = step5.plot_paths_over_speed_with_streamlines(
        speed_plot,
        vx_plot,
        vy_plot,
        result=result,
        i_y=1.0,
    )
    # colorbar axis is appended after main axis
    assert fig.axes[1].get_ylabel() == r"$|v|/q_{\mathrm{ref}}$"
    plt.close(fig)
