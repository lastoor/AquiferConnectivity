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
    t_min = np.ones((8, 8), dtype=float)
    return step5.PathSolveResult(fp=fp, fp_fmm=fp_fmm, fp_dijkstra=fp_dijkstra, lrp=lrp, t_min=t_min)


def test_select_run_input_and_load_logk(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    selected = step5.select_run_input(cfg, sweep="variance", value=0.1, field_size=200, mc_index=1)

    assert selected.run_paths.size_str == "200"
    assert selected.mc_label == 1
    field = step5.load_selected_logk(selected)
    assert field.shape == (8, 8)


def test_run_step5_saves_two_figures_with_mock_solver(pipeline_fixture, monkeypatch, tmp_path: Path) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])

    monkeypatch.setattr(step5, "solve_single_field", lambda *args, **kwargs: _mock_result())

    logk_path, tmin_path = step5.run_step5(
        cfg,
        sweep="variance",
        value=0.1,
        field_size=200,
        mc_index=1,
        output_dir=tmp_path,
        dh=None,
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
    assert logk_path.name.endswith("_logk_paths.png")
    assert tmin_path.name.endswith("_arrival_paths.png")


def test_plot_paths_over_field_returns_figure() -> None:
    fig = step5.plot_paths_over_field(
        np.zeros((8, 8), dtype=float),
        title="demo",
        cmap="viridis",
        colorbar_label="field",
        result=_mock_result(),
    )
    assert fig.axes
    plt.close(fig)
