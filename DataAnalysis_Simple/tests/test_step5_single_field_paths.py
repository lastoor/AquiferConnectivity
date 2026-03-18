from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import step1_index as step1
import step5_single_field_paths as step5


def _mock_result() -> step5.PathSolveResult:
    fp = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 3.0]], dtype=float)
    fp_fmm = np.array([[0.0, 1.0], [2.0, 2.2], [4.0, 3.2]], dtype=float)
    lrp_fmm = np.array([[0.0, 0.4], [2.0, 1.9], [4.0, 2.9]], dtype=float)
    linearized_fp_paths = [
        np.array([[0.0, 0.5], [2.0, 2.0], [4.0, 3.0]], dtype=float),
        np.array([[0.0, 0.7], [2.0, 2.1], [4.0, 3.1]], dtype=float),
    ]
    t_min = np.ones((8, 8), dtype=float)
    h = np.ones((8, 8), dtype=float)
    x = np.arange(0, 8, 1.0)
    y = np.arange(0, 8, 1.0)
    vx = np.ones((8, 8), dtype=float)
    vy = np.zeros((8, 8), dtype=float)
    source_xy = np.column_stack([np.zeros(6), np.arange(1, 7)])
    target_xy = np.column_stack([np.full(8, 7.0), np.arange(0, 8)])
    return step5.PathSolveResult(
        fp=fp,
        fp_fmm=fp_fmm,
        lrp_fmm=lrp_fmm,
        t_min=t_min,
        linearized_fp_paths=linearized_fp_paths,
        linearized_fp_indices=[1, 2],
        linearized_fp_final=linearized_fp_paths[-1],
        linearized_fp_history=[{"iter": 1.0, "residual_inf": 0.1}, {"iter": 2.0, "residual_inf": 0.01}],
        linearized_fp_failed_indices=[],
        linearized_fp_converged=True,
        h=h,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        source_xy=source_xy,
        target_xy=target_xy,
    )


def _mock_result_no_linearized_paths() -> step5.PathSolveResult:
    result = _mock_result()
    result.linearized_fp_paths = []
    result.linearized_fp_indices = []
    result.linearized_fp_final = None
    result.linearized_fp_failed_indices = [1, 2]
    result.linearized_fp_converged = False
    return result


def test_select_run_input_and_load_logk(pipeline_fixture) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    selected = step5.select_run_input(cfg, sweep="variance", value=0.1, field_size=200, mc_index=1)

    assert selected.run_paths.size_str == "200"
    assert selected.mc_label == 1
    field = step5.load_selected_logk(selected)
    assert field.shape == (8, 8)


def test_run_step5_saves_expected_figures_with_mock_solver(pipeline_fixture, monkeypatch, tmp_path: Path) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])

    monkeypatch.setattr(step5, "solve_single_field", lambda *args, **kwargs: _mock_result())

    logk_path, tmin_path, stream_path, balance_path, progression_path = step5.run_step5(
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
        fmm_backtracking="discrete",
        fmm_continuous_step_size=0.5,
        linearized_fp=True,
        linearized_scheme="linearized",
        linearized_outer_max=5,
        linearized_inner_max_sweeps=10,
        linearized_inner_tol=1e-6,
        linearized_eikonal_tol=1e-5,
        linearized_damping=0.8,
        linearized_max_deltat=None,
        linearized_monotone_residual=True,
        linearized_verbose=False,
        check_path_geometry=True,
        curvature_method="savgol",
        curvature_savgol_window=11,
        curvature_savgol_poly=3,
        curvature_circle_window=11,
        density_particle=20,
        num_particles=None,
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
    assert balance_path.exists()
    assert progression_path is not None
    assert progression_path.exists()
    assert logk_path.name.endswith("_logk_paths.png")
    assert tmin_path.name.endswith("_arrival_paths.png")
    assert stream_path.name.endswith("_speed_streamlines.png")
    assert balance_path.name.endswith("_curvature_balance.png")
    assert progression_path.name.endswith("_linearized_fp_progression.png")


def test_validate_particle_tracking_options_rejects_pollock_cell_mode() -> None:
    try:
        step5._validate_particle_tracking_options(solver="pollock", velocity_location="cell", backend="cpu")
    except ValueError as exc:
        assert "Pollock solver requires" in str(exc)
    else:
        raise AssertionError("Expected Pollock validation to reject cell velocity mode")


def test_validate_particle_tracking_options_rejects_rk45_gpu_mode() -> None:
    try:
        step5._validate_particle_tracking_options(solver="rk45", velocity_location="cell", backend="gpu")
    except ValueError as exc:
        assert "GPU backend" in str(exc)
    else:
        raise AssertionError("Expected RK45 validation to reject GPU backend")


def test_maybe_require_path_valid_can_skip_validation(monkeypatch) -> None:
    class _FakeSimPath:
        @staticmethod
        def is_path_valid(path, domain_box, len_min):
            return False

    step5._maybe_require_path_valid(
        _FakeSimPath(),
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        (0, 10, 0, 10),
        0,
        10,
        "dummy",
        check_path_geometry=False,
    )


def test_parse_args_accepts_no_check_path_geometry(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "step5_single_field_paths.py",
            "--config",
            "configs/visualization.default.yaml",
            "--sweep",
            "variance",
            "--value",
            "0.1",
            "--field-size",
            "200",
            "--mc-index",
            "0",
            "--no-check-path-geometry",
        ],
    )
    args = step5.parse_args()
    assert args.check_path_geometry is False


def test_parse_args_accepts_geometry_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "step5_single_field_paths.py",
            "--config",
            "configs/visualization.default.yaml",
            "--sweep",
            "variance",
            "--value",
            "0.1",
            "--field-size",
            "200",
            "--mc-index",
            "0",
            "--source-x",
            "10",
            "--target-x",
            "190",
            "--source-y-start",
            "20",
            "--source-y-end",
            "180",
            "--target-y-start",
            "0",
            "--target-y-end",
            "200",
        ],
    )
    args = step5.parse_args()
    assert args.source_x == 10
    assert args.target_x == 190
    assert args.source_y_start == 20
    assert args.source_y_end == 180
    assert args.target_y_start == 0
    assert args.target_y_end == 200


def test_parse_args_accepts_linearized_scheme(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "step5_single_field_paths.py",
            "--config",
            "configs/visualization.default.yaml",
            "--sweep",
            "variance",
            "--value",
            "0.1",
            "--field-size",
            "200",
            "--mc-index",
            "0",
            "--linearized-fp",
            "--linearized-scheme",
            "full_eikonal",
        ],
    )
    args = step5.parse_args()
    assert args.linearized_scheme == "full_eikonal"


def test_resolve_particle_count_prefers_exact_override() -> None:
    count = step5._resolve_particle_count(num_particles=1000, density_particle=20, y0=10, y1=50)
    assert count == 1000


def test_build_particle_seed_y_respects_half_open_source_span() -> None:
    seeds = step5._build_particle_seed_y(y_start=10, y_end=14, num_particles=4)
    assert np.all(seeds >= 10.0)
    assert np.all(seeds < 14.0)
    assert np.allclose(seeds, np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32))


def test_resolve_geometry_builds_shared_source_target_geometry() -> None:
    geometry = step5._resolve_geometry(
        field_size=20,
        source_x=2,
        target_x=17,
        source_y_start=3,
        source_y_end=9,
        target_y_start=0,
        target_y_end=20,
    )
    assert geometry.source_x == 2
    assert geometry.target_x == 17
    assert np.array_equal(geometry.source_rows, np.arange(3, 9, dtype=int))
    assert np.array_equal(geometry.target_rows, np.arange(0, 20, dtype=int))
    assert geometry.source_xy.shape == (6, 2)
    assert geometry.target_xy.shape == (20, 2)


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
    assert "FP (Particle)" in labels
    assert "FP (Dijkstra)" not in labels
    assert "LRP" not in labels
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


def test_plot_linearized_fp_progression_returns_figure() -> None:
    fig = step5.plot_linearized_fp_progression(
        np.ones((8, 8), dtype=float),
        cmap="viridis",
        colorbar_label="field",
        result=_mock_result(),
        i_y=1.0,
    )
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert "LRP (FMM)" in labels
    assert "FP (FMM)" in labels
    assert "FP (Particle)" in labels
    plt.close(fig)


def test_plot_linearized_fp_progression_returns_placeholder_for_empty_series() -> None:
    fig = step5.plot_linearized_fp_progression(
        np.ones((8, 8), dtype=float),
        cmap="viridis",
        colorbar_label="field",
        result=_mock_result_no_linearized_paths(),
        i_y=1.0,
    )
    labels = fig.axes[0].get_legend_handles_labels()[1]
    texts = [text.get_text() for text in fig.axes[0].texts]
    assert "LRP (FMM)" in labels
    assert "FP (FMM)" in labels
    assert "FP (Particle)" in labels
    assert r"No linearized FP$^i$ paths extracted" in texts
    plt.close(fig)


def test_extract_linearized_path_from_t_field_descends_to_source() -> None:
    t_field = np.array(
        [
            [5.2, 4.2, 3.2, 2.2, 1.2, 0.2],
            [5.1, 4.1, 3.1, 2.1, 1.1, 0.1],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [5.1, 4.1, 3.1, 2.1, 1.1, 0.1],
            [5.2, 4.2, 3.2, 2.2, 1.2, 0.2],
            [5.3, 4.3, 3.3, 2.3, 1.3, 0.3],
        ],
        dtype=float,
    )
    geometry = step5._resolve_geometry(
        field_size=6,
        source_x=0,
        target_x=5,
        source_y_start=1,
        source_y_end=4,
        target_y_start=1,
        target_y_end=4,
    )
    source_mask, target_mask, traversable_mask = step5._build_source_target_masks(
        field_shape=t_field.shape,
        source_rows=geometry.source_rows,
        target_rows=geometry.target_rows,
        x0=geometry.source_x,
        x1=geometry.target_x,
        traversable_mask=np.ones_like(t_field, dtype=bool),
    )
    path = step5._extract_linearized_path_from_t_field(
        t_field,
        source_mask=source_mask,
        target_mask=target_mask,
        traversable_mask=traversable_mask,
    )
    assert path.shape[0] >= 2
    assert int(path[0, 0]) == geometry.source_x
    assert int(path[-1, 0]) == geometry.target_x
    assert int(path[0, 1]) in geometry.source_rows
    assert int(path[-1, 1]) in geometry.target_rows


def test_extract_linearized_path_from_t_field_skips_dead_end_target_minimum() -> None:
    t_field = np.array(
        [
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            [10.0, 8.0, 7.0, 6.0, 5.0, 0.0],
            [10.0, 7.0, 6.0, 5.0, 4.0, 4.0],
            [10.0, 6.0, 5.0, 4.0, 3.0, 3.0],
            [10.0, 7.0, 6.0, 5.0, 4.0, 6.0],
            [10.0, 8.0, 7.0, 6.0, 5.0, 7.0],
        ],
        dtype=float,
    )
    geometry = step5._resolve_geometry(
        field_size=6,
        source_x=0,
        target_x=5,
        source_y_start=1,
        source_y_end=4,
        target_y_start=1,
        target_y_end=4,
    )
    source_mask, target_mask, traversable_mask = step5._build_source_target_masks(
        field_shape=t_field.shape,
        source_rows=geometry.source_rows,
        target_rows=geometry.target_rows,
        x0=geometry.source_x,
        x1=geometry.target_x,
        traversable_mask=np.ones_like(t_field, dtype=bool),
    )
    path = step5._extract_linearized_path_from_t_field(
        t_field,
        source_mask=source_mask,
        target_mask=target_mask,
        traversable_mask=traversable_mask,
    )
    assert path.shape[0] >= 2
    assert int(path[-1, 0]) == geometry.target_x
    assert int(path[-1, 1]) != 1


def test_extract_linearized_path_from_start_cell_prefers_drop_per_distance() -> None:
    t_field = np.array(
        [
            [8.9, 20.0, 20.0],
            [9.2, 10.0, 20.0],
            [20.0, 20.0, 20.0],
        ],
        dtype=float,
    )
    source_mask = np.zeros_like(t_field, dtype=bool)
    source_mask[1, 0] = True
    source_mask[0, 0] = True
    traversable_mask = np.ones_like(t_field, dtype=bool)

    path = step5._extract_linearized_path_from_start_cell(
        t_field,
        start_cell=(1, 1),
        source_mask=source_mask,
        traversable_mask=traversable_mask,
    )

    np.testing.assert_allclose(path[0], np.array([0.0, 1.0]))
    np.testing.assert_allclose(path[-1], np.array([1.0, 1.0]))


def test_compute_raw_geometry_uses_full_curvature_denominator() -> None:
    curvature_module = step5._import_curvature_module()
    t = np.linspace(-1.0, 1.0, 21)
    points = np.column_stack([t, t * t])

    tangent, normal, curvature = curvature_module.compute_raw_geometry(points)

    s = curvature_module.arclength_parameter(points)
    xs = np.gradient(points[:, 0], s, edge_order=2)
    ys = np.gradient(points[:, 1], s, edge_order=2)
    xss = np.gradient(xs, s, edge_order=2)
    yss = np.gradient(ys, s, edge_order=2)
    expected = (xs * yss - ys * xss) / np.maximum((xs * xs + ys * ys) ** 1.5, 1e-14)

    np.testing.assert_allclose(curvature, expected)
    assert tangent.shape == points.shape
    assert normal.shape == points.shape


def test_compute_linearized_fp_series_keeps_partial_paths_when_late_iterate_fails() -> None:
    slowness = np.ones((6, 6), dtype=float)
    geometry = step5._resolve_geometry(
        field_size=6,
        source_x=0,
        target_x=5,
        source_y_start=1,
        source_y_end=4,
        target_y_start=1,
        target_y_end=4,
    )
    iterates = [
        np.zeros((6, 6), dtype=float),
        np.array(
            [
                [5.2, 4.2, 3.2, 2.2, 1.2, 0.2],
                [5.1, 4.1, 3.1, 2.1, 1.1, 0.1],
                [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                [5.1, 4.1, 3.1, 2.1, 1.1, 0.1],
                [5.2, 4.2, 3.2, 2.2, 1.2, 0.2],
                [5.3, 4.3, 3.3, 2.3, 1.3, 0.3],
            ],
            dtype=float,
        ),
        np.full((6, 6), 2.0, dtype=float),
    ]

    class _FakeLinearSolver:
        @staticmethod
        def solve_eikonal_by_relinearization(**kwargs):
            return iterates[-1], [{"iter": 1, "residual_inf": 0.2}, {"iter": 2, "residual_inf": 0.1}], iterates

    def _fake_fast_marching(slowness_field, source_mask, h, traversable_mask):
        return np.zeros_like(slowness_field, dtype=float), None
    series = step5._compute_linearized_fp_series(
        linear_solver=_FakeLinearSolver(),
        slowness_unperturbed=slowness,
        slowness_true=slowness,
        geometry=geometry,
        fast_marching_2d_regions=_fake_fast_marching,
        extract_path_source_to_target=None,
        extract_path_continuous=None,
        backtracking="discrete",
        continuous_step_size=0.5,
        outer_max=4,
        inner_max_sweeps=10,
        inner_tol=1e-6,
        eikonal_tol=1e-5,
        damping=1.0,
        max_delta_t=None,
        monotone_residual=True,
        verbose=False,
    )

    assert len(series.paths) == 1
    assert series.path_indices == [1]
    assert series.failed_path_indices == [2]
    assert series.converged is False


def test_compute_linearized_fp_series_full_eikonal_uses_fmm_extraction() -> None:
    slowness_unperturbed = np.ones((6, 6), dtype=float)
    slowness_true = np.full((6, 6), 2.0, dtype=float)
    geometry = step5._resolve_geometry(
        field_size=6,
        source_x=0,
        target_x=5,
        source_y_start=1,
        source_y_end=4,
        target_y_start=1,
        target_y_end=4,
    )
    fmm_calls: list[np.ndarray] = []
    extract_calls = {"count": 0}

    def _fake_fast_marching(slowness_field, source_mask, h, traversable_mask):
        fmm_calls.append(np.asarray(slowness_field, dtype=float).copy())
        mean_value = float(np.mean(slowness_field))
        return np.full_like(slowness_field, mean_value, dtype=float), None

    def _fake_fmm_extract(
        t_field,
        *,
        source_mask,
        target_mask,
        extract_path_source_to_target,
        extract_path_continuous,
        backtracking,
        continuous_step_size,
        traversable_mask,
    ):
        extract_calls["count"] += 1
        return np.array([[0.0, 1.0], [2.0, 2.0], [5.0, 3.0]], dtype=float)

    original_fmm_extract = step5._extract_fmm_path_from_t_field
    try:
        step5._extract_fmm_path_from_t_field = _fake_fmm_extract
        series = step5._compute_linearized_fp_series(
            linear_solver=None,
            slowness_unperturbed=slowness_unperturbed,
            slowness_true=slowness_true,
            geometry=geometry,
            fast_marching_2d_regions=_fake_fast_marching,
            extract_path_source_to_target=None,
            extract_path_continuous=None,
            scheme="full_eikonal",
            backtracking="discrete",
            continuous_step_size=0.5,
            outer_max=3,
            inner_max_sweeps=10,
            inner_tol=1e-6,
            eikonal_tol=1e-5,
            damping=1.0,
            max_delta_t=None,
            monotone_residual=True,
            verbose=False,
        )
    finally:
        step5._extract_fmm_path_from_t_field = original_fmm_extract

    assert len(fmm_calls) == 4
    assert extract_calls["count"] == 3
    assert len(series.paths) == 3
    assert series.path_indices == [1, 2, 3]
    assert series.converged is True
    assert np.allclose(fmm_calls[1], (2.0 / 3.0) * slowness_unperturbed + (1.0 / 3.0) * slowness_true)
    assert np.allclose(fmm_calls[3], slowness_true)


def test_run_step5_saves_progression_figure_when_no_linearized_paths_exist(
    pipeline_fixture,
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])

    monkeypatch.setattr(step5, "solve_single_field", lambda *args, **kwargs: _mock_result_no_linearized_paths())

    _logk_path, _tmin_path, _stream_path, _balance_path, progression_path = step5.run_step5(
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
        fmm_backtracking="discrete",
        fmm_continuous_step_size=0.5,
        linearized_fp=True,
        linearized_scheme="linearized",
        linearized_outer_max=5,
        linearized_inner_max_sweeps=10,
        linearized_inner_tol=1e-6,
        linearized_eikonal_tol=1e-5,
        linearized_damping=0.8,
        linearized_max_deltat=None,
        linearized_monotone_residual=True,
        linearized_verbose=False,
        check_path_geometry=True,
        curvature_method="savgol",
        curvature_savgol_window=11,
        curvature_savgol_poly=3,
        curvature_circle_window=11,
        density_particle=20,
        num_particles=None,
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

    assert progression_path is not None
    assert progression_path.exists()


def test_run_step5_prints_linearized_fp_extraction_summary(
    pipeline_fixture,
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])
    result = _mock_result()
    result.linearized_fp_indices = [1, 2, 4]
    result.linearized_fp_failed_indices = [3, 5]

    monkeypatch.setattr(step5, "solve_single_field", lambda *args, **kwargs: result)

    step5.run_step5(
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
        fmm_backtracking="discrete",
        fmm_continuous_step_size=0.5,
        linearized_fp=True,
        linearized_scheme="linearized",
        linearized_outer_max=5,
        linearized_inner_max_sweeps=10,
        linearized_inner_tol=1e-6,
        linearized_eikonal_tol=1e-5,
        linearized_damping=0.8,
        linearized_max_deltat=None,
        linearized_monotone_residual=True,
        linearized_verbose=False,
        check_path_geometry=True,
        curvature_method="savgol",
        curvature_savgol_window=11,
        curvature_savgol_poly=3,
        curvature_circle_window=11,
        density_particle=20,
        num_particles=None,
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

    out = capsys.readouterr().out
    assert "Linearized FP^i extraction: 3 succeeded, 2 failed" in out
    assert "Successful FP^i iterations: [1, 2, 4]" in out
    assert "Failed FP^i iterations: [3, 5]" in out
