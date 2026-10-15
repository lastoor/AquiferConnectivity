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
    t_min_lrp = np.ones((8, 8), dtype=float)
    h = np.ones((8, 8), dtype=float)
    x = np.arange(0, 8, 1.0)
    y = np.arange(0, 8, 1.0)
    vx = np.ones((8, 8), dtype=float)
    vy = np.zeros((8, 8), dtype=float)
    source_xy = np.column_stack([np.zeros(6), np.arange(1, 7)])
    target_xy = np.column_stack([np.full(8, 7.0), np.arange(0, 8)])
    fp_pypar2 = np.array([[0.0, 0.2], [2.0, 2.05], [4.0, 3.05]], dtype=float)
    return step5.PathSolveResult(
        fp=fp,
        fp_arrival_time=4.0,
        fp_fmm=fp_fmm,
        lrp_fmm=lrp_fmm,
        t_min=t_min,
        t_min_lrp=t_min_lrp,
        fp_fsm=None,
        t_min_fsm=None,
        fp_fm_asr=None,
        fp_fm_asr_arrival_time=None,
        t_min_fm_asr=None,
        fm_asr_M=None,
        fm_asr_beta=None,
        fp_pypar2=fp_pypar2,
        fp_pypar2_arrival_time=4.5,
        fp_pypar2_error=None,
        pypar2_replay_meta={"status": "accepted"},
        pypar2_snapshots=None,
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
        vx_cell=vx,
        vy_cell=vy,
        source_xy=source_xy,
        target_xy=target_xy,
        snapshots=None,
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

    (logk_path, tmin_path, stream_path, balance_path,
     progression_path, isochrone_plume_path) = step5.run_step5(
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
    assert isochrone_plume_path is None  # mock result has snapshots=None


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




def test_parse_args_enables_pypar2_by_default(monkeypatch) -> None:
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
        ],
    )
    args = step5.parse_args()
    assert args.pypar2 is True


def test_parse_args_accepts_pypar2_defaults_and_overrides(monkeypatch) -> None:
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
            "--pypar2",
            "--pypar2-dt",
            "0.02",
            "--pypar2-steps",
            "123",
            "--pypar2-alpha-l",
            "0.2",
            "--pypar2-alpha-t",
            "0.03",
            "--pypar2-taichi-arch",
            "cpu",
            "--pypar2-num-particles",
            "50",
        ],
    )
    args = step5.parse_args()
    assert args.pypar2 is True
    assert args.pypar2_dt == 0.02
    assert args.pypar2_steps == 123
    assert args.pypar2_dm == 1e-9
    assert args.pypar2_alpha_l == 0.2
    assert args.pypar2_alpha_t == 0.03
    assert args.pypar2_taichi_arch == "cpu"
    assert args.pypar2_num_particles == 50


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
    assert r"$\gamma^*_{\mathcal{S}}$" in labels
    assert r"$\gamma^*_{\mathcal{E}}$" in labels
    assert r"$\gamma^*_{\mathcal{R}}$" in labels
    assert r"$\gamma^*_{\mathcal{P}}$" in labels
    assert "FP (Dijkstra)" not in labels
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
    assert fig.axes[1].get_ylabel() == r"$\| \mathbf{v} \|/U$"
    plt.close(fig)


def test_plot_isochrone_contour_applies_white_halo() -> None:
    fig, ax = plt.subplots()
    x = np.linspace(0.0, 4.0, 20)
    y = np.linspace(0.0, 4.0, 20)
    xx, yy = np.meshgrid(x, y)
    field = np.hypot(xx - 2.0, yy - 2.0)
    step5._plot_isochrone_contour(
        ax,
        x,
        y,
        field,
        1.0,
        color="red",
        linewidth=1.0,
    )
    assert ax.collections
    effects = ax.collections[0].get_path_effects()
    assert effects
    stroke = effects[0]
    assert stroke._gc.get("foreground") == "white"
    plt.close(fig)


def test_plot_snapshot_particles_line_draws_dashed_polyline() -> None:
    fig, ax = plt.subplots()
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 2.0 * np.pi, 80)
    x = 3.0 + t + 0.05 * rng.standard_normal(t.size)
    y = 4.0 + 0.5 * np.sin(t) + 0.05 * rng.standard_normal(t.size)
    ok = step5._plot_snapshot_particles_line(
        ax,
        x,
        y,
        i_y=1.0,
        color="red",
        linestyle="--",
        linewidth=1.0,
    )
    assert ok is True
    lines = ax.get_lines()
    assert len(lines) == 1
    assert lines[0].get_linestyle() == "--"
    assert len(lines[0].get_xdata()) == t.size
    effects = lines[0].get_path_effects()
    assert effects
    assert effects[0]._gc.get("foreground") == "white"
    plt.close(fig)


def test_plot_snapshot_density_contour_draws_plume() -> None:
    fig, ax = plt.subplots()
    rng = np.random.default_rng(0)
    ok = step5._plot_snapshot_density_contour(
        ax,
        3.0 + 0.2 * rng.standard_normal(200),
        4.0 + 0.2 * rng.standard_normal(200),
        x_grid=np.arange(0, 8, 1.0),
        y_grid=np.arange(0, 8, 1.0),
        i_y=1.0,
        color="red",
        linestyle="-",
        linewidth=1.0,
    )
    assert ok is True
    assert ax.collections
    assert not ax.collections[0].get_edgecolors().size  # filled plume, not contour lines
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
            [0.2, 1.2, 2.2, 3.2, 4.2, 5.2],
            [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
            [0.2, 1.2, 2.2, 3.2, 4.2, 5.2],
            [0.3, 1.3, 2.3, 3.3, 4.3, 5.3],
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
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 2.0, 3.0, 4.0, 5.0, 10.0],
            [0.0, 3.0, 4.0, 5.0, 5.0, 8.0],
            [0.0, 4.0, 5.0, 6.0, 6.0, 7.0],
            [0.0, 3.0, 4.0, 5.0, 6.0, 8.0],
            [0.0, 2.0, 3.0, 4.0, 5.0, 9.0],
        ],
        dtype=float,
    )
    geometry = step5._resolve_geometry(
        field_size=6,
        source_x=0,
        target_x=5,
        source_y_start=0,
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


def test_compute_savgol_geometry_on_parabola() -> None:
    """Savitzky-Golay curvature should approximate the analytical curvature of y=x²."""
    import step2_metrics as step2

    curvature_module = step5._import_curvature_module()
    # Dense parabola, then resample at uniform arc length (as done in the pipeline)
    t = np.linspace(-1.0, 1.0, 2001)
    raw_points = np.column_stack([t, t * t])
    points = step2.resample_by_arclength(raw_points, ds=0.01)

    s, pts, tangent, normal, curvature = curvature_module.compute_savgol_geometry(
        points, window=21, poly_order=3
    )

    # Analytical curvature of y = x²: κ(x) = 2 / (1 + 4x²)^1.5
    x_pts = pts[:, 0]
    kappa_exact = 2.0 / (1.0 + 4.0 * x_pts ** 2) ** 1.5

    # Interior only (avoid edge effects of SG filter)
    interior = slice(20, -20)
    rms_savgol = float(np.sqrt(np.mean((curvature[interior] - kappa_exact[interior]) ** 2)))
    assert rms_savgol < 0.05, f"SG curvature RMS error {rms_savgol:.4f} too large"
    assert tangent.shape == pts.shape
    assert normal.shape == pts.shape
    # Normal should be perpendicular to tangent
    dot = np.einsum("ij,ij->i", tangent, normal)
    np.testing.assert_allclose(dot, 0.0, atol=1e-12)


def test_normal_derivative_direct_on_linear_field() -> None:
    """_normal_derivative_direct should be exact for a linear field f(x,y) = ax + by."""
    x = np.linspace(0.0, 10.0, 50)
    y = np.linspace(0.0, 10.0, 50)
    xx, yy = np.meshgrid(x, y)
    a, b = 2.0, 3.0
    field = a * xx + b * yy  # linear field: ∇f = (a, b) everywhere

    # Path points and normals inside the domain
    pts = np.array([[3.0, 4.0], [5.0, 5.0], [7.0, 6.0]], dtype=float)
    # Normal pointing at 45 degrees
    angle = np.pi / 4.0
    normal = np.tile([np.cos(angle), np.sin(angle)], (3, 1))

    result = step5._normal_derivative_direct(field, x, y, pts, normal)
    expected = a * normal[:, 0] + b * normal[:, 1]  # dot(∇f, n) = a*nx + b*ny

    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_savgol_gradient_2d_on_linear_field() -> None:
    """_savgol_gradient_2d should recover the exact gradient of a linear field f=ax+by."""
    x = np.linspace(0.0, 10.0, 50)
    y = np.linspace(0.0, 10.0, 50)
    xx, yy = np.meshgrid(x, y)
    a, b = 2.0, 3.0
    field = a * xx + b * yy  # ∇f = (a, b) everywhere

    grad_x, grad_y = step5._savgol_gradient_2d(field, x, y, window=11, poly=3)

    # Interior only (avoid SG edge effects)
    interior = np.s_[5:-5, 5:-5]
    np.testing.assert_allclose(grad_x[interior], a, rtol=1e-6)
    np.testing.assert_allclose(grad_y[interior], b, rtol=1e-6)

    # Path points and normals
    pts = np.array([[3.0, 4.0], [5.0, 5.0], [7.0, 6.0]], dtype=float)
    angle = np.pi / 4.0
    normal = np.tile([np.cos(angle), np.sin(angle)], (3, 1))
    expected = a * normal[:, 0] + b * normal[:, 1]

    sampled_x = step5._bilinear_sample(grad_x, x, y, pts)
    sampled_y = step5._bilinear_sample(grad_y, x, y, pts)
    result = sampled_x * normal[:, 0] + sampled_y * normal[:, 1]
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_sample_directional_curvature_on_circular_velocity_field() -> None:
    """(v_hat·grad)v_hat has unit inward curvature on the unit circle."""
    x = np.linspace(-2.0, 2.0, 101)
    y = np.linspace(-2.0, 2.0, 101)
    xx, yy = np.meshgrid(x, y)
    vx = -yy
    vy = xx

    theta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    inward_normal = -points
    curvature = step5._sample_directional_curvature(
        vx,
        vy,
        x,
        y,
        points,
        inward_normal,
    )

    np.testing.assert_allclose(curvature, 1.0, atol=3e-3)


def test_spline2d_normal_derivative_on_linear_field() -> None:
    """_spline2d_normal_derivative should be exact for a linear field f(x,y) = ax + by."""
    x = np.linspace(0.0, 10.0, 50)
    y = np.linspace(0.0, 10.0, 50)
    xx, yy = np.meshgrid(x, y)
    a, b = 2.0, 3.0
    field = a * xx + b * yy

    pts = np.array([[3.0, 4.0], [5.0, 5.0], [7.0, 6.0]], dtype=float)
    angle = np.pi / 4.0
    normal = np.tile([np.cos(angle), np.sin(angle)], (3, 1))
    expected = a * normal[:, 0] + b * normal[:, 1]

    result = step5._spline2d_normal_derivative(field, x, y, pts, normal, smooth=0.0)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_mls_frenet_normal_derivative_on_linear_field() -> None:
    """_mls_frenet_normal_derivative should recover ∂f/∂n exactly for f(x,y) = ax + by."""
    x = np.linspace(0.0, 10.0, 50)
    y = np.linspace(0.0, 10.0, 50)
    xx, yy = np.meshgrid(x, y)
    a, b = 2.0, 3.0
    field = a * xx + b * yy

    pts = np.array([[3.0, 4.0], [5.0, 5.0], [7.0, 6.0]], dtype=float)
    angle = np.pi / 4.0
    tangent = np.tile([-np.sin(angle), np.cos(angle)], (3, 1))
    normal = np.tile([np.cos(angle), np.sin(angle)], (3, 1))
    expected = a * normal[:, 0] + b * normal[:, 1]

    result = step5._mls_frenet_normal_derivative(field, x, y, pts, tangent, normal, ns=2, nn=2)
    np.testing.assert_allclose(result, expected, atol=1e-5)


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
                [0.2, 1.2, 2.2, 3.2, 4.2, 5.2],
                [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
                [0.2, 1.2, 2.2, 3.2, 4.2, 5.2],
                [0.3, 1.3, 2.3, 3.3, 4.3, 5.3],
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
    expected_first = 1.0 / ((2.0 / 3.0) * slowness_unperturbed + (1.0 / 3.0) * slowness_true)
    assert np.allclose(fmm_calls[1], expected_first)
    assert np.allclose(fmm_calls[3], 1.0 / slowness_true)


def test_run_step5_saves_progression_figure_when_no_linearized_paths_exist(
    pipeline_fixture,
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = step1.load_config(pipeline_fixture["cfg_path"])

    monkeypatch.setattr(step5, "solve_single_field", lambda *args, **kwargs: _mock_result_no_linearized_paths())

    _logk_path, _tmin_path, _stream_path, _balance_path, progression_path, _isochrone_plume_path = step5.run_step5(
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


def test_run_pypar2_tracking_dispatch_payload() -> None:
    captured = {}

    class _FakeSimPt:
        @staticmethod
        def particle_tracking_summary(params):
            captured.update(params)
            return {
                "fastest": {
                    "x": np.array([0.0, 1.0, 2.0]),
                    "y": np.array([0.5, 0.6, 0.7]),
                    "travel_time": 3.25,
                },
                "replay": {"status": "accepted"},
                "snapshots": {
                    "snapshot_times": np.array([1.0, 2.0], dtype=float),
                    "x": np.array([[0.1, 0.2], [1.1, 1.2]], dtype=float),
                    "y": np.array([[0.3, 0.4], [1.3, 1.4]], dtype=float),
                },
            }

    geometry = step5._resolve_geometry(
        field_size=4,
        source_x=0,
        target_x=3,
        source_y_start=1,
        source_y_end=3,
        target_y_start=0,
        target_y_end=4,
    )
    path, travel_time, error, replay, snapshots = step5._run_pypar2_tracking(
        sim_pt=_FakeSimPt(),
        vx_face=np.ones((4, 5), dtype=float),
        vy_face=np.zeros((5, 4), dtype=float),
        x=geometry.x,
        y=geometry.y,
        geometry=geometry,
        density_particle=2,
        num_particles=None,
        pypar2_num_particles=6,
        pypar2_dt=0.02,
        pypar2_steps=12,
        pypar2_dm=1e-8,
        pypar2_alpha_l=0.2,
        pypar2_alpha_t=0.03,
        pypar2_velocity_floor=1e-6,
        pypar2_taichi_arch="cpu",
        pypar2_max_points=100,
        pypar2_trajectory_memory_limit=1000,
        pypar2_replay_top_k=3,
        pypar2_replay_particles_per_center=7,
        pypar2_replay_radius_cells=0.25,
        pypar2_replay_tolerance=1.01,
        pypar2_replay_max_redos=2,
        pypar2_seed=99,
        snapshot_mode="physical",
        snapshot_times=[1.0, 2.0],
    )

    assert error is None
    assert travel_time == 3.25
    assert replay == {"status": "accepted"}
    assert snapshots is not None
    np.testing.assert_allclose(snapshots.times, np.array([1.0, 2.0]))
    np.testing.assert_allclose(path, np.array([[0.0, 0.5], [1.0, 0.6], [2.0, 0.7]]))
    assert captured["method"] == "pypar2"
    assert captured["backend"] == "cpu"
    assert captured["dt"] == 0.02
    assert captured["steps"] == 12
    assert captured["molecular_diffusion"] == 1e-8
    assert captured["alpha_l"] == 0.2
    assert captured["alpha_t"] == 0.03
    assert captured["longitudinal_dispersivity"] == 0.2
    assert captured["transverse_dispersivity"] == 0.03
    assert captured["velocity_floor"] == 1e-6
    assert captured["replay_top_k"] == 3
    assert captured["replay_particles_per_center"] == 7
    assert captured["seed"] == 99
    assert captured["snapshot_mode"] == "physical"
    np.testing.assert_allclose(captured["snapshot_times"], np.array([1.0, 2.0]))
    assert captured["seeds_x"].shape == (6,)
    assert captured["seeds_y"].shape == (6,)
