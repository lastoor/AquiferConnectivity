from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import step5_single_field_paths as step5
import visualize_indicatrix_paths as indicatrix


def test_grf_generation_is_reproducible_and_seeded(monkeypatch) -> None:
    class FakeCovariance:
        def __init__(self, *, sigma: float, length_scale: float) -> None:
            self.sigma = sigma
            self.length_scale = length_scale

    class FakeGenerator:
        def __init__(self, mean, cov, pts, *, minpadding: int) -> None:
            self.mean = np.asarray(mean, dtype=float)
            self.cov = cov
            self.pts = pts
            self.minpadding = minpadding

        def sample(self) -> np.ndarray:
            return np.random.normal(loc=0.0, scale=self.cov.sigma, size=self.mean.shape)

    monkeypatch.setattr(
        step5,
        "_import_grf_module",
        lambda: SimpleNamespace(
            ExpCov=FakeCovariance,
            GaussianRandomFieldCircEmbed=FakeGenerator,
        ),
    )

    first = step5.generate_grf_logk(field_size=8, std=1.2, length_scale=0.1, seed=17)
    second = step5.generate_grf_logk(field_size=8, std=1.2, length_scale=0.1, seed=17)
    third = step5.generate_grf_logk(field_size=8, std=1.2, length_scale=0.1, seed=18)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_shared_glyph_scale_includes_scalar_vector_and_finsler_radii() -> None:
    shape = (4, 4)
    speed = np.ones(shape)
    conductivity_speed = np.full(shape, 2.0)
    vx = np.full(shape, 3.0)
    vy = np.zeros(shape)
    M = np.broadcast_to(np.eye(2), shape + (2, 2)).copy()
    beta = np.zeros(shape + (2,))

    def fake_randers_points(_M, _beta, *, K):
        theta = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
        return 4.0 * np.cos(theta), 4.0 * np.sin(theta)

    scale = indicatrix.compute_shared_glyph_scale(
        speed,
        conductivity_speed,
        vx,
        vy,
        M,
        beta,
        fake_randers_points,
        stride=2,
        directions=48,
    )

    assert scale.max_radius == 4.0
    assert np.isclose(scale.value, 0.4)


def _mock_path_result(size: int = 12) -> step5.PathSolveResult:
    x = np.arange(size, dtype=float)
    y = np.arange(size, dtype=float)
    path = np.column_stack((np.linspace(1.0, size - 2.0, 8), np.linspace(2.0, size - 3.0, 8)))
    source_xy = np.column_stack((np.zeros(size - 2), np.arange(1, size - 1, dtype=float)))
    target_xy = np.column_stack((np.full(size, size - 1.0), np.arange(size, dtype=float)))
    zeros = np.zeros((size, size), dtype=float)
    vx = np.ones((size, size), dtype=float)
    vy = np.full((size, size), 0.2, dtype=float)
    return step5.PathSolveResult(
        fp=path,
        fp_arrival_time=1.0,
        fp_fmm=path + np.array([0.0, 0.1]),
        lrp_fmm=path + np.array([0.0, -0.1]),
        t_min=zeros,
        t_min_lrp=zeros,
        fp_fsm=None,
        t_min_fsm=None,
        fp_fm_asr=path + np.array([0.0, 0.2]),
        fp_fm_asr_arrival_time=1.0,
        t_min_fm_asr=zeros,
        fm_asr_M=None,
        fm_asr_beta=None,
        fp_pypar2=None,
        fp_pypar2_arrival_time=None,
        fp_pypar2_error=None,
        pypar2_replay_meta=None,
        pypar2_snapshots=None,
        linearized_fp_paths=None,
        linearized_fp_indices=None,
        linearized_fp_final=None,
        linearized_fp_history=None,
        linearized_fp_failed_indices=None,
        linearized_fp_converged=None,
        h=zeros,
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


def test_generate_and_plot_saves_four_individual_figures(monkeypatch, tmp_path: Path) -> None:
    size = 24
    logk = np.zeros((size, size), dtype=float)
    result = _mock_path_result(size)

    def fake_build_randers(W, **_kwargs):
        speed = np.maximum(np.linalg.norm(W, axis=-1), 1e-12)
        M = np.zeros(W.shape[:2] + (2, 2), dtype=float)
        M[..., 0, 0] = 1.0 / speed**2
        M[..., 1, 1] = 1.0 / speed**2
        beta = np.zeros(W.shape[:2] + (2,), dtype=float)
        directions = np.zeros_like(W)
        directions[..., 0] = 1.0
        return speed, M, beta, directions

    def fake_randers_points(M, _beta, *, K):
        radius = 1.0 / np.sqrt(float(M[0, 0]))
        theta = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
        return radius * np.cos(theta), radius * np.sin(theta)

    monkeypatch.setattr(step5, "generate_grf_logk", lambda **_kwargs: logk.copy())
    monkeypatch.setattr(step5, "solve_single_field", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        step5,
        "_import_solver_modules",
        lambda: {
            "build_randers_from_W": fake_build_randers,
            "randers_indicatrix_points": fake_randers_points,
        },
    )

    cfg = SimpleNamespace(length_scale=0.05, dh=0.1, output_dir=tmp_path, figure_dpi=80)
    outputs = indicatrix.generate_and_plot(
        cfg,
        field_size=size,
        grf_std=1.0,
        grf_length_scale=0.1,
        grf_seed=23,
        output_dir=tmp_path,
        indicatrix_stride=4,
        sample_point_stride=6,
        indicatrix_directions=48,
    )

    paths = [outputs.speed, outputs.conductivity, outputs.streamline, outputs.finsler]
    assert len(paths) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert all("seed23" in path.name for path in paths)
    assert {path.name for path in tmp_path.glob("*_indicatrix_*.png")} == {path.name for path in paths}
    assert outputs.combined_pdf is not None
    assert outputs.combined_pdf.exists() and outputs.combined_pdf.stat().st_size > 0
