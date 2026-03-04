from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _write_main_output(path: Path, scale: float) -> None:
    np.savez(
        path,
        len_fp_m=10.0 * scale,
        len_fp_dijkstra_m=10.5 * scale,
        len_fp_fmm_m=9.8 * scale,
        len_lrp_m=9.6 * scale,
        fpt_m=5.0 * scale,
        fpt_dijkstra_m=5.2 * scale,
        fpt_fmm_m=4.9 * scale,
        mhr_m=4.7 * scale,
        endPointDist_fp_lrp_m=0.2 * scale,
        endPointDist_fp_fpDijkstra_m=0.3 * scale,
        endPointDist_fp_fpFMM_m=0.1 * scale,
    )


def _write_path_set(path_dir: Path, size: int, mc: int, y_shift: float) -> None:
    fp = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=float)
    lrp = np.array([[0.0, 0.0], [1.0, 1.1 + y_shift], [2.0, 2.1 + y_shift]], dtype=float)
    fp_dijkstra = np.array([[0.0, 0.0], [1.0, 0.9 + y_shift], [2.0, 1.9 + y_shift]], dtype=float)
    fp_fmm = np.array([[0.0, 0.0], [1.0, 1.05 + y_shift], [2.0, 2.05 + y_shift]], dtype=float)

    np.save(path_dir / f"fp_size{size}_mc{mc}.npy", fp)
    np.save(path_dir / f"lrp_size{size}_mc{mc}.npy", lrp)
    np.save(path_dir / f"fp_dijkstra_size{size}_mc{mc}.npy", fp_dijkstra)
    np.save(path_dir / f"fp_fmm_size{size}_mc{mc}.npy", fp_fmm)


@pytest.fixture()
def pipeline_fixture(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    run_dir = data_root / "run3"

    var_dir = run_dir / "var_0.1" / "lengthScale0.05_size200"
    std_dir = run_dir / "std_1.0" / "lengthScale0.05_size200"

    for base in [var_dir, std_dir]:
        (base / "main_output").mkdir(parents=True, exist_ok=True)
        (base / "path").mkdir(parents=True, exist_ok=True)
        (base / "logk_field").mkdir(parents=True, exist_ok=True)

    for mc in [1, 2]:
        _write_main_output(var_dir / "main_output" / f"mainOutput_size200_mc{mc}.npz", scale=1.0 + 0.1 * mc)
        _write_main_output(std_dir / "main_output" / f"mainOutput_size200_mc{mc}.npz", scale=1.0 + 0.05 * mc)

        _write_path_set(var_dir / "path", 200, mc, y_shift=0.05 * mc)
        _write_path_set(std_dir / "path", 200, mc, y_shift=0.08 * mc)

        np.save(var_dir / "logk_field" / f"logKField_{mc}.npy", np.ones((8, 8), dtype=float) * (1.0 + 0.2 * mc))
        np.save(std_dir / "logk_field" / f"logKField_{mc}.npy", np.ones((8, 8), dtype=float) * (1.0 + 0.1 * mc))

    output_dir = tmp_path / "outputs"
    cache_dir = tmp_path / "cache"
    cfg = {
        "data_root": str(data_root),
        "run_id": 3,
        "length_scale": 0.05,
        "dh": 0.1,
        "field_size_ref": 200,
        "k_g": 1.0,
        "variance_values": [0.1],
        "std_values": [1.0],
        "output_dir": str(output_dir),
        "figure_dpi": 120,
        "workers": -1,
        "save_formats": ["png"],
        "sections": ["times", "endpoint", "hausdorff", "std_endpoint", "std_hausdorff", "ratio_mean", "ratio_var", "wasserstein_len"],
        "distance_mode": {
            "end_point": "load_from_main_output",
            "hausdorff": "compute_from_paths",
            "frechet": "compute_from_paths",
        },
        "distance_enabled": {
            "end_point": True,
            "hausdorff": True,
            "frechet": False,
        },
        "distance_fallback_on_missing": "compute_from_paths",
        "distance_main_output_keys": {
            "end_point": {
                "fp_lrp": "endPointDist_fp_lrp_m",
                "fp_fpDijkstra": "endPointDist_fp_fpDijkstra_m",
                "fp_fpFMM": "endPointDist_fp_fpFMM_m",
            },
            "hausdorff": {
                "fp_lrp": "hausdorffDist_fp_lrp_m",
                "fp_fpDijkstra": "hausdorffDist_fp_fpDijkstra_m",
                "fp_fpFMM": "hausdorffDist_fp_fpFMM_m",
            },
            "frechet": {
                "fp_lrp": "frechetDist_fp_lrp_m",
                "fp_fpDijkstra": "frechetDist_fp_fpDijkstra_m",
                "fp_fpFMM": "frechetDist_fp_fpFMM_m",
            },
        },
        "font": {
            "family": "DejaVu Sans",
            "mathtext_fontset": "cm",
            "unicode_minus": False,
            "cmu_font_glob": "/tmp/no-font-needed*.ttf",
        },
    }

    cfg_path = tmp_path / "cfg.yaml"
    lines = [
        f"data_root: {cfg['data_root']}",
        "run_id: 3",
        "length_scale: 0.05",
        "dh: 0.1",
        "field_size_ref: 200",
        "k_g: 1.0",
        "variance_values: [0.1]",
        "std_values: [1.0]",
        f"output_dir: {output_dir}",
        "figure_dpi: 120",
        "workers: -1",
        "save_formats: [png]",
        "sections: [times, endpoint, hausdorff, std_endpoint, std_hausdorff, ratio_mean, ratio_var, wasserstein_len]",
        "distance_mode:",
        "  end_point: load_from_main_output",
        "  hausdorff: compute_from_paths",
        "  frechet: compute_from_paths",
        "distance_enabled:",
        "  end_point: true",
        "  hausdorff: true",
        "  frechet: false",
        "distance_fallback_on_missing: compute_from_paths",
        "distance_main_output_keys:",
        "  end_point:",
        "    fp_lrp: endPointDist_fp_lrp_m",
        "    fp_fpDijkstra: endPointDist_fp_fpDijkstra_m",
        "    fp_fpFMM: endPointDist_fp_fpFMM_m",
        "  hausdorff:",
        "    fp_lrp: hausdorffDist_fp_lrp_m",
        "    fp_fpDijkstra: hausdorffDist_fp_fpDijkstra_m",
        "    fp_fpFMM: hausdorffDist_fp_fpFMM_m",
        "  frechet:",
        "    fp_lrp: frechetDist_fp_lrp_m",
        "    fp_fpDijkstra: frechetDist_fp_fpDijkstra_m",
        "    fp_fpFMM: frechetDist_fp_fpFMM_m",
        "font:",
        "  family: DejaVu Sans",
        "  mathtext_fontset: cm",
        "  unicode_minus: false",
        "  cmu_font_glob: /tmp/no-font-needed*.ttf",
    ]
    cfg_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "repo_root": REPO_ROOT,
        "cfg_path": cfg_path,
        "cache_dir": cache_dir,
        "output_dir": output_dir,
    }
