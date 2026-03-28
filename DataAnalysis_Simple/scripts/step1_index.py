#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

DISTANCE_TYPES = ("end_point", "hausdorff", "frechet")
DISTANCE_PAIRS = ("fp_lrp", "fp_lrpFMM", "fp_fpDijkstra", "fp_fpFMM")
DISTANCE_MODE_VALUES = ("load_from_main_output", "compute_from_paths")
DISTANCE_FALLBACK_VALUES = ("compute_from_paths", "nan")
DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS: dict[str, dict[str, str]] = {
    "end_point": {
        "fp_lrp": "endpoint_dy_fp_vs_lrp",
        "fp_lrpFMM": "endpoint_dy_fp_vs_lrp_fmm",
        "fp_fpDijkstra": "endpoint_dy_fp_vs_dijkstra",
        "fp_fpFMM": "endpoint_dy_fp_vs_fmm",
    },
    "hausdorff": {
        "fp_lrp": "hausdorffDist_fp_lrp_m",
        "fp_lrpFMM": "hausdorffDist_fp_lrpFMM_m",
        "fp_fpDijkstra": "hausdorffDist_fp_fpDijkstra_m",
        "fp_fpFMM": "hausdorffDist_fp_fpFMM_m",
    },
    "frechet": {
        "fp_lrp": "frechetDist_fp_lrp_m",
        "fp_lrpFMM": "frechetDist_fp_lrpFMM_m",
        "fp_fpDijkstra": "frechetDist_fp_fpDijkstra_m",
        "fp_fpFMM": "frechetDist_fp_fpFMM_m",
    },
}
DISTANCE_MAIN_OUTPUT_KEY_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "end_point": {
        "fp_lrp": ("endpoint_dy_fp_vs_lrp", "endPointDist_fp_lrp_m"),
        "fp_lrpFMM": ("endpoint_dy_fp_vs_lrp_fmm", "endPointDist_fp_lrpFMM_m"),
        "fp_fpDijkstra": ("endpoint_dy_fp_vs_dijkstra", "endPointDist_fp_fpDijkstra_m"),
        "fp_fpFMM": ("endpoint_dy_fp_vs_fmm", "endPointDist_fp_fpFMM_m"),
    },
    "hausdorff": {
        "fp_lrp": ("hausdorff_fp_vs_lrp", "hausdorffDist_fp_lrp_m"),
        "fp_lrpFMM": ("hausdorff_fp_vs_lrp_fmm", "hausdorffDist_fp_lrpFMM_m"),
        "fp_fpDijkstra": ("hausdorff_fp_vs_dijkstra", "hausdorffDist_fp_fpDijkstra_m"),
        "fp_fpFMM": ("hausdorff_fp_vs_fmm", "hausdorffDist_fp_fpFMM_m"),
    },
    "frechet": {
        "fp_lrp": ("frechetDist_fp_lrp_m",),
        "fp_lrpFMM": ("frechetDist_fp_lrpFMM_m",),
        "fp_fpDijkstra": ("frechetDist_fp_fpDijkstra_m",),
        "fp_fpFMM": ("frechetDist_fp_fpFMM_m",),
    },
}


def distance_key_candidates(cfg: "VisualizationConfig", distance_type: str, pair: str) -> list[str]:
    candidates = [cfg.distance_main_output_keys[distance_type][pair]]
    for alias in DISTANCE_MAIN_OUTPUT_KEY_ALIASES.get(distance_type, {}).get(pair, ()):
        if alias not in candidates:
            candidates.append(alias)
    return candidates


@dataclass(slots=True)
class FontConfig:
    family: str = "CMU Sans Serif"
    mathtext_fontset: str = "cm"
    unicode_minus: bool = False
    cmu_font_glob: str = "/home1/binhaoli/.local/share/fonts/cmun*.ttf"


@dataclass(slots=True)
class VisualizationConfig:
    data_root: Path
    run_id: str
    length_scale: float
    dh: float
    field_size_ref: int
    variance_values: list[float]
    std_values: list[float]
    output_dir: Path
    figure_dpi: int = 400
    workers: int = -1
    save_formats: list[str] = field(default_factory=lambda: ["png"])
    sections: list[str] = field(
        default_factory=lambda: [
            "times",
            "times_box",
            "taylor",
            "scatter",
            "hausdorff",
            "hausdorff_box",
            "frechet",
            "endpoint",
            "std_hausdorff",
            "std_endpoint",
            "ratio_mean",
            "ratio_var",
            "wasserstein_len",
        ]
    )
    distance_mode: dict[str, str] = field(
        default_factory=lambda: {
            "end_point": "load_from_main_output",
            "hausdorff": "compute_from_paths",
            "frechet": "compute_from_paths",
        }
    )
    distance_enabled: dict[str, bool] = field(
        default_factory=lambda: {
            "end_point": True,
            "hausdorff": True,
            "frechet": True,
        }
    )
    distance_fallback_on_missing: str = "compute_from_paths"
    distance_main_output_keys: dict[str, dict[str, str]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS.items()}
    )
    k_g: float = 1.0
    analytical_percentile: float = 0.01
    font: FontConfig = field(default_factory=FontConfig)


@dataclass(slots=True)
class RunPaths:
    size_dir: Path
    size_value: float
    size_str: str
    mc_files: list[Path] = field(default_factory=list)
    main_output_files: list[Path] = field(default_factory=list)
    fp_files: list[Path] = field(default_factory=list)
    lrp_files: list[Path] = field(default_factory=list)
    lrp_fmm_files: list[Path] = field(default_factory=list)
    fp_dijkstra_files: list[Path] = field(default_factory=list)
    fp_fmm_files: list[Path] = field(default_factory=list)
    logk_files: list[Path] = field(default_factory=list)


class ConfigError(ValueError):
    pass


def _to_path(raw: str | Path) -> Path:
    return Path(raw).expanduser().resolve()


def _validate_non_empty(name: str, arr: list[Any]) -> None:
    if not arr:
        raise ConfigError(f"{name} must be non-empty")


def validate_config(cfg: VisualizationConfig) -> None:
    if cfg.dh <= 0:
        raise ConfigError("dh must be > 0")
    if cfg.length_scale <= 0:
        raise ConfigError("length_scale must be > 0")
    if cfg.field_size_ref <= 0:
        raise ConfigError("field_size_ref must be > 0")
    if cfg.figure_dpi <= 0:
        raise ConfigError("figure_dpi must be > 0")
    if not (0.0 < cfg.analytical_percentile < 1.0):
        raise ConfigError("analytical_percentile must be in (0, 1)")
    _validate_non_empty("variance_values", cfg.variance_values)
    _validate_non_empty("std_values", cfg.std_values)
    _validate_non_empty("save_formats", cfg.save_formats)
    _validate_non_empty("sections", cfg.sections)
    if cfg.distance_fallback_on_missing not in DISTANCE_FALLBACK_VALUES:
        raise ConfigError(f"distance_fallback_on_missing must be one of {DISTANCE_FALLBACK_VALUES}")
    if not any(cfg.distance_enabled.get(t, False) for t in DISTANCE_TYPES):
        raise ConfigError("At least one distance type must be enabled")
    for t in DISTANCE_TYPES:
        mode = cfg.distance_mode.get(t)
        if mode not in DISTANCE_MODE_VALUES:
            raise ConfigError(f"distance_mode[{t}] must be one of {DISTANCE_MODE_VALUES}")
        key_map = cfg.distance_main_output_keys.get(t, {})
        if not isinstance(key_map, dict):
            raise ConfigError(f"distance_main_output_keys[{t}] must be a mapping")
        for pair in DISTANCE_PAIRS:
            if pair not in key_map:
                raise ConfigError(f"distance_main_output_keys[{t}] missing pair key {pair}")


def load_config(path: str | Path) -> VisualizationConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to load YAML config files") from exc

    cfg_path = _to_path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    font_raw = raw.get("font", {})
    font = FontConfig(
        family=font_raw.get("family", "CMU Sans Serif"),
        mathtext_fontset=font_raw.get("mathtext_fontset", "cm"),
        unicode_minus=bool(font_raw.get("unicode_minus", False)),
        cmu_font_glob=font_raw.get("cmu_font_glob", "/home1/binhaoli/.local/share/fonts/cmun*.ttf"),
    )

    raw_mode = raw.get("distance_mode", {})
    raw_enabled = raw.get("distance_enabled", {})
    raw_keys = raw.get("distance_main_output_keys", {})

    cfg = VisualizationConfig(
        data_root=_to_path(raw["data_root"]),
        run_id=str(raw.get("run_id", 3)),
        length_scale=float(raw.get("length_scale", 0.05)),
        dh=float(raw.get("dh", 0.1)),
        field_size_ref=int(raw.get("field_size_ref", 200)),
        variance_values=[float(v) for v in raw.get("variance_values", [])],
        std_values=[float(v) for v in raw.get("std_values", [])],
        output_dir=_to_path(raw.get("output_dir", "./outputs/visualization")),
        figure_dpi=int(raw.get("figure_dpi", 400)),
        workers=int(raw.get("workers", -1)),
        save_formats=[str(v) for v in raw.get("save_formats", ["png"])],
        sections=[str(v) for v in raw.get("sections", [])]
        or [
            "times",
            "times_box",
            "taylor",
            "scatter",
            "hausdorff",
            "hausdorff_box",
            "frechet",
            "endpoint",
            "std_hausdorff",
            "std_endpoint",
            "ratio_mean",
            "ratio_var",
            "wasserstein_len",
        ],
        distance_mode={
            "end_point": str(raw_mode.get("end_point", "load_from_main_output")),
            "hausdorff": str(raw_mode.get("hausdorff", "compute_from_paths")),
            "frechet": str(raw_mode.get("frechet", "compute_from_paths")),
        },
        distance_enabled={
            "end_point": bool(raw_enabled.get("end_point", True)),
            "hausdorff": bool(raw_enabled.get("hausdorff", True)),
            "frechet": bool(raw_enabled.get("frechet", True)),
        },
        distance_fallback_on_missing=str(raw.get("distance_fallback_on_missing", "compute_from_paths")),
        distance_main_output_keys={
            "end_point": {
                "fp_lrp": str(raw_keys.get("end_point", {}).get("fp_lrp", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["end_point"]["fp_lrp"])),
                "fp_lrpFMM": str(raw_keys.get("end_point", {}).get("fp_lrpFMM", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["end_point"]["fp_lrpFMM"])),
                "fp_fpDijkstra": str(
                    raw_keys.get("end_point", {}).get("fp_fpDijkstra", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["end_point"]["fp_fpDijkstra"])
                ),
                "fp_fpFMM": str(raw_keys.get("end_point", {}).get("fp_fpFMM", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["end_point"]["fp_fpFMM"])),
            },
            "hausdorff": {
                "fp_lrp": str(raw_keys.get("hausdorff", {}).get("fp_lrp", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["hausdorff"]["fp_lrp"])),
                "fp_lrpFMM": str(raw_keys.get("hausdorff", {}).get("fp_lrpFMM", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["hausdorff"]["fp_lrpFMM"])),
                "fp_fpDijkstra": str(
                    raw_keys.get("hausdorff", {}).get("fp_fpDijkstra", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["hausdorff"]["fp_fpDijkstra"])
                ),
                "fp_fpFMM": str(raw_keys.get("hausdorff", {}).get("fp_fpFMM", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["hausdorff"]["fp_fpFMM"])),
            },
            "frechet": {
                "fp_lrp": str(raw_keys.get("frechet", {}).get("fp_lrp", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["frechet"]["fp_lrp"])),
                "fp_lrpFMM": str(raw_keys.get("frechet", {}).get("fp_lrpFMM", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["frechet"]["fp_lrpFMM"])),
                "fp_fpDijkstra": str(
                    raw_keys.get("frechet", {}).get("fp_fpDijkstra", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["frechet"]["fp_fpDijkstra"])
                ),
                "fp_fpFMM": str(raw_keys.get("frechet", {}).get("fp_fpFMM", DEFAULT_DISTANCE_MAIN_OUTPUT_KEYS["frechet"]["fp_fpFMM"])),
            },
        },
        k_g=float(raw.get("k_g", 1.0)),
        analytical_percentile=float(raw.get("analytical_percentile", 0.01)),
        font=font,
    )
    validate_config(cfg)
    return cfg


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def write_stage_metadata(path: Path, *, stage: str, config: VisualizationConfig, notes: dict[str, Any]) -> None:
    payload = {
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "notes": notes,
    }
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def default_cache_dir(cfg: VisualizationConfig) -> Path:
    return Path("outputs") / "cache" / f"run{cfg.run_id}"


def _size_sort_key(size_dir: Path) -> int:
    return int(size_dir.name.lower().split("size")[-1])


def _value_dir_candidates(value: float) -> list[str]:
    candidates = [f"{value}", f"{value:g}"]
    rounded = round(value)
    if abs(value - rounded) < 1e-12:
        candidates.extend([str(int(rounded)), f"{int(rounded)}.0"])
    out: list[str] = []
    for c in candidates:
        if c not in out:
            out.append(c)
    return out


def _resolve_value_dir(run_dir: Path, prefix: str, value: float) -> Path:
    for raw in _value_dir_candidates(value):
        candidate = run_dir / f"{prefix}_{raw}"
        if candidate.exists():
            return candidate
    return run_dir / f"{prefix}_{value:g}"


def _collect_run_paths(size_dir: Path, size_str: str) -> RunPaths:
    mc_files = sorted(size_dir.glob(f"size{size_str}_mc*.npy"))
    if mc_files:
        return RunPaths(
            size_dir=size_dir,
            size_value=float(size_str),
            size_str=size_str,
            mc_files=mc_files,
        )
    main_output_files = sorted((size_dir / "main_output").glob(f"mainOutput_size{size_str}_mc*.npz"))
    path_dir = size_dir / "path"
    return RunPaths(
        size_dir=size_dir,
        size_value=float(size_str),
        size_str=size_str,
        main_output_files=main_output_files,
        fp_files=sorted(path_dir.glob(f"fp_size{size_str}_mc*.npy")),
        lrp_files=sorted(path_dir.glob(f"lrp_size{size_str}_mc*.npy")),
        lrp_fmm_files=sorted(path_dir.glob(f"lrp_fmm_size{size_str}_mc*.npy")),
        fp_dijkstra_files=sorted(path_dir.glob(f"fp_dijkstra_size{size_str}_mc*.npy")),
        fp_fmm_files=sorted(path_dir.glob(f"fp_fmm_size{size_str}_mc*.npy")),
        logk_files=sorted((size_dir / "logk_field").glob("logKField_*.npy")),
    )


def _discover_size_dirs(base: Path, length_scale: float) -> list[Path]:
    patterns = [
        f"LengthScale{length_scale}_Size*",
        f"lengthScale{length_scale}_size*",
    ]
    found: dict[str, Path] = {}
    for pattern in patterns:
        for d in base.glob(pattern):
            if d.name not in found:
                found[d.name] = d
    return sorted(found.values(), key=_size_sort_key)


def _extract_size_str(dir_name: str) -> str:
    lower = dir_name.lower()
    return lower.split("size")[-1]


def discover_variance_runs(cfg: VisualizationConfig) -> dict[float, list[RunPaths]]:
    output: dict[float, list[RunPaths]] = {}
    run_dir = cfg.data_root / f"run{cfg.run_id}"
    for var_ in cfg.variance_values:
        base = _resolve_value_dir(run_dir, "var", var_)
        size_dirs = _discover_size_dirs(base, cfg.length_scale)
        runs: list[RunPaths] = []
        for size_dir in size_dirs:
            size_str = _extract_size_str(size_dir.name)
            runs.append(_collect_run_paths(size_dir, size_str))
        output[var_] = runs
    return output


def discover_std_runs(cfg: VisualizationConfig) -> dict[float, list[RunPaths]]:
    output: dict[float, list[RunPaths]] = {}
    run_dir = cfg.data_root / f"run{cfg.run_id}"
    for std_ in cfg.std_values:
        base = _resolve_value_dir(run_dir, "std", std_)
        size_dirs = _discover_size_dirs(base, cfg.length_scale)
        runs: list[RunPaths] = []
        for size_dir in size_dirs:
            size_str = _extract_size_str(size_dir.name)
            runs.append(_collect_run_paths(size_dir, size_str))
        output[std_] = runs
    return output


def runpaths_to_dict(rp: RunPaths) -> dict[str, Any]:
    return {
        "size_dir": str(rp.size_dir),
        "size_value": rp.size_value,
        "size_str": rp.size_str,
        "mc_files": [str(p) for p in rp.mc_files],
        "main_output_files": [str(p) for p in rp.main_output_files],
        "fp_files": [str(p) for p in rp.fp_files],
        "lrp_files": [str(p) for p in rp.lrp_files],
        "lrp_fmm_files": [str(p) for p in rp.lrp_fmm_files],
        "fp_dijkstra_files": [str(p) for p in rp.fp_dijkstra_files],
        "fp_fmm_files": [str(p) for p in rp.fp_fmm_files],
        "logk_files": [str(p) for p in rp.logk_files],
    }


def serialize_discovered(discovered: dict[float, list[RunPaths]]) -> dict[float, list[dict[str, Any]]]:
    return {float(k): [runpaths_to_dict(rp) for rp in v] for k, v in discovered.items()}


def run_stage1(cfg: VisualizationConfig, *, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or default_cache_dir(cfg)
    discovered_var = discover_variance_runs(cfg)
    discovered_std = discover_std_runs(cfg)

    payload = {
        "variance": serialize_discovered(discovered_var),
        "std": serialize_discovered(discovered_std),
    }
    cache_file = cache_dir / "stage1_index.pkl"
    save_pickle(cache_file, payload)

    notes = {
        "variance_entries": {str(k): len(v) for k, v in payload["variance"].items()},
        "std_entries": {str(k): len(v) for k, v in payload["std"].items()},
        "cache_file": str(cache_file),
    }
    write_stage_metadata(cache_dir / "stage1_index.meta.json", stage="stage1_index", config=cfg, notes=notes)
    return cache_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: discover and index data files")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cache_path = run_stage1(cfg, cache_dir=args.cache_dir)
    print(f"Wrote index cache: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
