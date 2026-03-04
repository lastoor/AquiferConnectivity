from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DISTANCE_TYPES = ("end_point", "hausdorff", "frechet")
DISTANCE_PAIRS = ("fp_lrp", "fp_fpDijkstra", "fp_fpFMM")
DISTANCE_MODE_VALUES = ("load_from_main_output", "compute_from_paths")
DISTANCE_FALLBACK_VALUES = ("compute_from_paths", "nan")


@dataclass(slots=True)
class FontConfig:
    family: str = "CMU Sans Serif"
    mathtext_fontset: str = "cm"
    unicode_minus: bool = False
    cmu_font_glob: str = "/home1/binhaoli/.local/share/fonts/cmun*.ttf"


@dataclass(slots=True)
class VisualizationConfig:
    data_root: Path
    run_id: int
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
            "taylor",
            "scatter",
            "hausdorff",
            "endpoint",
            "std_hausdorff",
            "std_endpoint",
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
        default_factory=lambda: {
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
        }
    )
    k_g: float = 1.0
    analytical_percentile: float = 0.01
    font: FontConfig = field(default_factory=FontConfig)


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
        run_id=int(raw.get("run_id", 3)),
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
            "taylor",
            "scatter",
            "hausdorff",
            "endpoint",
            "std_hausdorff",
            "std_endpoint",
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
                "fp_lrp": str(raw_keys.get("end_point", {}).get("fp_lrp", "endPointDist_fp_lrp_m")),
                "fp_fpDijkstra": str(raw_keys.get("end_point", {}).get("fp_fpDijkstra", "endPointDist_fp_fpDijkstra_m")),
                "fp_fpFMM": str(raw_keys.get("end_point", {}).get("fp_fpFMM", "endPointDist_fp_fpFMM_m")),
            },
            "hausdorff": {
                "fp_lrp": str(raw_keys.get("hausdorff", {}).get("fp_lrp", "hausdorffDist_fp_lrp_m")),
                "fp_fpDijkstra": str(raw_keys.get("hausdorff", {}).get("fp_fpDijkstra", "hausdorffDist_fp_fpDijkstra_m")),
                "fp_fpFMM": str(raw_keys.get("hausdorff", {}).get("fp_fpFMM", "hausdorffDist_fp_fpFMM_m")),
            },
            "frechet": {
                "fp_lrp": str(raw_keys.get("frechet", {}).get("fp_lrp", "frechetDist_fp_lrp_m")),
                "fp_fpDijkstra": str(raw_keys.get("frechet", {}).get("fp_fpDijkstra", "frechetDist_fp_fpDijkstra_m")),
                "fp_fpFMM": str(raw_keys.get("frechet", {}).get("fp_fpFMM", "frechetDist_fp_fpFMM_m")),
            },
        },
        k_g=float(raw.get("k_g", 1.0)),
        analytical_percentile=float(raw.get("analytical_percentile", 0.01)),
        font=font,
    )

    validate_config(cfg)
    return cfg
