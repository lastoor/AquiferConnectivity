from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    k_g: float = 1.0
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
    _validate_non_empty("variance_values", cfg.variance_values)
    _validate_non_empty("std_values", cfg.std_values)
    _validate_non_empty("save_formats", cfg.save_formats)
    _validate_non_empty("sections", cfg.sections)


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
        k_g=float(raw.get("k_g", 1.0)),
        font=font,
    )

    validate_config(cfg)
    return cfg
