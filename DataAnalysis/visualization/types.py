from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class PathSet:
    fp: np.ndarray | None = None
    lrp: np.ndarray | None = None
    fp_dijkstra: np.ndarray | None = None
    fp_fmm: np.ndarray | None = None


@dataclass(slots=True)
class RunPaths:
    size_dir: Path
    size_value: float
    size_str: str
    main_output_files: list[Path] = field(default_factory=list)
    fp_files: list[Path] = field(default_factory=list)
    lrp_files: list[Path] = field(default_factory=list)
    fp_dijkstra_files: list[Path] = field(default_factory=list)
    fp_fmm_files: list[Path] = field(default_factory=list)
    logk_files: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class SummaryStats:
    mean: float
    std: float
    count: int


@dataclass(slots=True)
class FigureArtifact:
    id: str
    filename: str
    path: Path
    section: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class VarianceResults:
    field_size: np.ndarray
    data: dict[str, Any]
    stats: dict[str, Any]
    line_integrals: dict[str, Any]


@dataclass(slots=True)
class StdResults:
    data_path: dict[str, Any]
    stats_path: dict[str, Any]
    stats_std: dict[str, np.ndarray]
