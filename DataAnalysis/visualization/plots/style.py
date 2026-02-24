from __future__ import annotations

import glob

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from visualization.config import VisualizationConfig

COLORS = {
    "fp": "black",
    "lrp": "#5278C4",
    "fpFMM": "#C46552",
    "fpDijkstra": "#B1C452",
}


def configure_style(cfg: VisualizationConfig) -> None:
    for p in glob.glob(cfg.font.cmu_font_glob):
        fm.fontManager.addfont(p)
    plt.rcParams["font.family"] = cfg.font.family
    plt.rcParams["mathtext.fontset"] = cfg.font.mathtext_fontset
    plt.rcParams["axes.unicode_minus"] = cfg.font.unicode_minus
