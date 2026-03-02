from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from visualization.plots.style import COLORS


def plot_characteristic_time(var: np.ndarray, stats_fpt: np.ndarray, stats_mhr: np.ndarray, stats_fpt_fmm: np.ndarray, stats_fpt_dijkstra: np.ndarray, t_ref: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.errorbar(var, stats_fpt[:, 0] / t_ref, stats_fpt[:, 1] / t_ref, color=COLORS["fp"], linestyle="-", marker="o", markerfacecolor="None", label="FP")
    ax.errorbar(var, stats_mhr[:, 0] / t_ref, stats_mhr[:, 1] / t_ref, color=COLORS["lrp"], linestyle="-", marker="o", markerfacecolor="None", label="LRP")
    ax.errorbar(var, stats_fpt_fmm[:, 0] / t_ref, stats_fpt_fmm[:, 1] / t_ref, color=COLORS["fpFMM"], linestyle="-", marker="o", markerfacecolor="None", label="FP (FMM)")
    ax.errorbar(var, stats_fpt_dijkstra[:, 0] / t_ref, stats_fpt_dijkstra[:, 1] / t_ref, color=COLORS["fpDijkstra"], linestyle="-", marker="o", markerfacecolor="None", label="FP (Dijkstra)")
    ax.set_yscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathbb{E}\left[ tU / I_Y \right]$")
    ax.legend()
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "characteristic_time"}


def plot_taylor_comparison(var: np.ndarray, stats_mhr: np.ndarray, stats_fpt: np.ndarray, stats_fpt_fmm: np.ndarray, stats_fpt_dijkstra: np.ndarray, taylor: dict[str, np.ndarray], t_ref: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.errorbar(var, stats_mhr[:, 0] / t_ref, stats_mhr[:, 1] / t_ref, color=COLORS["lrp"], linestyle="-", marker="o", markerfacecolor="None", label="$R_{\\mathrm{min}}$")
    ax.errorbar(var, stats_fpt[:, 0] / t_ref, stats_fpt[:, 1] / t_ref, color=COLORS["fp"], linestyle="-", marker="o", markerfacecolor="None", label="$t_{\\mathrm{min}}$")
    ax.errorbar(var, stats_fpt_fmm[:, 0] / t_ref, stats_fpt_fmm[:, 1] / t_ref, color=COLORS["fpFMM"], linestyle="-", marker="o", markerfacecolor="None", label="$t_{\\mathrm{min}}$ (FMM)")
    ax.errorbar(var, stats_fpt_dijkstra[:, 0] / t_ref, stats_fpt_dijkstra[:, 1] / t_ref, color=COLORS["fpDijkstra"], linestyle="-", marker="o", markerfacecolor="None", label="$t_{\\mathrm{min}}$ (Dijkstra)")

    ax.plot(var, taylor["lrp"] / t_ref, "--", color=COLORS["lrp"], linewidth=1.5, label="Taylor Expansion (LRP)")
    ax.plot(var, taylor["fp"] / t_ref, "--", color=COLORS["fp"], linewidth=1.5, label="Taylor Expansion (FP)")
    ax.plot(var, taylor["fp_fmm"] / t_ref, "--", color=COLORS["fpFMM"], linewidth=1.5, label="Taylor Expansion (FP, FMM)")
    ax.plot(var, taylor["fp_dijkstra"] / t_ref, "--", color=COLORS["fpDijkstra"], linewidth=1.5, label="Taylor Expansion (FP, Dijkstra)")

    ax.set_yscale("log")
    ax.set_ylim([10 ** (-2), 10 ** 2])
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\mathbb{E}[t_{\mathrm{min}}] U / I_Y$")
    ax.legend(fontsize="8", ncol=2)
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "taylor_comparison"}


def plot_scatter_tmin_vs_rmin(stats_fpt: np.ndarray, stats_fpt_fmm: np.ndarray, stats_fpt_dijkstra: np.ndarray, stats_mhr: np.ndarray, t_ref: float):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stats_fpt[:, 0] / t_ref, stats_fpt_fmm[:, 0] / t_ref, color="orange", label="$t_{\\text{min}}\\text{(FMM)}$", alpha=1)
    ax.scatter(stats_fpt[:, 0] / t_ref, stats_fpt_dijkstra[:, 0] / t_ref, color="blue", label="$t_{\\text{min}}\\text{(Dijkstra)}$", alpha=0.6)
    ax.scatter(stats_fpt[:, 0] / t_ref, stats_mhr[:, 0] / t_ref, color="darkred", marker="^", label="$R_{\\text{min}}$", alpha=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t_{\\text{min}}\\text{(RK45)}$ (s)")
    ax.set_ylabel("$\\tau$ (s)")
    ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Characteristic Time Scale")
    ax.legend()
    return fig, {"kind": "scatter_tmin_vs_rmin"}
