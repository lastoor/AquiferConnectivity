from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from visualization.plots.style import COLORS


def plot_hausdorff_vs_variance(var: np.ndarray, stats_hausdorff_lrp: np.ndarray, stats_hausdorff_fmm: np.ndarray, stats_hausdorff_dijkstra: np.ndarray, i_y: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.errorbar(var, stats_hausdorff_lrp[:, 0] / i_y, yerr=stats_hausdorff_lrp[:, 1] / i_y, label="$d_H$[FP, LRP]", linestyle="-", color=COLORS["lrp"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_hausdorff_fmm[:, 0] / i_y, yerr=stats_hausdorff_fmm[:, 1] / i_y, label="$d_H$[FP, FP(FMM)]", linestyle="-", color=COLORS["fpFMM"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_hausdorff_dijkstra[:, 0] / i_y, yerr=stats_hausdorff_dijkstra[:, 1] / i_y, label="$d_H$[FP, FP(Dijkstra)]", linestyle="-", color=COLORS["fpDijkstra"], marker="o", markerfacecolor="None")
    ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\\mathbb{E}\\left[ d_H / I_Y \\right]$")
    ax.legend(fontsize="9", loc="upper left")
    ax.grid(True, which="both")
    return fig, {"kind": "hausdorff_variance"}


def plot_endpoint_vs_variance(var: np.ndarray, stats_end_lrp: np.ndarray, stats_end_fmm: np.ndarray, stats_end_dijkstra: np.ndarray, i_y: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.errorbar(var, stats_end_lrp[:, 0] / i_y, yerr=stats_end_lrp[:, 1] / i_y, label="$d_{EP}$[FP, LRP]", linestyle="-", color=COLORS["lrp"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_end_fmm[:, 0] / i_y, yerr=stats_end_fmm[:, 1] / i_y, label="$d_{EP}$[FP, FP(FMM)]", linestyle="-", color=COLORS["fpFMM"], marker="o", markerfacecolor="None")
    ax.errorbar(var, stats_end_dijkstra[:, 0] / i_y, yerr=stats_end_dijkstra[:, 1] / i_y, label="$d_{EP}$[FP, FP(Dijkstra)]", linestyle="-", color=COLORS["fpDijkstra"], marker="o", markerfacecolor="None")
    ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel(r"$\\mathbb{E}\\left[ d_{EP} / I_Y \\right]$")
    ax.legend(fontsize="9", loc="upper left")
    ax.grid(True, which="both")
    return fig, {"kind": "endpoint_variance"}


def plot_std_hausdorff(x_var: np.ndarray, stats_std: dict[str, np.ndarray], i_y: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    y = np.asarray(stats_std["hausdorffDist"]["fp_lrp"][:, 0, 0]) / i_y
    yerr = np.asarray(stats_std["hausdorffDist"]["fp_lrp"][:, 1, 0]) / i_y
    ax.errorbar(x_var, y, yerr=yerr, label="$d(\\text{FP}, \\text{LRP})$", linestyle="-", color="blue", marker="o", markerfacecolor="None")

    y = np.asarray(stats_std["hausdorffDist"]["fp_fpFMM"][:, 0, 0]) / i_y
    yerr = np.asarray(stats_std["hausdorffDist"]["fp_fpFMM"][:, 1, 0]) / i_y
    ax.errorbar(x_var, y, yerr=yerr, label="$d[\\text{FP}, \\text{FP(FMM)}]$", linestyle="-", color="orange", marker="o", markerfacecolor="None")

    y = np.asarray(stats_std["hausdorffDist"]["fp_fpDijkstra"][:, 0, 0]) / i_y
    yerr = np.asarray(stats_std["hausdorffDist"]["fp_fpDijkstra"][:, 1, 0]) / i_y
    ax.errorbar(x_var, y, yerr=yerr, label="$d[\\text{FP}, \\text{FP(Dijkstra)}]$", linestyle="-", color="red", marker="o", markerfacecolor="None")

    ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel("$\\mathbb{E}\\left[ \\frac{d_{\\text{Hausdorff}}}{I_Y} \\right]$")
    ax.set_title("Variation of Hausdorff Distance with Variance")
    ax.legend(fontsize="9", loc="upper left")
    ax.grid(True, which="both")
    return fig, {"kind": "hausdorff_std"}


def plot_std_endpoint(x_var: np.ndarray, stats_std: dict[str, np.ndarray], i_y: float):
    fig, ax = plt.subplots(figsize=[5, 4])
    y = np.asarray(stats_std["endPointDist"]["fp_lrp"][:, 0, 0]) / i_y
    yerr = np.asarray(stats_std["endPointDist"]["fp_lrp"][:, 1, 0]) / i_y
    ax.errorbar(x_var, y, yerr=yerr, label="$d(\\text{FP}, \\text{LRP})$", linestyle="-", color="blue", marker="o", markerfacecolor="None")

    y = np.asarray(stats_std["endPointDist"]["fp_fpFMM"][:, 0, 0]) / i_y
    yerr = np.asarray(stats_std["endPointDist"]["fp_fpFMM"][:, 1, 0]) / i_y
    ax.errorbar(x_var, y, yerr=yerr, label="$d[\\text{FP}, \\text{FP(FMM)}]$", linestyle="-", color="orange", marker="o", markerfacecolor="None")

    y = np.asarray(stats_std["endPointDist"]["fp_fpDijkstra"][:, 0, 0]) / i_y
    yerr = np.asarray(stats_std["endPointDist"]["fp_fpDijkstra"][:, 1, 0]) / i_y
    ax.errorbar(x_var, y, yerr=yerr, label="$d[\\text{FP}, \\text{FP(Dijkstra)}]$", linestyle="-", color="red", marker="o", markerfacecolor="None")

    ax.set_xscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel("$\\mathbb{E}\\left[ \\frac{d_{\\text{end-point}}}{I_Y} \\right]$")
    ax.set_title("Variation of End-Point Distance with Variance")
    ax.legend(fontsize="9", loc="upper left")
    ax.grid(True, which="both")
    return fig, {"kind": "endpoint_std"}
