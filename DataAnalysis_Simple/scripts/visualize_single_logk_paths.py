#!/usr/bin/env python3
"""Load and visualize the logK field and paths from a single MC realization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cmcrameri.cm as cmc
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ALL_PATH_KEYS = ["fp", "fp_dijkstra", "fp_fmm", "lrp", "lrp_fmm"]

PATH_LABELS = {
    "fp": r"$\gamma^*$",
    "fp_fmm": r"$\gamma^*_{\mathcal{E}}$ (FMM)",
    "fp_dijkstra": r"$\gamma^*_{\mathcal{E}}$ (Dijkstra)",
    "lrp_fmm": r"$\gamma^*_{\mathcal{R}}$ (FMM)",
    "lrp": r"$\gamma^*_{\mathcal{R}}$ (Dijkstra)",
}

# COLORS = {
#     "fp": "#000000",
#     "fp_fmm": "#000000",
#     "fp_dijkstra": "#000000",
#     "lrp": "#00008B",
#     "lrp_fmm": "#6A3FB6",
# }

COLORS = {
    "fp": "#FF6B78",
    "fp_fmm": "#3F73B8",
    "fp_dijkstra": "#3F73B8",
    "lrp": "#F9B43F",
    "lrp_fmm": "#F9B43F",
}

STROKE_COLORS = {
    "fp": "#000000",
    "fp_fmm": "#000000",
    "fp_dijkstra": "#000000",
    "lrp": "#000000",
    "lrp_fmm": "#000000",
}

LINESTYLES = {
    "fp": "-",
    "fp_fmm": "--",
    "fp_dijkstra": ":",
    "lrp": ":",
    "lrp_fmm": "--",
}

LINEWIDTH = 1.2


def load_consolidated(npy_path: Path) -> dict[str, Any]:
    return np.load(npy_path, allow_pickle=True).item()


def discover_npy_file(
    config_path: Path,
    sweep: str,
    value: float,
    field_size: int,
    mc_index: int,
) -> Path:
    import step1_index as step1

    cfg = step1.load_config(config_path)
    if sweep == "variance":
        discovered = step1.discover_variance_runs(cfg)
    elif sweep == "std":
        discovered = step1.discover_std_runs(cfg)
    else:
        raise ValueError(f"Unsupported sweep={sweep}; expected 'variance' or 'std'.")

    best_key = min(discovered.keys(), key=lambda k: abs(float(k) - value))
    runs = discovered[best_key]
    if not runs:
        raise ValueError(f"No runs found for sweep={sweep}, value={best_key}.")

    size_str = str(field_size)
    rp = None
    for r in runs:
        if r.size_str == size_str or int(r.size_value) == field_size:
            rp = r
            break
    if rp is None:
        available = [r.size_str for r in runs]
        raise ValueError(f"No run with field_size={field_size}. Available: {available}")

    if not rp.mc_files:
        raise ValueError(f"No consolidated .npy files found in {rp.size_dir}")
    if mc_index < 0 or mc_index >= len(rp.mc_files):
        raise ValueError(
            f"mc_index={mc_index} out of range; {len(rp.mc_files)} MC files available."
        )
    return rp.mc_files[mc_index]


def print_summary(data: dict[str, Any]) -> None:
    meta = data.get("meta", {})
    print(f"\n{'=' * 60}")
    print(f"  MC Index: {meta.get('mc_index', '?')}    "
          f"Trial: {meta.get('trial_used', '?')}    "
          f"Status: {meta.get('status_message', '?')}")
    print(f"{'=' * 60}")

    paths = data.get("paths", {})
    if paths:
        print(f"\n{'Path':<18s} {'Length':>10s} {'Travel Time':>12s} {'Resistance':>12s}")
        print("-" * 54)
        for key in ALL_PATH_KEYS:
            p = paths.get(key, {})
            if not p:
                continue
            length = p.get("length", float("nan"))
            tt = p.get("travel_time", float("nan"))
            res = p.get("resistance", float("nan"))
            tt_str = f"{tt:.4f}" if np.isfinite(tt) else "-"
            res_str = f"{res:.4f}" if np.isfinite(res) else "-"
            print(f"  {PATH_LABELS.get(key, key):<16s} {length:>10.4f} {tt_str:>12s} {res_str:>12s}")

    hausdorff = data.get("hausdorff", {})
    if hausdorff:
        print(f"\n{'Hausdorff':<22s} {'Max':>8s} {'Mean':>8s} {'Std':>8s} {'Median':>8s}")
        print("-" * 58)
        for key in ("fp_vs_lrp", "fp_vs_lrp_fmm", "fp_vs_dijkstra", "fp_vs_fmm"):
            h = hausdorff.get(key, {})
            if not h:
                continue
            print(f"  {key:<20s} {h.get('max', float('nan')):>8.3f} "
                  f"{h.get('mean', float('nan')):>8.3f} "
                  f"{h.get('std', float('nan')):>8.3f} "
                  f"{h.get('median', float('nan')):>8.3f}")

    renyi = data.get("renyi_entropy", {})
    if renyi:
        s1 = np.asarray(renyi.get("S1", []))
        s2 = np.asarray(renyi.get("S2", []))
        print(f"\nRenyi Entropy (profiles over x):")
        print(f"  S1: shape={s1.shape}, mean={np.nanmean(s1):.6f}")
        print(f"  S2: shape={s2.shape}, mean={np.nanmean(s2):.6f}")
    print()


def plot_logk_paths(
    data: dict[str, Any],
    *,
    path_keys: list[str],
    cmap: str = "cmc.vik",
    show_legend: bool = True,
) -> plt.Figure:
    logk = data["fields"]["logk"]
    field_size = logk.shape[0]
    meta = data.get("meta", {})
    mc_index = meta.get("mc_index", "?")

    cmap_obj = getattr(cmc, cmap.replace("cmc.", ""), None) if cmap.startswith("cmc.") else cmap
    if cmap_obj is None:
        cmap_obj = cmc.vik

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        logk,
        origin="lower",
        cmap=cmap_obj,
        interpolation="nearest",
        extent=[0, field_size, 0, field_size],
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log\,K$")

    paths = data.get("paths", {})
    lw = LINEWIDTH
    for key in path_keys:
        p = paths.get(key, {})
        coords = p.get("coords")
        if coords is None:
            continue
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[0] < 2:
            continue
        line, = ax.plot(
            coords[:, 0],
            coords[:, 1],
            color=COLORS.get(key, "gray"),
            linestyle=LINESTYLES.get(key, "-"),
            linewidth=lw,
            label=PATH_LABELS.get(key, key),
            zorder=10
        )
        line.set_path_effects([
            pe.Stroke(linewidth=lw + 2, foreground=STROKE_COLORS.get(key, "black")),
            pe.Normal(),
        ])
        line.set_solid_capstyle("round")
        line.set_dash_capstyle("round")

    if show_legend:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"MC {mc_index}  ({field_size}×{field_size})")
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize logK field and paths from a single MC realization."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--npy-file", type=Path,
        help="Direct path to a consolidated size{N}_mc{i}.npy file.",
    )
    source.add_argument(
        "--config", type=Path,
        help="Path to visualization config YAML (discovery mode).",
    )

    parser.add_argument("--sweep", choices=["variance", "std"])
    parser.add_argument("--value", type=float)
    parser.add_argument("--field-size", type=int)
    parser.add_argument("--mc-index", type=int)

    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--cmap", type=str, default="cmc.lajolla")
    parser.add_argument("--no-legend", action="store_true", default=False)
    parser.add_argument(
        "--paths", nargs="+", default=ALL_PATH_KEYS,
        choices=ALL_PATH_KEYS,
        help="Which paths to plot (default: all).",
    )

    args = parser.parse_args()
    if args.config is not None:
        missing = []
        if args.sweep is None:
            missing.append("--sweep")
        if args.value is None:
            missing.append("--value")
        if args.field_size is None:
            missing.append("--field-size")
        if args.mc_index is None:
            missing.append("--mc-index")
        if missing:
            parser.error(
                f"Discovery mode (--config) requires: {', '.join(missing)}"
            )

    return args


def main() -> int:
    args = parse_args()

    if args.npy_file is not None:
        npy_path = args.npy_file
    else:
        npy_path = discover_npy_file(
            args.config, args.sweep, args.value, args.field_size, args.mc_index,
        )

    print(f"Loading: {npy_path}")
    data = load_consolidated(npy_path)

    meta = data.get("meta", {})
    if meta.get("status_code", 9) != 0:
        print(f"WARNING: MC realization status = {meta.get('status_message', 'unknown')}")

    if "fields" not in data or "logk" not in data.get("fields", {}):
        print("ERROR: No logK field found in the consolidated file.")
        return 1

    print_summary(data)

    fig = plot_logk_paths(
        data,
        path_keys=args.paths,
        cmap=args.cmap,
        show_legend=not args.no_legend,
    )

    mc_index = meta.get("mc_index", 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"logk_paths_mc{mc_index}.{args.format}"
    fig.savefig(str(out_file), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
