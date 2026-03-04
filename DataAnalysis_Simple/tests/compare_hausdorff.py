#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import step1_index as step1
import step2_metrics as step2

_MC_PATTERN = re.compile(r"_mc(-?\d+)$")
_PAIRS = ("fp_lrp", "fp_fpDijkstra", "fp_fpFMM")


def _parse_mc_label(path: Path) -> int | None:
    match = _MC_PATTERN.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _to_mc_map(paths: list[Path]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in paths:
        label = _parse_mc_label(path)
        if label is not None:
            out[label] = path
    return out


def _resolve_variance_key(discovered: dict[float, list[step1.RunPaths]], value: float) -> float:
    if value in discovered:
        return float(value)
    for key in discovered:
        if abs(float(key) - float(value)) < 1e-12:
            return float(key)
    available = ", ".join(str(k) for k in sorted(discovered))
    raise ValueError(f"Variance {value} was not found. Available values: [{available}]")


def _select_run_paths_by_size(runs: list[step1.RunPaths], field_size: int) -> step1.RunPaths:
    for run_paths in runs:
        if int(round(run_paths.size_value)) == int(field_size):
            return run_paths
        if run_paths.size_str == str(field_size):
            return run_paths
    available = ", ".join(rp.size_str for rp in runs)
    raise ValueError(f"Field size {field_size} was not found. Available sizes: [{available}]")


def _hausdorff_shapely(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if step2.hausdorff_distance is None:
        return float("nan")
    ls_a = step2.path_to_linestring(a)
    ls_b = step2.path_to_linestring(b)
    if ls_a is None or ls_b is None:
        return float("nan")
    return float(step2.hausdorff_distance(ls_a, ls_b))


def _hausdorff_nn_max(a: np.ndarray | None, b: np.ndarray | None, *, workers: int) -> float:
    if a is None or b is None:
        return float("nan")
    d, _, _ = step2.distance_distribution(a, b, ds=1.0, workers=workers)
    if d.size == 0:
        return float("nan")
    return float(np.max(d))


def _fmt(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.8g}"


def _available_labels_text(paths: list[Path]) -> str:
    labels = sorted({label for label in (_parse_mc_label(p) for p in paths) if label is not None})
    if labels:
        return ", ".join(str(v) for v in labels)
    return "<none>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Hausdorff distance computed by shapely vs nn_max for one variance/size/mc sample."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to visualization YAML config")
    parser.add_argument("--variance", required=True, type=float, help="Variance value to select")
    parser.add_argument("--field-size", required=True, type=int, help="Exact field size to select")
    parser.add_argument("--mc", required=True, type=int, help="MC label parsed from filenames (e.g. *_mc3.npy -> 3)")
    parser.add_argument("--workers", type=int, default=None, help="Nearest-neighbor worker count (defaults to config workers)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if step2.hausdorff_distance is None:
        print("Shapely is not available. Install it first to run this comparison: pip install shapely")
        return 2

    cfg = step1.load_config(args.config)
    workers = cfg.workers if args.workers is None else int(args.workers)

    discovered = step1.discover_variance_runs(cfg)
    try:
        variance_key = _resolve_variance_key(discovered, args.variance)
        run_paths = _select_run_paths_by_size(discovered[variance_key], args.field_size)
    except ValueError as exc:
        print(f"Selection error: {exc}")
        return 2

    fp_map = _to_mc_map(run_paths.fp_files)
    lrp_map = _to_mc_map(run_paths.lrp_files)
    fp_d_map = _to_mc_map(run_paths.fp_dijkstra_files)
    fp_fmm_map = _to_mc_map(run_paths.fp_fmm_files)

    if args.mc not in fp_map:
        print(f"Selection error: MC label {args.mc} was not found in FP files.")
        print(f"Available FP MC labels: {_available_labels_text(run_paths.fp_files)}")
        return 2

    pathset = step2.PathSet(
        fp=np.load(fp_map[args.mc]),
        lrp=np.load(lrp_map[args.mc]) if args.mc in lrp_map else None,
        fp_dijkstra=np.load(fp_d_map[args.mc]) if args.mc in fp_d_map else None,
        fp_fmm=np.load(fp_fmm_map[args.mc]) if args.mc in fp_fmm_map else None,
    )

    print("Selected sample")
    print(f"  variance: {variance_key}")
    print(f"  field_size: {run_paths.size_str}")
    print(f"  mc: {args.mc}")
    print(f"  size_dir: {run_paths.size_dir}")
    print(f"  workers: {workers}")
    print("")

    header = f"{'pair':<16} {'shapely':>14} {'nn_max':>14} {'abs_diff':>14} {'rel_diff':>14} {'note'}"
    print(header)
    print("-" * len(header))

    eps = 1e-12
    for pair in _PAIRS:
        a_name, b_name = step2.PAIR_TO_FIELDS[pair]
        a = getattr(pathset, a_name)
        b = getattr(pathset, b_name)

        shapely_value = _hausdorff_shapely(a, b)
        nn_value = _hausdorff_nn_max(a, b, workers=workers)

        if np.isnan(shapely_value) or np.isnan(nn_value):
            abs_diff = float("nan")
            rel_diff = float("nan")
        else:
            abs_diff = abs(shapely_value - nn_value)
            rel_diff = abs_diff / max(abs(shapely_value), eps)

        note = ""
        if b is None:
            note = f"missing {b_name} path for mc={args.mc}"
        elif np.isnan(shapely_value):
            note = "invalid shapely geometry"
        elif np.isnan(nn_value):
            note = "invalid nn_max geometry"

        print(
            f"{pair:<16} {_fmt(shapely_value):>14} {_fmt(nn_value):>14} "
            f"{_fmt(abs_diff):>14} {_fmt(rel_diff):>14} {note}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
