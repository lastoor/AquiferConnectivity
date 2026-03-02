from __future__ import annotations

import numpy as np
from tqdm import tqdm

from visualization.config import VisualizationConfig
from visualization.io.discovery import discover_variance_runs
from visualization.io.loaders import load_logk_field, load_main_output, load_paths
from visualization.metrics.distance_resolver import resolve_distances
from visualization.metrics.line_integrals import compute_mc_line_integrals, summarize_integral_values
from visualization.stats.aggregate import mean_std_matrix
from visualization.types import VarianceResults


def _init_lengths() -> dict[str, list[np.ndarray]]:
    return {"fp": [], "fp_dijkstra": [], "fp_fmm": [], "lrp": []}


def _init_times() -> dict[str, list[np.ndarray]]:
    return {"fpt": [], "fpt_dijkstra": [], "fpt_fmm": [], "mhr": []}


def _init_distances() -> dict[str, dict[str, list[np.ndarray]]]:
    return {
        "hausdorff": {"fp_lrp": [], "fp_fpDijkstra": [], "fp_fpFMM": []},
        "frechet": {"fp_lrp": [], "fp_fpDijkstra": [], "fp_fpFMM": []},
        "end_point": {"fp_lrp": [], "fp_fpDijkstra": [], "fp_fpFMM": []},
    }


def _append_stat_row(container: dict[str, list[np.ndarray]], key: str, values: list[float]) -> None:
    container[key].append(np.asarray(values, dtype=float))


def _summary_stats(lengths, times, distances):
    return {
        "lengths": {
            "fp": mean_std_matrix(lengths["fp"]),
            "fp_dijkstra": mean_std_matrix(lengths["fp_dijkstra"]),
            "fp_fmm": mean_std_matrix(lengths["fp_fmm"]),
            "lrp": mean_std_matrix(lengths["lrp"]),
        },
        "times": {
            "fpt": mean_std_matrix(times["fpt"], filter_positive=True),
            "fpt_dijkstra": mean_std_matrix(times["fpt_dijkstra"], filter_positive=True),
            "fpt_fmm": mean_std_matrix(times["fpt_fmm"], filter_positive=True),
            "mhr": mean_std_matrix(times["mhr"], filter_positive=True),
        },
        "distances": {
            "hausdorff": {
                "fp_lrp": mean_std_matrix(distances["hausdorff"]["fp_lrp"]),
                "fp_fpDijkstra": mean_std_matrix(distances["hausdorff"]["fp_fpDijkstra"], filter_positive=True),
                "fp_fpFMM": mean_std_matrix(distances["hausdorff"]["fp_fpFMM"], filter_positive=True),
            },
            "frechet": {
                "fp_lrp": mean_std_matrix(distances["frechet"]["fp_lrp"]),
                "fp_fpDijkstra": mean_std_matrix(distances["frechet"]["fp_fpDijkstra"], filter_positive=True),
                "fp_fpFMM": mean_std_matrix(distances["frechet"]["fp_fpFMM"], filter_positive=True),
            },
            "end_point": {
                "fp_lrp": mean_std_matrix(distances["end_point"]["fp_lrp"]),
                "fp_fpDijkstra": mean_std_matrix(distances["end_point"]["fp_fpDijkstra"]),
                "fp_fpFMM": mean_std_matrix(distances["end_point"]["fp_fpFMM"]),
            },
        },
    }


def _compute_line_integrals_for_var(
    cfg: VisualizationConfig,
    var_: float,
    run_paths,
    *,
    max_mc_per_size: int | None = None,
) -> dict[str, dict[str, float | list[float]]]:
    target = next((rp for rp in run_paths if int(rp.size_value) == int(cfg.field_size_ref)), None)
    if target is None:
        empty = summarize_integral_values([])
        return {
            "int_logk_fp": empty,
            "int_logk2_fp": empty,
            "int_logk_lrp": empty,
            "int_logk2_lrp": empty,
            "int_logk_fp_dijkstra": empty,
            "int_logk2_fp_dijkstra": empty,
            "int_logk_fp_fmm": empty,
            "int_logk2_fp_fmm": empty,
        }

    values = {
        "fp": {"logk": [], "logk2": []},
        "lrp": {"logk": [], "logk2": []},
        "fp_dijkstra": {"logk": [], "logk2": []},
        "fp_fmm": {"logk": [], "logk2": []},
    }

    n_mc = len(target.logk_files)
    if max_mc_per_size is not None:
        n_mc = min(n_mc, max_mc_per_size)

    for mc_idx in range(n_mc):
        field = load_logk_field(target, mc_idx)
        if field is None:
            continue
        pathset = load_paths(target, mc_idx)
        ints = compute_mc_line_integrals(pathset, field, grid_spacing=1.0)
        for key, pair in ints.items():
            if pair[0] is None:
                continue
            values[key]["logk"].append(pair[0])
            values[key]["logk2"].append(pair[1])

    return {
        "int_logk_fp": summarize_integral_values(values["fp"]["logk"]),
        "int_logk2_fp": summarize_integral_values(values["fp"]["logk2"]),
        "int_logk_lrp": summarize_integral_values(values["lrp"]["logk"]),
        "int_logk2_lrp": summarize_integral_values(values["lrp"]["logk2"]),
        "int_logk_fp_dijkstra": summarize_integral_values(values["fp_dijkstra"]["logk"]),
        "int_logk2_fp_dijkstra": summarize_integral_values(values["fp_dijkstra"]["logk2"]),
        "int_logk_fp_fmm": summarize_integral_values(values["fp_fmm"]["logk"]),
        "int_logk2_fp_fmm": summarize_integral_values(values["fp_fmm"]["logk2"]),
    }


def run_variance_analysis_from_discovered(
    cfg: VisualizationConfig,
    discovered,
    *,
    show_progress: bool = False,
    max_var_values: int | None = None,
    max_sizes_per_var: int | None = None,
    max_mc_per_size: int | None = None,
) -> tuple[VarianceResults, list[str]]:
    data: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    line_integrals: dict[str, dict] = {}
    warnings: list[str] = []
    warning_seen: set[str] = set()

    var_values = cfg.variance_values[:max_var_values] if max_var_values is not None else cfg.variance_values
    var_iter = var_values
    if show_progress:
        var_iter = tqdm(var_values, desc="Variance sweep", unit="var", leave=False)

    for var_ in var_iter:
        run_paths = discovered.get(var_, [])
        if max_sizes_per_var is not None:
            run_paths = run_paths[:max_sizes_per_var]
        field_size_list: list[float] = []
        lengths = _init_lengths()
        times = _init_times()
        distances = _init_distances()

        run_iter = run_paths
        if show_progress:
            run_iter = tqdm(run_paths, desc=f"var={var_} sizes", unit="size", leave=False)

        for rp in run_iter:
            field_size_list.append(rp.size_value)
            if not rp.main_output_files:
                warnings.append(f"No mainOutput files found for var={var_}, size={rp.size_str}")
                continue

            len_fp_m = []
            len_lrp_m = []
            len_fp_dijkstra_m = []
            len_fp_fmm_m = []
            fpt_m = []
            fpt_dijkstra_m = []
            fpt_fmm_m = []
            mhr_m = []

            haus_fp_lrp = []
            haus_fp_d = []
            haus_fp_fmm = []
            fre_fp_lrp = []
            fre_fp_d = []
            fre_fp_fmm = []
            end_fp_lrp = []
            end_fp_d = []
            end_fp_fmm = []

            main_output_files = rp.main_output_files
            if max_mc_per_size is not None:
                main_output_files = main_output_files[:max_mc_per_size]

            mc_iter = enumerate(main_output_files)
            if show_progress:
                mc_iter = enumerate(
                    tqdm(
                        main_output_files,
                        desc=f"var={var_} size={rp.size_str} MC",
                        unit="mc",
                        leave=False,
                    )
                )

            for mc_idx, f in mc_iter:
                output_data = load_main_output(f)
                len_fp_m.append(output_data["len_fp_m"])
                len_fp_dijkstra_m.append(output_data["len_fp_dijkstra_m"])
                len_fp_fmm_m.append(output_data["len_fp_fmm_m"])
                len_lrp_m.append(output_data["len_lrp_m"])
                fpt_m.append(output_data["fpt_m"])
                fpt_dijkstra_m.append(output_data["fpt_dijkstra_m"])
                fpt_fmm_m.append(output_data["fpt_fmm_m"])
                mhr_m.append(output_data["mhr_m"])
                dist_values, distance_warnings = resolve_distances(
                    cfg,
                    output_data,
                    None,
                    workers=cfg.workers,
                    hausdorff_compute_method="shapely",
                    pathset_loader=lambda rp=rp, mc_idx=mc_idx: load_paths(rp, mc_idx),
                )
                for w in distance_warnings:
                    if w not in warning_seen:
                        warning_seen.add(w)
                        warnings.append(w)

                haus_fp_lrp.append(dist_values["hausdorff"]["fp_lrp"])
                haus_fp_d.append(dist_values["hausdorff"]["fp_fpDijkstra"])
                haus_fp_fmm.append(dist_values["hausdorff"]["fp_fpFMM"])
                fre_fp_lrp.append(dist_values["frechet"]["fp_lrp"])
                fre_fp_d.append(dist_values["frechet"]["fp_fpDijkstra"])
                fre_fp_fmm.append(dist_values["frechet"]["fp_fpFMM"])
                end_fp_lrp.append(dist_values["end_point"]["fp_lrp"])
                end_fp_d.append(dist_values["end_point"]["fp_fpDijkstra"])
                end_fp_fmm.append(dist_values["end_point"]["fp_fpFMM"])

            _append_stat_row(lengths, "fp", len_fp_m)
            _append_stat_row(lengths, "fp_dijkstra", len_fp_dijkstra_m)
            _append_stat_row(lengths, "fp_fmm", len_fp_fmm_m)
            _append_stat_row(lengths, "lrp", len_lrp_m)
            _append_stat_row(times, "fpt", fpt_m)
            _append_stat_row(times, "fpt_dijkstra", fpt_dijkstra_m)
            _append_stat_row(times, "fpt_fmm", fpt_fmm_m)
            _append_stat_row(times, "mhr", mhr_m)

            distances["hausdorff"]["fp_lrp"].append(np.asarray(haus_fp_lrp, dtype=float))
            distances["hausdorff"]["fp_fpDijkstra"].append(np.asarray(haus_fp_d, dtype=float))
            distances["hausdorff"]["fp_fpFMM"].append(np.asarray(haus_fp_fmm, dtype=float))
            distances["frechet"]["fp_lrp"].append(np.asarray(fre_fp_lrp, dtype=float))
            distances["frechet"]["fp_fpDijkstra"].append(np.asarray(fre_fp_d, dtype=float))
            distances["frechet"]["fp_fpFMM"].append(np.asarray(fre_fp_fmm, dtype=float))
            distances["end_point"]["fp_lrp"].append(np.asarray(end_fp_lrp, dtype=float))
            distances["end_point"]["fp_fpDijkstra"].append(np.asarray(end_fp_d, dtype=float))
            distances["end_point"]["fp_fpFMM"].append(np.asarray(end_fp_fmm, dtype=float))

        key = f"var_{var_}"
        data[key] = {
            "field_size": np.asarray(field_size_list, dtype=float),
            "lengths": lengths,
            "times": times,
            "distances": distances,
        }
        stats[key] = _summary_stats(lengths, times, distances)
        line_integrals[key] = _compute_line_integrals_for_var(
            cfg,
            var_,
            run_paths,
            max_mc_per_size=max_mc_per_size,
        )

    first_key = f"var_{var_values[0]}" if var_values else ""
    field_size = data.get(first_key, {}).get("field_size", np.asarray([], dtype=float))

    return VarianceResults(field_size=field_size, data=data, stats=stats, line_integrals=line_integrals), warnings


def run_variance_analysis(
    cfg: VisualizationConfig,
    *,
    max_var_values: int | None = None,
    max_sizes_per_var: int | None = None,
    max_mc_per_size: int | None = None,
) -> tuple[VarianceResults, list[str]]:
    discovered = discover_variance_runs(cfg)
    return run_variance_analysis_from_discovered(
        cfg,
        discovered,
        max_var_values=max_var_values,
        max_sizes_per_var=max_sizes_per_var,
        max_mc_per_size=max_mc_per_size,
    )
