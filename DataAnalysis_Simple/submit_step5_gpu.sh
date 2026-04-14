#!/bin/bash
set -euo pipefail

logk_source="grf" # Options: dataset, synthetic, grf
sweep="variance"
value="5"
field_size="200"
mc_index="34"
synthetic_amplitude="6.0"
synthetic_frequency_x="2.0"
synthetic_frequency_y="0.5"
synthetic_offset="-3.0"

# GRF logK source (used when logk_source="grf")
grf_std="2.0"
grf_length_scale=""   # leave empty to use cfg.length_scale
grf_seed="2"
grf_minpadding="0"

run_tag="3_synthetic_plume"
num_particles="1000000"
dh=""
fsm_fp="true"
fsm_max_sweeps="100"
fsm_tol="1e-8"
fsm_mse_tol=""
fsm_convergence_mode=""
fsm_mse_check_every="5"
fsm_mse_activation_factor="10.0"
fsm_consecutive_passes="2"
fsm_dt="0.5"
fsm_max_steps="10000"

linearized_fp="false"
linearized_scheme="linearized" # Options: linearized, full_eikonal
linearized_outer_max="10"
linearized_inner_max_sweeps="200"
linearized_inner_tol="1e-8"
linearized_eikonal_tol="1e-6"
linearized_damping="0.8"
linearized_max_deltat=""
linearized_monotone_residual="true"
linearized_verbose="true"
check_path_geometry="false"
source_x=""
target_x=""
source_y_start=""
source_y_end=""
target_y_start=""
target_y_end=""
curvature_resample_ds="1.0"  # Arc-length resampling interval (grid cells) before curvature; smaller = denser points
curvature_method="savgol" # Options: spline, savgol, circle
curvature_spline_mode="approximation" # Options: approximation, interpolation
curvature_spline_smooth="1"
curvature_spline_degree="4"
curvature_field_smooth_method="none" # Options: none, gaussian
curvature_field_smooth_sigma="2.0"
curvature_circle_window="20"
curvature_savgol_window="3"
curvature_savgol_poly="2"
normal_deriv_method="hessian_T"     # Options: direct, savgol2d, spline2d, mls_frenet, hessian_T
normal_deriv_spline2d_smooth="1.0"   # s=0: interpolating spline; s>0: smoothing spline
normal_deriv_mls_ns="2"              # MLS Frenet: half-width of patch in tangent direction
normal_deriv_mls_nn="2"              # MLS Frenet: half-width of patch in normal direction
normal_deriv_hessian_smooth="0.0"    # hessian_T: Gaussian pre-smooth sigma in grid cells (0=none)
normal_deriv_hessian_window="5"      # hessian_T: SG stencil window (odd, >= 3; window=3 = original Sobel)
normal_deriv_hessian_poly="3"        # hessian_T: SG polynomial degree (>= 2)
snapshot_mode="fraction_of_first_arrival"  # Options: fraction_of_first_arrival, dt_interval, physical, or "" to disable
snapshot_times="0.25 0.5 0.75 1.0"

time="00:05:00"
cpus_per_task="2"
mem_per_cpu="2GB"
gpus_per_task="1"

account="fbarros_324"
partition="gpu"

repo_root="/home1/binhaoli/FMM_FastestPath/DataAnalysis_Simple"
config_path="${repo_root}/configs/visualization.default.yaml"
run_root="${repo_root}/outputs/visualization/path/${run_tag}"
log_dir="${run_root}/slurm_logs"
mkdir -p "${log_dir}"

dh_args=""
if [[ -n "${dh}" ]]; then
    dh_args="--dh ${dh}"
fi

check_path_geometry_args=""
if [[ "${check_path_geometry}" == "true" ]]; then
    check_path_geometry_args="--check-path-geometry"
else
    check_path_geometry_args="--no-check-path-geometry"
fi

fsm_args=""
if [[ "${fsm_fp}" == "true" ]]; then
    fsm_args+=" --fsm-fp"
else
    fsm_args+=" --no-fsm-fp"
fi
fsm_args+=" --fsm-max-sweeps ${fsm_max_sweeps}"
fsm_args+=" --fsm-tol ${fsm_tol}"
if [[ -n "${fsm_mse_tol}" ]]; then
    fsm_args+=" --fsm-mse-tol ${fsm_mse_tol}"
fi
if [[ -n "${fsm_convergence_mode}" ]]; then
    fsm_args+=" --fsm-convergence-mode ${fsm_convergence_mode}"
fi
fsm_args+=" --fsm-mse-check-every ${fsm_mse_check_every}"
fsm_args+=" --fsm-mse-activation-factor ${fsm_mse_activation_factor}"
fsm_args+=" --fsm-consecutive-passes ${fsm_consecutive_passes}"
fsm_args+=" --fsm-dt ${fsm_dt}"
fsm_args+=" --fsm-max-steps ${fsm_max_steps}"

linearized_args=""
if [[ "${linearized_fp}" == "true" ]]; then
    linearized_args+=" --linearized-fp"
    linearized_args+=" --linearized-scheme ${linearized_scheme}"
    linearized_args+=" --linearized-outer-max ${linearized_outer_max}"
    linearized_args+=" --linearized-inner-max-sweeps ${linearized_inner_max_sweeps}"
    linearized_args+=" --linearized-inner-tol ${linearized_inner_tol}"
    linearized_args+=" --linearized-eikonal-tol ${linearized_eikonal_tol}"
    linearized_args+=" --linearized-damping ${linearized_damping}"
    if [[ -n "${linearized_max_deltat}" ]]; then
        linearized_args+=" --linearized-max-deltat ${linearized_max_deltat}"
    fi
    if [[ "${linearized_monotone_residual}" == "true" ]]; then
        linearized_args+=" --linearized-monotone-residual"
    else
        linearized_args+=" --no-linearized-monotone-residual"
    fi
    if [[ "${linearized_verbose}" == "true" ]]; then
        linearized_args+=" --linearized-verbose"
    else
        linearized_args+=" --no-linearized-verbose"
    fi
fi

snapshot_args=""
if [[ -n "${snapshot_mode}" ]]; then
    snapshot_args+=" --snapshot-mode ${snapshot_mode}"
    snapshot_args+=" --snapshot-times ${snapshot_times}"
fi

geometry_args=""
if [[ -n "${source_x}" ]]; then
    geometry_args+=" --source-x ${source_x}"
fi
if [[ -n "${target_x}" ]]; then
    geometry_args+=" --target-x ${target_x}"
fi
if [[ -n "${source_y_start}" ]]; then
    geometry_args+=" --source-y-start ${source_y_start}"
fi
if [[ -n "${source_y_end}" ]]; then
    geometry_args+=" --source-y-end ${source_y_end}"
fi
if [[ -n "${target_y_start}" ]]; then
    geometry_args+=" --target-y-start ${target_y_start}"
fi
if [[ -n "${target_y_end}" ]]; then
    geometry_args+=" --target-y-end ${target_y_end}"
fi

logk_args="--logk-source ${logk_source} --field-size ${field_size}"
job_suffix=""
if [[ "${logk_source}" == "dataset" ]]; then
    logk_args+=" --sweep ${sweep}"
    logk_args+=" --value ${value}"
    logk_args+=" --mc-index ${mc_index}"
    job_suffix="${sweep}_${value}"
    output_dir="${run_root}/${sweep}_${value}_size${field_size}_mc${mc_index}"
elif [[ "${logk_source}" == "synthetic" ]]; then
    logk_args+=" --synthetic-amplitude ${synthetic_amplitude}"
    logk_args+=" --synthetic-frequency-x ${synthetic_frequency_x}"
    logk_args+=" --synthetic-frequency-y ${synthetic_frequency_y}"
    logk_args+=" --synthetic-offset ${synthetic_offset}"
    job_suffix="synthetic_amp${synthetic_amplitude}_fx${synthetic_frequency_x}_fy${synthetic_frequency_y}"
    output_dir="${run_root}/synthetic_size${field_size}_amp${synthetic_amplitude}_fx${synthetic_frequency_x}_fy${synthetic_frequency_y}_offset${synthetic_offset}"
elif [[ "${logk_source}" == "grf" ]]; then
    logk_args+=" --grf-std ${grf_std}"
    logk_args+=" --grf-seed ${grf_seed}"
    logk_args+=" --grf-minpadding ${grf_minpadding}"
    if [[ -n "${grf_length_scale}" ]]; then
        logk_args+=" --grf-length-scale ${grf_length_scale}"
    fi
    job_suffix="grf_std${grf_std}_seed${grf_seed}"
    output_dir="${run_root}/grf_size${field_size}_std${grf_std}_seed${grf_seed}"
else
    echo "Unsupported logk_source=${logk_source}; expected dataset, synthetic, or grf." >&2
    exit 1
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --job-name=step5_gpu_${job_suffix}
#SBATCH --output=${log_dir}/output_${job_suffix}_%j.txt
#SBATCH --error=${log_dir}/error_${job_suffix}_%j.txt
#SBATCH --partition=${partition}
#SBATCH --time=${time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=${gpus_per_task}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem-per-cpu=${mem_per_cpu}
#SBATCH --chdir=${repo_root}

echo "Running step5 GPU Pollock on node \$SLURM_NODELIST"
echo "Start Time: \$(date)"
echo "logK source: ${logk_source}"
echo "FSM FP: ${fsm_fp} (max_sweeps=${fsm_max_sweeps}, tol=${fsm_tol}, mse_tol=${fsm_mse_tol:-disabled}, mode=${fsm_convergence_mode:-auto}, mse_check_every=${fsm_mse_check_every}, mse_activation_factor=${fsm_mse_activation_factor}, consecutive_passes=${fsm_consecutive_passes}, dt=${fsm_dt}, max_steps=${fsm_max_steps})"
echo "Linearized FP: ${linearized_fp}"
echo "Linearized scheme: ${linearized_scheme}"
echo "Check path geometry: ${check_path_geometry}"
echo "Curvature method: ${curvature_method} (mode=${curvature_spline_mode}, smooth=${curvature_spline_smooth}, degree=${curvature_spline_degree}, circle_window=${curvature_circle_window}, savgol_window=${curvature_savgol_window}, savgol_poly=${curvature_savgol_poly}), normal_deriv_method=${normal_deriv_method}"
echo "Snapshot mode: ${snapshot_mode:-disabled}, times: ${snapshot_times:-n/a}"
echo "Curvature field smoothing: method=${curvature_field_smooth_method}, sigma=${curvature_field_smooth_sigma}"
echo "Step5 geometry overrides: source_x=${source_x:-default}, target_x=${target_x:-default}, source_y=[${source_y_start:-default}, ${source_y_end:-default}), target_y=[${target_y_start:-default}, ${target_y_end:-default})"
if [[ "${logk_source}" == "dataset" ]]; then
  echo "Dataset selection: sweep=${sweep}, value=${value}, mc_index=${mc_index}"
else
  echo "Synthetic logK: amplitude=${synthetic_amplitude}, freq_x=${synthetic_frequency_x}, freq_y=${synthetic_frequency_y}, offset=${synthetic_offset}"
fi
if [[ "${linearized_fp}" == "true" && "${linearized_scheme}" == "full_eikonal" ]]; then
  echo "Note: full_eikonal ignores linearized-only controls: inner_max_sweeps, inner_tol, eikonal_tol, damping, max_deltat, monotone_residual."
fi

output_dir="${output_dir}"
mkdir -p "\${output_dir}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda deactivate || true
conda activate taichi_env
which python

export PYTHONPATH=./scripts

python ./scripts/step5_single_field_paths.py \\
  --config "${config_path}" \\
  ${logk_args} \\
  --output-dir "\${output_dir}" \\
  ${dh_args} \\
  --solver pollock \\
  --backend gpu \\
  --num-particles "${num_particles}" \\
  --velocity-location face \\
  --no-align-with-modflow-velocity \\
  --fmm-backtracking continuous \\
  --fmm-continuous-step-size 0.1 \\
  ${fsm_args} \\
  ${linearized_args} \\
  ${check_path_geometry_args} \\
  ${geometry_args} \\
  --curvature-resample-ds "${curvature_resample_ds}" \\
  --curvature-method "${curvature_method}" \\
  --curvature-spline-mode "${curvature_spline_mode}" \\
  --curvature-spline-smooth "${curvature_spline_smooth}" \\
  --curvature-spline-degree "${curvature_spline_degree}" \\
  --curvature-field-smooth-method "${curvature_field_smooth_method}" \\
  --curvature-field-smooth-sigma "${curvature_field_smooth_sigma}" \\
  --curvature-circle-window "${curvature_circle_window}" \\
  --curvature-savgol-window "${curvature_savgol_window}" \\
  --curvature-savgol-poly "${curvature_savgol_poly}" \\
  --normal-deriv-method "${normal_deriv_method}" \\
  --normal-deriv-spline2d-smooth "${normal_deriv_spline2d_smooth}" \\
  --normal-deriv-mls-ns "${normal_deriv_mls_ns}" \\
  --normal-deriv-mls-nn "${normal_deriv_mls_nn}" \\
  --normal-deriv-hessian-smooth "${normal_deriv_hessian_smooth}" \\
  --normal-deriv-hessian-window "${normal_deriv_hessian_window}" \\
  --normal-deriv-hessian-poly "${normal_deriv_hessian_poly}" \\
  ${snapshot_args}

echo "End Time: \$(date)"
EOF
