#!/bin/bash
set -euo pipefail

sweep="variance"
value="2.5"
field_size="200"
mc_index="10"

run_tag="1_full_eikonal_iteration"
num_particles="1000000"
dh=""
linearized_fp="true"
linearized_scheme="full_eikonal" # Options: linearized, full_eikonal
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
curvature_method="spline"
curvature_spline_mode="approximation"
curvature_spline_smooth="10"
curvature_spline_degree="3"
curvature_field_smooth_method="gaussian"
curvature_field_smooth_sigma="1.0"
curvature_circle_window="10"

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

sbatch <<EOF
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --job-name=step5_gpu_${sweep}_${value}
#SBATCH --output=${log_dir}/output_${sweep}_${value}_%j.txt
#SBATCH --error=${log_dir}/error_${sweep}_${value}_%j.txt
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
echo "Linearized FP: ${linearized_fp}"
echo "Linearized scheme: ${linearized_scheme}"
echo "Check path geometry: ${check_path_geometry}"
echo "Curvature method: ${curvature_method} (mode=${curvature_spline_mode}, smooth=${curvature_spline_smooth}, degree=${curvature_spline_degree}, circle_window=${curvature_circle_window})"
echo "Curvature field smoothing: method=${curvature_field_smooth_method}, sigma=${curvature_field_smooth_sigma}"
echo "Step5 geometry overrides: source_x=${source_x:-default}, target_x=${target_x:-default}, source_y=[${source_y_start:-default}, ${source_y_end:-default}), target_y=[${target_y_start:-default}, ${target_y_end:-default})"
if [[ "${linearized_fp}" == "true" && "${linearized_scheme}" == "full_eikonal" ]]; then
  echo "Note: full_eikonal ignores linearized-only controls: inner_max_sweeps, inner_tol, eikonal_tol, damping, max_deltat, monotone_residual."
fi

output_dir="${run_root}/${sweep}_${value}_size${field_size}_mc${mc_index}"
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
  --sweep "${sweep}" \\
  --value "${value}" \\
  --field-size "${field_size}" \\
  --mc-index "${mc_index}" \\
  --output-dir "\${output_dir}" \\
  ${dh_args} \\
  --solver pollock \\
  --backend gpu \\
  --num-particles "${num_particles}" \\
  --velocity-location face \\
  --no-align-with-modflow-velocity \\
  --fmm-backtracking continuous \\
  --fmm-continuous-step-size 0.5 \\
  ${linearized_args} \\
  ${check_path_geometry_args} \\
  ${geometry_args} \\
  --curvature-method "${curvature_method}" \\
  --curvature-spline-mode "${curvature_spline_mode}" \\
  --curvature-spline-smooth "${curvature_spline_smooth}" \\
  --curvature-spline-degree "${curvature_spline_degree}" \\
  --curvature-field-smooth-method "${curvature_field_smooth_method}" \\
  --curvature-field-smooth-sigma "${curvature_field_smooth_sigma}" \\
  --curvature-circle-window "${curvature_circle_window}"

echo "End Time: \$(date)"
EOF
