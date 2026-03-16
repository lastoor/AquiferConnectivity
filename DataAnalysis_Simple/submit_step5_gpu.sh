#!/bin/bash
set -euo pipefail

sweep="variance"
value="2.5"
field_size="200"
mc_index="40"

run_tag="step5_gpu_pollock"
num_particles="1000000"
dh=""

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
  --fmm-backtracking continuous \
  --fmm-continuous-step-size 0.5 \\
  --fmm-velocity-layout face \\
  --fmm-face-mapping to_cell_avg \\
  --fmm-zero-policy obstacle \\
  --fmm-min-speed 1e-12 \\
  --curvature-method savgol \\
  --curvature-savgol-window 9 \\
  --curvature-savgol-poly 3 \\
  --curvature-circle-window 7

echo "End Time: \$(date)"
EOF
