#!/bin/bash
set -euo pipefail

VAR=("0.1" "0.5" "1" "1.5" "2" "2.5" "3" "3.5" "4" "4.5" "5")
# VAR=("0.1" "0.5" "1.5" "2" "2.5" "3" "3.5" "4" "4.5")
# VAR=("1" "5")
length_scale="0.05"
run="12p5_fmasr_ratio0p01_factor50"
dh="0.1"

num_mc="1000"
field_size="200"
num_particles="1000000"
cpu_workers="6"
fsm_solver="fd" # Options: fipy, sweep, legacy, new, fd
fsm_max_sweeps="100000"
fsm_tol="1e-8"
fsm_mse_tol=""
fsm_convergence_mode=""
fsm_mse_check_every="5"
fsm_mse_activation_factor="10.0"
fsm_consecutive_passes="2"
fsm_dt="0.5"
fsm_max_steps="1000000"

# FM-ASR trajectory extraction (run_fmm_gpu_worker.py). Set fm_asr_enabled=0 to disable.
# Leave fm_asr_min_speed empty to use --fmm_min_speed for FM-ASR (same as worker default).
# Override via export before running this script if needed.
fm_asr_enabled="${fm_asr_enabled:-1}"
fm_asr_perp_ratio="${fm_asr_perp_ratio:-0.01}"
fm_asr_back_factor="${fm_asr_back_factor:-50.0}"
fm_asr_max_steps="${fm_asr_max_steps:-1000000}"
fm_asr_min_speed="${fm_asr_min_speed:-}"

time="08:00:00"
mem_per_cpu="2GB"
cpus_per_task="8"
gpus_per_task="1"

account="fbarros_324"
partition="gpu"

repo_root="/home1/binhaoli/FMM_FastestPath/Algorithm"
run_root="/project2/fbarros_324/binhaoli/task8_rMin_tMin/FastMarchingMethod/run${run}"
log_dir="${run_root}/slurm_logs"
mkdir -p "${log_dir}"

# MSE/convergence-mode args are not used by solver=fd
fsm_extra_args=""
if [[ "${fsm_solver}" != "fd" ]]; then
    fsm_extra_args+=" --fsm_mse_check_every ${fsm_mse_check_every}"
    fsm_extra_args+=" --fsm_mse_activation_factor ${fsm_mse_activation_factor}"
    fsm_extra_args+=" --fsm_consecutive_passes ${fsm_consecutive_passes}"
    if [[ -n "${fsm_mse_tol}" ]]; then
        fsm_extra_args+=" --fsm_mse_tol ${fsm_mse_tol}"
    fi
    if [[ -n "${fsm_convergence_mode}" ]]; then
        fsm_extra_args+=" --fsm_convergence_mode ${fsm_convergence_mode}"
    fi
fi

fm_asr_args=""
if [[ "${fm_asr_enabled}" == "0" ]]; then
    fm_asr_args+=" --no-fm-asr"
else
    fm_asr_args+=" --fm_asr_perp_ratio ${fm_asr_perp_ratio}"
    fm_asr_args+=" --fm_asr_back_factor ${fm_asr_back_factor}"
    fm_asr_args+=" --fm_asr_max_steps ${fm_asr_max_steps}"
    if [[ -n "${fm_asr_min_speed}" ]]; then
        fm_asr_args+=" --fm_asr_min_speed ${fm_asr_min_speed}"
    fi
fi

for var in "${VAR[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --job-name=gpu_var_${var}
#SBATCH --output=${log_dir}/output_var${var}_%j.txt
#SBATCH --error=${log_dir}/error_var${var}_%j.txt
#SBATCH --partition=${partition}
#SBATCH --time=${time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=${gpus_per_task}
#SBATCH --mem-per-cpu=${mem_per_cpu}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --chdir=${repo_root}

echo "Running GPU var=${var} on node \$SLURM_NODELIST"
echo "Start Time: \$(date)"

output_dir="${run_root}/var_${var}"
mkdir -p "\${output_dir}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/.bashrc
source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate taichi_env

export PYTHONPATH=./scripts
TAICHI_PY=/home1/binhaoli/.conda/envs/taichi_env/bin/python

./scripts/run_in_fmm_env.sh "\$TAICHI_PY" ./scripts/run_fmm_gpu_worker.py \\
  --path_workdir "\${output_dir}" \\
  --num_mc ${num_mc} \\
  --field_size ${field_size} \\
  --length_scale ${length_scale} \\
  --minpadding 0 \\
  --var ${var} \\
  --dh ${dh} \\
  --seed 0 \\
  --seed_mode strict_mc \\
  --variance_seed_policy same \\
  --solver pollock \\
  --backend gpu \\
  --num_particles ${num_particles} \\
  --cpu_workers ${cpu_workers} \\
  --velocity-location face \\
  --no-align-with-modflow-velocity \\
  --fmm_solver custom \\
  --fmm_backtracking continuous \\
  --fmm_velocity_layout face \\
  --fmm_face_output full \\
  --fmm_face_mapping to_cell_avg \\
  --fmm_zero_policy obstacle \\
  --fmm_min_speed 1e-12 \\
  --fsm_solver ${fsm_solver} \\
  --fsm_max_sweeps ${fsm_max_sweeps} \\
  --fsm_tol ${fsm_tol} \\
  --fsm_dt ${fsm_dt} \\
  --fsm_max_steps ${fsm_max_steps} \\
  ${fm_asr_args} \\
  ${fsm_extra_args}

echo "End Time: \$(date)"
EOF
done
