#!/bin/bash
set -euo pipefail

VAR=("0" "0.1" "0.5" "1" "1.5" "2" "2.5" "3" "3.5" "4" "4.5" "5")
length_scale="0.05"
run="8_pollock_gpu_1e6_discrete"
dh="0.1"

num_mc="1000"
field_size="200"
num_particles="1000000"
cpu_workers="6"

time="00:30:00"
mem_per_cpu="2GB"
cpus_per_task="8"
gpus_per_task="1"

account="fbarros_324"
partition="gpu"

repo_root="/home1/binhaoli/FMM_FastestPath/Algorithm"
run_root="/project2/fbarros_324/binhaoli/task8_rMin_tMin/FastMarchingMethod/run${run}"
log_dir="${run_root}/slurm_logs"
mkdir -p "${log_dir}"

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
conda deactivate
conda activate taichi_env
which python

./scripts/run_in_fmm_env.sh python ./scripts/run_fmm_gpu_worker.py \\
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
  --fmm_backtracking discrete \\
  --fmm_velocity_layout face \\
  --fmm_face_output full \\
  --fmm_face_mapping to_cell_avg \\
  --fmm_zero_policy obstacle \\
  --fmm_min_speed 1e-12

echo "End Time: \$(date)"
EOF
done
