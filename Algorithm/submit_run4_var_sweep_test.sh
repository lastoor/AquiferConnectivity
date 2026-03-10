#!/bin/bash
set -euo pipefail

VAR=("0.1" "5") 
# VAR=("0.1" "5")
length_scale="0.05"
run="4_test"
dh="0.1"

# Keep original workload profile
num_mc="1000"
# num_mc="10"
field_size="200"

# Keep original Slurm CPU profile
time="1:00:00"
# time="00:10:00"
mem_per_cpu="8GB"
cpus_per_task="10"
# cpus_per_task="2"

account="fbarros_324"
partition="main"

repo_root="/home1/binhaoli/FMM_FastestPath/Algorithm"
run_root="/project2/fbarros_324/binhaoli/task8_rMin_tMin/FastMarchingMethod/run${run}"
# run_root="/home1/binhaoli/FMM_FastestPath/Algorithm/output/bias_check_taichi200_50_fix"
log_dir="${run_root}/slurm_logs"
mkdir -p "${log_dir}"

for var in "${VAR[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --job-name=var_${var}
#SBATCH --output=${log_dir}/output_var${var}_%j.txt
#SBATCH --error=${log_dir}/error_var${var}_%j.txt
#SBATCH --partition=${partition}
#SBATCH --time=${time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=${mem_per_cpu}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --chdir=${repo_root}

echo "Running var=${var} on node \$SLURM_NODELIST"
echo "Start Time: \$(date)"

# Keep threading controlled for stability
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda deactivate
conda activate taichi_env
which python

./scripts/run_in_fmm_env.sh python ./scripts/run_fmm.py \\
  --path_workdir ${run_root}/var_${var} \\
  --num_mc ${num_mc} \\
  --field_size ${field_size} \\
  --length_scale ${length_scale} \\
  --minpadding 0 \\
  --var ${var} \\
  --dh ${dh} \\
  --pt_velocity_layout face \\
  --seed 0 \\
  --seed_mode strict_mc \\
  --variance_seed_policy same \\
  --fmm_velocity_layout face \\
  --fmm_face_mapping to_cell_avg \\
  --fmm_zero_policy obstacle \\
  --fmm_min_speed 1e-12

echo "End Time: \$(date)"
EOF
done
