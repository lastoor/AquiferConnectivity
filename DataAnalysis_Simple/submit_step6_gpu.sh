#!/bin/bash
set -euo pipefail

field_size="500"
length_scale="0.02"
mode="var"
mode_value="1.0"
n_mc="50"
seed_start="0"

run_tag="gpu_pollock"
num_particles="1000000"
dh=""
k_g=""
source_x="49"
target_x="449"
source_y_start="49"
source_y_end="449"
target_y_start="0"
target_y_end="499"

time="00:30:00"
cpus_per_task="8"
mem_per_cpu="4GB"
gpus_per_task="1"

account="fbarros_324"
partition="gpu"

repo_root="/home1/binhaoli/FMM_FastestPath/DataAnalysis_Simple"
config_path="${repo_root}/configs/visualization.default.yaml"
run_root="/project2/fbarros_324/binhaoli/FMM_FastestPath/outputs/visualization/cdf_travel_time/${run_tag}"
# run_root="${repo_root}/outputs/visualization/cdf_travel_time/${run_tag}"
log_dir="${run_root}/slurm_logs"
mkdir -p "${log_dir}"

if [[ "${mode}" == "var" ]]; then
    stat_args="--var ${mode_value}"
elif [[ "${mode}" == "std" ]]; then
    stat_args="--std ${mode_value}"
else
    echo "mode must be 'var' or 'std'" >&2
    exit 1
fi

length_scale_args=""
if [[ -n "${length_scale}" ]]; then
    length_scale_args="--length-scale ${length_scale}"
fi

dh_args=""
if [[ -n "${dh}" ]]; then
    dh_args="--dh ${dh}"
fi

k_g_args=""
if [[ -n "${k_g}" ]]; then
    k_g_args="--k-g ${k_g}"
fi

source_x_args=""
if [[ -n "${source_x}" ]]; then
    source_x_args="--source-x ${source_x}"
fi

target_x_args=""
if [[ -n "${target_x}" ]]; then
    target_x_args="--target-x ${target_x}"
fi

source_y_start_args=""
if [[ -n "${source_y_start}" ]]; then
    source_y_start_args="--source-y-start ${source_y_start}"
fi

source_y_end_args=""
if [[ -n "${source_y_end}" ]]; then
    source_y_end_args="--source-y-end ${source_y_end}"
fi

target_y_start_args=""
if [[ -n "${target_y_start}" ]]; then
    target_y_start_args="--target-y-start ${target_y_start}"
fi

target_y_end_args=""
if [[ -n "${target_y_end}" ]]; then
    target_y_end_args="--target-y-end ${target_y_end}"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --job-name=step6_gpu_${mode}_${mode_value}
#SBATCH --output=${log_dir}/output_${mode}_${mode_value}_%j.txt
#SBATCH --error=${log_dir}/error_${mode}_${mode_value}_%j.txt
#SBATCH --partition=${partition}
#SBATCH --time=${time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=${gpus_per_task}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem-per-cpu=${mem_per_cpu}
#SBATCH --chdir=${repo_root}

echo "Running step6 GPU Pollock on node \$SLURM_NODELIST"
echo "Start Time: \$(date)"
echo "Step6 geometry: source_x=${source_x:-default}, target_x=${target_x:-default}, source_y=[${source_y_start:-default},${source_y_end:-default}), target_y=[${target_y_start:-default},${target_y_end:-default})"
echo "Note: target-y bounds are currently saved as metadata only; arrival stopping still uses target_x."

output_dir="${run_root}/${mode}_${mode_value}_size${field_size}_seed${seed_start}_mc${n_mc}"
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

python ./scripts/step6_single_field_arrival_hist.py \\
  --config "${config_path}" \\
  --field-size "${field_size}" \\
  ${length_scale_args} \\
  ${stat_args} \\
  ${dh_args} \\
  ${k_g_args} \\
  ${source_x_args} \\
  ${target_x_args} \\
  ${source_y_start_args} \\
  ${source_y_end_args} \\
  ${target_y_start_args} \\
  ${target_y_end_args} \\
  --n-mc "${n_mc}" \\
  --seed-start "${seed_start}" \\
  --solver pollock \\
  --backend gpu \\
  --num-particles "${num_particles}" \\
  --velocity-location face \\
  --no-align-with-modflow-velocity \\
  --refine-tau-grid \\
  --tau-split 10 \\
  --tau-grid-fine 1024 \\
  --tau-grid-coarse 256 \\
  --y-exp 0.3 \\
  --pdf-hist-bins 48 \\
  --save-pdf-plot \\
  --save-npz \\
  --save-json \\
  --output-dir "\${output_dir}"

echo "End Time: \$(date)"
EOF
