#!/bin/bash
set -euo pipefail

# Execution mode: slurm (default) or reload (local visualization from saved NPZ).
run_mode="slurm"

# Reload mode settings (used when run_mode="reload").
load_npz=""
reload_output_dir=""
figure_dpi=""

field_size="500"
length_scale="0.02"
mode="var"
mode_value="5.0"
n_mc="100"
seed_start="0"

run_tag="cdf_fp_lrp"
num_particles="1000000"
dh=""
k_g=""
source_x="49"
target_x="449"
source_y_start="49"
source_y_end="449"
target_y_start="0"
target_y_end="499"

time="00:60:00" #"00:30:00"
cpus_per_task="8"
mem_per_cpu="4GB"
gpus_per_task="1"

account="fbarros_324"
partition="gpu"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="${repo_root}/configs/visualization.default.yaml"
run_root="/project2/fbarros_324/binhaoli/FMM_FastestPath/outputs/visualization/cdf_travel_time/${run_tag}"
# run_root="${repo_root}/outputs/visualization/cdf_travel_time/${run_tag}"
log_dir="${run_root}/slurm_logs"

step6_python="${STEP6_PYTHON:-python3}"
step6_pythonpath="${STEP6_PYTHONPATH:-${repo_root}/scripts}"
script_path="${repo_root}/scripts/step6_single_field_arrival_hist.py"

tau_split="10"
tau_grid_fine="1024"
tau_grid_coarse="256"
y_exp="0.3"
pdf_hist_bins="48"

usage() {
    cat <<'EOF' >&2
Usage:
  submit_step6_gpu.sh
  submit_step6_gpu.sh reload NPZ_PATH [OUTPUT_DIR]

Modes:
  slurm   Submit a full Step 6 GPU solve via sbatch (default).
  reload  Regenerate CDF/PDF figures locally from a saved *_arrival_times.npz.

Environment overrides:
  STEP6_LOAD_NPZ=/path/to/file.npz   Force reload mode.
  STEP6_PYTHON=/path/to/python       Python interpreter for local reload.
  STEP6_PYTHONPATH=/path/to/scripts  PYTHONPATH for local reload.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "reload" ]]; then
    run_mode="reload"
    if [[ $# -lt 2 ]]; then
        echo "reload mode requires an NPZ path." >&2
        usage
        exit 1
    fi
    load_npz="${2}"
    reload_output_dir="${3:-}"
    shift $#
fi

if [[ -n "${STEP6_LOAD_NPZ:-}" ]]; then
    run_mode="reload"
    load_npz="${STEP6_LOAD_NPZ}"
    if [[ -z "${reload_output_dir}" && -n "${STEP6_RELOAD_OUTPUT_DIR:-}" ]]; then
        reload_output_dir="${STEP6_RELOAD_OUTPUT_DIR}"
    fi
fi

reload_plot_args=(
  --refine-tau-grid
  --tau-split "${tau_split}"
  --tau-grid-fine "${tau_grid_fine}"
  --tau-grid-coarse "${tau_grid_coarse}"
  --y-exp "${y_exp}"
  --save-pdf-plot
)

figure_dpi_args=()
if [[ -n "${figure_dpi}" ]]; then
    figure_dpi_args=(--figure-dpi "${figure_dpi}")
fi

reload_output_args=()
if [[ -n "${reload_output_dir}" ]]; then
    reload_output_args=(--output-dir "${reload_output_dir}")
fi

activate_step6_env() {
    if [[ -x "${HOME}/.conda/envs/taichi_env/bin/python" ]]; then
        step6_python="${HOME}/.conda/envs/taichi_env/bin/python"
        return 0
    fi
    if [[ -f /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh ]]; then
        set +u
        # shellcheck disable=SC1091
        source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
        set -u
        conda activate taichi_env
        step6_python="$(command -v python)"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        set +u
        # shellcheck disable=SC1090
        [[ -f "${HOME}/.bashrc" ]] && source "${HOME}/.bashrc"
        set -u
        conda deactivate >/dev/null 2>&1 || true
        conda activate taichi_env
        step6_python="$(command -v python)"
    fi
}

run_reload_local() {
    if [[ -z "${load_npz}" ]]; then
        echo "reload mode requires load_npz or STEP6_LOAD_NPZ." >&2
        usage
        exit 1
    fi
    if [[ ! -f "${load_npz}" ]]; then
        echo "NPZ file not found: ${load_npz}" >&2
        exit 1
    fi

    if [[ -n "${reload_output_dir}" ]]; then
        mkdir -p "${reload_output_dir}"
    fi

    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="${step6_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"

    activate_step6_env

    echo "Start Time: $(date)"
    echo "Execution mode: reload (local, visualization only)"
    echo "NPZ path: ${load_npz}"
    if [[ -n "${reload_output_dir}" ]]; then
        echo "Output dir: ${reload_output_dir}"
    else
        echo "Output dir: $(cd "$(dirname "${load_npz}")" && pwd)"
    fi

    "${step6_python}" "${script_path}" \
      --load-npz "${load_npz}" \
      "${reload_plot_args[@]}" \
      "${reload_output_args[@]}" \
      "${figure_dpi_args[@]}"

    echo "End Time: $(date)"
}

run_slurm() {
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
  --tau-split ${tau_split} \\
  --tau-grid-fine ${tau_grid_fine} \\
  --tau-grid-coarse ${tau_grid_coarse} \\
  --y-exp ${y_exp} \\
  --pdf-hist-bins ${pdf_hist_bins} \\
  --save-pdf-plot \\
  --save-npz \\
  --save-json \\
  --output-dir "\${output_dir}"

echo "End Time: \$(date)"
EOF
}

case "${run_mode}" in
    reload)
        run_reload_local
        ;;
    slurm)
        if ! command -v sbatch >/dev/null 2>&1; then
            echo "sbatch not found; use reload mode or run on a Slurm login node." >&2
            usage
            exit 1
        fi
        run_slurm
        ;;
    *)
        echo "Unsupported run_mode=${run_mode}; expected slurm or reload." >&2
        exit 1
        ;;
esac
