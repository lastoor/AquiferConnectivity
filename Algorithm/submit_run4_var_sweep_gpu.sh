#!/bin/bash
set -euo pipefail

# VAR=("0.1" "0.5" "1" "1.5" "2" "2.5" "3" "3.5" "4" "4.5" "5")
VAR=("0.1" "0.5" "1.5" "2" "2.5" "3" "3.5" "4" "4.5")
# VAR=("1" "5")
length_scale="0.05"
run="16_pypar2-al0p1-at0p1_asr-ratio0p05-factor10"
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
fm_asr_perp_ratio="${fm_asr_perp_ratio:-0.05}"
fm_asr_back_factor="${fm_asr_back_factor:-10.0}"
fm_asr_velocity_multiplier="${fm_asr_velocity_multiplier:-1.0}"
fm_asr_max_steps="${fm_asr_max_steps:-1000000}"
fm_asr_min_speed="${fm_asr_min_speed:-}"

# Mixed Randers/conic FM-ASR comparison. The wind is the flow velocity; the
# default self metric is the PyPar2 dispersion velocity ellipse.
fm_asr_conic_enabled="${fm_asr_conic_enabled:-1}"
fm_asr_conic_self_velocity="${fm_asr_conic_self_velocity:-dispersion}"
fm_asr_conic_velocity_multiplier="${fm_asr_conic_velocity_multiplier:-${fm_asr_velocity_multiplier}}"
fm_asr_conic_lambda_threshold="${fm_asr_conic_lambda_threshold:-0.0}"
fm_asr_conic_dispersion_c="${fm_asr_conic_dispersion_c:-1.0}"
fm_asr_conic_dispersion_tau="${fm_asr_conic_dispersion_tau:-1.0}"
fm_asr_conic_dispersion_eigenvalue_floor="${fm_asr_conic_dispersion_eigenvalue_floor:-1e-12}"
fm_asr_conic_stencil_radius="${fm_asr_conic_stencil_radius:-20}"
fm_asr_conic_edge_samples="${fm_asr_conic_edge_samples:-100}"
fm_asr_conic_max_steps="${fm_asr_conic_max_steps:-1000000}"

# PyPar2 Taichi random-walk particle tracking. Set pypar2_enabled=0 to disable.
# Dispersion defaults: weak physical dispersion (Dm=1e-9, alpha_l=0.1, alpha_t=0.01).
# Leave pypar2_velocity_floor empty to let pypar2 auto-derive from Dm/alpha_l.
pypar2_enabled="${pypar2_enabled:-1}"
pypar2_dt="${pypar2_dt:-0.01}"
pypar2_steps="${pypar2_steps:-200000}"
pypar2_dm="${pypar2_dm:-1e-9}"
pypar2_alpha_l="${pypar2_alpha_l:-0.1}"
pypar2_alpha_t="${pypar2_alpha_t:-0.1}"
pypar2_velocity_floor="${pypar2_velocity_floor:-}"
pypar2_taichi_arch="${pypar2_taichi_arch:-gpu}"
pypar2_max_points="${pypar2_max_points:-2000}"
# PyPar2 pilot pass is trajectory-free, so it scales to the same particle
# count as Pollock. Trajectory storage only happens in the small pass-2
# replay around top-K release points, sized by the replay_* parameters below.
pypar2_num_particles="${pypar2_num_particles:-1000000}"
pypar2_trajectory_memory_limit="${pypar2_trajectory_memory_limit:-5000000}"
# Pilot-then-replay parameters. Pass 1 picks the top-K fastest pilot particles;
# pass 2 spawns particles_per_center particles inside a disk of radius
# (radius_cells * dx) around each release point and confirms the trajectory.
# Retried up to max_redos additional times until the replay's fastest time is
# within tolerance * pilot's fastest time.
pypar2_replay_top_k="${pypar2_replay_top_k:-5}"
pypar2_replay_particles_per_center="${pypar2_replay_particles_per_center:-40}"
pypar2_replay_radius_cells="${pypar2_replay_radius_cells:-0.5}"
pypar2_replay_tolerance="${pypar2_replay_tolerance:-1.05}"
pypar2_replay_max_redos="${pypar2_replay_max_redos:-3}"

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
    fm_asr_args+=" --fm_asr_velocity_multiplier ${fm_asr_velocity_multiplier}"
    fm_asr_args+=" --fm_asr_max_steps ${fm_asr_max_steps}"
    if [[ -n "${fm_asr_min_speed}" ]]; then
        fm_asr_args+=" --fm_asr_min_speed ${fm_asr_min_speed}"
    fi
fi

fm_asr_conic_args=""
if [[ "${fm_asr_conic_enabled}" == "0" ]]; then
    fm_asr_conic_args+=" --no-fm-asr-conic"
else
    fm_asr_conic_args+=" --fm_asr_conic"
    fm_asr_conic_args+=" --fm_asr_conic_self_velocity ${fm_asr_conic_self_velocity}"
    fm_asr_conic_args+=" --fm_asr_conic_velocity_multiplier ${fm_asr_conic_velocity_multiplier}"
    fm_asr_conic_args+=" --fm_asr_conic_lambda_threshold ${fm_asr_conic_lambda_threshold}"
    fm_asr_conic_args+=" --fm_asr_conic_dispersion_c ${fm_asr_conic_dispersion_c}"
    fm_asr_conic_args+=" --fm_asr_conic_dispersion_tau ${fm_asr_conic_dispersion_tau}"
    fm_asr_conic_args+=" --fm_asr_conic_dispersion_eigenvalue_floor ${fm_asr_conic_dispersion_eigenvalue_floor}"
    fm_asr_conic_args+=" --fm_asr_conic_stencil_radius ${fm_asr_conic_stencil_radius}"
    fm_asr_conic_args+=" --fm_asr_conic_edge_samples ${fm_asr_conic_edge_samples}"
    fm_asr_conic_args+=" --fm_asr_conic_max_steps ${fm_asr_conic_max_steps}"
fi

pypar2_args=""
if [[ "${pypar2_enabled}" == "0" ]]; then
    pypar2_args+=" --no-pypar2"
else
    pypar2_args+=" --pypar2"
    pypar2_args+=" --pypar2_dt ${pypar2_dt}"
    pypar2_args+=" --pypar2_steps ${pypar2_steps}"
    pypar2_args+=" --pypar2_dm ${pypar2_dm}"
    pypar2_args+=" --pypar2_alpha_l ${pypar2_alpha_l}"
    pypar2_args+=" --pypar2_alpha_t ${pypar2_alpha_t}"
    pypar2_args+=" --pypar2_taichi_arch ${pypar2_taichi_arch}"
    pypar2_args+=" --pypar2_max_points ${pypar2_max_points}"
    pypar2_args+=" --pypar2_num_particles ${pypar2_num_particles}"
    pypar2_args+=" --pypar2_trajectory_memory_limit ${pypar2_trajectory_memory_limit}"
    pypar2_args+=" --pypar2_replay_top_k ${pypar2_replay_top_k}"
    pypar2_args+=" --pypar2_replay_particles_per_center ${pypar2_replay_particles_per_center}"
    pypar2_args+=" --pypar2_replay_radius_cells ${pypar2_replay_radius_cells}"
    pypar2_args+=" --pypar2_replay_tolerance ${pypar2_replay_tolerance}"
    pypar2_args+=" --pypar2_replay_max_redos ${pypar2_replay_max_redos}"
    if [[ -n "${pypar2_velocity_floor}" ]]; then
        pypar2_args+=" --pypar2_velocity_floor ${pypar2_velocity_floor}"
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
  ${fm_asr_conic_args} \\
  ${pypar2_args} \\
  ${fsm_extra_args}

echo "End Time: \$(date)"
EOF
done
