#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
flowsim_path="${FLOWSIM_PATH:-$(cd "${repo_root}/../../FlowSimulation" && pwd)}"
conda_env="${CONDA_ENV:-taichi}"

export PYTHONPATH="${repo_root}/src:${flowsim_path}${PYTHONPATH:+:${PYTHONPATH}}"

output_dir="${OUTPUT_DIR:-${repo_root}/output/fm_asr_demo}"
field_size="${FIELD_SIZE:-100}"
seed="${SEED:-12345}"
covariance="${COVARIANCE:-exp}"
length_scale="${LENGTH_SCALE:-0.1}"
std="${STD:-2.0}"
fbm_alpha="${FBM_ALPHA:-0.8}"
fbm_c="${FBM_C:-1.0}"
minpadding="${MINPADDING:-0}"
domain_margin="${DOMAIN_MARGIN:-}"
dh="${DH:-0.01}"
num_particles="${NUM_PARTICLES:-20000}"
perp_ratio="${PERP_RATIO:-0.2}"
back_factor="${BACK_FACTOR:-5.0}"
min_speed="${MIN_SPEED:-1e-12}"
fmm_solver="${FMM_SOLVER:-scikit}"
fmm_backtracking="${FMM_BACKTRACKING:-continuous}"
max_points="${MAX_POINTS:-100000}"
max_iters="${MAX_ITERS:-1000000}"
asr_max_steps="${ASR_MAX_STEPS:-200000}"
conic_asr_enabled="${CONIC_ASR:-1}"
conic_asr_velocity_multiplier="${CONIC_ASR_VELOCITY_MULTIPLIER:-1.0}"
conic_asr_self_velocity="${CONIC_ASR_SELF_VELOCITY:-dispersion}"
conic_asr_min_speed="${CONIC_ASR_MIN_SPEED:-1.000000001}"
conic_asr_min_wind_factor="${CONIC_ASR_MIN_WIND_FACTOR:-1.000000001}"
conic_asr_dispersion_eigenvalue_floor="${CONIC_ASR_DISPERSION_EIGENVALUE_FLOOR:-1e-12}"
conic_asr_stencil_radius="${CONIC_ASR_STENCIL_RADIUS:-8}"
conic_asr_edge_samples="${CONIC_ASR_EDGE_SAMPLES:-64}"
pypar2_enabled="${PYPAR2:-1}"
pypar2_dt="${PYPAR2_DT:-0.01}"
pypar2_steps="${PYPAR2_STEPS:-200000}"
pypar2_dm="${PYPAR2_DM:-1e-9}"
pypar2_alpha_l="${PYPAR2_ALPHA_L:-0.1}"
pypar2_alpha_t="${PYPAR2_ALPHA_T:-0.01}"
pypar2_velocity_floor="${PYPAR2_VELOCITY_FLOOR:-}"
pypar2_taichi_arch="${PYPAR2_TAICHI_ARCH:-gpu}"
pypar2_max_points="${PYPAR2_MAX_POINTS:-2000}"
pypar2_num_particles="${PYPAR2_NUM_PARTICLES:-}"
pypar2_trajectory_memory_limit="${PYPAR2_TRAJECTORY_MEMORY_LIMIT:-5000000}"
pypar2_replay_top_k="${PYPAR2_REPLAY_TOP_K:-5}"
pypar2_replay_particles_per_center="${PYPAR2_REPLAY_PARTICLES_PER_CENTER:-40}"
pypar2_replay_radius_cells="${PYPAR2_REPLAY_RADIUS_CELLS:-0.5}"
pypar2_replay_tolerance="${PYPAR2_REPLAY_TOLERANCE:-1.05}"
pypar2_replay_max_redos="${PYPAR2_REPLAY_MAX_REDOS:-3}"
indicatrix_stride="${INDICATRIX_STRIDE:-9}"
indicatrix_glyph_scale="${INDICATRIX_GLYPH_SCALE:-400}"

align_flag="--no-align-with-modflow-velocity"
if [[ "${ALIGN_WITH_MODFLOW_VELOCITY:-0}" == "1" || "${ALIGN_WITH_MODFLOW_VELOCITY:-false}" == "true" ]]; then
  align_flag="--align-with-modflow-velocity"
fi

cmd=(
  conda run -n "${conda_env}" python "${script_dir}/compare_fm_asr_velocity_paths.py"
  --output-dir "${output_dir}"
  --field-size "${field_size}"
  --seed "${seed}"
  --covariance "${covariance}"
  --length-scale "${length_scale}"
  --std "${std}"
  --fbm-alpha "${fbm_alpha}"
  --fbm-c "${fbm_c}"
  --minpadding "${minpadding}"
  --dh "${dh}"
  --num-particles "${num_particles}"
  --perp-ratio "${perp_ratio}"
  --back-factor "${back_factor}"
  --min-speed "${min_speed}"
  --fmm-solver "${fmm_solver}"
  --fmm-backtracking "${fmm_backtracking}"
  "${align_flag}"
  --max-points "${max_points}"
  --max-iters "${max_iters}"
  --asr-max-steps "${asr_max_steps}"
  --conic-asr-velocity-multiplier "${conic_asr_velocity_multiplier}"
  --conic-asr-self-velocity "${conic_asr_self_velocity}"
  --conic-asr-min-speed "${conic_asr_min_speed}"
  --conic-asr-min-wind-factor "${conic_asr_min_wind_factor}"
  --conic-asr-dispersion-eigenvalue-floor "${conic_asr_dispersion_eigenvalue_floor}"
  --conic-asr-stencil-radius "${conic_asr_stencil_radius}"
  --conic-asr-edge-samples "${conic_asr_edge_samples}"
  --pypar2-dt "${pypar2_dt}"
  --pypar2-steps "${pypar2_steps}"
  --pypar2-dm "${pypar2_dm}"
  --pypar2-alpha-l "${pypar2_alpha_l}"
  --pypar2-alpha-t "${pypar2_alpha_t}"
  --pypar2-taichi-arch "${pypar2_taichi_arch}"
  --pypar2-max-points "${pypar2_max_points}"
  --pypar2-trajectory-memory-limit "${pypar2_trajectory_memory_limit}"
  --pypar2-replay-top-k "${pypar2_replay_top_k}"
  --pypar2-replay-particles-per-center "${pypar2_replay_particles_per_center}"
  --pypar2-replay-radius-cells "${pypar2_replay_radius_cells}"
  --pypar2-replay-tolerance "${pypar2_replay_tolerance}"
  --pypar2-replay-max-redos "${pypar2_replay_max_redos}"
  --indicatrix-stride "${indicatrix_stride}"
  --indicatrix-glyph-scale "${indicatrix_glyph_scale}"
)

if [[ "${conic_asr_enabled}" == "0" || "${conic_asr_enabled}" == "false" ]]; then
  cmd+=(--no-conic-asr)
else
  cmd+=(--conic-asr)
fi

if [[ "${pypar2_enabled}" == "0" || "${pypar2_enabled}" == "false" ]]; then
  cmd+=(--no-pypar2)
else
  cmd+=(--pypar2)
fi

if [[ -n "${pypar2_velocity_floor}" ]]; then
  cmd+=(--pypar2-velocity-floor "${pypar2_velocity_floor}")
fi

if [[ -n "${pypar2_num_particles}" ]]; then
  cmd+=(--pypar2-num-particles "${pypar2_num_particles}")
fi

if [[ -n "${domain_margin}" ]]; then
  cmd+=(--domain-margin "${domain_margin}")
fi

if [[ "${NO_INDICATRIX:-0}" == "1" || "${NO_INDICATRIX:-false}" == "true" ]]; then
  cmd+=(--no-indicatrix)
fi

cmd+=("$@")
exec "${cmd[@]}"
