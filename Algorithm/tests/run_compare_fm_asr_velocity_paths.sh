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
perp_ratio="${PERP_RATIO:-0.25}"
back_factor="${BACK_FACTOR:-4.0}"
min_speed="${MIN_SPEED:-1e-12}"
fmm_solver="${FMM_SOLVER:-scikit}"
fmm_backtracking="${FMM_BACKTRACKING:-continuous}"
max_points="${MAX_POINTS:-100000}"
max_iters="${MAX_ITERS:-1000000}"
asr_max_steps="${ASR_MAX_STEPS:-200000}"
indicatrix_stride="${INDICATRIX_STRIDE:-8}"
indicatrix_glyph_scale="${INDICATRIX_GLYPH_SCALE:-150}"

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
  --indicatrix-stride "${indicatrix_stride}"
  --indicatrix-glyph-scale "${indicatrix_glyph_scale}"
)

if [[ -n "${domain_margin}" ]]; then
  cmd+=(--domain-margin "${domain_margin}")
fi

if [[ "${NO_INDICATRIX:-0}" == "1" || "${NO_INDICATRIX:-false}" == "true" ]]; then
  cmd+=(--no-indicatrix)
fi

cmd+=("$@")
exec "${cmd[@]}"
