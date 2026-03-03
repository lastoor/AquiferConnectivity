#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
flowsim_path_default="$(cd "${repo_root}/../../FlowSimulation" && pwd)"
flowsim_path="${FLOWSIM_PATH:-${flowsim_path_default}}"
src_path="${repo_root}/src"

if [[ ! -d "${flowsim_path}" ]]; then
  echo "FlowSimulation path not found: ${flowsim_path}" >&2
  exit 1
fi

if [[ ! -d "${src_path}" ]]; then
  echo "Local src path not found: ${src_path}" >&2
  exit 1
fi

export PYTHONPATH="${src_path}:${flowsim_path}${PYTHONPATH:+:${PYTHONPATH}}"
exec conda run -n taichi_env "$@"
