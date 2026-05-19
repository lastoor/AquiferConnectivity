#!/bin/bash
set -euo pipefail

# Local Step 5 launcher for Apple Silicon Macs (M-series GPU via Taichi/Metal).
# Override any setting at invocation time, e.g.:
#   STEP5_PYTHON=/path/to/python STEP5_MAC_BACKEND=gpu bash DataAnalysis/submit_step5_mac_m4.sh

logk_source="${LOGK_SOURCE:-synthetic}" # Options: dataset, synthetic, grf
sweep="${SWEEP:-variance}"
value="${VALUE:-5}"
field_size="${FIELD_SIZE:-200}"
mc_index="${MC_INDEX:-34}"
synthetic_amplitude="${SYNTHETIC_AMPLITUDE:-6.0}"
synthetic_frequency_x="${SYNTHETIC_FREQUENCY_X:-2.0}"
synthetic_frequency_y="${SYNTHETIC_FREQUENCY_Y:-0.5}"
synthetic_offset="${SYNTHETIC_OFFSET:--3.0}"

grf_std="${GRF_STD:-2.25}"
grf_length_scale="${GRF_LENGTH_SCALE:-}"
grf_seed="${GRF_SEED:-2}"
grf_minpadding="${GRF_MINPADDING:-0}"

run_tag="${RUN_TAG:-mac_m4_step5}"
num_particles="${NUM_PARTICLES:-1000000}"
dh="${DH:-}"

fsm_fp="${FSM_FP:-true}"
fsm_max_sweeps="${FSM_MAX_SWEEPS:-100000}"
fsm_tol="${FSM_TOL:-1e-8}"
fsm_mse_tol="${FSM_MSE_TOL:-}"
fsm_convergence_mode="${FSM_CONVERGENCE_MODE:-}"
fsm_mse_check_every="${FSM_MSE_CHECK_EVERY:-5}"
fsm_mse_activation_factor="${FSM_MSE_ACTIVATION_FACTOR:-10.0}"
fsm_consecutive_passes="${FSM_CONSECUTIVE_PASSES:-2}"
fsm_solver="${FSM_SOLVER:-fd}" # Options: sweep, fipy, legacy, new, fd
fsm_dt="${FSM_DT:-0.5}"
fsm_max_steps="${FSM_MAX_STEPS:-1000000}"

fm_asr_fp="${FM_ASR_FP:-true}"
fm_asr_perp_ratio="${FM_ASR_PERP_RATIO:-0.1}"
fm_asr_back_factor="${FM_ASR_BACK_FACTOR:-10.0}"
fm_asr_min_speed="${FM_ASR_MIN_SPEED:-1e-12}"
fm_asr_max_steps="${FM_ASR_MAX_STEPS:-200000}"

linearized_fp="${LINEARIZED_FP:-false}"
linearized_scheme="${LINEARIZED_SCHEME:-linearized}" # Options: linearized, full_eikonal
linearized_outer_max="${LINEARIZED_OUTER_MAX:-10}"
linearized_inner_max_sweeps="${LINEARIZED_INNER_MAX_SWEEPS:-200}"
linearized_inner_tol="${LINEARIZED_INNER_TOL:-1e-8}"
linearized_eikonal_tol="${LINEARIZED_EIKONAL_TOL:-1e-6}"
linearized_damping="${LINEARIZED_DAMPING:-0.8}"
linearized_max_deltat="${LINEARIZED_MAX_DELTAT:-}"
linearized_monotone_residual="${LINEARIZED_MONOTONE_RESIDUAL:-true}"
linearized_verbose="${LINEARIZED_VERBOSE:-true}"

check_path_geometry="${CHECK_PATH_GEOMETRY:-false}"
source_x="${SOURCE_X:-}"
target_x="${TARGET_X:-}"
source_y_start="${SOURCE_Y_START:-}"
source_y_end="${SOURCE_Y_END:-}"
target_y_start="${TARGET_Y_START:-}"
target_y_end="${TARGET_Y_END:-}"

curvature_resample_ds="${CURVATURE_RESAMPLE_DS:-1.0}"
curvature_method="${CURVATURE_METHOD:-savgol}" # Options: spline, savgol, circle
curvature_spline_mode="${CURVATURE_SPLINE_MODE:-approximation}" # Options: approximation, interpolation
curvature_spline_smooth="${CURVATURE_SPLINE_SMOOTH:-1}"
curvature_spline_degree="${CURVATURE_SPLINE_DEGREE:-4}"
curvature_field_smooth_method="${CURVATURE_FIELD_SMOOTH_METHOD:-none}" # Options: none, gaussian
curvature_field_smooth_sigma="${CURVATURE_FIELD_SMOOTH_SIGMA:-2.0}"
curvature_circle_window="${CURVATURE_CIRCLE_WINDOW:-20}"
curvature_savgol_window="${CURVATURE_SAVGOL_WINDOW:-3}"
curvature_savgol_poly="${CURVATURE_SAVGOL_POLY:-2}"
normal_deriv_method="${NORMAL_DERIV_METHOD:-hessian_T}" # Options: direct, savgol2d, spline2d, mls_frenet, hessian_T
normal_deriv_spline2d_smooth="${NORMAL_DERIV_SPLINE2D_SMOOTH:-1.0}"
normal_deriv_mls_ns="${NORMAL_DERIV_MLS_NS:-2}"
normal_deriv_mls_nn="${NORMAL_DERIV_MLS_NN:-2}"
normal_deriv_hessian_smooth="${NORMAL_DERIV_HESSIAN_SMOOTH:-0.0}"
normal_deriv_hessian_window="${NORMAL_DERIV_HESSIAN_WINDOW:-5}"
normal_deriv_hessian_poly="${NORMAL_DERIV_HESSIAN_POLY:-3}"
fm_asr_hamiltonian_t_hessian="${FM_ASR_HAMILTONIAN_T_HESSIAN:-grid_savgol}" # Options: grid_savgol, frenet_quadratic
fm_asr_hamiltonian_p_dot="${FM_ASR_HAMILTONIAN_P_DOT:-canonical}" # Options: canonical, hessian_T

snapshot_mode="${SNAPSHOT_MODE:-fraction_of_first_arrival}" # Options: fraction_of_first_arrival, dt_interval, physical, or "" to disable
snapshot_times="${SNAPSHOT_TIMES:-0.25 0.5 0.75 1.0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="${CONFIG_PATH:-${repo_root}/configs/visualization.default.yaml}"
run_root="${RUN_ROOT:-${repo_root}/outputs/visualization/path/${run_tag}}"

flowsim_default="$(cd "${repo_root}/../.." && pwd)/FlowSimulation"
export FLOWSIM_PATH="${FLOWSIM_PATH:-${flowsim_default}}"

step5_python="${STEP5_PYTHON:-/Users/binhaoli/.conda/envs/taichi/bin/python}"
step5_pythonpath="${STEP5_PYTHONPATH:-${repo_root}/scripts}"
script_path="${repo_root}/scripts/step5_single_field_paths.py"
backend="${STEP5_MAC_BACKEND:-gpu}" # Options: gpu, cpu

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This launcher is intended for macOS/Apple Silicon. Use submit_step5_gpu.sh for Slurm/Linux." >&2
    exit 1
fi

if [[ ! -f "${config_path}" ]]; then
    echo "Config does not exist: ${config_path}" >&2
    exit 1
fi

if [[ ! -d "${FLOWSIM_PATH}" ]]; then
    echo "FlowSimulation path does not exist: ${FLOWSIM_PATH}" >&2
    echo "Set FLOWSIM_PATH=/path/to/FlowSimulation and rerun." >&2
    exit 1
fi

dh_args=()
if [[ -n "${dh}" ]]; then
    dh_args=(--dh "${dh}")
fi

check_path_geometry_args=(--no-check-path-geometry)
if [[ "${check_path_geometry}" == "true" ]]; then
    check_path_geometry_args=(--check-path-geometry)
fi

fsm_args=()
if [[ "${fsm_fp}" == "true" ]]; then
    fsm_args+=(--fsm-fp)
else
    fsm_args+=(--no-fsm-fp)
fi
fsm_args+=(--fsm-max-sweeps "${fsm_max_sweeps}")
fsm_args+=(--fsm-tol "${fsm_tol}")
if [[ "${fsm_solver}" != "fd" ]]; then
    if [[ -n "${fsm_mse_tol}" ]]; then
        fsm_args+=(--fsm-mse-tol "${fsm_mse_tol}")
    fi
    if [[ -n "${fsm_convergence_mode}" ]]; then
        fsm_args+=(--fsm-convergence-mode "${fsm_convergence_mode}")
    fi
    fsm_args+=(--fsm-mse-check-every "${fsm_mse_check_every}")
    fsm_args+=(--fsm-mse-activation-factor "${fsm_mse_activation_factor}")
    fsm_args+=(--fsm-consecutive-passes "${fsm_consecutive_passes}")
fi
fsm_args+=(--fsm-solver "${fsm_solver}")
fsm_args+=(--fsm-dt "${fsm_dt}")
fsm_args+=(--fsm-max-steps "${fsm_max_steps}")

fm_asr_args=()
if [[ "${fm_asr_fp}" == "true" ]]; then
    fm_asr_args+=(--fm-asr-fp)
else
    fm_asr_args+=(--no-fm-asr-fp)
fi
fm_asr_args+=(--fm-asr-perp-ratio "${fm_asr_perp_ratio}")
fm_asr_args+=(--fm-asr-back-factor "${fm_asr_back_factor}")
fm_asr_args+=(--fm-asr-min-speed "${fm_asr_min_speed}")
fm_asr_args+=(--fm-asr-max-steps "${fm_asr_max_steps}")

linearized_args=()
if [[ "${linearized_fp}" == "true" ]]; then
    linearized_args+=(--linearized-fp)
    linearized_args+=(--linearized-scheme "${linearized_scheme}")
    linearized_args+=(--linearized-outer-max "${linearized_outer_max}")
    linearized_args+=(--linearized-inner-max-sweeps "${linearized_inner_max_sweeps}")
    linearized_args+=(--linearized-inner-tol "${linearized_inner_tol}")
    linearized_args+=(--linearized-eikonal-tol "${linearized_eikonal_tol}")
    linearized_args+=(--linearized-damping "${linearized_damping}")
    if [[ -n "${linearized_max_deltat}" ]]; then
        linearized_args+=(--linearized-max-deltat "${linearized_max_deltat}")
    fi
    if [[ "${linearized_monotone_residual}" == "true" ]]; then
        linearized_args+=(--linearized-monotone-residual)
    else
        linearized_args+=(--no-linearized-monotone-residual)
    fi
    if [[ "${linearized_verbose}" == "true" ]]; then
        linearized_args+=(--linearized-verbose)
    else
        linearized_args+=(--no-linearized-verbose)
    fi
fi

snapshot_args=()
if [[ -n "${snapshot_mode}" ]]; then
    snapshot_args+=(--snapshot-mode "${snapshot_mode}")
    read -r -a snapshot_time_values <<< "${snapshot_times}"
    snapshot_args+=(--snapshot-times "${snapshot_time_values[@]}")
fi

geometry_args=()
if [[ -n "${source_x}" ]]; then geometry_args+=(--source-x "${source_x}"); fi
if [[ -n "${target_x}" ]]; then geometry_args+=(--target-x "${target_x}"); fi
if [[ -n "${source_y_start}" ]]; then geometry_args+=(--source-y-start "${source_y_start}"); fi
if [[ -n "${source_y_end}" ]]; then geometry_args+=(--source-y-end "${source_y_end}"); fi
if [[ -n "${target_y_start}" ]]; then geometry_args+=(--target-y-start "${target_y_start}"); fi
if [[ -n "${target_y_end}" ]]; then geometry_args+=(--target-y-end "${target_y_end}"); fi

logk_args=(--logk-source "${logk_source}" --field-size "${field_size}")
if [[ "${logk_source}" == "dataset" ]]; then
    logk_args+=(--sweep "${sweep}" --value "${value}" --mc-index "${mc_index}")
    output_dir="${OUTPUT_DIR:-${run_root}/${sweep}_${value}_size${field_size}_mc${mc_index}}"
elif [[ "${logk_source}" == "synthetic" ]]; then
    logk_args+=(--synthetic-amplitude "${synthetic_amplitude}")
    logk_args+=(--synthetic-frequency-x "${synthetic_frequency_x}")
    logk_args+=(--synthetic-frequency-y "${synthetic_frequency_y}")
    logk_args+=(--synthetic-offset "${synthetic_offset}")
    output_dir="${OUTPUT_DIR:-${run_root}/synthetic_size${field_size}_amp${synthetic_amplitude}_fx${synthetic_frequency_x}_fy${synthetic_frequency_y}_offset${synthetic_offset}}"
elif [[ "${logk_source}" == "grf" ]]; then
    logk_args+=(--grf-std "${grf_std}" --grf-seed "${grf_seed}" --grf-minpadding "${grf_minpadding}")
    if [[ -n "${grf_length_scale}" ]]; then
        logk_args+=(--grf-length-scale "${grf_length_scale}")
    fi
    output_dir="${OUTPUT_DIR:-${run_root}/grf_size${field_size}_std${grf_std}_seed${grf_seed}}"
else
    echo "Unsupported LOGK_SOURCE=${logk_source}; expected dataset, synthetic, or grf." >&2
    exit 1
fi

if [[ "${backend}" != "gpu" && "${backend}" != "cpu" ]]; then
    echo "Unsupported STEP5_MAC_BACKEND=${backend}; expected gpu or cpu." >&2
    exit 1
fi

if [[ "${backend}" == "gpu" ]]; then
    if ! env PYTHONPATH="${step5_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}" FLOWSIM_PATH="${FLOWSIM_PATH}" \
        "${step5_python}" -c 'import taichi as ti; ti.init(arch=ti.gpu, verbose=False)' >/dev/null 2>&1; then
        echo "Taichi GPU preflight failed. On Apple Silicon, install/use a Python env with Taichi Metal support or run STEP5_MAC_BACKEND=cpu." >&2
        exit 1
    fi
elif [[ ${#snapshot_args[@]} -gt 0 ]]; then
    echo "Snapshot capture requires --backend gpu; disabling snapshots for CPU run." >&2
    snapshot_args=()
fi

mkdir -p "${output_dir}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${step5_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Start Time: $(date)"
echo "Execution mode: local-mac backend=${backend}"
echo "Python: ${step5_python}"
echo "Repo root: ${repo_root}"
echo "Config path: ${config_path}"
echo "Output dir: ${output_dir}"
echo "FlowSimulation path: ${FLOWSIM_PATH}"
echo "logK source: ${logk_source}"
echo "FSM FP: ${fsm_fp} (solver=${fsm_solver}, max_sweeps=${fsm_max_sweeps}, tol=${fsm_tol}, dt=${fsm_dt}, max_steps=${fsm_max_steps})"
echo "FM-ASR FP: ${fm_asr_fp} (perp_ratio=${fm_asr_perp_ratio}, back_factor=${fm_asr_back_factor}, min_speed=${fm_asr_min_speed}, max_steps=${fm_asr_max_steps}, T_hessian=${fm_asr_hamiltonian_t_hessian}, p_dot=${fm_asr_hamiltonian_p_dot})"
echo "Linearized FP: ${linearized_fp} (${linearized_scheme})"
echo "Snapshot mode: ${snapshot_mode:-disabled}, times: ${snapshot_times:-n/a}"

"${step5_python}" "${script_path}" \
    --config "${config_path}" \
    "${logk_args[@]}" \
    --output-dir "${output_dir}" \
    ${dh_args[@]+"${dh_args[@]}"} \
    --solver pollock \
    --num-particles "${num_particles}" \
    --velocity-location face \
    --no-align-with-modflow-velocity \
    --fmm-backtracking continuous \
    --fmm-continuous-step-size 0.1 \
    "${fsm_args[@]}" \
    "${fm_asr_args[@]}" \
    ${linearized_args[@]+"${linearized_args[@]}"} \
    "${check_path_geometry_args[@]}" \
    ${geometry_args[@]+"${geometry_args[@]}"} \
    --curvature-resample-ds "${curvature_resample_ds}" \
    --curvature-method "${curvature_method}" \
    --curvature-spline-mode "${curvature_spline_mode}" \
    --curvature-spline-smooth "${curvature_spline_smooth}" \
    --curvature-spline-degree "${curvature_spline_degree}" \
    --curvature-field-smooth-method "${curvature_field_smooth_method}" \
    --curvature-field-smooth-sigma "${curvature_field_smooth_sigma}" \
    --curvature-circle-window "${curvature_circle_window}" \
    --curvature-savgol-window "${curvature_savgol_window}" \
    --curvature-savgol-poly "${curvature_savgol_poly}" \
    --normal-deriv-method "${normal_deriv_method}" \
    --normal-deriv-spline2d-smooth "${normal_deriv_spline2d_smooth}" \
    --normal-deriv-mls-ns "${normal_deriv_mls_ns}" \
    --normal-deriv-mls-nn "${normal_deriv_mls_nn}" \
    --normal-deriv-hessian-smooth "${normal_deriv_hessian_smooth}" \
    --normal-deriv-hessian-window "${normal_deriv_hessian_window}" \
    --normal-deriv-hessian-poly "${normal_deriv_hessian_poly}" \
    --fm-asr-hamiltonian-t-hessian "${fm_asr_hamiltonian_t_hessian}" \
    --fm-asr-hamiltonian-p-dot "${fm_asr_hamiltonian_p_dot}" \
    --backend "${backend}" \
    ${snapshot_args[@]+"${snapshot_args[@]}"}

echo "End Time: $(date)"
