import csv
import os
import re
import subprocess
from subprocess import call
import yaml
import time
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
import sys
import glob
import shutil
from IPython.display import clear_output
from pathlib import Path
import logging
import gc
import threading
import cProfile
import pstats
import uuid
import warnings
import argparse
from scipy.spatial import cKDTree

import petsc4py
petsc4py.init()
from petsc4py import PETSc
# from numba import cuda, float32, int32
from numba import njit
import math
import cupy as cp
from cupy.fft import fftfreq, fft2, fftshift
import multiprocessing as mp
from scipy.fft import fftn, dctn, next_fast_len
import numpy as np
import heapq
from collections import Counter

# Usage: python /home1/binhaoli/flow_simulation/test11_op_pressureField/run_cpu.py --path_workdir /home1/binhaoli/flow_simulation/test11_op_pressureField --num_mc 10 --field_size 100 --length_scale 0.1 --minpadding 0 --std 1

# Add libgfortran.so.5 and libstdc++.so.6 in /apps/generic/gcc/13.3.0/lib64 as an environment variable
os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
global_queue_gpu = None
global_result_dict = None
global_grf_gen = None

FAR, NARROW, ACCEPTED = 0, 1, 2 # For FMM

def init_worker(queue_gpu, result_dict, grf_gen):
    """Initializer for simulation pool workers.
       Sets the shared GPU task queue and result dictionary as globals."""
    global global_queue_gpu, global_result_dict, global_grf_gen
    global_queue_gpu = queue_gpu
    global_result_dict = result_dict
    global_grf_gen = grf_gen

def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--path_workdir', required=True, type=str, help="Path to the working directory where flow packages lie in: str")
    parser.add_argument('--num_mc', required=True, nargs='+', type=int, help='Number of MC: int list')
    parser.add_argument('--field_size', required=True, nargs='+', type=int, help='Field size: int list')
    parser.add_argument('--length_scale', required=True, type=float, help="Normalized length scale (0, 1)")
    parser.add_argument('--minpadding', type=int, default=0, help="Minimal padding for GRF generation")
    parser.add_argument('--dh', type=float, default=0.01, help="Hydraulic head x-gradient")
    group.add_argument('--std', type=float, default=None, help="Standard deviation for GRF generation")
    group.add_argument('--var', type=float, default=None, help="Variance for GRF generation")
    args = parser.parse_args()
    print("\nAll input parameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("\n")

    # print("CUDA available:", cuda.is_available())
        
    mp.set_start_method('spawn')
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    path_workdir = Path(args.path_workdir).expanduser().resolve()

    # Base configuration dictionary
    config_base = {
        'num_mc': args.num_mc,
        'dx': 1, 
        'dy': 1, 
        'dz': 1,
        'field_size': args.field_size,
        'grf_arg': {
            'length_scale': args.length_scale,
            'std': args.std if (args.std is not None) else (np.sqrt(args.var)),
            'minpadding': args.minpadding,
        },
        'path_simulation': path_workdir,
        'is_save_lenpath_mc': True,
        'headSolver_arg': {
            'dh': args.dh, #0.01, 
            'rtol': 1e-10, 
            'atol': 1e-20, 
            'max_it': 10000
        }
    }

    config_base['particleTracking_arg'] = {
        'density_particle': 20, 
        'eps_speed': 1e-30, 
        'ds_min': 1e-20, 
        'ds_max': 0.5*min(config_base['dx'], config_base['dy']), 
        'tol': 1e-4, 
        'y_policy': "stop", 
        'max_points': 100000, 
        'max_iters': 1000000, 
        'threads_per_block': 256,
    }
    config_base['path_log'] = config_base['path_simulation'] / 'log'
    config_base['path_mainlogfile'] = config_base['path_log'] / f'simulation_{timestamp}.log'
    config_base['path_gpulogfile'] = config_base['path_log'] / f'gpu_profile_{timestamp}.log'
    config_base['path_cpulogfile'] = config_base['path_log'] / f'cpu_profile_{timestamp}.log'
    
    # Ensure output directory exists
    config_base['path_simulation'].mkdir(parents=True, exist_ok=True)
    config_base['path_log'].mkdir(parents=True, exist_ok=True)

    # Setup main logger
    logger = get_logger(path_logfile = str(config_base['path_mainlogfile']))
    logger.info(f"[MAIN] Starting simulation.")
    
    # Setup GPU profiling
    event_stop_gpuprofiler = threading.Event()
    thread_gpuprofiler = threading.Thread(target=log_gpu_usage, kwargs={
        'path_logfile': str(config_base['path_gpulogfile']),
        'interval': 1.0,
        'event_stop': event_stop_gpuprofiler
    })
    thread_gpuprofiler.start()

    # Setup CPU profiling
    cpu_profiler = cProfile.Profile()
    cpu_profiler.enable()

    # Setup GPU Worker process
    manager = mp.Manager()
    result_dict = manager.dict()
    queue_gpu = mp.Queue()
    event_stop_gpuworker = mp.Event()  # Separate event for the GPU worker process
    process_gpu_worker = mp.Process(target=gpu_worker, args=(config_base, event_stop_gpuworker, queue_gpu, result_dict), name="GPU-Worker")
    process_gpu_worker.start()
    
    # Run simulations for each field size sequentially
    print(config_base['field_size'])
    for idx_size in range(len(config_base['field_size'])):
        run_field_size_simulation(idx_size, config_base, queue_gpu, result_dict)

    # End of simulation
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_seconds = elapsed_time.total_seconds()
    elapsed_hours = round(elapsed_seconds / 3600, 2)

    # Signal GPU worker to stop and wait for it
    event_stop_gpuworker.set()
    process_gpu_worker.join()

    # Stop CPU profiling
    cpu_profiler.disable()
    with open(str(config_base['path_cpulogfile']), "w") as f:
        stats_stream = pstats.Stats(cpu_profiler, stream=f).sort_stats('time')
        stats_stream.print_stats()
        
    # Stop GPU profiling
    event_stop_gpuprofiler.set()
    thread_gpuprofiler.join()

    # End of main logger
    logger.info(f"[MAIN] Completed simulation. {elapsed_hours} hours elapsed from {start_time::%Y-%m-%d %H:%M:%S} to {end_time::%Y-%m-%d %H:%M:%S}.")

def run_field_size_simulation(idx_size, config_base, queue_gpu, result_dict):
    """Run simulations for a given field size with parallelized Monte Carlo runs."""

    # Update config for this field size
    field_size_ = config_base['field_size'][idx_size]
    logger = get_logger(path_logfile = str(config_base['path_mainlogfile']))
    logger.info(f"[SIZE {field_size_}] Starting simulation.")
    config = config_base.copy()
    config.update({
        'idx_size': idx_size,
        'field_size_': field_size_,
        'lx': int(field_size_), 'ly': int(field_size_), 'lz': 1,
    })
    config.update({
        'source_vertex': [[config['lx']//20 - 1, config['ly']//20 - 1, 0],
                          [config['lx']//20 - 1, config['ly'] - config['ly']//20 - 1, 1]],
        'target_vertex': [[config['lx'] - config['lx']//20 - 1, 0, 0],
                          [config['lx'] - config['lx']//20 - 1, config['ly'] - 1, 1]]
    })
    config['x_cp'] = config['target_vertex'][0][0]
    config['x'] = np.arange(0, field_size_, config['dx'])
    config['y'] = np.arange(0, field_size_, config['dy'])
    
    config['path_size'] = config['path_simulation'] / f"lengthScale{config['grf_arg']['length_scale']}_size{field_size_}"
    config['path_size_mainOutput'] = config['path_size'] / 'main_output'
    config['path_size_path'] = config['path_size'] / 'path'
    config['path_size_logk'] = config['path_size'] / 'logk_field'

    if config['path_size'].exists() and config['path_size'].is_dir():
        shutil.rmtree(config['path_size'])
    config['path_size'].mkdir(parents=True, exist_ok=True)
    config['path_size_mainOutput'].mkdir(parents=True, exist_ok=True)
    config['path_size_path'].mkdir(parents=True, exist_ok=True)
    config['path_size_logk'].mkdir(parents=True, exist_ok=True)
        
    cov = ExpCov(sigma=config['grf_arg']['std'], length_scale=config['grf_arg']['length_scale'])
    pts=(np.linspace(0, 1, field_size_),)*2
    mean = np.zeros((field_size_, field_size_))
    grf_gen = GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=config['grf_arg']['minpadding'])
        
    # Run Monte Carlo simulations in parallel
    print(config['num_mc'], idx_size)
    num_mc_ = config['num_mc'][idx_size]
    num_process = min(num_mc_, len(os.sched_getaffinity(0)))
    with mp.Pool(processes=num_process, initializer=init_worker, initargs=(queue_gpu, result_dict, grf_gen)) as pool:
        mc_result = pool.starmap(run_unit_simulation, [(config, i) for i in range(num_mc_)])

    # Collect and save length of paths
    len_lrp_sm = np.array([r for r in mc_result[0]])
    len_fp_sm = np.array([r for r in mc_result[1]])
    fpt_sm = mc_result[2] # first passage time
    mhr_sm = mc_result[3] # minimal hydraulic resistance
    mainOutput = {
        'len_lrp_sm': len_lrp_sm,
        'len_fp_sm': len_fp_sm,
        'fpt_sm': fpt_sm,
        'mhr_sm': mhr_sm
    }
    np.savez(str(config['path_size_mainOutput'] / f'mainOutput_size{field_size_}.npz'), **mainOutput)
    logger.info(f"[SIZE {field_size_}] Main outputs saved.")
    logger.info(f"[SIZE {field_size_}] Completed simulation.")
    
    return idx_size, len_lrp_sm, len_fp_sm
    
def run_unit_simulation(config, num_mc_, num_trial=200):
    """
    Run a single Monte Carlo simulation with controlled retry logic.
    """
    logger = get_logger(path_logfile=str(config['path_mainlogfile']))
    logger.info(f"[MC {num_mc_}] Starting simulation (up to {num_trial} trials).")

    config_mc = config.copy()
    config_mc['path_mc'] = config['path_size'] / f"mc{num_mc_}"

    # Domain limits
    x0 = config_mc['source_vertex'][0][0]
    x1 = config_mc['target_vertex'][1][0]
    y0 = config_mc['source_vertex'][0][1]
    y1 = config_mc['source_vertex'][1][1]
    domain_box = (x0, x1, y0, y1)

    # Try multiple realizations if needed
    for trial in range(num_trial):
        logger.info(f"[MC {num_mc_}] Trial {trial + 1}/{num_trial} starting.")

        # === 1. Generate random log-conductivity field ===
        logk_field, return_msg_k = generate_logk_field(config_mc)
        np.save(config_mc['path_size_logk'] / f"logKField_size{config_mc['field_size_']}_mc{num_mc_}.npy", logk_field)
        k_field = np.exp(logk_field)
        logger.info(f"[MC {num_mc_}] {return_msg_k}")

        # === 2 Compute LRP ===
        logger.info(f"[MC {num_mc_}] Computing LRP.")
        source = [(y, x0) for y in range(y0, y1)]
        target = [(y, x1) for y in range(0, config_mc['ly']-1)]
        lrp, total_R, s_node, e_node, Rmap = dijkstra_2d_least_resistance(k_field, source, target, average='harmonic') 
        len_lrp = euclidean_length(lrp)

        # Check LRP geometry
        if not is_path_valid(lrp, domain_box, len_min=np.abs(x1 - x0)):
            logger.warning(f"[MC {num_mc_}] Invalid LRP, retrying next trial.")
            continue

        # === 3. Compute FP ===
        logger.info(f"[MC {num_mc_}] Computing FP.")
        h, msg, is_successful = solve_head(k_field, config_mc['x'], config_mc['y'], 
                                            dh=config_mc['headSolver_arg']['dh'], 
                                            rtol=config_mc['headSolver_arg']['rtol'], 
                                            atol=config_mc['headSolver_arg']['atol'], 
                                            max_it=config_mc['headSolver_arg']['max_it'])
        if not is_successful:
            logger.warning(f"[MC {num_mc_}] Head not converged, retrying next trial. Error: {msg}")
            continue
        # Velocity field    
        vx, vy = compute_velocity(k_field, config_mc['x'], config_mc['y'], h)

        # === 3.1 Compute FP via v_mag (Dijkstra) ===
        logger.info(f"[MC {num_mc_}] Computing FP (Dijkstra).")
        v_mag = np.sqrt(vx**2 + vy**2)
        fp_dijkstra, total_t, s_node, e_node, Rmap = dijkstra_2d_least_resistance(v_mag, source, target, average='arithmetic')
        len_fp_dijkstra = euclidean_length(fp_dijkstra)
        
        # Check FP (Dijkstra) geometry
        if not is_path_valid(fp_dijkstra, domain_box, len_min=np.abs(x1 - x0)):
            logger.warning(f"[MC {num_mc_}] Invalid FP (Dijkstra), retrying next trial.")
            continue
            
        # === 3.2 Compute FP via v_mag (FMM) ===
        n = logk_field.shape[0]
        l = n
        source_mask = np.zeros((n, n), dtype=bool)
        source_mask[y0:y1, x0] = True
        target_mask = np.zeros((n, n), dtype=bool)
        target_mask[:, x1] = True
        T, _ = fast_marching_2d_regions(v_mag, source_mask, h=config_mc['dx'])
        fp_fmm, chosen_target = extract_path_source_to_target(T, source_mask, target_mask)
        fp_fmm = np.array(fp_fmm)[:, ::-1]
        len_fp_fmm = euclidean_length(fp_fmm)
        
        # Check FP (FMM) geometry
        if not is_path_valid(fp_fmm, domain_box, len_min=np.abs(x1 - x0)):
            logger.warning(f"[MC {num_mc_}] Invalid FP (FMM), retrying next trial.")
            continue
            
        # Particle tracking
        nP = int(config_mc['particleTracking_arg']['density_particle'] * np.abs(y1 - y0))
        seeds_x = np.full(nP, x0, np.float32)
        seeds_y = np.linspace(y0, y1, nP).astype(np.float32)
        args_particleTracking = {
            'x': config_mc['x'], 
            'y': config_mc['y'], 
            'vx': vx, 
            'vy': vy, 
            'seeds_x': seeds_x, 
            'seeds_y': seeds_y, 
            'right_x': x1, 
            'eps_speed': config_mc['particleTracking_arg']['eps_speed'], 
            'ds_min': config_mc['particleTracking_arg']['ds_min'], 
            'ds_max': config_mc['particleTracking_arg']['ds_max'], 
            'tol': config_mc['particleTracking_arg']['tol'], 
            'y_policy': config_mc['particleTracking_arg']['y_policy'], 
            'max_points': config_mc['particleTracking_arg']['max_points'], 
            'max_iters': config_mc['particleTracking_arg']['max_iters'], 
            'threads_per_block': config_mc['particleTracking_arg']['threads_per_block'], 
            'dtype': np.float64
        }
        px, py, pt, L, st = particle_tracking_cpu(args_particleTracking)
        # 0=success, 1=stagnation, 4=other errors
        count = Counter(st.tolist())
        if count[0]/nP < 0.5 or (count[1] + count[4])/nP > 0.1:
            logger.warning(f"[MC {num_mc_}] Invalid velocity field, retrying next trial. Error: {count}")
            continue
            
        # Fastest path
        res = fastest_trajectory(px, py, pt, L, st)
        fp = np.vstack((res["x"], res["y"])).T
        fp = fp[fp[:,0] <= config_mc['x_cp']] # Trim the traj
        len_fp = euclidean_length(fp)

        # Check FP geometry
        if not is_path_valid(fp, domain_box, len_min=np.abs(x1 - x0)):
            logger.warning(f"[MC {num_mc_}] Invalid FP, retrying next trial.")
            continue

        # === 4. Save and exit successfully ===
        # Compare FP and LRP
        histDist_fp_lrp, _, _ = distance_distribution(fp, lrp, ds=1.0)

        # Compare FP and FP (Dijkstra)
        histDist_fp_fpDijkstra, _, _ = distance_distribution(fp, fp_dijkstra, ds=1.0)

        # Compare FP and FP (FMM)
        histDist_fp_fpFMM, _, _ = distance_distribution(fp, fp_fmm, ds=1.0)

        mainOutput_m = {
            'len_lrp_m': len_lrp,
            'len_fp_m': len_fp,
            'len_fp_dijkstra_m': len_fp_dijkstra,
            'len_fp_fmm_m': len_fp_fmm,
            
            'fpt_m': float(res['travel_time']),
            'fpt_dijkstra_m': float(T[fp_fmm[0][1], fp_fmm[0][0]]), # T(y,x)
            'fpt_fmm_m': float(total_t),
            'mhr_m': float(total_R),
            
            'hausdorffDist_fp_lrp_m': max(histDist_fp_lrp),
            'hausdorffDist_fp_fpDijkstra_m': max(histDist_fp_fpDijkstra),
            'hausdorffDist_fp_fpFMM_m': max(histDist_fp_fpFMM),
            
            'endPointDist_fp_lrp_m': fp[-1,1] - lrp[-1,1],
            'endPointDist_fp_fpDijkstra_m': fp[-1,1] - fp_dijkstra[-1,1],
            'endPointDist_fp_fpFMM_m': fp[-1,1] - fp_fmm[-1,1]
        }
        np.savez(str(config['path_size_mainOutput'] / f'mainOutput_size{config_mc['field_size_']}_mc{num_mc_}.npz'), **mainOutput_m)
    
        np.save(config_mc['path_size_path'] / f"lrp_size{config_mc['field_size_']}_mc{num_mc_}.npy", lrp)
        np.save(config_mc['path_size_path'] / f"fp_dijkstra_size{config_mc['field_size_']}_mc{num_mc_}.npy", fp_dijkstra)
        np.save(config_mc['path_size_path'] / f"fp_fmm_size{config_mc['field_size_']}_mc{num_mc_}.npy", fp_fmm)
        np.save(config_mc['path_size_path'] / f"fp_size{config_mc['field_size_']}_mc{num_mc_}.npy", fp)
        logger.info(f"[MC {num_mc_}] Paths saved.")

        logger.info(f"[MC {num_mc_}] Trial {trial + 1} successful.")
        break

    else:
        len_lrp = np.nan
        len_fp = np.nan
        logger.error(f"[MC {num_mc_}] All {num_trial} trials failed to produce valid outputs.")

    return len_lrp, len_fp, res['travel_time'], total_R
    
def generate_rectangular_array(vertex_pair):
    '''Generate a rectangular grid of points in a given dimension.'''

    # Ensure p0 and p1 are NumPy arrays
    p0 = np.array(vertex_pair[0], dtype=int)
    p1 = np.array(vertex_pair[1], dtype=int)
    
    # Determine the number of dimensions
    dim = p0.shape[0]
    
    # Create the grid points for each dimension
    grid_points = []
    
    for i in range(dim):
        grid_points.append(np.arange(p0[i], p1[i] + 1))
    
    # Create the meshgrid for the given dimension
    grid = np.meshgrid(*grid_points, indexing='ij')
    
    # Stack the grids into a single array (this will be the final grid)
    grid_stacked = np.stack(grid, axis=-1)
    
    return grid_stacked

def euclidean_length(trajectory):
    """
    Calculate the Euclidean length of a trajectory using CuPy for GPU acceleration.
    Args:
        trajectory (np.ndarray): array of shape (N, D)
    Returns:
        float: Euclidean length of the trajectory.
    """

    try:
        import cupy as cp
        _cupy_available = True
    except ImportError:
        _cupy_available = False

    # Select backend
    use_gpu = False
    if _cupy_available:
        try:
            cp.cuda.runtime.getDevice()  # test GPU availability
            use_gpu = True
            xp = cp
        except Exception:
            xp = np
    else:
        xp = np

    # Move data to chosen backend
    trajectory_xp = xp.asarray(trajectory)

    # Compute differences and length
    diff = trajectory_xp[1:] - trajectory_xp[:-1]
    length = xp.sum(xp.sqrt(xp.sum(diff ** 2, axis=-1)))

    # Convert back to NumPy float
    if use_gpu:
        length_np = cp.asnumpy(length)
    else:
        length_np = float(length)

    # Cleanup
    del trajectory_xp, diff, length
    gc.collect()

    return float(length_np)

def get_logger(name='simulation', path_logfile='simulation.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s')
        file_handler = logging.FileHandler(path_logfile, mode='a')
        console_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def get_gpu_usage():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        lines = result.stdout.strip().split('\n')
        usage_data = []
        for line in lines:
            used, free = [item.strip() for item in line.split(',')]
            usage_data.append(f"Memory: {int(used):,}MB used, {int(free):,}MB free.")
        return ' || '.join(usage_data)
    except Exception as e:
        return f"GPU info error: {e}"

def generate_logk_field(config):
    """Generate the logk field based on the chosen method."""
    start_time_grf = time.time()
    try:
        logk_field = global_grf_gen.sample()
    except Exception as e:
        elapsed_time_grf = time.time() - start_time_grf
        return_msg = f'GRF field generated unsuccessfully {np.round(elapsed_time_grf, 4)} seconds: {e}'
        return None, return_msg

    elapsed_time_grf = time.time() - start_time_grf
    return_msg = f'GRF field generated successfully after {np.round(elapsed_time_grf, 4)} seconds.'
    return logk_field, return_msg

def is_path_valid(path, box, len_min):
    """
    Check if a path stays within a bounding box and exceeds a minimum length.
    box = (x0, x1, y0, y1)
    """
    x0, x1, y0, y1 = box
    x, y = path[:, 0], path[:, 1]
    # print(f'minx = {min(x)}, maxx = {max(x)}, miny = {min(y)}, maxy = {max(y)}')
    # print(f'x0 = {x0}, x1 = {x1}, y0 = {y0}, y1 = {y1}')
    in_box = (min(x) >= x0) & (max(x) <= x1) & (min(y) >= y0) & (max(y) <= y1)
    # in_box = (min(y) >= y0) & (max(y) <= y1)
    long_enough = euclidean_length(path) >= len_min
    # print(in_box, long_enough)
    return in_box and long_enough

def log_gpu_usage(path_logfile='gpu_profile.log', interval=1.0, event_stop=None):
    """Periodically logs GPU usage to a file using nvidia-smi."""
    with open(path_logfile, 'w') as f:
        f.write("timestamp, gpu_utilization [%], memory_used [MB], memory_free [MB]\n")
        while not event_stop.is_set():
            try:
                result = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=timestamp,utilization.gpu,memory.used,memory.free',
                        '--format=csv,noheader,nounits'
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                f.write(result.stdout.strip() + '\n')
                time.sleep(interval)
            except Exception as e:
                f.write(f"# Error: {e}\n")
                break

def gpu_worker(config, event_stop_gpuworker, queue_gpu, result_dict):
    """
    GPU worker process: pulls tasks from the GPU task queue, processes them,
    and writes results into the shared result dictionary.
    """
    logger = get_logger(path_logfile=str(config['path_mainlogfile']))
    logger.info('[GPU-Worker] Starting with 1 GPU.')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 only.
    while not event_stop_gpuworker.is_set() or not queue_gpu.empty():
        try:
            task = queue_gpu.get(timeout=0.5)
        except Exception:
            continue
        task_id = task.get('id')
        func_name = task.get('func')
        args = task.get('args', tuple())
        kwargs = task.get('kwargs', {})
        log_msg = task.get('log_msg')

        # Dispatch GPU tasks
        logger.info(f"[GPU-Worker] Starting task '{log_msg}'.")
        if func_name == 'generate_logk_field':
            result = generate_logk_field(*args, **kwargs)
        elif func_name == 'run_par2':
            result = run_par2(*args, **kwargs)
        elif func_name == 'particle_tracking_rk45':
            result = particle_tracking_rk45(*args, **kwargs)
        else:
            logger.info('[GPU-Worker] Unmatched `func_name`. Please choose either `generate_logk_field` or `run_par2`.')
            result = None
        # Put the result into the shared dictionary with task_id as key.
        result_dict[task_id] = result
    logger.info('[GPU-Worker] Shutting down.')

def gpu_call(func_name, config, log_msg):
    task_id = uuid.uuid4().hex  # Unique task id
    task = {
        'id': task_id,
        'func': func_name,
        'args': (config,),  # pass config as argument
        'kwargs': {},
        'log_msg': log_msg
    }
    global_queue_gpu.put(task)  # Submit task to the GPU queue

    # Poll until the result arrives in the shared dictionary.
    while task_id not in global_result_dict:
        time.sleep(0.01)
        
    # Retrieve and remove the result.
    return global_result_dict.pop(task_id)

class GaussianRandomFieldCircEmbed:
    """
    Gaussian random field via circulant embedding.
    """
    def __init__(self, mean, cov, pts,
                 minpadding=0):
        self.mean = mean
        self.cov  = cov
        self.pts  = pts
        
        normedpts = [p - p[0] for p in pts]
        
        # compute minimal circulant‐embedding size in each dim
        pad = minpadding
        if np.isscalar(pad):
            pads = [pad]*len(pts) # if minpadding is scalar, use same in each dim
        else:
            pads = pad
        dims = [
            self.circulant_minsize(cov, normedpts[i], pads[i])
            for i in range(len(pts))
        ]
        
        # compute the embedded covariance eigenvalues
        Λ = self.circulant_eigvals(cov, normedpts, dims)
        
        # zero out any small negative λ due to numerical error,
        #    track how many and how negative, then scale by sqrt(λ/M)
        M = np.prod(dims)
        neg_mask = Λ < 0
        n_neg = int(neg_mask.sum())
        λ_min = None
        if n_neg:
            λ_min = float(Λ[neg_mask].min())
            Λ[neg_mask] = 0.0
        pos_mask = ~neg_mask
        Λ[pos_mask] = np.sqrt(Λ[pos_mask] / M)
        
        if n_neg:
            warnings.warn(
                f"{n_neg} negative eigenvalues ≥ {λ_min:.3e} were set to zero; "
                "consider increasing padding."
            )
        
        # store data:  v = sqrt(eigs/M)  and  shape_emb = tuple(dims)
        self.data = (Λ, tuple(dims))
        
    def sample(self):
        """
        Single realization.  xi must be shape == self.data[1].
        """
        Λ, shape_emb = self.data
        xi = np.random.randn(*shape_emb)
        # 1) spectral weights
        w = Λ * xi
        # 2) forward FFT over all dims
        w = fftn(w)
        # 3) crop back to original grid and extract real+imag
        orig_shape = tuple(len(p) for p in self.pts)
        slices = tuple(slice(0, n) for n in orig_shape)
        w_crop = w[slices]
        return self.mean + (w_crop.real + w_crop.imag)

    @staticmethod
    def circulant_minsize(cov, pts, minpadding):
        n = len(pts)
        base = n + minpadding - (1 if cov.is_even else 0)
        return 2 * next_fast_len(base) # SciPy’s next_fast_len returns the smallest ≥ base with 2,3,5 factors

    @staticmethod
    def circulant_eigvals(cov, pts, dims):
        pts = [np.asarray(p) for p in pts]
        starts = [p[0] for p in pts]
        steps = [p[1] - p[0] for p in pts]
        N = len(dims)
        is_even = getattr(cov, 'is_even', False)
        
        # Vectorized covariance evaluation
        cov_vec = np.vectorize(cov, signature='(n)->()')
        
        if is_even:
            half_shape = tuple(d//2 + 1 for d in dims)
            coords = [starts[d] + np.arange(half_shape[d]) * steps[d] for d in range(N)]
            mesh = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1)
            C = cov_vec(mesh)
            C = dctn(C, type=1, norm=None)
            idx = [np.minimum(np.arange(d), d - np.arange(d)) for d in dims]
            mesh_idx = np.ix_(*idx)
            return C[mesh_idx]
        else:
            C = np.zeros(dims, dtype=float)
            dims2 = [d + 2 for d in dims]
            mids  = [d2 // 2 for d2 in dims2]
            grid = [np.arange(d) for d in dims]
            # Build signed offsets
            coords = []
            for d in range(N):
                idx = grid[d]
                pos = np.where(idx < mids[d], idx, idx - dims[d])
                coords.append(starts[d] + pos * steps[d])
            mesh = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1)
            mask = np.ones(dims, dtype=bool)
            for d in range(N):
                mask &= ((grid[d] + 1) != mids[d]).reshape(
                    *(dims[j] if j == d else 1 for j in range(N))
                )
            C[mask] = cov_vec(mesh[mask])
            return np.real(np.fft.fftn(C))

class ExpCov:
    def __init__(self, sigma=1.0, length_scale=0.1):
        self.sigma = sigma
        self.length_scale = length_scale
        self.is_even = True
    def __call__(self, x):
        r = np.linalg.norm(x)
        return self.sigma**2 * np.exp(-np.abs(r) / self.length_scale)

def solve_head(K, x, y, dh=0.01, rtol=1e-6, atol=1e-20, max_it=10000):
    """
    Solve ∇·(K∇h) = 0 in a 2D domain with:
    - Dirichlet BCs on left/right: h(left)=dh*x_len, h(right)=0
    - No-flow BCs on top/bottom.
    
    Parameters
    ----------
    K : 2D np.ndarray
        Hydraulic conductivity at cell centers, shape (ny, nx).
    x, y : 1D np.ndarray
        Cell-center coordinates in x and y directions.
    dh : float
        Mean hydraulic head gradient between left and right boundaries.
    
    Returns
    -------
    h : 2D np.ndarray
        Computed hydraulic head field.
    msg : str
        PETSc convergence message.
    """
    ny, nx = K.shape
    dx = np.diff(np.concatenate(([0], x + 0.5*(x[1]-x[0]))))
    dy = np.diff(np.concatenate(([0], y + 0.5*(y[1]-y[0]))))
    dz = 1.0

    def lid(i, j): return i*nx + j
    def hmean(a, b): return 2*a*b/(a+b + 1e-300)

    dx_face = np.empty(nx+1)
    dx_face[1:-1] = 0.5*(dx[:-1] + dx[1:])
    dx_face[0] = dx[0]
    dx_face[-1] = dx[-1]

    dy_face = np.empty(ny+1)
    dy_face[1:-1] = 0.5*(dy[:-1] + dy[1:])
    dy_face[0] = dy[0]
    dy_face[-1] = dy[-1]

    # PETSc assembly
    N = nx * ny
    A = PETSc.Mat().create()
    A.setSizes((N, N))
    A.setType(PETSc.Mat.Type.AIJ)
    A.setPreallocationNNZ(5)

    b = PETSc.Vec().createMPI(N)
    b.set(0.0)

    LEFT_VAL, RIGHT_VAL = dh*np.abs(x[-1]-x[0]), 0.0

    rstart, rend = A.getOwnershipRange()
    for row in range(rstart, rend):
        i, j = divmod(row, nx)
        is_left = (j == 0)
        is_right = (j == nx-1)

        if is_left or is_right:
            A.setValue(row, row, 1.0)
            b.setValue(row, LEFT_VAL if is_left else RIGHT_VAL)
            continue

        Ax = dy[i]*dz
        Ay = dx[j]*dz

        KW = hmean(K[i, j], K[i, j-1])
        GW = KW * Ax / dx_face[j]

        KE = hmean(K[i, j], K[i, j+1])
        GE = KE * Ax / dx_face[j+1]

        GS = hmean(K[i, j], K[i-1, j]) * Ay / dy_face[i] if i > 0 else 0.0
        GN = hmean(K[i, j], K[i+1, j]) * Ay / dy_face[i+1] if i < ny-1 else 0.0

        diag = GE + GW + GN + GS
        A.setValue(row, row, diag)

        if j-1 == 0:
            b.setValue(row, GW*LEFT_VAL, addv=True)
        else:
            A.setValue(row, lid(i, j-1), -GW)

        if j+1 == nx-1:
            b.setValue(row, GE*RIGHT_VAL, addv=True)
        else:
            A.setValue(row, lid(i, j+1), -GE)

        if i > 0:
            A.setValue(row, lid(i-1, j), -GS)
        if i < ny-1:
            A.setValue(row, lid(i+1, j), -GN)

    A.assemblyBegin(); A.assemblyEnd()
    b.assemblyBegin(); b.assemblyEnd()

    x_vec = b.duplicate()
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    pc = ksp.getPC()
    try:
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")
    except PETSc.Error:
        pc.setType(PETSc.PC.Type.GAMG)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    ksp.setFromOptions()
    ksp.solve(b, x_vec)

    h_flat = x_vec.getArray()
    h = h_flat.reshape((ny, nx))

    msg = (f"ConvergedReason: {ksp.getConvergedReason()}, "
           f"Iterations: {ksp.getIterationNumber()}, "
           f"Residual: {ksp.getResidualNorm():.3e}")
    is_success = True if (ksp.getConvergedReason() > 0) else False
    return h, msg, is_success

def compute_velocity(K, x, y, h):
    """
    Compute Darcy velocity components (vx, vy) from head field h(x,y).

    Parameters
    ----------
    K : 2D np.ndarray
        Hydraulic conductivity at cell centers, shape (ny, nx)
    x, y : 1D np.ndarray
        Cell-center coordinates in x and y directions
    h : 2D np.ndarray
        Hydraulic head field, shape (ny, nx)

    Returns
    -------
    vx, vy : 2D np.ndarray
        Darcy velocity components in x and y directions
        vx = -K * ∂h/∂x, vy = -K * ∂h/∂y
    """
    ny, nx = h.shape
    dhdx = np.zeros_like(h)
    dhdy = np.zeros_like(h)

    # --- x-derivative (central interior, one-sided edges)
    for i in range(ny):
        dhdx[i, 1:-1] = (h[i, 2:] - h[i, :-2]) / (x[2:] - x[:-2])
        dhdx[i, 0] = (h[i, 1] - h[i, 0]) / (x[1] - x[0])
        dhdx[i, -1] = (h[i, -1] - h[i, -2]) / (x[-1] - x[-2])

    # --- y-derivative (central interior, one-sided edges)
    for j in range(nx):
        dhdy[1:-1, j] = (h[2:, j] - h[:-2, j]) / (y[2:] - y[:-2])
        dhdy[0, j] = (h[1, j] - h[0, j]) / (y[1] - y[0])
        dhdy[-1, j] = (h[-1, j] - h[-2, j]) / (y[-1] - y[-2])

    # --- Darcy velocity
    vx = -K * dhdx
    vy = -K * dhdy
    return vx, vy

# BEGIN: RK45 particle tracking
@njit(inline='always')
def find_le_index(arr, val):
    n = arr.size
    if val <= arr[0]:
        return 0
    if val >= arr[n-2]:
        return n-2
    lo, hi = 0, n-1
    while hi - lo > 1:
        mid = (hi + lo) >> 1
        if arr[mid] <= val:
            lo = mid
        else:
            hi = mid
    return lo

@njit(inline='always')
def bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, x, y):
    if x <= xgrid[0]:
        ix = 0
    elif x >= xgrid[nx-1]:
        ix = nx - 2
    else:
        ix = find_le_index(xgrid, x)

    if y <= ygrid[0]:
        iy = 0
    elif y >= ygrid[ny-1]:
        iy = ny - 2
    else:
        iy = find_le_index(ygrid, y)

    x0, x1 = xgrid[ix], xgrid[ix+1]
    y0, y1 = ygrid[iy], ygrid[iy+1]

    tx = (x - x0) / (x1 - x0 + 1e-30)
    ty = (y - y0) / (y1 - y0 + 1e-30)

    i00 = iy*nx + ix
    i10 = iy*nx + ix + 1
    i01 = (iy+1)*nx + ix
    i11 = (iy+1)*nx + ix + 1

    vx0 = vx[i00] + tx*(vx[i10] - vx[i00])
    vx1 = vx[i01] + tx*(vx[i11] - vx[i01])
    vy0 = vy[i00] + tx*(vy[i10] - vy[i00])
    vy1 = vy[i01] + tx*(vy[i11] - vy[i01])

    return vx0 + ty*(vx1 - vx0), vy0 + ty*(vy1 - vy0)

@njit
def norm2(v0, v1):
    """Simple L2 norm function."""
    return math.sqrt(v0**2 + v1**2)

@njit
def rhs_arclength(x, y, xgrid, ygrid, vx, vy, nx, ny, eps_speed):
    """
    For z=[x,y,t], dz/ds = [vx/|v|, vy/|v|, 1/|v|].
    Runs on a single CPU thread.
    """
    # 1. CALL AND CAPTURE THE RETURNED VELOCITIES (vx, vy)
    vx_at_p, vy_at_p = bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, x, y)

    # 2. CALCULATE SPEED USING THE RETURNED VALUES
    sp = norm2(vx_at_p, vy_at_p)

    if not math.isfinite(sp) or sp < eps_speed:
        return 0.0, 0.0, 1e30, sp

    inv = 1.0 / sp
    
    # 3. CALCULATE RATES using the captured values
    rate_x = vx_at_p * inv
    rate_y = vy_at_p * inv
    rate_t = inv
    
    return rate_x, rate_y, rate_t, sp
    
@njit
def rk45_one_particle(xgrid, ygrid,
                      vx, vy, nx, ny,
                      x0, y0,
                      right_x, xmin, xmax, ymin, ymax,
                      eps_speed, ds_min, ds_max, tol,
                      y_policy,
                      max_points, max_iters,
                      A, B5, B4):
    """
    Returns: x_path, y_path, t_path, length, status
    """
    x = x0
    y = y0
    t = 0.0
    ds = ds_max * 0.25
    sp = 0.0
    st = 4  # default fail

    x_path = np.zeros(max_points, np.float32)
    y_path = np.zeros(max_points, np.float32)
    t_path = np.zeros(max_points, np.float32)

    x_path[0] = x
    y_path[0] = y
    t_path[0] = t
    count = 1

    for it in range(max_iters):
        if count >= max_points:
            break

        # Boundary conditions
        if x >= right_x:
            st = 0
            break
        if x <= xmin or x >= xmax:
            st = 2
            break

        if y < ymin or y > ymax:
            if y_policy == 0:   # stop
                st = 3
                break
            elif y_policy == 1:  # reflect
                span = ymax - ymin
                Y = ymin + (y-ymin) % (2*span)
                if Y < ymin: Y += 2*span
                if Y > ymax: Y = ymax - (Y-ymax)
                y = Y
            elif y_policy == 2:  # clip
                y = min(max(y, ymin), ymax)

        # Guard step
        ds = min(max(ds, ds_min), ds_max)

        # RK45 stage buffers
        kx = np.zeros(7, np.float32)
        ky = np.zeros(7, np.float32)
        kt = np.zeros(7, np.float32)

        # Stage k1
        kx[0], ky[0], kt[0], sp = rhs_arclength(x, y, xgrid, ygrid,
                                                vx, vy, nx, ny, eps_speed)
        if sp < eps_speed or not math.isfinite(sp):
            st = 1
            break

        # Stages k2..k6
        for s in range(1, 6):
            xi, yi, ti = x, y, t
            for j in range(s):
                xi += ds * A[s,j] * kx[j]
                yi += ds * A[s,j] * ky[j]
                ti += ds * A[s,j] * kt[j]

            kx[s], ky[s], kt[s], _ = rhs_arclength(xi, yi, xgrid, ygrid,
                                                   vx, vy, nx, ny, eps_speed)

        # 5th order solution
        x5, y5, t5 = x, y, t
        for j in range(6):
            x5 += ds * B5[j] * kx[j]
            y5 += ds * B5[j] * ky[j]
            t5 += ds * B5[j] * kt[j]

        # 4th order estimate
        x4, y4, t4 = x, y, t
        for j in range(6):
            x4 += ds * B4[j] * kx[j]
            y4 += ds * B4[j] * ky[j]
            t4 += ds * B4[j] * kt[j]
        x4 += ds * B4[6] * kx[0]
        y4 += ds * B4[6] * ky[0]
        t4 += ds * B4[6] * kt[0]

        # error
        err = math.sqrt((x5-x4)**2 + (y5-y4)**2) + 1e-30

        if err <= tol:  # accept step
            x, y, t = x5, y5, t5

            x_path[count] = x
            y_path[count] = y
            t_path[count] = t
            count += 1

            ds = ds * min(5.0, max(0.2, 0.9 * (tol/err)**0.2))
        else:           # reject
            ds = ds * max(0.1, 0.5 * (tol/err)**0.25)

        # Detect crossing
        if x < right_x <= x5:
            tau = (right_x - x) / ((x5-x) + 1e-30)
            tau = min(max(tau, 0), 1)
            x = right_x
            y = y + tau * (y5-y)
            t = t + tau * (t5-t)

            x_path[count] = x
            y_path[count] = y
            t_path[count] = t
            count += 1
            st = 0
            break

    return x_path, y_path, t_path, count, st

def particle_tracking_cpu(params):
    x = params["x"].astype(np.float32)
    y = params["y"].astype(np.float32)
    vx = params["vx"].astype(np.float32)
    vy = params["vy"].astype(np.float32)
    seeds_x = params["seeds_x"].astype(np.float32)
    seeds_y = params["seeds_y"].astype(np.float32)

    nx = vx.shape[1]
    ny = vx.shape[0]
    vx_flat = vx.ravel()
    vy_flat = vy.ravel()

    right_x = params.get("right_x", x[-1])
    eps_speed = params.get("eps_speed", 1e-16)
    ds_min = params.get("ds_min", 1e-5)
    ds_max = params.get("ds_max", 1.0)
    tol = params.get("tol", 1e-4)
    y_policy = {"stop":0, "reflect":1, "clip":2}[params.get("y_policy", "stop")]
    max_points = params.get("max_points", 2000)
    max_iters = params.get("max_iters", 200000)

    # RK45 coefficients
    A = np.zeros((6,6),np.float32)
    A[1,0] = 1/5
    A[2,0:2] = [3/40, 9/40]
    A[3,0:3] = [44/45,-56/15,32/9]
    A[4,0:4] = [19372/6561,-25360/2187,64448/6561,-212/729]
    A[5,0:5] = [9017/3168,-355/33,46732/5247,49/176,-5103/18656]

    B5 = np.array([35/384, 0., 500/1113, 125/192,-2187/6784,11/84], np.float32)
    B4 = np.array([5179/57600,0.,7571/16695,393/640,-92097/339200,187/2100,1/40], np.float32)

    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]

    nP = seeds_x.size

    paths_x = np.zeros((nP, max_points), np.float32)
    paths_y = np.zeros((nP, max_points), np.float32)
    paths_t = np.zeros((nP, max_points), np.float32)
    lengths = np.zeros(nP, np.int32)
    status  = np.zeros(nP, np.int32)

    for i in range(nP):
        xp, yp, tp, L, st = rk45_one_particle(
            x, y, vx_flat, vy_flat, nx, ny,
            seeds_x[i], seeds_y[i],
            right_x, xmin, xmax, ymin, ymax,
            eps_speed, ds_min, ds_max, tol,
            y_policy,
            max_points, max_iters,
            A, B5, B4
        )
        paths_x[i,:] = xp
        paths_y[i,:] = yp
        paths_t[i,:] = tp
        lengths[i] = L
        status[i] = st

    return paths_x, paths_y, paths_t, lengths, status
# END: RK45 particle tracking

def dijkstra_2d_least_resistance(K, start_region, end_region, average='harmonic'):
    """
    Finds the least-resistance path on a 2D conductivity lattice using Dijkstra's algorithm,
    considering 8-neighbor connectivity and user-selected averaging (harmonic or arithmetic)
    of bond conductivities.

    Args:
        K (np.ndarray): 2D array of conductivities (higher = easier to move).
        start_region (list[tuple[int,int]]): List of (row, col) coordinates defining the start region.
        end_region (list[tuple[int,int]]): List of (row, col) coordinates defining the end region.
        average (str): Either 'harmonic' or 'arithmetic'. Determines how to average conductivities
                       between neighboring sites.

    Returns:
        tuple:
            - path (np.ndarray): Optimal path as list of (row, col) coordinates.
            - total_resistance (np.longdouble): Total resistance along the optimal path.
            - actual_start_node (tuple[int,int]): The specific node where the path starts.
            - actual_end_node (tuple[int,int]): The specific node where the path ends.
            - resistance_map (np.ndarray): 2D array of minimal cumulative resistance.
    """

    Lx, Ly = K.shape
    pq = []
    resist = {}
    predecessors = {}
    visited = set()

    # --- Initialize all nodes with infinite resistance ---
    for r in range(Lx):
        for c in range(Ly):
            resist[(r, c)] = np.longdouble(np.inf)

    # --- Start region initialization ---
    for r, c in start_region:
        resist[(r, c)] = 0.0
        heapq.heappush(pq, (0.0, r, c))

    # --- 8-neighbor moves ---
    moves = [
        (-1, 0), (1, 0), (0, -1), (0, 1),     # N, S, W, E
        (-1, -1), (-1, 1), (1, -1), (1, 1)    # NW, NE, SW, SE
    ]

    actual_end_node = None

    while pq:
        current_res, r, c = heapq.heappop(pq)
        if (r, c) in visited:
            continue
        visited.add((r, c))

        # --- Check if reached end region ---
        if (r, c) in end_region:
            actual_end_node = (r, c)
            break

        # --- Loop over neighbors ---
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < Lx and 0 <= nc < Ly):
                continue

            k1, k2 = K[r, c], K[nr, nc]
            if k1 <= 0 or k2 <= 0:
                continue  # skip invalid or zero conductivity

            # --- Compute bond conductivity ---
            if average == 'harmonic':
                k_bond = 2.0 * k1 * k2 / (k1 + k2)
            elif average == 'arithmetic':
                k_bond = 0.5 * (k1 + k2)
            else:
                raise ValueError("average must be either 'harmonic' or 'arithmetic'")

            if k_bond <= 0:
                continue

            # --- Distance weighting (diagonals longer) ---
            length_factor = np.sqrt(2.0) if dr != 0 and dc != 0 else 1.0

            # --- Edge resistance = path length / bond conductivity ---
            edge_res = length_factor / k_bond
            new_res = current_res + edge_res

            if new_res < resist[(nr, nc)]:
                resist[(nr, nc)] = new_res
                predecessors[(nr, nc)] = (r, c)
                heapq.heappush(pq, (new_res, nr, nc))

    # --- Backtrack to reconstruct path ---
    path = []
    total_resistance = np.longdouble(np.inf)
    actual_start_node = None

    if actual_end_node is not None:
        total_resistance = resist[actual_end_node]
        node = actual_end_node
        while node is not None:
            path.append(node)
            if node in start_region:
                actual_start_node = node
                break
            node = predecessors.get(node)
        path.reverse()

    # --- Convert resistance dict to 2D array ---
    resistance_map = np.full_like(K, np.inf, dtype=np.longdouble)
    for (r, c), val in resist.items():
        resistance_map[r, c] = val

    return np.array(path)[:,[1,0]], total_resistance, actual_start_node, actual_end_node, resistance_map

def fastest_trajectory(px, py, pt, L, st):
    nP = L.shape[0]

    # Valid if success and at least 1 point
    valid = (st == 0) & (L > 0)

    # Initialize with NaN so invalid ones don’t get picked
    travel_times = np.full(nP, np.nan, dtype=pt.dtype)

    # Extract final time for each valid trajectory
    idx = np.where(valid)[0]
    if idx.size > 0:
        # vectorized gather of last times
        travel_times[idx] = pt[idx, L[idx] - 1]

    # Find fastest (minimum travel time)
    if np.all(np.isnan(travel_times)):
        raise RuntimeError("No successful trajectories with valid travel time.")

    fastest_id = int(np.nanargmin(travel_times))
    fastest_time = float(travel_times[fastest_id])

    # Extract the polyline coordinates/time for the fastest trajectory
    k = L[fastest_id]
    x_fast = px[fastest_id, :k].copy()
    y_fast = py[fastest_id, :k].copy()
    t_fast = pt[fastest_id, :k].copy()

    return {
        "particle_id": fastest_id,
        "travel_time": fastest_time,
        "x": x_fast,
        "y": y_fast,
        "t": t_fast,
        "all_travel_times": travel_times  # useful for histograms, optional
    }
    
# FMM
def _solve_quadratic_update(ax, ay, f, h):
    """
    Upwind isotropic update for |∇T| = f (where f = 1/c).
    """
    if ax > ay:
        ax, ay = ay, ax
    fh = f * h

    if (ay - ax) >= fh:
        return ax + fh

    disc = 2.0 * fh * fh - (ay - ax) * (ay - ax)
    if disc < 0:
        disc = 0.0
    return 0.5 * (ax + ay + math.sqrt(disc))


def fast_marching_2d_regions(speed, source_mask, h=1.0):
    """
    Solve |∇T| = 1/c for isotropic medium with FMM, with T=0 on source_mask.

    Parameters
    ----------
    speed : (n,n) array, positive speeds c(x,y)>0
    source_mask : (n,n) bool array, True where source region is
    h : float, grid spacing

    Returns
    -------
    T : (n,n) travel time field
    state : (n,n) int8, FAR/NARROW/ACCEPTED (useful for debugging)
    """
    speed = np.asarray(speed, dtype=float)
    if speed.ndim != 2 or speed.shape[0] != speed.shape[1]:
        raise ValueError("speed must be a square 2D array (n x n).")
    if np.any(speed <= 0):
        raise ValueError("All speed values must be > 0.")

    n = speed.shape[0]
    source_mask = np.asarray(source_mask, dtype=bool)
    if source_mask.shape != (n, n):
        raise ValueError("source_mask must have same shape as speed.")
    if not np.any(source_mask):
        raise ValueError("source_mask is empty (no True entries).")

    f = 1.0 / speed  # slowness

    T = np.full((n, n), np.inf, dtype=float)
    state = np.full((n, n), FAR, dtype=np.int8)
    heap = []

    def push(i, j, val):
        if val < T[i, j]:
            T[i, j] = val
            heapq.heappush(heap, (val, i, j))
            state[i, j] = NARROW

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 1) Initialize: accept all source cells with T=0
    src_idx = np.argwhere(source_mask)
    for i, j in src_idx:
        T[i, j] = 0.0
        state[i, j] = ACCEPTED

    # 2) Initialize narrow band from source boundary
    for i, j in src_idx:
        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and state[ni, nj] != ACCEPTED:
                # update (ni, nj) using accepted neighbors
                ax = np.inf
                ay = np.inf

                # x-dir neighbors
                if nj - 1 >= 0 and state[ni, nj - 1] == ACCEPTED:
                    ax = min(ax, T[ni, nj - 1])
                if nj + 1 < n and state[ni, nj + 1] == ACCEPTED:
                    ax = min(ax, T[ni, nj + 1])

                # y-dir neighbors
                if ni - 1 >= 0 and state[ni - 1, nj] == ACCEPTED:
                    ay = min(ay, T[ni - 1, nj])
                if ni + 1 < n and state[ni + 1, nj] == ACCEPTED:
                    ay = min(ay, T[ni + 1, nj])

                if np.isinf(ax) and np.isinf(ay):
                    continue

                newT = _solve_quadratic_update(ax, ay, f[ni, nj], h)
                push(ni, nj, newT)

    # 3) Main FMM loop
    while heap:
        t, i, j = heapq.heappop(heap)
        if state[i, j] == ACCEPTED:
            continue
        state[i, j] = ACCEPTED
        T[i, j] = t

        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if not (0 <= ni < n and 0 <= nj < n):
                continue
            if state[ni, nj] == ACCEPTED:
                continue

            ax = np.inf
            ay = np.inf

            if nj - 1 >= 0 and state[ni, nj - 1] == ACCEPTED:
                ax = min(ax, T[ni, nj - 1])
            if nj + 1 < n and state[ni, nj + 1] == ACCEPTED:
                ax = min(ax, T[ni, nj + 1])

            if ni - 1 >= 0 and state[ni - 1, nj] == ACCEPTED:
                ay = min(ay, T[ni - 1, nj])
            if ni + 1 < n and state[ni + 1, nj] == ACCEPTED:
                ay = min(ay, T[ni + 1, nj])

            if np.isinf(ax) and np.isinf(ay):
                continue

            newT = _solve_quadratic_update(ax, ay, f[ni, nj], h)
            push(ni, nj, newT)

    return T, state


def extract_path_source_to_target(T, source_mask, target_mask, max_steps=200000):
    """
    Backtrack from the best target cell (min T over target_mask) to the source region
    by descending T on the 4-neighborhood.

    Returns
    -------
    path : list of (i,j) indices from target -> source (you can reverse if you want)
    chosen_target : (i,j)
    """
    T = np.asarray(T, dtype=float)
    n = T.shape[0]
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    if target_mask.shape != (n, n) or source_mask.shape != (n, n):
        raise ValueError("source_mask and target_mask must match T shape.")
    if not np.any(target_mask):
        raise ValueError("target_mask is empty (no True entries).")

    # choose target with minimal T
    Ti = T.copy()
    Ti[~target_mask] = np.inf
    flat = np.argmin(Ti)
    ti = Ti.flat[flat]
    if not np.isfinite(ti):
        raise RuntimeError("No reachable target cell (T=inf on target_mask).")

    i, j = np.unravel_index(flat, T.shape)
    chosen_target = (i, j)

    path = [(i, j)]
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(max_steps):
        if source_mask[i, j]:
            break

        best_val = T[i, j]
        best_ij = (i, j)

        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n:
                if T[ni, nj] < best_val:
                    best_val = T[ni, nj]
                    best_ij = (ni, nj)

        if best_ij == (i, j):
            # can't descend further (plateau or discretization issue)
            break

        i, j = best_ij
        path.append((i, j))

    return path, chosen_target

def resample_by_arclength(P, ds=1.0):
    """P: Nx2 polyline. Returns roughly uniform-arc-length resampling."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    L = s[-1]
    if L == 0:  # degenerate
        return P[[0]].copy()
    s_new = np.arange(0, L + 1e-9, ds)
    # linear interp on cumulative arclength
    Px = np.interp(s_new, s, P[:,0])
    Py = np.interp(s_new, s, P[:,1])
    return np.column_stack([Px, Py])

def directed_nn_distances(A, B):
    """For each point in A, distance to nearest point in B."""
    tree = cKDTree(B)
    d, _ = tree.query(A, k=1, workers=-1)
    return d

def distance_distribution(A, B, ds=1.0):
    """Compute symmetric distribution + standard metrics."""
    Au = resample_by_arclength(A, ds=ds)
    Bu = resample_by_arclength(B, ds=ds)

    dAB = directed_nn_distances(Au, Bu)
    dBA = directed_nn_distances(Bu, Au)
    D = np.concatenate([dAB, dBA])
    
    return D, dAB, dBA
    
### Main Execution

if __name__ == '__main__':
    main()