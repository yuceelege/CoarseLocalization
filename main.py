import argparse
import time
import os
import sys
import random
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
import configs
from tools.utils import create_ellipse, compute_recovery_signal

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', type=int, default=1)
parser.add_argument('--max_steps',  type=int, default=1000)
parser.add_argument('--experiment_name', type=str, default='default_experiment', 
                    help='Name of the experiment to organize results')
args = parser.parse_args()

# Create experiment-specific directory
experiment_dir = f'results/{args.experiment_name}'
os.makedirs(experiment_dir, exist_ok=True)

A = np.array(configs.A, dtype=np.longdouble)
B = np.array(configs.B, dtype=np.longdouble)
s_dx, s_dy = configs.red_width/2, configs.red_height/2
cx, cy = configs.red_center
r = configs.sensor_radius
dx_y, dy_y = configs.yellow_width/2, configs.yellow_height/2
USE_AUTO_CONTROL = True
N_CONSTELLATION = 6

# Precompute matrix powers for speed
print(f"Precomputing matrix powers up to {args.max_steps}...")
A_powers = [np.eye(2, dtype=np.longdouble)]  # A^0 = I
for k in range(1, args.max_steps + 1):
    A_powers.append(A @ A_powers[-1])
print("Matrix powers precomputed.")

# Get fixed landmark region center from config
landmark_region_center = np.array(configs.landmark_region_center)
print(f'Fixed landmark region center: {landmark_region_center}')

for trial in range(args.num_trials):
    tic = time.time()
    initial_corners = np.array([[cx-s_dx, cy-s_dy],
                                [cx+s_dx, cy-s_dy],
                                [cx+s_dx, cy+s_dy],
                                [cx-s_dx, cy+s_dy]])
    region_static = Polygon(initial_corners)

    # Select landmark_center from within the fixed landmark region
    while True:
        landmark_center = np.array([random.uniform(landmark_region_center[0]-dx_y, landmark_region_center[0]+dx_y),
                                    random.uniform(landmark_region_center[1]-dy_y, landmark_region_center[1]+dy_y)])
        state0 = np.array([random.uniform(cx-s_dx, cx+s_dx),
                           random.uniform(cy-s_dy, cy+s_dy)])
        if np.linalg.norm(state0 - landmark_center) <= r:
            break
    state = state0.reshape(2,1)
    signal_times = []
    controls = []
    in_recovery = False
    recovery_ctrls = []
    rec_idx = 0
    last_region_dyn = None
    last_u = None
    M = np.eye(2, dtype=np.longdouble)
    offset = np.zeros((2,1), dtype=np.longdouble)

    trial_file = open(f'{experiment_dir}/diam_trial_{trial}.csv','w')
    trial_file.write('i,diameter,landmark_diameter\n')

    # Initialize landmark region (full yellow rectangle) centered at fixed landmark_region_center
    landmark_region = Polygon([
        (landmark_region_center[0]-dx_y, landmark_region_center[1]-dy_y),
        (landmark_region_center[0]+dx_y, landmark_region_center[1]-dy_y),
        (landmark_region_center[0]+dx_y, landmark_region_center[1]+dy_y),
        (landmark_region_center[0]-dx_y, landmark_region_center[1]+dy_y)
    ])

    print(f'Starting trial {trial}: initial_state={state0}, landmark_center={landmark_center}')
    
    i = 0
    trial_failed = False
    while True:
        inside = np.linalg.norm(state.flatten() - landmark_center) <= r
        if inside:
            signal_times.append(i)
            if len(signal_times) > 1:
                available_indices = list(range(len(signal_times) - 1))
                selected_indices = [0]
                if len(available_indices) > 10:
                    remaining_indices = list(range(1, len(available_indices)))
                    selected_indices.extend(random.sample(remaining_indices, 10))
                else:
                    selected_indices.extend(range(1, len(available_indices)))
                
                for j_idx in selected_indices:
                    j = signal_times[j_idx]
                    A_i = A_powers[i]
                    A_j = A_powers[j]
                    ΔA = A_i - A_j
                    sum_term1 = sum(A_powers[i-1-k] @ B @ controls[k] for k in range(i))
                    sum_term2 = sum(A_powers[j-1-k] @ B @ controls[k] for k in range(j))
                    sum_term = sum_term1 - sum_term2
                    μ64 = -np.linalg.inv(ΔA.astype(np.float64)) @ sum_term.astype(np.float64)
                    μ_ij = μ64.astype(np.longdouble)
                    P_ij = (ΔA.T @ ΔA) / (4 * r * r)
                    ellipse = create_ellipse(P_ij, μ_ij)
                    if region_static.intersects(ellipse):
                        intersection = region_static.intersection(ellipse)
                        if not intersection.is_empty:
                            region_static = intersection.buffer(0.0)
                            # Ensure validity
                            if not region_static.is_valid:
                                region_static = region_static.buffer(0.0)
                            if region_static.is_empty or not region_static.is_valid:
                                print(f'Trial {trial} Step {i}: region_static became invalid, breaking')
                                trial_failed = True
                                break
                if trial_failed:
                    break
            if in_recovery:
                in_recovery = False
                rec_idx = 0
                recovery_ctrls = []

        if not inside and not in_recovery and last_region_dyn is not None:
            in_recovery = True
            frozen_pts = [tuple(pt) for pt in last_region_dyn.exterior.coords]
            recovery_ctrls, _, _ = compute_recovery_signal(r, A, B, frozen_pts, last_u.flatten(), N_CONSTELLATION)

        if in_recovery:
            if rec_idx < N_CONSTELLATION:
                u = recovery_ctrls[rec_idx]
                rec_idx += 1
            else:
                print(f'Trial {trial} FAIL at step {i}: recovery sequence exhausted')
                sys.exit(1)
        else:
            u = np.array([[random.uniform(-1,1)], [random.uniform(-1,1)]]) if USE_AUTO_CONTROL else controls[-1]

        controls.append(u)
        state = A @ state + B @ u
        M = A @ M
        offset = A @ offset + B @ u
        
        # Ensure region_static is valid before transformation
        if not region_static.is_valid:
            region_static = region_static.buffer(0.0)
        if region_static.is_empty or not region_static.is_valid:
            print(f'Trial {trial} Step {i}: region_static is invalid, breaking')
            trial_failed = True
            break
        
        params = [float(M[0,0]), float(M[0,1]),
                  float(M[1,0]), float(M[1,1]),
                  float(offset[0,0]), float(offset[1,0])]
        region_dyn = affine_transform(region_static, params)

        # Update landmark region: use 50 randomly selected positive measurement times
        if inside:
            # Randomly select 50 indices from positive measurements (or all if less than 50)
            if len(signal_times) <= 50:
                selected_times = signal_times
            else:
                selected_times = random.sample(signal_times, 50)
            
            # For selected past positive measurement times, bloat and intersect
            for past_time in selected_times:
                if past_time <= i:
                    # Reconstruct M and offset for past_time
                    M_past = np.eye(2, dtype=np.longdouble)
                    offset_past = np.zeros((2,1), dtype=np.longdouble)
                    for k in range(past_time):
                        M_past = A @ M_past
                        offset_past = A @ offset_past + B @ controls[k]
                    
                    params_past = [float(M_past[0,0]), float(M_past[0,1]),
                                  float(M_past[1,0]), float(M_past[1,1]),
                                  float(offset_past[0,0]), float(offset_past[1,0])]
                    region_dyn_past = affine_transform(region_static, params_past)
                    
                    # Bloat and intersect
                    dyn_buffer = region_dyn_past.buffer(r).intersection(landmark_region)
                    if not dyn_buffer.is_empty:
                        landmark_region = dyn_buffer
            
            land_buffer = landmark_region.buffer(r).intersection(region_dyn)
            if not land_buffer.is_empty:
                region_dyn = land_buffer
            last_region_dyn = region_dyn
            last_u = u.copy()

        if trial_failed:
            trial_file.close()
            print(f'Trial {trial} FAILED at step {i}: region_static became invalid')
            break
        
        bounds = region_static.bounds
        diam_current = np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])
        
        # Compute landmark diameter
        landmark_bounds = landmark_region.bounds
        last_landmark_diam = np.hypot(landmark_bounds[2] - landmark_bounds[0], 
                                     landmark_bounds[3] - landmark_bounds[1])
        
        trial_file.write(f"{i},{diam_current},{last_landmark_diam}\n")
        print(f'Trial {trial} Step {i}: diameter={diam_current:.4f}, landmark_diam={last_landmark_diam:.4f}, inside={inside}')

        i += 1
        if i >= args.max_steps:
            trial_file.close()
            print(f'Finished trial {trial}')
            toc = time.time()
            print(f'Trial {trial} time elapsed: {toc-tic:.2f} seconds')
            break
