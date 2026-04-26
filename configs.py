from pickle import TRUE
import numpy as np

N = 6
eta = 0.5
n = 2

def compute_D_bar(A, N, I):
    max_norm = 0.0
    A_power = A.copy()
    for k in range(1, N + 1):
        A_power = A_power @ A
        norm_k = np.linalg.norm(A_power - I, ord=2)
        max_norm = max(max_norm, norm_k)
    return max_norm

def check_A_feasible(A, yellow_width, sensor_radius):
    I = np.eye(2)
    diam_M = yellow_width * np.sqrt(2)
    max_D_bar = (1 - eta) * np.sqrt(2 * (n + 1) / n) / ((diam_M / sensor_radius + 2) * (3 - eta))
    
    D_bar_actual = compute_D_bar(A, N, I)
    eigenvals = np.linalg.eigvals(A)
    min_eigenval = np.min(eigenvals)
    abs_eigenvals = np.abs(eigenvals)
    eigenvals_different = not np.isclose(abs_eigenvals[0], abs_eigenvals[1], atol=1e-6)
    
    lhs = diam_M / sensor_radius
    rhs = (1 - eta) / (D_bar_actual * (3 - eta)) * np.sqrt(2 * (n + 1) / n) - 2
    condition_satisfied = lhs <= rhs
    
    feasible = (D_bar_actual <= max_D_bar and min_eigenval > 1.0 and 
                eigenvals_different and condition_satisfied)
    return feasible, D_bar_actual, max_D_bar, min_eigenval

def generate_A_matrix(yellow_width, sensor_radius, min_eigenval=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    I = np.eye(2)
    diam_M = yellow_width * np.sqrt(2)
    max_D_bar = (1 - eta) * np.sqrt(2 * (n + 1) / n) / ((diam_M / sensor_radius + 2) * (3 - eta))
    if min_eigenval is None:
        min_eigenval = 1.001

    feasible_As = []

    for attempt in range(10000):
        e1 = min_eigenval + np.random.uniform(0.0, 0.08)
        e2 = min_eigenval + np.random.uniform(0.0, 0.08)
        if abs(e1 - e2) < 1e-6:
            continue

        D = np.diag([e1, e2])
        P = np.random.randn(2, 2)
        Q, _ = np.linalg.qr(P)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1

        A_test = Q @ D @ Q.T
        D_bar_test = compute_D_bar(A_test, N, I)
        eigenvals_test = np.linalg.eigvals(A_test)
        min_eigenval_test = np.min(eigenvals_test)
        abs_eigenvals_test = np.abs(eigenvals_test)
        eigenvals_different_test = not np.isclose(abs_eigenvals_test[0], abs_eigenvals_test[1], atol=1e-6)

        lhs = diam_M / sensor_radius
        rhs = (1 - eta) / (D_bar_test * (3 - eta)) * np.sqrt(2 * (n + 1) / n) - 2
        condition_satisfied = lhs <= rhs

        if (D_bar_test <= max_D_bar and min_eigenval_test >= min_eigenval and 
            eigenvals_different_test and condition_satisfied):
            feasible_As.append((A_test, min_eigenval_test))

    if feasible_As:
        selected_A, _ = feasible_As[np.random.randint(len(feasible_As))]
        return selected_A

    raise ValueError(f"Could not find feasible A with min_eigenval ≥ {min_eigenval}")


yellow_width = 1
yellow_height = 1
sensor_radius = 2

USE_CUSTOM_A = True
CUSTOM_A = np.array([
    [1.0143, -0.0003],
    [-0.0003, 1.0143]
])
# CUSTOM_A = np.array([
#     [1.0105, -0.0005],
#     [-0.0005, 1.0105]
# ])


if USE_CUSTOM_A:
    feasible, D_bar_actual, max_D_bar, min_eigenval = check_A_feasible(CUSTOM_A, yellow_width, sensor_radius)
    if feasible:
        A = CUSTOM_A
        print(f"Custom A is feasible")
        print(f"A = \n{A}")
        print(f"Minimum eigenvalue: {min_eigenval:.6f}")
        print(f"D̄ = {D_bar_actual:.6f} (max allowed: {max_D_bar:.6f})")
    else:
        print(f"Custom A is NOT feasible")
        print(f"Generating random A instead...")
        A = generate_A_matrix(yellow_width, sensor_radius)
        feasible, D_bar_actual, max_D_bar, min_eigenval = check_A_feasible(A, yellow_width, sensor_radius)
        print(f"Generated A matrix:")
        print(f"A = \n{A}")
        print(f"Minimum eigenvalue: {min_eigenval:.6f}")
        print(f"D̄ = {D_bar_actual:.6f} (max allowed: {max_D_bar:.6f})")
else:
    A = generate_A_matrix(yellow_width, sensor_radius)
    feasible, D_bar_actual, max_D_bar, min_eigenval = check_A_feasible(A, yellow_width, sensor_radius)
    print(f"Generated A matrix:")
    print(f"A = \n{A}")
    print(f"Minimum eigenvalue: {min_eigenval:.6f}")
    print(f"D̄ = {D_bar_actual:.6f} (max allowed: {max_D_bar:.6f})")

B = np.random.uniform(0, 1, (2, 2))

red_center   = [0.2, -0.2]
red_width    = 3.5
red_height   = 3.5

initial_state = [0.5, -0.6]

landmark_center = [2.0, 0.0]  # Legacy, not used in main.py
landmark_region_center = [0.5, 0.5]  # Fixed center for landmark region (green region)

frame_count  = 50
interval_ms  = 200
fig_size     = (9, 9)
margin       = 1.0