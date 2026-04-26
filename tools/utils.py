# utils.py
import numpy as np
import random
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, Point

__all__ = ['minimum_enclosing_circle']

def _circle_from(R):
    if not R:
        return np.array([0.0, 0.0]), 0.0
    elif len(R) == 1:
        return np.array(R[0]), 0.0
    elif len(R) == 2:
        a, b = map(np.array, R)
        center = (a + b) / 2
        radius = np.linalg.norm(a - b) / 2
        return center, radius
    else:
        a, b, c = map(np.array, R)
        d = 2 * (a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
        ux = (np.dot(a,a)*(b[1]-c[1]) +
              np.dot(b,b)*(c[1]-a[1]) +
              np.dot(c,c)*(a[1]-b[1])) / d
        uy = (np.dot(a,a)*(c[0]-b[0]) +
              np.dot(b,b)*(a[0]-c[0]) +
              np.dot(c,c)*(b[0]-a[0])) / d
        center = np.array([ux, uy])
        radius = np.linalg.norm(center - a)
        return center, radius

def _welzl(P, R):
    if not P or len(R) == 3:
        return _circle_from(R)
    p = P.pop(random.randrange(len(P)))
    center, radius = _welzl(P, R)
    if np.linalg.norm(p - center) <= radius:
        P.append(p)
        return center, radius
    R.append(p)
    center, radius = _welzl(P, R)
    R.pop()
    P.append(p)
    return center, radius

def minimum_enclosing_circle(points):
    """
    Compute the minimum enclosing circle of a set of 2D points.
    Returns (center: np.ndarray of shape (2,), radius: float).
    """
    P = points.tolist()
    random.shuffle(P)
    return _welzl(P, [])


def create_ellipse(P_ij, mu_ij):
    unit_circle = Point(0, 0).buffer(1, resolution=128)
    # Convert to float64 for eigh computation
    P_ij_float64 = P_ij.astype(np.float64)
    E_vals, E_vecs = np.linalg.eigh(P_ij_float64)
    L = E_vecs @ np.diag(1.0 / np.sqrt(E_vals)) @ E_vecs.T
    tx, ty = float(mu_ij[0]), float(mu_ij[1])
    a, b = L[0,0], L[0,1]
    d, e = L[1,0], L[1,1]
    return affine_transform(unit_circle, [a, b, d, e, tx, ty])


def get_constellation_points(num_points=6):
    indices = np.random.permutation(num_points)
    return [np.array([np.cos(2 * np.pi * i / num_points), np.sin(2 * np.pi * i / num_points)]) for i in indices]

# def get_constellation_points(num_points=6):
#     return [np.array([np.cos(2 * np.pi * i / num_points), np.sin(2 * np.pi * i / num_points)]) for i in range(num_points)]


def compute_recovery_signal(r, A, B, X_points, u0, N=6):
    c0, R0 = _welzl(list(X_points), [])
    n = A.shape[0]
    I = np.eye(n, dtype=np.longdouble)
    B_inv = np.linalg.inv(B.astype(np.float64)).astype(np.longdouble)
    c0 = np.asarray(c0, dtype=np.longdouble).reshape(n, 1)
    u0 = np.asarray(u0, dtype=np.longdouble).reshape(n, 1)
    u_list = [u0]
    ps = get_constellation_points(num_points=N)
    for i in range(1, N+1):
        p_i = ps[i-1].reshape(n, 1)
        A_pow = np.linalg.matrix_power(A, i+1)

        # use float64 for the spectral‐norm
        norm_term = np.linalg.norm((A_pow - I).astype(np.float64), ord=2)

        term1 = (r - norm_term * R0) * p_i
        term2 = (A_pow - I) @ c0

        sum_term = np.zeros((n, 1), dtype=np.longdouble)
        for j, u_j in enumerate(u_list):
            sum_term += np.linalg.matrix_power(A, i - j) @ B @ u_j  # both (n,1)

        Bu_i = term1 - term2 - sum_term
        u_i = B_inv @ Bu_i
        u_list.append(u_i)
    return u_list[1:], c0, R0
