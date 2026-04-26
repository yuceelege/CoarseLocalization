import numpy as np
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


# ---------- Optimization Part ----------
def max_pairwise_inner_prod(flat_vecs, n, N):
    vecs = flat_vecs.reshape(N, n)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    inner = np.dot(vecs, vecs.T)
    mask = np.triu(np.ones_like(inner), 1)
    return np.max(inner[mask == 1])


def find_min_inner_product_vectors(n=3, N=8, seed=0):
    np.random.seed(seed)
    x0 = np.random.randn(N * n)
    cons = [{'type': 'eq', 'fun': lambda x, i=i: np.linalg.norm(x[i*n:(i+1)*n]) - 1}
            for i in range(N)]
    res = minimize(max_pairwise_inner_prod, x0, args=(n, N),
                   constraints=cons, method='SLSQP',
                   options={'maxiter': 2000, 'ftol': 1e-6, 'disp': False})
    vecs = res.x.reshape(N, n)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs, res.fun


# ---------- Compute Voronoi ----------
def compute_spherical_voronoi(P):
    """Compute spherical Voronoi for 3D case"""
    sv = SphericalVoronoi(P, radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()

    m_values = []
    for i, region in enumerate(sv.regions):
        vertices = sv.vertices[region]
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
        dots = np.dot(vertices, P[i])
        m_i = np.min(dots)
        m_values.append(m_i)
    return sv, np.array(m_values)


def compute_circular_voronoi(P):
    """Compute 2D circular Voronoi vertices using bisectors"""
    N = len(P)
    angles = np.arctan2(P[:, 1], P[:, 0])
    order = np.argsort(angles)
    P = P[order]
    angles = angles[order]

    # Compute mid-angles (bisectors)
    mid_angles = (angles + np.roll(angles, -1)) / 2
    # Wrap around if crossing 2π boundary
    mask = np.where(np.diff(angles, append=angles[0] + 2*np.pi) < -np.pi)[0]
    mid_angles[mask] += np.pi
    mid_angles = np.mod(mid_angles, 2*np.pi)

    vertices = np.stack([np.cos(mid_angles), np.sin(mid_angles)], axis=1)
    m_values = np.array([np.min(np.dot(vertices, p)) for p in P])

    return vertices, m_values, P


# ---------- Visualization ----------
def plot_svc(P, sv=None, vertices=None, m_values=None):
    n = P.shape[1]

    # ---------------------- 2D visualization ----------------------
    if n == 2:
        theta = np.linspace(0, 2*np.pi, 400)
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        plt.plot(circle[:,0], circle[:,1], 'k--', alpha=0.3)

        # Plot generators
        plt.scatter(P[:,0], P[:,1], c='r', s=60, label="Generators (p_i)", zorder=5)
        for i, v in enumerate(P):
            plt.arrow(0, 0, v[0], v[1], head_width=0.05, color='b', alpha=0.6)
            if m_values is not None:
                plt.text(v[0]*1.1, v[1]*1.1, f"m{i+1}={m_values[i]:.2f}", ha='center', va='center')

        # Plot Voronoi vertices
        plt.scatter(vertices[:,0], vertices[:,1], c='k', s=20, label="Voronoi Vertices")

        plt.axis('equal')
        plt.legend()
        plt.title("2D Circular Voronoi Diagram (SVC projection)")
        plt.show()
        return

    # ---------------------- 3D visualization ----------------------
    cmap = get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(P)))
    plotter = pv.Plotter(window_size=[900,900])

    # Smooth sphere with cell coloring
    sphere = pv.Sphere(radius=1.0, theta_resolution=512, phi_resolution=256)
    cell_centers = sphere.cell_centers().points
    labels = np.argmax(cell_centers.dot(P.T), axis=1)

    base_cmap = plt.colormaps["tab20"](np.linspace(0, 1, len(P)))
    light = 0.25
    light_cmap = base_cmap.copy()
    light_cmap[:, :3] = light_cmap[:, :3] + (1 - light_cmap[:, :3]) * light
    rgb = (light_cmap[:, :3] * 255).astype(np.uint8)
    sphere.cell_data["RGB"] = rgb[labels]

    plotter.add_mesh(sphere, scalars="RGB", rgb=True, show_scalar_bar=False,
                     smooth_shading=False, lighting=False, show_edges=False)

    # Add grid lines (optional)
    tube_radius = 0.0008
    n_lines = 2
    n_parallels = 9 * n_lines
    n_meridians = 12 * n_lines
    par_angles = np.linspace(-np.pi/2 + np.pi/(n_parallels+1),
                             np.pi/2 - np.pi/(n_parallels+1),
                             n_parallels)
    for phi_lat in par_angles:
        theta = np.linspace(0, 2*np.pi, 360)
        x = np.cos(phi_lat)*np.cos(theta)
        y = np.cos(phi_lat)*np.sin(theta)
        z = np.full_like(theta, np.sin(phi_lat))
        line = pv.lines_from_points(np.column_stack([x, y, z]), close=True)
        plotter.add_mesh(line.tube(radius=tube_radius, n_sides=12),
                         color='dimgray', lighting=False)

    mer_angles = np.linspace(0, 2*np.pi, n_meridians, endpoint=False)
    for theta0 in mer_angles:
        phi_vals = np.linspace(-np.pi/2, np.pi/2, 181)
        x = np.cos(phi_vals)*np.cos(theta0)
        y = np.cos(phi_vals)*np.sin(theta0)
        z = np.sin(phi_vals)
        line = pv.lines_from_points(np.column_stack([x, y, z]), close=False)
        plotter.add_mesh(line.tube(radius=tube_radius, n_sides=12),
                         color='dimgray', lighting=False)

    # Add generators (red) and vertices (black)
    plotter.add_points(P, color="red", point_size=24, render_points_as_spheres=True)
    if sv is not None:
        verts = sv.vertices / np.linalg.norm(sv.vertices, axis=1, keepdims=True)
        plotter.add_points(verts, color="black", point_size=12, render_points_as_spheres=True)

    # Labels
    if m_values is not None:
        for i, p in enumerate(P):
            plotter.add_point_labels(p[None, :],
                                     [f"m{i+1}={m_values[i]:.2f}"],
                                     font_size=10, text_color='black')

    plotter.background_color = 'white'
    plotter.camera_position = 'yz'
    plotter.show(title="3D Spherical Voronoi Cells (SVC) with Vertices")


# ---------- Main ----------
if __name__ == "__main__":
    n, N = 3, 12  # (2,_) or (3, _)
    P, max_ip = find_min_inner_product_vectors(n, N)
    print(f"Maximum pairwise inner product: {max_ip:.4f}")
    print("Vectors:\n", P)
    if n == 3:
        sv, m_values = compute_spherical_voronoi(P)
        plot_svc(P, sv=sv, m_values=m_values)
    elif n == 2:
        vertices, m_values, P = compute_circular_voronoi(P)
        plot_svc(P, vertices=vertices, m_values=m_values)
