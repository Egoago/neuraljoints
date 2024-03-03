import numpy as np
import polyscope as ps


def add_bezier():
    # Define control points for the Bezier curve
    control_points = np.array([
        [0.0, 0.0, 0.0],  # P0
        [1.0, 2.0, 0.0],  # P1
        [2.0, -1.0, 0.0],  # P2
        [3.0, 0.0, 0.0]  # P3
    ])

    # Compute points on the Bezier curve
    num_samples = 100
    t_values = np.linspace(0.0, 1.0, num_samples)
    curve_points = np.zeros((num_samples, 3))
    for i, t in enumerate(t_values):
        curve_points[i] = np.sum([((1 - t) ** (3 - j)) * (t ** j) * control_points[j] for j in range(4)], axis=0)

    edges = np.stack([np.arange(len(curve_points) - 1),
                      np.arange(len(curve_points) - 1) + 1], axis=-1)

    # Register the curve with Polyscope
    ps_curve = ps.register_curve_network("Bezier Curve", curve_points, edges)
    ps.register_point_cloud("Control Points", control_points)
