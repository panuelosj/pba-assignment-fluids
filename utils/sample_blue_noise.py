import numpy as np
import random
from math import ceil, sqrt

def sample_blue_noise(n_points, center, size, k=30):
    """
    Generate approximately n blue-noise (Poisson-disk) samples in a 3D box.

    Args:
        n_points (int): Desired number of samples (approximate).
        center (array-like, shape (3,)): Center of the box.
        size (array-like, shape (3,)): Size of the box along each axis.
        k (int): Number of candidates per active point (Poisson disk parameter).

    Returns:
        np.ndarray: Array of shape (m, 3) of sample positions.
    """
    center = np.array(center, dtype=float)
    size = np.array(size, dtype=float)

    # Box extents
    mins = center - size * 0.5
    maxs = center + size * 0.5

    # Estimate radius based on desired number of samples
    box_volume = np.prod(size)
    r = (box_volume / n_points) ** (1/3) * 0.7  # scaling factor for spacing

    # Cell size for grid
    cell_size = r / sqrt(3)
    grid_shape = np.ceil(size / cell_size).astype(int)
    grid = -np.ones(grid_shape, dtype=int)

    samples = []
    active = []

    def grid_coords(p):
        return np.floor((p - mins) / cell_size).astype(int)

    def in_box(p):
        return np.all((p >= mins) & (p <= maxs))

    def too_close(p):
        gx, gy, gz = grid_coords(p)
        rx = range(max(0, gx-2), min(grid_shape[0], gx+3))
        ry = range(max(0, gy-2), min(grid_shape[1], gy+3))
        rz = range(max(0, gz-2), min(grid_shape[2], gz+3))
        for ix in rx:
            for iy in ry:
                for iz in rz:
                    idx = grid[ix, iy, iz]
                    if idx != -1:
                        if np.linalg.norm(samples[idx] - p) < r:
                            return True
        return False

    # Initialize with one random point
    p0 = np.random.uniform(mins, maxs)
    samples.append(p0)
    active.append(0)
    grid[tuple(grid_coords(p0))] = 0

    # Main loop
    while active and len(samples) < n_points:
        idx = random.choice(active)
        base = samples[idx]
        found = False

        for _ in range(k):
            dir = np.random.normal(size=3)
            dir /= np.linalg.norm(dir)
            mag = np.random.uniform(r, 2 * r)
            p = base + dir * mag
            if in_box(p) and not too_close(p):
                samples.append(p)
                grid[tuple(grid_coords(p))] = len(samples) - 1
                active.append(len(samples) - 1)
                found = True
                break

        if not found:
            active.remove(idx)

    return np.array(samples)
