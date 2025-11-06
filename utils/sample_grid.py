import numpy as np
from math import ceil

def sample_grid(n_points, center, size, k=30):
    """
    Generate approximately n grid samples in a 3D box.

    Args:
        n_points (int): Desired number of samples (approximate).
        center (array-like, shape (3,)): Center of the box.
        size (array-like, shape (3,)): Size of the box along each axis.
        k (int): Unused parameter (kept for interface compatibility).

    Returns:
        np.ndarray: Array of shape (m, 3) of sample positions.
    """
    center = np.array(center, dtype=float)
    size = np.array(size, dtype=float)

    # Box extents
    mins = center - size * 0.5
    maxs = center + size * 0.5

    # Calculate number of grid points per dimension to get approximately n samples
    # n ≈ nx * ny * nz, and assuming roughly cubic grid spacing
    grid_points_per_dim = int(np.ceil(n_points ** (1/3)))
    
    # Create grid points along each axis
    x_vals = np.linspace(mins[0], maxs[0], grid_points_per_dim)
    y_vals = np.linspace(mins[1], maxs[1], grid_points_per_dim)
    z_vals = np.linspace(mins[2], maxs[2], grid_points_per_dim)
    
    # Generate all combinations using meshgrid
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    
    # Flatten and combine into (m, 3) array
    samples = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    
    return samples

