import numpy as np
from utils.sample_grid import sample_grid

def sample_jittered_grid(n_points, center, size, k=30):
    """
    Generate approximately n jittered grid samples in a 3D box.
    Starts with a regular grid and adds random jitter to each point.

    Args:
        n_points (int): Desired number of samples (approximate).
        ccenter (array-like, shape (3,)): Center of the box.
        size (array-like, shape (3,)): Size of the box along each axis.
        k (float): Jitter amount as a fraction of grid cell size (0.0 = no jitter, 1.0 = full cell jitter).

    Returns:
        np.ndarray: Array of shape (m, 3) of sample positions.
    """
    # Get regular grid samples (grid_sample_3d uses n_points, center, size)
    grid_samples = sample_grid(n_points, center, size, k=0)  # k=0 since we don't need k for grid generation
    
    center = np.array(center, dtype=float)
    size = np.array(size, dtype=float)
    
    # Calculate grid cell size
    grid_points_per_dim = int(np.ceil(n_points ** (1/3)))
    cell_size = size / grid_points_per_dim
    
    # Generate random jitter for each sample
    # Jitter is uniformly distributed in [-k*cell_size/2, k*cell_size/2]
    jitter_scale = k / 100.0  # Convert k to a fraction (k=30 means 30% jitter)
    jitter = np.random.uniform(
        -jitter_scale * cell_size / 2.0,
        jitter_scale * cell_size / 2.0,
        size=grid_samples.shape
    )
    
    # Apply jitter to grid samples
    jittered_samples = grid_samples + jitter
    
    # Clamp to box bounds to ensure all points stay within the box
    mins = center - size * 0.5
    maxs = center + size * 0.5
    jittered_samples = np.clip(jittered_samples, mins, maxs)
    
    return jittered_samples

