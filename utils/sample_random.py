import numpy as np

def sample_random(n_points, center, size, k=30):
    """
    Generate n random samples uniformly distributed in a 3D box.

    Args:
        n_points (int): Number of samples to generate.
        center (array-like, shape (3,)): Center of the box.
        size (array-like, shape (3,)): Size of the box along each axis.
        k (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (n_points, 3) of sample positions.
    """
    center = np.array(center, dtype=float)
    size = np.array(size, dtype=float)

    # Box extents
    mins = center - size * 0.5
    maxs = center + size * 0.5

    # Set random seed
    np.random.seed(k)

    # Generate random samples uniformly distributed in the box
    samples = np.random.uniform(mins, maxs, size=(n_points, 3))

    return samples

