import warp as wp
from utils.structs import *

@wp.kernel
def p2g(state: StateStruct, model: ModelStruct):
    """
    Particle-to-Grid (P2G) transfer using cubic B-spline interpolation.
    
    PURPOSE:
    This function transfers particle mass and momentum to the grid using a cubic B-spline
    interpolation kernel. Each particle distributes its mass and momentum to a 3×3×3
    neighborhood of grid cells, with weights determined by the particle's position relative
    to the grid nodes. 
    
    INPUT VARIABLES (from state):
    - particle_x[p]: vec3 - World-space position of particle p
    - particle_mass[p]: float - Mass of particle p
    - particle_v[p]: vec3 - Momentum vector of particle p
    
    INPUT VARIABLES (from model):
    - inv_dx: float - Inverse of grid cell size (1/dx), used to convert world coordinates to grid coordinates
    - grid_dim_x, grid_dim_y, grid_dim_z: int - Grid dimensions in each direction (typically all equal to n_grid)
    
    OUTPUT VARIABLES (modified in state):
    - grid_m[ix, iy, iz]: float - Grid mass array (accumulated via atomic_add)
        Each grid cell accumulates mass from all particles in its 3×3×3 neighborhood.
        After all particles are processed, grid_m[i,j,k] contains the total mass at grid cell (i,j,k).
    
    - grid_v_in[ix, iy, iz]: vec3 - Grid momentum array (accumulated via atomic_add)
        Each grid cell accumulates momentum (velocity*mass) from all particles in its neighborhood.
        After all particles are processed, grid_v_in[i,j,k] contains the total momentum at grid cell (i,j,k)
    
    BOUNDARY HANDLING:
    - Particles near or outside grid boundaries are handled by skipping out-of-bounds grid cells
    - This prevents illegal memory access
    
    PARALLELIZATION:
    - This kernel runs in parallel, with one thread per particle (p = wp.tid())
    - Atomic operations (atomic_add) are used to safely accumulate mass and velocities
      when multiple particles contribute to the same grid cell
    - Note that you won't need to do the normalization of weights inside this file, this is done at copy_v_in_to_v_out.
    """
    p = wp.tid()