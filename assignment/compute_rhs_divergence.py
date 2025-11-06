import warp as wp
from utils.structs import *

@wp.kernel
def compute_rhs_divergence(
    state: StateStruct, 
    model: ModelStruct, 
    dt: float
):
    """
    Compute divergence of velocity field at cell centers using staggered grid (MAC).
    
    PURPOSE:
    This function computes the divergence RHS ((1/dt)∇·v) at each grid cell center
    using finite differences on a staggered (MAC) grid. The divergence measures how much the
    velocity field is "spreading out" or "converging" at each location.
    
    INPUT VARIABLES (from state):
    - grid_v_out[grid_x, grid_y, grid_z]: vec3 - Grid velocity at cell (grid_x, grid_y, grid_z)
        Velocity components are stored at cell faces on a staggered grid:
        - vx is stored at face (i-1/2, j, k) between cells (i-1) and (i)
        - vy is stored at face (i, j-1/2, k) between cells (j-1) and (j)
        - vz is stored at face (i, j, k-1/2) between cells (k-1) and (k)
    - grid_occupancy[grid_x, grid_y, grid_z]: int - Grid occupancy at cell (grid_x, grid_y, grid_z)
        Values: 1 = fluid, 0 = air, -1 = solid. Used to determine which cells to compute
        divergence for and to handle boundary conditions at solid interfaces.
    
    INPUT VARIABLES (function parameter):
    - dt: float - Time step size
        Used to scale the divergence for the Poisson equation: ∇²p = ∇·v / dt
    
    INPUT VARIABLES (from model):
    - inv_dx: float - Inverse of grid cell size (1/dx)
        Used to compute finite differences in the divergence calculation
    - grid_dim_x, grid_dim_y, grid_dim_z: int - Grid dimensions in each direction
        Used for bounds checking when accessing neighboring cells
    
    OUTPUT VARIABLES (modified in state):
    - grid_divergence[grid_x, grid_y, grid_z]: float - Divergence RHS at cell center (grid_x, grid_y, grid_z)
        Computed as (1/dt)∇·v = (∂vx/∂x + ∂vy/∂y + ∂vz/∂z) / dt for fluid cells.
        Set to 0.0 for non-fluid cells (air or solid).
    """
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_divergence[grid_x, grid_y, grid_z] = 0.0

