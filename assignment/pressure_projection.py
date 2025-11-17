import warp as wp
from utils.structs import *

@wp.kernel
def pressure_projection(
    state: StateStruct,
    model: ModelStruct,
    dt: float
):
    """
    Project velocity using pressure gradient to enforce incompressibility.
    
    PURPOSE:
    This function performs the pressure projection step that corrects the velocity field to be
    divergence-free (incompressible). It subtracts the pressure gradient from the velocity
    at each face location on the staggered grid: v_new = v_old - ∇p * dt / ρ. 
    
    INPUT VARIABLES (from state):
    - grid_v_out[grid_x, grid_y, grid_z]: vec3 - Current grid velocity at cell (grid_x, grid_y, grid_z)
        Velocity components are stored at cell faces on a staggered grid:
        - vx is stored at face (i-1/2, j, k) between cells (i-1) and (i)
        - vy is stored at face (i, j-1/2, k) between cells (j-1) and (j)
        - vz is stored at face (i, j, k-1/2) between cells (k-1) and (k)
    - grid_pressure[grid_x, grid_y, grid_z]: float - Pressure at cell center (grid_x, grid_y, grid_z)
        This is the pressure after the Poisson problem is solved (ie it satisfies ∇²p = (1/dt)∇·v).
        Used to compute pressure gradients at face locations.
    - grid_occupancy[grid_x, grid_y, grid_z]: int - Grid occupancy at cell (grid_x, grid_y, grid_z)
        Values: 1 = fluid, 0 = air, -1 = solid. Used to determine boundary conditions:
        - If either adjacent cell is solid (-1), set velocity to 0 at that face
        - If at least one adjacent cell is fluid (1), apply pressure projection
        - If both adjacent cells are air (0), keep velocity unchanged
    
    INPUT VARIABLES (function parameter):
    - dt: float - Time step size
        Used to scale the pressure gradient correction: v_new = v_old - ∇p * dt / ρ
    
    INPUT VARIABLES (from model):
    - inv_dx: float - Inverse of grid cell size (1/dx)
        Used to compute pressure gradients at face locations using finite differences
    - grid_dim_x, grid_dim_y, grid_dim_z: int - Grid dimensions in each direction
        Used for bounds checking when accessing neighboring cells
    
    OUTPUT VARIABLES (modified in state):
    - grid_v_out[grid_x, grid_y, grid_z]: vec3 - Updated grid velocity at cell (grid_x, grid_y, grid_z)
        The velocity after pressure projection. Each component is corrected independently:
        - vx_new = vx_old - (∂p/∂x) * dt / ρ at face (i-1/2, j, k)
        - vy_new = vy_old - (∂p/∂y) * dt / ρ at face (i, j-1/2, k)
        - vz_new = vz_old - (∂p/∂z) * dt / ρ at face (i, j, k-1/2)
        Velocity is set to 0 at faces adjacent to solid cells.
    """
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_v_out[grid_x, grid_y, grid_z] = state.grid_v_in[grid_x, grid_y, grid_z]

