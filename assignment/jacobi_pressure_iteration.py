import warp as wp
from utils.structs import *

@wp.kernel
def jacobi_pressure_iteration(
    state: StateStruct,
    model: ModelStruct,
    alpha: float,
    beta: float
):
    """
    One Jacobi iteration for solving the Poisson equation.
    
    PURPOSE:
    This function performs one iteration of the Jacobi method to solve the Poisson equation
    ∇²p = (1/dt)∇·v for pressure. 
    
    INPUT VARIABLES (from state):
    - grid_pressure_old[grid_x, grid_y, grid_z]: float - Pressure at cell center from previous iteration
    - grid_divergence[grid_x, grid_y, grid_z]: float - Divergence RHS at cell center
        The right-hand side of the Poisson equation: (1/dt)∇·v
    - grid_occupancy[grid_x, grid_y, grid_z]: int - Grid occupancy at cell (grid_x, grid_y, grid_z)
        Values: 1 = fluid, 0 = air (free surface), -1 = solid. Used to apply boundary conditions:
        - Air cells (0): Set pressure to 0 (Dirichlet boundary condition)
        - Solid cells (-1): Use mirror condition (p_neighbor = p_center) for Neumann boundary condition
        - Fluid cells (1): Use neighbor's actual pressure value
    
    INPUT VARIABLES (function parameters):
    - alpha: float - Relaxation parameter for Jacobi update
        Controls how much of the new value to use: alpha = 1.0 is standard Jacobi, alpha < 1.0 adds damping
    - beta: float - Damping parameter
        Typically beta = 1.0 - alpha. Controls how much of the old value to retain for stability
    
    INPUT VARIABLES (from model):
    - dx: float - Grid cell size
        Used to compute dx² term in the Laplacian discretization
    - grid_dim_x, grid_dim_y, grid_dim_z: int - Grid dimensions in each direction
        Used for bounds checking when accessing neighboring cells
    
    OUTPUT VARIABLES (modified in state):
    - grid_pressure[grid_x, grid_y, grid_z]: float - Updated pressure at cell center (grid_x, grid_y, grid_z)
    """
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_pressure[grid_x, grid_y, grid_z] = 0.0

