import warp as wp
from utils.structs import *

@wp.kernel
def gauss_seidel_sor_pressure_iteration(
    state: StateStruct,
    model: ModelStruct,
    omega: float,
    color: int
):
    """
    One Gauss-Seidel SOR iteration for solving the Poisson equation.
    
    PURPOSE:
    This function performs one iteration of the Gauss-Seidel method with Successive Over-Relaxation
    (SOR) to solve the Poisson equation ∇²p = (1/dt)∇·v for pressure. Gauss-Seidel uses updated
    neighbor values immediately (unlike Jacobi), which converges faster but requires careful
    ordering. Red-black ordering is used to enable parallel execution: cells are divided into
    two colors (red and black) based on (i+j+k) % 2, and only cells of the specified color
    are processed in each iteration.
    
    INPUT VARIABLES (from state):
    - grid_pressure[grid_x, grid_y, grid_z]: float - Pressure at cell center (may be updated or old)
        For neighbors of opposite color (already updated this iteration), use this array.
        For neighbors of same color (not yet updated), use grid_pressure_old.
    - grid_pressure_old[grid_x, grid_y, grid_z]: float - Pressure at cell center from previous iteration
        Used for the current cell and neighbors of the same color (not yet updated this iteration)
    - grid_divergence[grid_x, grid_y, grid_z]: float - Divergence RHS at cell center
        The right-hand side of the Poisson equation: (1/dt)∇·v
    - grid_occupancy[grid_x, grid_y, grid_z]: int - Grid occupancy at cell (grid_x, grid_y, grid_z)
        Values: 1 = fluid, 0 = air (free surface), -1 = solid. Used to apply boundary conditions:
        - Air cells (0): Set pressure to 0 (Dirichlet boundary condition)
        - Solid cells (-1): Use mirror condition for Neumann boundary condition
        - Fluid cells (1): Use neighbor's pressure value (updated or old depending on color)
    
    INPUT VARIABLES (function parameters):
    - omega: float - SOR relaxation parameter
        Controls over-relaxation: omega = 1.0 is standard Gauss-Seidel, omega > 1.0 (typically 1.5-1.9)
        is over-relaxation that can accelerate convergence
    - color: int - Color of cells to process in this iteration
        Values: 0 for red cells ((i+j+k) % 2 == 0), 1 for black cells ((i+j+k) % 2 == 1)
        Only cells matching this color are processed; others return immediately
    
    INPUT VARIABLES (from model):
    - dx: float - Grid cell size
        Used to compute dx² term in the Laplacian discretization
    - grid_dim_x, grid_dim_y, grid_dim_z: int - Grid dimensions in each direction
        Used for bounds checking when accessing neighboring cells
    
    OUTPUT VARIABLES (modified in state):
    - grid_pressure[grid_x, grid_y, grid_z]: float - Updated pressure at cell center (grid_x, grid_y, grid_z)
        Only cells matching the specified color are updated. Air cells (occupancy == 0) are set to 0.0.
    """
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_pressure[grid_x, grid_y, grid_z] = 0.0

