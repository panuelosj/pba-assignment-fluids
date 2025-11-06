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
    
    # Red-black ordering: only process cells of the specified color
    cell_color = (grid_x + grid_y + grid_z) % 2
    if cell_color != color:
        return
    
    # Get occupancy and current pressure at this node
    occupancy = state.grid_occupancy[grid_x, grid_y, grid_z]
    
    # Set boundary conditions based on occupancy
    if occupancy == 0:  # Air (free surface): Dirichlet BC, pressure = 0
        state.grid_pressure[grid_x, grid_y, grid_z] = 0.0
        return
    
    # Fluid cell: solve Poisson equation
    divergence = state.grid_divergence[grid_x, grid_y, grid_z]
    p_center_old = state.grid_pressure_old[grid_x, grid_y, grid_z]
    
    # Check each neighbor and apply boundary conditions based on occupancy
    # For Gauss-Seidel: use updated values (grid_pressure) for neighbors of opposite color
    # and old values (grid_pressure_old) for neighbors of same color
    p_x_plus = p_center_old
    p_x_minus = p_center_old
    p_y_plus = p_center_old
    p_y_minus = p_center_old
    p_z_plus = p_center_old
    p_z_minus = p_center_old
    
    # X+ direction
    if grid_x + 1 < model.grid_dim_x:
        occ_neighbor = state.grid_occupancy[grid_x + 1, grid_y, grid_z]
        if occ_neighbor == 1:  # Fluid
            neighbor_color = (grid_x + 1 + grid_y + grid_z) % 2
            if neighbor_color != color:
                # Use updated value (already computed)
                p_x_plus = state.grid_pressure[grid_x + 1, grid_y, grid_z]
            else:
                # Use old value (will be updated later)
                p_x_plus = state.grid_pressure_old[grid_x + 1, grid_y, grid_z]
        elif occ_neighbor == 0:  # Air (free surface): p = 0
            p_x_plus = 0.0
    
    # X- direction
    if grid_x - 1 >= 0:
        occ_neighbor = state.grid_occupancy[grid_x - 1, grid_y, grid_z]
        if occ_neighbor == 1:  # Fluid
            neighbor_color = (grid_x - 1 + grid_y + grid_z) % 2
            if neighbor_color != color:
                p_x_minus = state.grid_pressure[grid_x - 1, grid_y, grid_z]
            else:
                p_x_minus = state.grid_pressure_old[grid_x - 1, grid_y, grid_z]
        elif occ_neighbor == 0:  # Air (free surface): p = 0
            p_x_minus = 0.0
    
    # Y+ direction
    if grid_y + 1 < model.grid_dim_y:
        occ_neighbor = state.grid_occupancy[grid_x, grid_y + 1, grid_z]
        if occ_neighbor == 1:  # Fluid
            neighbor_color = (grid_x + grid_y + 1 + grid_z) % 2
            if neighbor_color != color:
                p_y_plus = state.grid_pressure[grid_x, grid_y + 1, grid_z]
            else:
                p_y_plus = state.grid_pressure_old[grid_x, grid_y + 1, grid_z]
        elif occ_neighbor == 0:  # Air (free surface): p = 0
            p_y_plus = 0.0
    
    # Y- direction
    if grid_y - 1 >= 0:
        occ_neighbor = state.grid_occupancy[grid_x, grid_y - 1, grid_z]
        if occ_neighbor == 1:  # Fluid
            neighbor_color = (grid_x + grid_y - 1 + grid_z) % 2
            if neighbor_color != color:
                p_y_minus = state.grid_pressure[grid_x, grid_y - 1, grid_z]
            else:
                p_y_minus = state.grid_pressure_old[grid_x, grid_y - 1, grid_z]
        elif occ_neighbor == 0:  # Air (free surface): p = 0
            p_y_minus = 0.0
    
    # Z+ direction
    if grid_z + 1 < model.grid_dim_z:
        occ_neighbor = state.grid_occupancy[grid_x, grid_y, grid_z + 1]
        if occ_neighbor == 1:  # Fluid
            neighbor_color = (grid_x + grid_y + grid_z + 1) % 2
            if neighbor_color != color:
                p_z_plus = state.grid_pressure[grid_x, grid_y, grid_z + 1]
            else:
                p_z_plus = state.grid_pressure_old[grid_x, grid_y, grid_z + 1]
        elif occ_neighbor == 0:  # Air (free surface): p = 0
            p_z_plus = 0.0
            
    # Z- direction
    if grid_z - 1 >= 0:
        occ_neighbor = state.grid_occupancy[grid_x, grid_y, grid_z - 1]
        if occ_neighbor == 1:  # Fluid
            neighbor_color = (grid_x + grid_y + grid_z - 1) % 2
            if neighbor_color != color:
                p_z_minus = state.grid_pressure[grid_x, grid_y, grid_z - 1]
            else:
                p_z_minus = state.grid_pressure_old[grid_x, grid_y, grid_z - 1]
        elif occ_neighbor == 0:  # Air (free surface): p = 0
            p_z_minus = 0.0
    
    # Sum of neighbors (with boundary conditions applied)
    p_neighbors = p_x_plus + p_x_minus + p_y_plus + p_y_minus + p_z_plus + p_z_minus
    
    # dx² term from Laplacian
    dx_squared = model.dx * model.dx
    
    # Gauss-Seidel update: p_new = (p_neighbors - dx² * div) / 6
    p_gs = (p_neighbors - dx_squared * divergence) / 6.0
    
    # SOR (Successive Over-Relaxation): p_new = omega * p_gs + (1 - omega) * p_old
    # omega = 1.0 is standard Gauss-Seidel, omega > 1.0 is over-relaxation
    state.grid_pressure[grid_x, grid_y, grid_z] = omega * p_gs + (1.0 - omega) * p_center_old

