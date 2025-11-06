import warp as wp
from utils.structs import *

@wp.kernel
def apply_gravity(
    state: StateStruct, model: ModelStruct, dt: float
):
    """
    Apply gravitational body force to grid velocity field.
    
    PURPOSE:
    This function applies gravitational acceleration to the grid velocity field 
    via a simple forward Euler timestepping scheme.
    This is a body force that acts uniformly throughout the fluid volume, accelerating the
    fluid in the direction specified by the gravitational acceleration vector.
    
    INPUT VARIABLES (from state):
    - grid_v_out[grid_x, grid_y, grid_z]: vec3 - Current grid velocity at cell (grid_x, grid_y, grid_z)
    - grid_occupancy[grid_x, grid_y, grid_z]: int - Grid occupancy at cell (grid_x, grid_y, grid_z)
        Used to check if the cell contains fluid. Values: 1 = fluid, 0 = air, -1 = solid.
        Only cells with occupancy == 1 (fluid) have gravity applied.
    
    INPUT VARIABLES (function parameter):
    - dt: float - Time step size
    
    INPUT VARIABLES (from model):
    - gravitational_acceleration: vec3 - Gravitational acceleration vector
        Typically (0, -9.8, 0) for downward gravity in the Y direction.
    
    OUTPUT VARIABLES (modified in state):
    - grid_v_out[grid_x, grid_y, grid_z]: vec3 - Updated grid velocity at cell (grid_x, grid_y, grid_z)
    """
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

