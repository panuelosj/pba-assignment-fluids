import warp as wp
from utils.structs import *

@wp.kernel
def g2p(state: StateStruct, model: ModelStruct, alpha: float):
    """
    Grid-to-particle (G2P) transfer using PIC-FLIP blending.
    
    PURPOSE:
    This function transfers the updated grid velocity field back to particles using a
    blended PIC-FLIP interpolation scheme. 
    
    INPUT VARIABLES (from state):
    - particle_x[p]: vec3 - World-space position of particle p
        Used to determine which grid cells to interpolate from
    - particle_v[p]: vec3 - Current velocity of particle p
        The particle's velocity from the previous step, used in the FLIP component
    - grid_v_out[ix, iy, iz]: vec3 - Updated grid velocity at cell (ix, iy, iz)
        The grid velocity after pressure projection, used for the PIC component
    - grid_v_in[ix, iy, iz]: vec3 - Previous grid velocity at cell (ix, iy, iz)
        The grid velocity before pressure projection, used to compute velocity change for FLIP
    
    INPUT VARIABLES (function parameters):
    - alpha: float - PIC-FLIP blending parameter
        Controls the blend between PIC and FLIP:
        - alpha = 1.0: Pure PIC (directly use interpolated new grid velocity)
        - alpha = 0.0: Pure FLIP (add velocity change to old particle velocity)
        - 0 < alpha < 1: Blended PIC-FLIP (typical values: ~0.1 for mostly FLIP with small PIC)
    
    INPUT VARIABLES (from model):
    - inv_dx: float - Inverse of grid cell size (1/dx)
        Used to convert particle world position to grid coordinates
    - grid_dim_x, grid_dim_y, grid_dim_z: int - Grid dimensions in each direction
        Used for bounds checking when accessing grid cells
    
    OUTPUT VARIABLES (modified in state):
    - particle_v[p]: vec3 - Updated velocity of particle p
        The new particle velocity computed by blending PIC and FLIP:
        v_new = alpha * v_PIC + (1 - alpha) * v_FLIP
    """
    p = wp.tid()
    state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)

