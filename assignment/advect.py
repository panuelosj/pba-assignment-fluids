import warp as wp
from utils.structs import *

@wp.kernel
def advect(state: StateStruct, model: ModelStruct, dt: float):
    """
    Particle advection using forward Euler time integration.
    
    PURPOSE:
    This function advects (moves) particles through space based on their current velocities
    via a simple forward Euler integration step, updating each particle's position
    by moving it along its velocity vector for the duration of the time step.
    
    INPUT VARIABLES (from state):
    - particle_x[p]: vec3 - Current world-space position of particle p
    - particle_v[p]: vec3 - Velocity vector of particle p
    
    INPUT VARIABLES (function parameter):
    - dt: float - Time step size
    
    OUTPUT VARIABLES (modified in state):
    - particle_x[p]: vec3 - Updated world-space position of particle p
        The particle's new position after advection. 
    """
    p = wp.tid()
