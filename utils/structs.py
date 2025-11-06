import warp as wp

@wp.struct
class ModelStruct:
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    gravitational_acceleration: wp.vec3

@wp.struct
class StateStruct:
    ###### essential #####
    # particle
    particle_x: wp.array(dtype=wp.vec3)
    particle_v: wp.array(dtype=wp.vec3)
    particle_vol: wp.array(dtype=float)
    particle_mass: wp.array(dtype=float)
    particle_density: wp.array(dtype=float)

    # grid
    grid_occupancy: wp.array(dtype=int, ndim=3)
    grid_m: wp.array(dtype=float, ndim=3)
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3)
    grid_v_out: wp.array(dtype=wp.vec3, ndim=3)

    grid_pressure_old: wp.array(dtype=float, ndim=3)
    grid_pressure: wp.array(dtype=float, ndim=3)
    grid_divergence: wp.array(dtype=float, ndim=3)
