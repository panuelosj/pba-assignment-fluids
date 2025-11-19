import warp as wp
import torch
from utils.structs import *

@wp.kernel
def zero_grid(state: StateStruct, model: ModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_pressure[grid_x, grid_y, grid_z] = 0.0
    state.grid_pressure_old[grid_x, grid_y, grid_z] = 0.0
    state.grid_divergence[grid_x, grid_y, grid_z] = 0.0

@wp.kernel
def enforce_solid_boundaries(state: StateStruct, model: ModelStruct):
    grid_x, grid_y, grid_z = wp.tid()

    if state.grid_occupancy[grid_x, grid_y, grid_z] == -1:
        state.grid_v_out[grid_x, grid_y, grid_z][0] = 0.0
        state.grid_v_out[grid_x, grid_y, grid_z][1] = 0.0
        state.grid_v_out[grid_x, grid_y, grid_z][2] = 0.0

@wp.kernel
def copy_v_in_to_v_out(
    state: StateStruct
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        state.grid_v_in[grid_x, grid_y, grid_z] = state.grid_v_in[grid_x, grid_y, grid_z] / state.grid_m[grid_x, grid_y, grid_z]
        state.grid_v_out[grid_x, grid_y, grid_z] = state.grid_v_in[grid_x, grid_y, grid_z]
    else:
        # No mass, just copy (or set to zero)
        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def copy_pressure_to_old(
    state: StateStruct
):
    """Copy pressure grid to pressure_old grid for ping-pong buffer in Jacobi iteration.
    Pressure stored at cell centers for staggered grid (MAC)."""
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_pressure_old[grid_x, grid_y, grid_z] = state.grid_pressure[grid_x, grid_y, grid_z]

@wp.func
def hash_int(x: int) -> int:
    # 32-bit xorshift hash
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x85ebca6b) & 0xFFFFFFFF
    x = (x ^ (x >> 13)) & 0xFFFFFFFF
    x = (x * 0xc2b2ae35) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return x

@wp.kernel
def compute_grid_occupancy(
    state: StateStruct,
    model: ModelStruct,
    padding: int
):
    """
    Fill grid_occupancy array: -1 = solid, 0 = air, 1 = fluid.
    
    Args:
        state: State containing grid mass and occupancy
        model: Model containing grid dimensions
        padding: Number of cells of solid padding from outermost extents
    """
    grid_x, grid_y, grid_z = wp.tid()
    
    # Initialize to air
    occupancy = 0
    
    # Check if within padding distance from any boundary -> solid
    if (grid_x < padding or grid_x >= model.grid_dim_x - padding or
        grid_y < padding or grid_y >= model.grid_dim_y - padding or
        grid_z < padding or grid_z >= model.grid_dim_z - padding):
        occupancy = -1  # solid
    else:
        # Check if has mass -> fluid
        if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
            occupancy = 1  # fluid
        # else remains 0 (air)
    
    state.grid_occupancy[grid_x, grid_y, grid_z] = occupancy

@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]

def torch2warp_vec3(t, copy=False, dtype=wp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = wp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value

@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def set_solid_boxes(
    state: StateStruct,
    model: ModelStruct,
    center: wp.vec3,
    size: wp.vec3
):
    """
    Mark grid cells inside an axis-aligned bounding box as solid (occupancy = -1).
    
    Args:
        center: World space center of the box (vec3)
        size: World space size of the box along each axis (vec3)
    """
    grid_x, grid_y, grid_z = wp.tid()
    
    # Convert grid cell center to world space
    world_pos = wp.vec3(
        (wp.float(grid_x) + 0.5) * model.dx,
        (wp.float(grid_y) + 0.5) * model.dx,
        (wp.float(grid_z) + 0.5) * model.dx
    )
    
    # Calculate AABB bounds
    half_size = size * 0.5
    min_bound = center - half_size
    max_bound = center + half_size
    
    # Check if grid cell center is inside the box
    if (world_pos[0] >= min_bound[0] and world_pos[0] <= max_bound[0] and
        world_pos[1] >= min_bound[1] and world_pos[1] <= max_bound[1] and
        world_pos[2] >= min_bound[2] and world_pos[2] <= max_bound[2]):
        # Mark as solid
        state.grid_occupancy[grid_x, grid_y, grid_z] = -1
