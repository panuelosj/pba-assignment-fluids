# This file constains the main application code for the assignment
# DO NOT MODIFY THIS FILE, to complete the assignment you need
# only correctly modify the files in the ./assignment directory
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import warp as wp
import torch
import argparse 
from assignment import *
from utils import *
from sim_wrapper import *

wp.init() #initialize warp

sim = None

#Global variables for UI
simulating = False
scene_file_path = None

# Point cloud point sizes
POINT_SIZE_PARTICLES = 0.05
POINT_SIZE_FLUID_OCCUPANCY = 0.04
POINT_SIZE_SOLID_OCCUPANCY = 0.01

#callback to run one simulation step
def simulation_step():
    sim.step()

def read_positions():
    # Register particle positions as point cloud
    particle_positions = sim.get_positions()
    if isinstance(particle_positions, torch.Tensor):
        particle_positions_np = particle_positions.detach().cpu().numpy()
    else:
        particle_positions_np = particle_positions
    
    # Ensure positions are in the right shape (N x 3)
    if len(particle_positions_np.shape) == 1:
        particle_positions_np = particle_positions_np.reshape(-1, 3)
    
    ps.register_point_cloud("particles", particle_positions_np)
    ps.get_point_cloud("particles").set_radius(POINT_SIZE_PARTICLES, relative=False)

def read_occupancy():
    fluid_occupancy = sim.get_fluid_occupancy()
    if isinstance(fluid_occupancy, torch.Tensor):
        fluid_occupancy_np = fluid_occupancy.detach().cpu().numpy()
    else:
        fluid_occupancy_np = fluid_occupancy
    
    # Ensure positions are in the right shape (N x 3)
    if len(fluid_occupancy_np.shape) == 1:
        fluid_occupancy_np = fluid_occupancy_np.reshape(-1, 3)
    
    ps.register_point_cloud("fluid occupancy", fluid_occupancy_np)
    ps.get_point_cloud("fluid occupancy").set_radius(POINT_SIZE_FLUID_OCCUPANCY, relative=False)


    solid_occupancy = sim.get_solid_occupancy()
    if isinstance(solid_occupancy, torch.Tensor):
        solid_occupancy_np = solid_occupancy.detach().cpu().numpy()
    else:
        solid_occupancy_np = solid_occupancy
    
    # Ensure positions are in the right shape (N x 3)
    if len(solid_occupancy_np.shape) == 1:
        solid_occupancy_np = solid_occupancy_np.reshape(-1, 3)
    
    ps.register_point_cloud("solid occupancy", solid_occupancy_np)
    ps.get_point_cloud("solid occupancy").set_radius(POINT_SIZE_SOLID_OCCUPANCY, relative=False)

def simulation_init(scene_file=None, device="cuda:0"):
    global sim
    print("Initialized Sim")
    if scene_file:
        print(f"Loading scene from: {scene_file}")
        sim = Sim_Wrapper(scene_file=scene_file, device=device)
    else:
        sim = Sim_Wrapper(device=device)
    read_positions()
    read_occupancy()
    # Disable occupancy point clouds by default (only once at initialization)
    ps.get_point_cloud("fluid occupancy").set_enabled(False)
    ps.get_point_cloud("solid occupancy").set_enabled(False)
    
def ui_callback():
    global simulating,q, qm1
    # start top simulation button with text box for end time and dt
    #checkbox for write to USD
    changed_sim, simulating = psim.Checkbox("Start Simulation", simulating)

    #button to run one step of the simulation 
    if psim.Button("Step"):
        simulation_step()
        read_positions()
        read_occupancy()

    #reset button
    if psim.Button("Reset Simulation"):
        print("Resetting Simulation")
        global scene_file_path, sim
        # Use the same device as the current simulation
        device = sim.device if sim else "cuda:0"
        simulation_init(scene_file=scene_file_path, device=device)

    if simulating:
        simulation_step()
        read_positions()
        read_occupancy()

if __name__ == "__main__":
    #check arguments, load approriate model and test configuration
    parser = argparse.ArgumentParser(description="Physics-Based Animation Assignment program")
    parser.add_argument("--scene", help="Path to the scene file")
    parser.add_argument("--usd_output", help="Path to usd output directory, requires num_steps parameters")
    parser.add_argument("--num_steps", help="Number of steps to simulate", type=int)
    parser.add_argument("--device", help="Device to use", type=str, default="cpu")
    args = parser.parse_args()

    print("CUDA version: ", torch.version.cuda)
    print("CUDA available: ", torch.cuda.is_available())
    print("cuDNN version: ", torch.backends.cudnn.version())

    if args.device == "cpu":
        sim_dtype = torch.float64
        sim_device = "cpu"
    elif args.device == "cuda":
        sim_dtype = torch.float64
        sim_device = "cuda"

    sim_device_wp = wp.device_from_torch(sim_device)
    wp.set_device(sim_device_wp)
        
    # initialize polyscope
    ps.init()
    camera_position = np.array([10.0, 10.0, 10.0])   # where the camera is
    look_at_target  = np.array([0.1, 0.3, 0.1])   # what the camera looks at
    up_direction    = np.array([0.0, 1.0, 0.0])   # which direction is "up"
    ps.look_at(camera_position, look_at_target)

    # Store scene file globally for reset functionality
    scene_file_path = args.scene
    # Convert device format: "cuda" -> "cuda:0", "cpu" -> "cpu"
    device_str = sim_device if sim_device == "cpu" else f"{sim_device}:0"

    simulation_init(scene_file=args.scene, device=device_str)

    if args.usd_output:
        if args.num_steps: 
            print("Writing to USD")
          
            # Get initial particle count to register point cloud
            initial_positions = sim.get_positions()
            if isinstance(initial_positions, torch.Tensor):
                initial_positions_np = initial_positions.detach().cpu().numpy()
            else:
                initial_positions_np = initial_positions
            
            if len(initial_positions_np.shape) == 1:
                initial_positions_np = initial_positions_np.reshape(-1, 3)
            
            num_points = initial_positions_np.shape[0]
            
            w = USDMultiMeshWriter(args.usd_output, fps=1/sim.scene.dt, stage_up="Y", sim_up="Y", write_velocities=True)
            w.open()

            # Register point cloud (using add_mesh with no faces - just points)
            # USDMultiMeshWriter can handle point clouds by registering with num_points but no faces
            face_counts = torch.zeros(0, dtype=torch.int32)  # Empty face counts for point cloud
            w.add_mesh("Particles", face_counts=face_counts.numpy(), face_indices=np.array([], dtype=np.int32), num_points=num_points)
                
            for k in range(args.num_steps):
                print("Step "+str(k))
                simulation_step()
                
                # Get particle positions for this frame
                positions = sim.get_positions()
                if isinstance(positions, torch.Tensor):
                    positions_np = positions.detach().cpu().numpy()
                else:
                    positions_np = positions
                
                # Ensure correct shape
                if len(positions_np.shape) == 1:
                    positions_np = positions_np.reshape(-1, 3)
                
                # Write point cloud positions
                w.write_points("Particles", positions_np, timecode=k)

            w.close()
                        
            exit()
        else:
            print("Num steps not provided, skipping USD output")
            exit()


    ps.set_user_callback(ui_callback)

    #turn off polyscope ground plane
    ps.set_ground_plane_mode("none")
    #ps.set_ground_plane_height(0.1)
    #ps.set_automatically_compute_scene_extents(False)
    #ps.set_length_scale(1.)
    #ps.set_bounding_box((0., 0., 0.),(2., 2., 2.))
    ps.show()
