import warp as wp
from simulator import Simulator_WARP
from utils.engine_utils import *
import numpy as np
from typing import Optional
from utils.scene import Scene
from utils import *
wp.init()
wp.config.verify_cuda = True

class Sim_Wrapper:
    def __init__(self, scene=None, scene_file=None, device="cuda:0"):
        """
        Initialize simulation wrapper.
        
        Args:
            scene: Scene object to use (if provided)
            scene_file: Path to JSON scene file (if provided, scene is ignored)
            device: Device to use for simulation (default: "cuda:0")
        """
        self.device = device
        if scene_file:
            self.scene = Scene.from_json(scene_file)
        elif scene:
            self.scene = scene
        else:
            # Default scene
            self.scene = Scene(dt=0.02, n_grid=100, grid_lim=1.0, 
                              gravity=[0.0, -9.8, 0.0], density=1000.0)
            # Initial particles
            self.scene.initial_particles = {
                "n_particles": 10000,
                "center": [0.25, 0.5, 0.25],
                "size": [0.25, 0.25, 0.25],
                "sampling_type": "random",
                "k": 100
            }
            
        # Initialize solver with scene parameters
        self.solver = Simulator_WARP(device=device)
        
        # Set solver parameters from scene
        self.solver.use_gauss_seidel = self.scene.use_gauss_seidel
        self.solver.num_jacobi_iterations = self.scene.num_jacobi_iterations
        self.solver.num_gauss_seidel_iterations = self.scene.num_gauss_seidel_iterations
        self.solver.gauss_seidel_omega = self.scene.gauss_seidel_omega
        self.solver.pic_flip_alpha = self.scene.pic_flip_alpha
        self.solver.jacobi_alpha = self.scene.jacobi_alpha
        
        # Load initial particles if specified
        if self.scene.initial_particles:
            init = self.scene.initial_particles
            if init["sampling_type"] == "grid":
                positions = sample_grid(init["n_particles"], init["center"], init["size"], init.get("k", 30))
            elif init["sampling_type"] == "jittered_grid":
                positions = sample_jittered_grid(init["n_particles"], init["center"], init["size"], init.get("k", 30))
            elif init["sampling_type"] == "blue_noise":
                positions = sample_blue_noise(init["n_particles"], init["center"], init["size"], init.get("k", 30))
            elif init["sampling_type"] == "random":
                positions = sample_random(init["n_particles"], init["center"], init["size"], init.get("k", 30))
            else:
                raise ValueError(f"Unknown sampling_type: {init['sampling_type']}")
            
            volumes = np.ones(positions.shape[0]) * 2.5e-8 * 71363. / 1000.
            self.solver.load_from_array(positions, volumes, 
                                       n_grid=self.scene.n_grid, 
                                       grid_lim=self.scene.grid_lim,
                                       device=self.device)
        
        # Set material parameters
        material_params = {
            "material": "fluid",
            'g': self.scene.gravity,
            "density": self.scene.density
        }
        self.solver.set_parameters_dict(material_params, device=self.device)
        
        # Add solid boxes
        for center, size in self.scene.solid_boxes:
            self.solver.add_solid_box(center, size, device=self.device)
        
        # Track current frame
        self.current_frame = 0
        
        # Process frame 0 particle events immediately (before first step)
        # This avoids resetting velocities during step()
        events = self.scene.get_particle_events_at_frame(0)
        for event in events:
            self.add_particle_box(event.n_particles, event.center, event.size, event.sampling_type)

    def add_particle_box(self, n_particles, center, size, sampling_type="grid"):
        """
        Add particles in a box using the specified sampling method.
        
        Args:
            n_particles (int): Desired number of particles (approximate).
            center (array-like, shape (3,)): Center of the box.
            size (array-like, shape (3,)): Size of the box along each axis.
            sampling_type (str): Sampling method to use. Options: "grid", "jittered_grid", "blue_noise", "random". Default: "grid".
        """
        if sampling_type == "grid":
            positions = sample_grid(n_particles, center, size)
        elif sampling_type == "jittered_grid":
            positions = sample_jittered_grid(n_particles, center, size)
        elif sampling_type == "blue_noise":
            positions = sample_blue_noise(n_particles, center, size)
        elif sampling_type == "random":
            positions = sample_random(n_particles, center, size)
        else:
            raise ValueError(f"Unknown sampling_type: {sampling_type}. Must be one of: 'grid', 'jittered_grid', 'blue_noise', 'random'")
        
        box_volume = size[0] * size[1] * size[2]
        particle_volume = box_volume / positions.shape[0]
        volumes = np.ones(positions.shape[0]) * particle_volume
        self.solver.add_particles(positions, volumes, device=self.device)
        
        material_params = {
            "material": "fluid",
            'g': self.scene.gravity,
            "density": self.scene.density
        }
        self.solver.set_parameters_dict(material_params, device=self.device)

    def step(self):
        # Check for particle events at current frame (skip frame 0, already processed in __init__)
        if self.current_frame > 0:
            events = self.scene.get_particle_events_at_frame(self.current_frame)
            for event in events:
                self.add_particle_box(event.n_particles, event.center, event.size, event.sampling_type)

        for i in range(self.scene.frames_per_output):
            # Step simulation with scene dt
            self.solver.p2g2p(1, self.scene.dt, device=self.device)
            
            # Increment frame counter
            self.current_frame += 1
    
    def get_positions(self):
        position = self.solver.export_particle_x_to_torch().cpu().numpy()*10.
        return position
    
    def get_fluid_occupancy(self):
        position = self.solver.export_fluid_occupancy_to_torch().cpu().numpy()*10.
        return position
    
    def get_solid_occupancy(self):
        position = self.solver.export_solid_occupancy_to_torch().cpu().numpy()*10.
        return position