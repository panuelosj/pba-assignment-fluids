import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import *

from assignment import *

class Simulator_WARP:
    def __init__(self, n_particles=0, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.initialize(n_particles, n_grid, grid_lim, device=device)
        self.time_profile = {}
        self.i = 0
        self.use_gauss_seidel = True  # Flag to switch between Jacobi and Gauss-Seidel solvers
        self.num_jacobi_iterations = 1000  # Number of Jacobi pressure solver iterations
        self.num_gauss_seidel_iterations = 200  # Number of Gauss-Seidel pressure solver iterations
        self.gauss_seidel_omega = 1.7  # SOR relaxation parameter for Gauss-Seidel
        self.pic_flip_alpha = 0.1  # PIC-FLIP blending parameter for g2p
        self.jacobi_alpha = 1.0  # Relaxation parameter for Jacobi
        self.solid_boxes = []  # List of (center, size) tuples for solid bounding boxes

    def zero_out_fields(self, device="cuda:0"):
        self.state.grid_occupancy = wp.zeros(
            shape=(self.model.n_grid, self.model.n_grid, self.model.n_grid),
            dtype=int,
            device=device,
        )
        self.state.grid_m = wp.zeros(
            shape=(self.model.n_grid, self.model.n_grid, self.model.n_grid),
            dtype=float,
            device=device,
        )
        self.state.grid_v_in = wp.zeros(
            shape=(self.model.n_grid, self.model.n_grid, self.model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.state.grid_v_out = wp.zeros(
            shape=(self.model.n_grid, self.model.n_grid, self.model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        
        # Pressure projection arrays
        self.state.grid_pressure = wp.zeros(
            shape=(self.model.n_grid, self.model.n_grid, self.model.n_grid),
            dtype=float,
            device=device,
        )
        self.state.grid_pressure_old = wp.zeros(
            shape=(self.model.n_grid, self.model.n_grid, self.model.n_grid),
            dtype=float,
            device=device,
        )
        self.state.grid_divergence = wp.zeros(
            shape=(self.model.n_grid, self.model.n_grid, self.model.n_grid),
            dtype=float,
            device=device,
        )

    def initialize(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.n_particles = n_particles

        self.model = ModelStruct()
        self.model.grid_lim = grid_lim
        self.model.n_grid = n_grid
        self.model.grid_dim_x = self.model.n_grid
        self.model.grid_dim_y = self.model.n_grid
        self.model.grid_dim_z = self.model.n_grid
        (
            self.model.dx,
            self.model.inv_dx,
        ) = float(
            self.model.grid_lim / self.model.n_grid
        ), float(
            self.model.n_grid / self.model.grid_lim
        )

        self.model.gravitational_acceleration = wp.vec3(0.0, 0.0, 0.0)
        self.state = StateStruct()

        self.state.particle_x = wp.empty(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # current position

        self.state.particle_v = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity


        self.state.particle_vol = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle volume
        self.state.particle_mass = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle mass
        self.state.particle_density = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.zero_out_fields(device)

        self.time = 0.0

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []

    def add_particles(
        self, positions, volumes, device="cuda:0"
    ):
        """
        Add new particles to the simulation.
        
        Args:
            positions: np.ndarray of shape (m, 3) containing positions of new particles
            volumes: np.ndarray of shape (m,) containing volumes of new particles (default: 1.0 for each)
            device: Device to use for warp arrays
        """
        # Ensure positions is a numpy array with correct shape
        positions = np.asarray(positions, dtype=np.float32)
        if len(positions.shape) != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be shape (m, 3), got {positions.shape}")
        
        m = positions.shape[0]  # Number of new particles
        
        # Get device from existing arrays if they exist, otherwise use provided device
        if hasattr(self, 'state') and self.state.particle_x is not None:
            device = self.state.particle_x.device
        
        # Get current particle data as numpy arrays
        if hasattr(self, 'state') and self.state.particle_x is not None and self.n_particles > 0:
            old_x = self.state.particle_x.numpy()
            old_vol = self.state.particle_vol.numpy()
            
            # Concatenate with new particles
            combined_x = np.concatenate([old_x, positions], axis=0)
            
            # Handle volumes
            volumes = np.asarray(volumes, dtype=np.float32)
            if len(volumes.shape) > 1:
                volumes = volumes.flatten()
            if volumes.shape[0] != m:
                raise ValueError(f"volumes must have shape (m,), got {volumes.shape}")
            new_volumes = volumes
            
            combined_vol = np.concatenate([old_vol, new_volumes], axis=0)
        else:
            # No existing particles, just use new particles
            combined_x = positions
            
            # Handle volumes
            if volumes is None:
                combined_vol = np.ones(m, dtype=np.float32)
            else:
                volumes = np.asarray(volumes, dtype=np.float32)
                if len(volumes.shape) > 1:
                    volumes = volumes.flatten()
                if volumes.shape[0] != m:
                    raise ValueError(f"volumes must have shape (m,), got {volumes.shape}")
                combined_vol = volumes
        
        # Get current grid parameters if they exist
        if hasattr(self, 'model') and self.model is not None:
            n_grid = self.model.n_grid
            grid_lim = self.model.grid_lim
        else:
            n_grid = 100
            grid_lim = 1.0
        
        # Reinitialize system using load_from_array
        self.load_from_array(combined_x, combined_vol, n_grid=n_grid, grid_lim=grid_lim, device=device)


    def load_from_array(
            self, x_array, vol_array, n_grid=100, grid_lim=1.0, device="cuda:0"
    ):
        # Ensure x_array is numpy array with correct shape
        x_array = np.asarray(x_array, dtype=np.float32)
        vol_array = np.asarray(vol_array, dtype=np.float32)
        
        # Ensure x_array is shape (n_particles, dim)
        if len(x_array.shape) == 1:
            x_array = x_array.reshape(-1, 3)
        elif len(x_array.shape) == 2 and x_array.shape[1] != 3:
            raise ValueError(f"x_array must have shape (n, 3) or be transposable to it, got {x_array.shape}")
        
        # Transpose if needed (to ensure (n_particles, dim) format)
        if x_array.shape[0] < x_array.shape[1] and x_array.shape[1] == 3:
            x_array = x_array.transpose()
        
        self.dim, self.n_particles = x_array.shape[1], x_array.shape[0]
        
        # Ensure volume array matches particle count
        if len(vol_array.shape) > 1:
            vol_array = np.squeeze(vol_array, 0) if vol_array.shape[0] == 1 else np.squeeze(vol_array)
        
        assert x_array.shape[0] == vol_array.shape[0], f"x_array and vol_array must have same number of particles: {x_array.shape[0]} vs {vol_array.shape[0]}"
        
        self.initialize(self.n_particles, n_grid, grid_lim, device=device)
        
        print(
            "Particles loaded from arrays. Simulator is re-initialized for the correct n_particles"
        )
        
        self.state.particle_x = wp.from_numpy(
            x_array, dtype=wp.vec3, device=device
        )  # initialize warp array from np
        
        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.state.particle_v],
            device=device,
        )
        
        self.state.particle_vol = wp.from_numpy(
            vol_array, dtype=float, device=device
        )
        
        print("Particles initialized from arrays.")
        print("Total particles: ", self.n_particles)


    def set_parameters_dict(self, kwargs={}, device="cuda:0"):
        if "grid_lim" in kwargs:
            self.model.grid_lim = kwargs["grid_lim"]
        if "n_grid" in kwargs:
            self.model.n_grid = kwargs["n_grid"]
        self.model.grid_dim_x = self.model.n_grid
        self.model.grid_dim_y = self.model.n_grid
        self.model.grid_dim_z = self.model.n_grid
        (
            self.model.dx,
            self.model.inv_dx,
        ) = self.model.grid_lim / self.model.n_grid, float(
            self.model.n_grid / self.model.grid_lim
        )
        self.zero_out_fields(device)

        if "g" in kwargs:
            self.model.gravitational_acceleration = wp.vec3(kwargs["g"][0], kwargs["g"][1], kwargs["g"][2])

        if "density" in kwargs:
            density_value = kwargs["density"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.state.particle_density, density_value],
                device=device,
            )
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.state.particle_density,
                    self.state.particle_vol,
                    self.state.particle_mass,
                ],
                device=device,
            )


    def p2g2p(self, step, dt, device="cuda:0"):
        grid_size = (
            self.model.grid_dim_x,
            self.model.grid_dim_y,
            self.model.grid_dim_z,
        )
        wp.launch(
            kernel=zero_grid,
            dim=(grid_size),
            inputs=[self.state, self.model],
            device=device,
        )

        # p2g
        wp.launch(
            kernel=p2g,
            dim=self.n_particles,
            inputs=[self.state, self.model],
            device=device,
        )  # apply p2g'

        # Copy current pressure to pressure_old for ping-pong buffer
        wp.launch(
            kernel=copy_v_in_to_v_out,
            dim=(grid_size),
            inputs=[self.state],
            device=device,
        )

        # compute occupancy
        wp.launch(
            kernel=compute_grid_occupancy,
            dim=(grid_size),
            inputs=[self.state, self.model, 1],
            device=device,
        )
        
        # Apply any solid boxes defined by the user
        self._apply_solid_boxes(device)
        
        # gravity
        wp.launch(
            kernel=apply_gravity,
            dim=(grid_size),
            inputs=[self.state, self.model, dt],
            device=device,
        )

        # set solid boundaries to zero
        # Step 1: Initialize pressure and divergence grids
        wp.launch(
            kernel=enforce_solid_boundaries,
            dim=(grid_size),
            inputs=[self.state, self.model],
            device=device,
        )

        # Pressure projection for incompressible fluids
        # Step 2: Compute velocity divergence
        wp.launch(
            kernel=compute_rhs_divergence,
            dim=(grid_size),
            inputs=[self.state, self.model, dt],
            device=device,
        )
        
        # Step 3: Solve Poisson equation for pressure
        if self.use_gauss_seidel:
            # Gauss-Seidel SOR parameters
            for iter in range(self.num_gauss_seidel_iterations):
                # Copy current pressure to pressure_old for ping-pong buffer
                wp.launch(
                    kernel=copy_pressure_to_old,
                    dim=(grid_size),
                    inputs=[self.state],
                    device=device,
                )
                # Update red cells (color = 0)
                wp.launch(
                    kernel=gauss_seidel_sor_pressure_iteration,
                    dim=(grid_size),
                    inputs=[self.state, self.model, self.gauss_seidel_omega, 0],
                    device=device,
                )
                # Update black cells (color = 1)
                wp.launch(
                    kernel=gauss_seidel_sor_pressure_iteration,
                    dim=(grid_size),
                    inputs=[self.state, self.model, self.gauss_seidel_omega, 1],
                    device=device,
                )
        else:
            # Jacobi iteration parameters
            beta = 1.0 - self.jacobi_alpha  # Damping parameter (0 = standard Jacobi)
            
            for iter in range(self.num_jacobi_iterations):
                # Copy current pressure to pressure_old for ping-pong buffer
                wp.launch(
                    kernel=copy_pressure_to_old,
                    dim=(grid_size),
                    inputs=[self.state],
                    device=device,
                )
                wp.launch(
                    kernel=jacobi_pressure_iteration,
                    dim=(grid_size),
                    inputs=[self.state, self.model, self.jacobi_alpha, beta],
                    device=device,
                )

        # Step 4: Project velocity using pressure gradient
        wp.launch(
            kernel=pressure_projection,
            dim=(grid_size),
            inputs=[self.state, self.model, dt],
            device=device,
        )

        # g2p
        wp.launch(
            kernel=g2p,
            dim=self.n_particles,
            inputs=[self.state, self.model, self.pic_flip_alpha],
            device=device,
        )

        wp.launch(
            kernel=advect,
            dim=self.n_particles,
            inputs=[self.state, self.model, dt],
            device=device,
        )

        self.time = self.time + dt

    # clone = True makes a copy, not necessarily needed
    def import_particle_x_from_torch(self, tensor_x, clone=True, device="cuda:0"):
        if tensor_x is not None:
            if clone:
                tensor_x = tensor_x.clone().detach()
            self.state.particle_x = torch2warp_vec3(tensor_x, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_v_from_torch(self, tensor_v, clone=True, device="cuda:0"):
        if tensor_v is not None:
            if clone:
                tensor_v = tensor_v.clone().detach()
            self.state.particle_v = torch2warp_vec3(tensor_v, dvc=device)

    def export_particle_x_to_torch(self):
        return wp.to_torch(self.state.particle_x)

    def export_particle_v_to_torch(self):
        return wp.to_torch(self.state.particle_v)

    def export_fluid_occupancy_to_torch(self):
        """
        Export positions of all fluid cells (occupancy == 1) as a torch tensor.
        Returns a (N, 3) torch tensor where each row is the world position of a fluid cell center.
        """
        # Convert occupancy array to numpy
        occupancy_np = self.state.grid_occupancy.numpy()
        
        # Find all fluid cells (occupancy == 1)
        fluid_indices = np.where(occupancy_np == 1)
        
        # Get grid cell size
        dx = self.model.dx
        
        # Convert grid indices to world positions
        # Cell centers are at (i + 0.5) * dx, assuming grid starts at origin
        positions = np.zeros((len(fluid_indices[0]), 3), dtype=np.float32)
        positions[:, 0] = (fluid_indices[0] + 0.5) * dx
        positions[:, 1] = (fluid_indices[1] + 0.5) * dx
        positions[:, 2] = (fluid_indices[2] + 0.5) * dx
        
        # Convert to torch tensor
        return torch.from_numpy(positions)

    def export_solid_occupancy_to_torch(self):
        """
        Export positions of all fluid cells (occupancy == 1) as a torch tensor.
        Returns a (N, 3) torch tensor where each row is the world position of a fluid cell center.
        """
        # Convert occupancy array to numpy
        occupancy_np = self.state.grid_occupancy.numpy()
        
        # Find all fluid cells (occupancy == 1)
        solid_indices = np.where(occupancy_np == -1)
        
        # Get grid cell size
        dx = self.model.dx
        
        # Convert grid indices to world positions
        # Cell centers are at (i + 0.5) * dx, assuming grid starts at origin
        positions = np.zeros((len(solid_indices[0]), 3), dtype=np.float32)
        positions[:, 0] = (solid_indices[0] + 0.5) * dx
        positions[:, 1] = (solid_indices[1] + 0.5) * dx
        positions[:, 2] = (solid_indices[2] + 0.5) * dx
        
        # Convert to torch tensor
        return torch.from_numpy(positions)

    def add_solid_box(self, center, size, device="cuda:0"):
        """
        Add a solid axis-aligned bounding box to the simulation.
        The box will be re-applied after each occupancy computation.
        
        Args:
            center: 3-vector (list, tuple, or numpy array) representing world space center
            size: 3-vector (list, tuple, or numpy array) representing size along each axis
            device: Device to use for kernel launch
        """
        # Convert to numpy arrays for consistency
        center = np.asarray(center, dtype=np.float32)
        size = np.asarray(size, dtype=np.float32)
        
        if center.shape != (3,) or size.shape != (3,):
            raise ValueError("center and size must be 3-element arrays")
        
        # Store the box for re-application
        self.solid_boxes.append((center.copy(), size.copy()))
        
        # Apply the box immediately
        self._apply_solid_boxes(device)
    
    def _apply_solid_boxes(self, device="cuda:0"):
        """
        Apply all stored solid boxes to the grid occupancy.
        Called internally after compute_grid_occupancy.
        """
        grid_size = (
            self.model.grid_dim_x,
            self.model.grid_dim_y,
            self.model.grid_dim_z,
        )
        
        for center, size in self.solid_boxes:
            center_vec = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
            size_vec = wp.vec3(float(size[0]), float(size[1]), float(size[2]))
            
            wp.launch(
                kernel=set_solid_boxes,
                dim=(grid_size),
                inputs=[self.state, self.model, center_vec, size_vec],
                device=device,
            )
    