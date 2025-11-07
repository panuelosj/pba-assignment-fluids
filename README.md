# Physics-Based Animation: PIC-FLIP Fluids 

In this assignment you will learn to implement a standard hybrid particle-grid fluid simulator that solves on the GPU.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c5857055-06ff-42be-b50a-3d6bfef6cf15" width="100%">
</p>

**Rendered assignment output played back at high-speed**


**WARNING:** Do not create public repos or forks of this assignment or your solution. Do not post code to your answers online or in the class discussion board. Doing so will result in a 20% deduction from your final grade. 

## Checking out the code and setting up the python environment
These instructions use [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) for virtual environment. If you do not have it installed, follow the 
instructions at the preceeding link for your operating system

Checkout the code ```git clone git@github.com:panuelosj/pba-assignment-fluids.git {ROOT_DIR}```, where **{ROOT_DIR}*** is a directory you specify for the source code. 

Next create a virtual environment and install relevant python dependencies install.
```
cd {ROOT_DIR}
conda create -n csc417  python=3.12 -c conda-forge
conda activate csc417
pip install -e . 
```
Optionally, if you have an NVIDIA GPU you might need to install CUDA if you want to use the GPU settings
```
conda install cuda -c nvidia/label/cuda-12.1.0
```
Assignment code templates are stored in the ```{ROOT_DIR}/assignment``` directory. 

**WINDOWS NOTE:** If you want to run the assignments using your GPU you may have to force install torch with CUDA support using 
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Installation without conda
If you are having too many problems with Conda or prefer to use another package manager, we recommend using [UV](https://docs.astral.sh/uv/getting-started/installation/). If you do not have it installed, follow the instructions at the preceeding link for your operating system

Next, create a virtual environment and install relevant python dependencies:

```
cd {ROOT_DIR}
uv venv
uv pip install -e . 
```

If you opted to use UV, you can run the examples using:

```
uv run python main.py <arguments-for-tests>
```

## Tools You Will Use
1. [NVIDIA Warp](https://github.com/NVIDIA/warp) -- python library for kernel programming
2. [PyTorch](https://pytorch.org/) -- python library for array management, deep learning etc ...
   
## Running the Assignment Code
```
cd {ROOT_DIR}
python main.py --scene=tests/{SCENE_JSON_FILE}.json
```
By default the assignment code runs on the cpu, you can run it using your GPU via:
```
python main.py --scene=tests/{SCENE_JSON_FILE}.json --device=cuda
```
Finally, the code runs, headless and can write results to a USD file which can be viewed in [Blender](https://www.blender.org/):
```
python main.py --scene=tests/{SCENE_JSON_FILE}.json --usd_output={FULL_PATH_AND_NAME}.usd --num_steps={Number of steps to run}
```
**Occasionaly on windows the assignment will fail to run the first time, but subsequent attempts should work fine**
## Assignment Structure and Instructions
1. You are responsible for implementing all functions found in the [assignments](./assignment) subdirectory.
2. The [tests](./tests) subdirectory contains the scenes, specified as python files,  we will validate your code against.
3. This [Google Drive link](https://drive.google.com/drive/folders/TODO_EXPORT_SOLUTIONS) (to be posted...) contains output from the solution code that you can use to validate your code. The output consists of **USD (Universal Scene Description)** files which contain simulated results. These can be played back in any USD viewer. I use [Blender](https://www.blender.org/). You can output your own simulations as USD files, load both files in blender and examine the simulations side-by-side. A blender file is available at the blender folder if you want to render out with the same material as the gif above.

## Background and Resources
PIC-FLIP was introduced into graphics in [this paper](https://www.cs.ubc.ca/~rbridson/docs/zhu-siggraph05-sandfluid.pdf) by Zhu and Bridson. The [SIGGRAPH Coursenotes](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf) is also a good resource for implementing this technique, as is a [recent video](https://www.youtube.com/watch?v=Q78wvrQ9xsU) by [Sebastian Lague](https://www.youtube.com/@SebastianLague).

## PIC-FLIP Pipeline
In this assignment, you will be implementing a fluid simulator according to the following PIC-FLIP Pipeline:

<img width="931" height="367" alt="image" src="https://github.com/user-attachments/assets/d06464e9-11ef-43dd-bc74-3bca453a4de0" />

This method involves doing operations in two data structures: particles and a grid. As you learned in class, and as presented in the pipeline above, certain operations are more amenable to either one of the data structures. You will be responsible for implementing the operations for advection, body force (gravity), pressure projection, and the grid-to-particle and particle-to-grid interpolations that transfers data between the particle and the grid data structures.

## Advection 
Advection moves particles through the velocity field. For each particle, update its position using the simple forward Euler timestepping:

<img width="397" height="360" alt="image" src="https://github.com/user-attachments/assets/9f655313-3c90-4a52-be7c-9c02e14354fb" />

Advection is simply physically moving the particle forward by shifting its position based on its current velocity. The positions of these particles are markers that dictate which portion of the domain is considered as fluid (any grid cell that contains a particle is considered as being fluid, whose velocity is then solved for in subsequent steps).

## Body Force (Gravity)
Body forces are external forces that act uniformly throughout the fluid volume. In this assignment, we primarily consider gravity, which accelerates the fluid downwards.
Body forces are applied with simple forward Euler again, this time operating on velocities and, taking `g` to be the gravitational acceleration vector (remember that acceleration is the time-derivative of velocity):

<img width="652" height="138" alt="image" src="https://github.com/user-attachments/assets/2baca344-1c32-4e0f-b238-0fe9b6193ab7" />

Note that this can be done in either the particle or grid regimes. In this assignment, the implementation uses it inside the grid (the warp tid's are given to you and should make that clear).

## The Staggered Grid
As previously mentioned, we need to operate on both a particle data structure as well as a grid data structure. The grid data structure, because of the spatial derivatives involved, have to be of a specific form.
The PIC-FLIP method uses a **staggered grid** (also called a MAC grid - Marker-and-Cell grid), which visually have the following schematic:

<img width="447" height="350" alt="image" src="https://github.com/user-attachments/assets/8c6a30c5-910f-431a-9f08-d0ae1ffcdd7c" />

Here, pressure and velocity components all "live" in different places in space. More specifically:
- **Pressures** (green squares) are stored at cell centers (i, j, k)
- **Velocity components** are stored at cell faces:
  - x-component (vx, orange circles) is stored at faces between cells in the x-direction: (i-1/2, j, k)
  - y-component (vy, yellow circles) is stored at faces between cells in the y-direction: (i, j-1/2, k)
  - z-component (vz, not pictured) is stored at faces between cells in the z-direction: (i, j, k-1/2)
  
Because we cannot have half indexing, we simply shift our indexing by the appropriate amounts for the four different grids (see diagram for integer indexing). Notice that from the point of view of a pressure sample (i, j, k), the **left** velocity sample is (i, j, k) and the **right** velocity sample is (i+1, j, k). Likewise, from the point of view of a velocity sample (i, j, k), the **left** pressure sample is (i-1, j, k) and the **right** pressure sample is (i, j, k).
This specific layout ensures that taking spatial derivatives end up placed where they need to be in space (the divergence of velocities naturally falls on cell centers where it can be added to pressures for example), and avoids nasty nullspaces that causes instabilities. 

## Pressure Projection
Pressure projection is the step that enforces incompressibility (∇·u = 0) by correcting the velocity field using the pressure gradient. Note that this will be the most involved part of the assignment, and involves solving the following equations:

<img width="338" height="133" alt="image" src="https://github.com/user-attachments/assets/b08a963f-f156-4a77-9dd1-84649a6d8136" />

and

<img width="439" height="142" alt="image" src="https://github.com/user-attachments/assets/03af478b-f4b1-4544-beb8-222f707a19cf" />

Numerically, this involves three steps:

1. **Computing the divergence** of the current velocity field at each cell center, then scaling it by ρ / dt (RHS of the top equation).
2. **Solving the Poisson equation** ∇²p = (ρ/dt)∇·u to find the pressure field (notice that the RHS is exactly what's computed at step 1).
3. **Correcting the velocity** using the pressure gradient: u_new = u - (dt/ρ) ∇p (the bottom equation).

Note that boundary conditions need to be handled carefully: set velocities to zero at solid boundaries, and set pressures to zero at free surface (air) boundaries.
`state.grid_occupancy` is already computed for you and will give the type of cell at each grid center (valued at 1 for fluid, 0 for air, and -1 for solid).

To perform this solve, you will need to use discrete operators for divergence, gradient, and the Laplacian.

### Divergence Operator
The divergence operator computes ∇·v at each cell center from the staggered velocity field, essentially being a measure of how much fluid is entering vs exiting some cell (and thus "converging" vs "spreading out"). Note that this value "lives" at **cell centers** (so the same spot as pressures).

For a cell at (i, j, k), the divergence is:

<img width="945" height="404" alt="image" src="https://github.com/user-attachments/assets/30319504-4e8f-4803-99c4-d4afdfbd96e7" />

where:
- u[i,j] is the x-velocity at face (i-1/2, j, k)
- u[i+1,j] is the x-velocity at face (i+1/2, j, k)
- v[i,j] is the y-velocity at face (i, j-1/2, k)
- v[i,j+1] is the y-velocity at face (i, j+1/2, k)
- Similar for z-component in 3D.

For incompressible flow, we want ∇·u=0, which is enforced by the pressure projection step. Only compute divergence in fluid cells; set it to zero for air and solid cells (since we won't need to update velocities at those cells).

### Laplace Operator
The Laplace operator (∇²) is used in the Poisson equation to solve for pressure. Notice that this too must exist at **cell centers** (since we have ∇²p = (ρ/dt)∇·u and we know that RHS must be defined on cell centers, and the LHS must agree in the domain). 

The discrete Laplacian at cell center (i, j, k) is:

<img width="931" height="517" alt="image" src="https://github.com/user-attachments/assets/ec07f820-0ab7-4edb-bb92-28b7f43366d0" />

This is the standard 5-point stencil in 2D, you can derive a similar 7-point stencil in 3D. The Poisson equation ∇²p = divergence is solved iteratively using either the Jacobi method or Gauss-Seidel with Successive Over-Relaxation (SOR) (see later section). Boundary conditions must be handled:
- **Free surface (air cells)**: p = 0 (Dirichlet boundary condition)
- **Solid cells**: p_solid = p_fluid (Neumann boundary condition, mirror the fluid pressure)

### Gradient Operator
The gradient operator computes ∇p for the pressure projection step. This now must live on **cell faces** since you need to compute central finite differences of pressures which live on cell centers.

<img width="786" height="391" alt="image" src="https://github.com/user-attachments/assets/286d3167-a18a-4706-8bfb-0317ff4fc402" />

On a staggered grid, the gradient is computed at face locations:

- **At x-face (i-1/2, j, k)**: ∇p_x = (p[i] - p[i-1]) / dx
- **At y-face (i, j-1/2, k)**: ∇p_y = (p[j] - p[j-1]) / dy  
- **At z-face (i, j, k-1/2)**: ∇p_z = (p[k] - p[k-1]) / dz

The velocity is then updated using: v_new = v_old - ∇p × dt / ρ. This correction subtracts the divergent component of the velocity, making it incompressible. Handle boundaries carefully: if either adjacent cell is solid, set the velocity component at that face to the solid velocity (zero for purposes of this assignment).

### Jacobi Iteration
We want to find a solution for pressures that solves the following equation:

<img width="536" height="238" alt="image" src="https://github.com/user-attachments/assets/058aa55e-f5a3-437a-b4af-8c1b1ad7fd4e" />

Which if we discretize according to the above Laplacian operator, looks like:

<img width="994" height="254" alt="image" src="https://github.com/user-attachments/assets/9280107b-e059-4bc4-9ba4-aad2c99e86c7" />

where P is the pressure at our current grid cell, sum(P_nbr) is the sum of pressures at the neighbouring grid cells (i±1, j±1, k±1), and n_nbr is the number of these neighbours (4 in 2D, 6 in 3D). Assuming we have an accurate estimate for the neighbouring pressures, this means we can isolate for the pressure at our current point:

<img width="984" height="310" alt="image" src="https://github.com/user-attachments/assets/d13d5a24-5107-40de-a77a-7c16d6529061" />

Taking this to be true everywhere, this means that we can start off with some guess for the pressures everywhere, and iteratively compute new pressures. This gives increasingly better estimates for the solution pressure that satisfies this equation, which hopefully converges to the true solution. This iterative method of guessing and recomputing is called Jacobi iteration, which updates all grid cells in parallel using values from the previous iteration.

**Key Implementation Details:**
1. **Read from old values**: All cells read from `p_old` simultaneously, making this method highly parallelizable
2. **Boundary conditions**: 
   - Air cells (free surface): Set p = 0 (Dirichlet boundary condition)
   - Solid cells: Use mirror condition (p_neighbor = p_center) for Neumann boundary condition
   - Fluid neighbors: Use their actual pressure values
3. **Relaxation**: Apply weighted averaging: p_new = α × p_jacobi + β × p_old, where α + β = 1. Standard Jacobi uses α = 1, β = 0, but relaxation can improve convergence. This is exposed as a parameter in the scene files.
4. **Iteration**: Repeat this process many times (might take ~500-1000 iterations to have the fluid not lose volume) until convergence

The Jacobi method is simple and parallelizable but converges slowly.

## Particle-to-Grid
Particle-to-grid (P2G) transfers particle mass and momentum to the grid. For each particle:

1. **Find the base grid cell** containing the particle (convert particle position to grid coordinates)
2. **Compute interpolation weights** using cubic B-spline kernels (or simpler linear weights) for a 3×3×3 neighborhood (see diagram below)
3. **Transfer mass and velocity** to each grid node in the neighborhood:
   - Mass: m_grid += weight × m_particle
   - Velocity: u_grid += weight × v_particle

<img width="945" height="528" alt="image" src="https://github.com/user-attachments/assets/15d197a1-3c25-47a9-9ed3-d54c493b57e6" />

<img width="947" height="531" alt="image" src="https://github.com/user-attachments/assets/2bcd2bbb-4c70-4ea3-b2a8-49ea52c67c6f" />

## Grid-to-Particle
Grid-to-particle (G2P) transfers the updated grid velocity back to particles. This is where PIC-FLIP blending happens:

1. **Interpolate the new grid velocity** at the particle position using trilinear interpolation (PIC component) ( u_PIC = interp(u_new_grid) )
2. **Interpolate the change in grid velocity** at the particle position using trilinear interpolation (FLIP component) ( Δv_FLIP = interp(u_new_grid - u_old_grid) )
3. **Apply FLIP blending**: u_particle = α × u_PIC + (1-α) × (v_particle_old + Δv_FLIP)

where:
- **PIC (α = 1.0)**: Directly use the interpolated new grid velocity
- **FLIP (α = 0.0)**: Add the velocity change to the old particle velocity
- **PIC-FLIP (0 < α < 1)**: Blend both approaches for better stability and detail preservation

Typical values for α are around 0.01-0.1, giving mostly FLIP with a small PIC component for stability. 

For interpolation, use trilinear interpolation weights based on the fractional position of the particle within its containing grid cell:

<img width="936" height="524" alt="image" src="https://github.com/user-attachments/assets/ddf1e72e-5f53-4e50-8600-04dc31627960" />

## Debugging Hints

1. I suggest starting by implementing advect and apply_gravity, then g2p and p2g. These files can be debugged independently of pressure projection (it should just let your particles fall down). Implementing pressure projection is near impossible without getting these files working.
2. I've included a visual in polyscope for previewing the fluid and solid cells. You can make use of these to visually reason on where grid points are. A lot of the discrete operators (gradient, divergence, laplacian) involve local stencils that simply need to query the nearby grid cells. It is very easy to have an off-by-one error here. Draw yourself a diagram on what each grid stencil should look like.
3. If you are getting memory access issues, it's likely because you're trying to query a grid point that's out of bounds. Make sure to include bounds checks that your indexing doesn't go below 0 or go above the grid limits.
4. You can test all files minus jacobi_pressure_iteration.py by switching the "use_gauss_seidel" flag on the json scene files to true. Instead of using your implementation of jacobi_pressure_iteration, it replaces it with a Gauss-Seidel solve. This method converges faster and will be helpful for checking if the rest of your simulation is working correctly.

## Admissable Code and Libraries
You are allowed to use any functions in the warp and warp.sparse packages. You ARE NOT allowed to use code from other warp packages like warp.fem. You cannot use code from any other external simulation library.  

## Hand-In
We will collect and grade the assignment using [MarkUs](https://markus.teach.cs.toronto.edu/markus/courses/109)

## Late Penalty
The late penalty is the same as for the course, specified on the [main github page](https://github.com/dilevin/CSC417-physics-based-animation).

















