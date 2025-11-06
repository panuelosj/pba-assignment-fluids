"""
Scene data structure for managing simulation parameters, particle addition events, and solid boundaries.
"""
import json
import numpy as np
from typing import List, Dict, Optional


class ParticleBoxEvent:
    """Represents a particle box addition event at a specific frame."""
    def __init__(self, frame: int, n_particles: int, center: List[float], 
                 size: List[float], sampling_type: str = "grid", k: int = 30):
        self.frame = frame
        self.n_particles = n_particles
        self.center = center
        self.size = size
        self.sampling_type = sampling_type
        self.k = k


class Scene:
    """Scene data structure containing simulation parameters and events."""
    
    def __init__(self, dt: float = 0.02, n_grid: int = 100, grid_lim: float = 1.0,
                 gravity: Optional[List[float]] = None, density: float = 1000.0,
                 initial_particles: Optional[Dict] = None,
                 use_gauss_seidel: bool = True,
                 num_iterations: int = 200,
                 omega: float = 1.7,
                 pic_flip_alpha: float = 0.1,
                 jacobi_alpha: float = 1.0,
                 frames_per_output: int = 100):
        """
        Initialize a scene.
        
        Args:
            dt: Time step size
            n_grid: Grid resolution
            grid_lim: Grid limit (world space extent)
            gravity: Gravity vector [x, y, z]
            density: Fluid density
            initial_particles: Dict with keys: n_particles, center, size, sampling_type, k
            use_gauss_seidel: Use Gauss-Seidel SOR solver (True) or Jacobi solver (False)
            num_iterations: Number of pressure solver iterations
            omega: SOR relaxation parameter for Gauss-Seidel (1.0 = standard GS, >1.0 = over-relaxation)
            pic_flip_alpha: PIC-FLIP blending parameter for g2p (0.0 = pure FLIP, 1.0 = pure PIC)
            jacobi_alpha: Relaxation parameter for Jacobi iteration (1.0 = standard Jacobi)
            frames_per_output: Number of simulation frames to run between outputs
        """
        self.dt = dt
        self.n_grid = n_grid
        self.grid_lim = grid_lim
        self.gravity = gravity if gravity is not None else [0.0, -9.8, 0.0]
        self.density = density
        
        # Pressure solver parameters
        self.use_gauss_seidel = use_gauss_seidel
        self.num_iterations = num_iterations
        self.omega = omega
        self.pic_flip_alpha = pic_flip_alpha
        self.jacobi_alpha = jacobi_alpha
        self.frames_per_output = frames_per_output
        
        # Initial particles (added at frame 0, before first step)
        self.initial_particles = initial_particles
        
        # Particle box addition events
        self.particle_events: List[ParticleBoxEvent] = []
        
        # Solid bounding boxes: list of (center, size) tuples
        self.solid_boxes: List[tuple] = []
    
    def add_particle_box_event(self, frame: int, n_particles: int, center: List[float],
                               size: List[float], sampling_type: str = "grid", k: int = 30):
        """Add a particle box addition event at a specific frame."""
        event = ParticleBoxEvent(frame, n_particles, center, size, sampling_type, k)
        self.particle_events.append(event)
        # Sort by frame
        self.particle_events.sort(key=lambda e: e.frame)
    
    def add_solid_box(self, center: List[float], size: List[float]):
        """Add a solid bounding box."""
        self.solid_boxes.append((center, size))
    
    def get_particle_events_at_frame(self, frame: int) -> List[ParticleBoxEvent]:
        """Get all particle events scheduled for a specific frame."""
        return [event for event in self.particle_events if event.frame == frame]
    
    def to_dict(self) -> Dict:
        """Convert scene to dictionary for JSON serialization."""
        return {
            "dt": self.dt,
            "n_grid": self.n_grid,
            "grid_lim": self.grid_lim,
            "gravity": self.gravity,
            "density": self.density,
            "use_gauss_seidel": self.use_gauss_seidel,
            "num_iterations": self.num_iterations,
            "omega": self.omega,
            "pic_flip_alpha": self.pic_flip_alpha,
            "jacobi_alpha": self.jacobi_alpha,
            "frames_per_output": self.frames_per_output,
            "initial_particles": self.initial_particles,
            "particle_events": [
                {
                    "frame": e.frame,
                    "n_particles": e.n_particles,
                    "center": e.center,
                    "size": e.size,
                    "sampling_type": e.sampling_type,
                    "k": e.k
                }
                for e in self.particle_events
            ],
            "solid_boxes": [
                {"center": center, "size": size}
                for center, size in self.solid_boxes
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Scene':
        """Create scene from dictionary."""
        scene = cls(
            dt=data.get("dt", 0.02),
            n_grid=data.get("n_grid", 100),
            grid_lim=data.get("grid_lim", 1.0),
            gravity=data.get("gravity", [0.0, -9.8, 0.0]),
            density=data.get("density", 1000.0),
            initial_particles=data.get("initial_particles", None),
            use_gauss_seidel=data.get("use_gauss_seidel", True),
            num_iterations=data.get("num_iterations", 200),
            omega=data.get("omega", 1.7),
            pic_flip_alpha=data.get("pic_flip_alpha", 0.1),
            jacobi_alpha=data.get("jacobi_alpha", 1.0),
            frames_per_output=data.get("frames_per_output", 100)
        )
        
        # Add particle events
        for event_data in data.get("particle_events", []):
            scene.add_particle_box_event(
                frame=event_data["frame"],
                n_particles=event_data["n_particles"],
                center=event_data["center"],
                size=event_data["size"],
                sampling_type=event_data.get("sampling_type", "grid"),
                k=event_data.get("k", 30)
            )
        
        # Add solid boxes
        for box_data in data.get("solid_boxes", []):
            scene.add_solid_box(box_data["center"], box_data["size"])
        
        return scene
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Scene':
        """Load scene from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json(self, filepath: str):
        """Save scene to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

