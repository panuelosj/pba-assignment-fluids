# usd_scene_writer.py
from __future__ import annotations
from typing import Iterable, Optional, Sequence, Dict, Tuple
import re
import numpy as np
from pxr import Usd, UsdGeom, Sdf, Gf, Vt


# ----------------------------
# Small utilities
# ----------------------------

def _vec3f(xyz) -> Gf.Vec3f:
    x, y, z = xyz
    # Gf.Vec3f ctor wants Python floats (not numpy.float32)
    return Gf.Vec3f(float(x), float(y), float(z))

def _sanitize_name(name: str, fallback: str) -> str:
    """Make a USD-legal path component: [A-Za-z0-9_] only."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(name or fallback))
    return safe or fallback


class USDMultiMeshWriter:
    """
    Sim-agnostic, multi-object OpenUSD writer for deforming triangle meshes.

    Usage pattern A (explicit object registration once, then stream frames):
        w = USDSceneWriter("out.usdc", fps=60, stage_up="Z", sim_up="Y",
                           write_velocities=True)
        w.open()
        w.add_mesh("Cloth", face_counts, face_indices, num_points=N0)
        w.add_mesh("Bunny", face_counts2, face_indices2, num_points=N1)

        for k in range(num_frames):
            w.write_points("Cloth", cloth_vertices_k, timecode=k)   # (N0,3)
            w.write_points("Bunny", bunny_vertices_k, timecode=k)   # (N1,3)
        w.close()

    Usage pattern B (define-if-missing on first call):
        w = USDSceneWriter("out.usdc", ...)
        w.open()
        for k in range(num_frames):
            w.write_mesh_frame("Cloth", faces, V_k, timecode=k)     # defines once, then streams
        w.close()

    Notes:
    - stage_up:   'Y' or 'Z' (Blender is Z-up; recommend 'Z' for sanity)
    - sim_up:     'Y' or 'Z' (how your coordinates are expressed)
    - We remap sim coords → stage coords when they differ:
        Y→Z: (x, y, z) -> (x, -z, y)    [rotate +90° about X]
        Z→Y: (x, y, z) -> (x,  z,-y)    [rotate -90° about X]
    - Topology is assumed constant per object. We validate and error if it changes.
    """

    def __init__(self,
                 output_path: str,
                 fps: float = 24.0,
                 stage_up: str = "Z",
                 meters_per_unit: float = 1.0,
                 root_path: str = "/World",
                 sim_up: str = "Y",
                 write_velocities: bool = False,
                 flush_every_frame: bool = True):
        self.output_path = output_path
        self.fps = float(fps)
        self.stage_up = stage_up.strip().upper()   # 'Y' or 'Z'
        self.meters_per_unit = float(meters_per_unit)
        self.root_path = Sdf.Path(root_path)
        self.sim_up = sim_up.strip().upper()       # 'Y' or 'Z'
        self.write_velocities = write_velocities
        self.flush_every_frame = flush_every_frame

        self._stage: Optional[Usd.Stage] = None
        self._root_xf: Optional[UsdGeom.Xform] = None

        # Per-object state
        self._meshes: Dict[str, UsdGeom.Mesh] = {}             # name -> prim
        self._num_points: Dict[str, int] = {}                  # name -> N
        self._topo_key: Dict[str, Tuple[int, Tuple[int, ...]]] = {}  # name -> (N, flat_indices)
        self._prev_points_sim: Dict[str, Optional[np.ndarray]] = {}  # name -> prev (sim) or None

    # ----------------------------
    # Stage lifecycle
    # ----------------------------

    def open(self):
        """Create a new stage, set metrics, and define /World as defaultPrim."""
        st = Usd.Stage.CreateNew(self.output_path)
        st.SetTimeCodesPerSecond(self.fps)
        st.SetStartTimeCode(0.0)
        st.SetEndTimeCode(0.0)

        # Stage metrics
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z if self.stage_up == "Z" else UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(st, self.meters_per_unit)

        # Root xform + defaultPrim
        self._root_xf = UsdGeom.Xform.Define(st, self.root_path)
        st.SetDefaultPrim(self._root_xf.GetPrim())

        self._stage = st
        st.GetRootLayer().Save()

    def close(self):
        if self._stage:
            self._stage.GetRootLayer().Save()
        self._stage = None
        self._root_xf = None
        self._meshes.clear()
        self._num_points.clear()
        self._topo_key.clear()
        self._prev_points_sim.clear()

    # ----------------------------
    # Object registration & validation
    # ----------------------------

    def add_mesh(self,
                 name: str,
                 face_counts: Sequence[int],
                 face_indices: Sequence[int],
                 num_points: int):
        """
        Define a Mesh prim under root once, with constant topology.
        You must call this before write_points(), unless you use write_mesh_frame().
        """
        assert self._stage is not None and self._root_xf is not None, "Call open() first."

        name = _sanitize_name(name, "Obj")
        if name in self._meshes:
            raise ValueError(f"Mesh '{name}' already exists.")

        # Typed Vt arrays (avoid VtValue list mismatch)
        fc = Vt.IntArray([int(x) for x in face_counts])
        fi = Vt.IntArray([int(x) for x in face_indices])

        # Basic sanity
        if sum(fc) != len(fi):
            raise ValueError(f"'{name}': sum(face_counts) != len(face_indices) ({sum(fc)} != {len(fi)})")
        if num_points <= 0:
            raise ValueError(f"'{name}': num_points must be > 0")

        path = self.root_path.AppendChild(name)
        mesh = UsdGeom.Mesh.Define(self._stage, path)

        # Author topology once (no time samples)
        mesh.GetFaceVertexCountsAttr().Set(fc)
        mesh.GetFaceVertexIndicesAttr().Set(fi)

        # Ensure there are no authored transforms on the mesh prim
        UsdGeom.Xformable(mesh.GetPrim()).ClearXformOpOrder()

        # Track object state
        self._meshes[name] = mesh
        self._num_points[name] = int(num_points)
        self._topo_key[name] = (int(num_points), tuple(int(i) for i in fi))
        self._prev_points_sim[name] = None  # first frame → zero velocity

        # Persist the stage (optional)
        self._stage.GetRootLayer().Save()

    # ----------------------------
    # Frame authoring
    # ----------------------------

    @staticmethod
    def _to_numpy(P: Iterable) -> np.ndarray:
        """Accept numpy / torch / warp / list-like → (N,3) float32 numpy array."""
        try:
            import torch  # type: ignore
            if isinstance(P, torch.Tensor):
                P = P.detach().cpu().numpy()
        except Exception:
            pass
        try:
            import warp as wp  # type: ignore
            if hasattr(wp, "array") and isinstance(P, wp.array):  # type: ignore[attr-defined]
                try:
                    P = P.numpy()
                except Exception:
                    P = P.to_torch().cpu().numpy()
        except Exception:
            pass

        P = np.asarray(P, dtype=np.float32)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError(f"Expected (N,3) array; got {P.shape}")
        return P

    def _remap_to_stage(self, P: np.ndarray) -> np.ndarray:
        """Map sim-space coords (sim_up) → stage coords (stage_up)."""
        su = self.sim_up
        tu = self.stage_up
        if su == tu:
            return P
        if su == "Y" and tu == "Z":
            # Rx(+90°): (x, y, z) -> (x, -z, y)
            return np.column_stack((P[:, 0], -P[:, 2], P[:, 1]))
        if su == "Z" and tu == "Y":
            # Rx(-90°): (x, y, z) -> (x,  z, -y)
            return np.column_stack((P[:, 0],  P[:, 2], -P[:, 1]))
        return P  # shouldn't happen

    def write_points(self, name: str, points_now: Iterable, timecode: float):
        """
        Time-sample the 'points' (and extent, and optionally velocities) for an existing mesh.
        Call once per object you want to write at 'timecode'.
        """
        if self._stage is None:
            raise RuntimeError("Call open() first.")
        if name not in self._meshes:
            raise KeyError(f"Mesh '{name}' is not defined. Call add_mesh(...) or use write_mesh_frame(...).")

        P_sim = self._to_numpy(points_now)

        # Validate vertex count
        #N_expected = self._num_points[name]
        #if P_sim.shape[0] != N_expected:
        #    raise ValueError(f"'{name}': got {P_sim.shape[0]} vertices, expected {N_expected}")

        # Remap to stage frame (so Blender / other Z-up tools don't rotate it on import)
        P_stage = self._remap_to_stage(P_sim)

        t = float(timecode)
        mesh = self._meshes[name]

        # points (time-sampled)
        pts = Vt.Vec3fArray([_vec3f(p) for p in P_stage])
        mesh.GetPointsAttr().Set(pts, time=t)

        # extent (time-sampled)
        mn = P_stage.min(axis=0); mx = P_stage.max(axis=0)
        mesh.GetExtentAttr().Set([_vec3f(mn), _vec3f(mx)], time=t)

        # velocities (time-sampled, units/sec in stage frame)
        
        if self.write_velocities:
            prev_sim = self._prev_points_sim[name]
            if prev_sim is None:
                V_sim = np.zeros_like(P_sim, dtype=np.float32)
            else:
                # Handle variable vertex counts by cloning missing rows
                if prev_sim.shape[0] != P_sim.shape[0]:
                    if prev_sim.shape[0] < P_sim.shape[0]:
                        # Previous frame has fewer points: clone the last row to match current size
                        num_to_add = P_sim.shape[0] - prev_sim.shape[0]
                        last_row = prev_sim[-1:].copy()  # Get last row
                        cloned_rows = np.repeat(last_row, num_to_add, axis=0)
                        prev_sim_padded = np.vstack([prev_sim, cloned_rows])
                    else:
                        # Previous frame has more points: truncate to current size
                        prev_sim_padded = prev_sim[:P_sim.shape[0]]
                    V_sim = (P_sim - prev_sim_padded) * self.fps
                else:
                    V_sim = (P_sim - prev_sim) * self.fps
            V_stage = self._remap_to_stage(V_sim)
            vlist = Vt.Vec3fArray([_vec3f(v) for v in V_stage])
            mesh.GetVelocitiesAttr().Set(vlist, time=t)
            self._prev_points_sim[name] = P_sim

        # Maintain stage time range; keep the max endTimeCode
        cur_end = self._stage.GetEndTimeCode()
        if t > cur_end:
            self._stage.SetEndTimeCode(t)

        if self.flush_every_frame:
            self._stage.GetRootLayer().Save()

    def write_mesh_frame(self,
                         name: str,
                         faces_tri_mx3: Iterable,
                         points_now: Iterable,
                         timecode: float):
        """
        Convenience: define mesh on first call (using faces_tri_mx3), then stream points.
        """
        name = _sanitize_name(name, "Obj")
        if name not in self._meshes:
            # Define topology from faces on the first call
            F = np.asarray(faces_tri_mx3, dtype=np.int32)
            if F.ndim != 2 or F.shape[1] != 3:
                raise ValueError(f"faces must be (M,3) triangles; got {F.shape}")
            counts = [3] * int(F.shape[0])
            indices = F.flatten().tolist()

            # Peek vertex count from points_now
            P_sim = self._to_numpy(points_now)
            self.add_mesh(name, counts, indices, num_points=P_sim.shape[0])

        # Now write the time sample
        self.write_points(name, points_now, timecode)