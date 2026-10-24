"""SAM 3D Body vertices + faces -> upright, centered, single-color GLB."""

from pathlib import Path

import numpy as np
import trimesh

# same light blue as the upstream overlay renderer, so the 3D viewer matches the overlay panel
MESH_COLOR = (0.65098039, 0.74117647, 0.85882353, 1.0)

# camera coords have Y down and Z forward; 180 deg about X makes Y up and faces the viewer
FLIP_X = np.diag([1.0, -1.0, -1.0])


def to_glb(vertices: np.ndarray, faces: np.ndarray, path: Path) -> Path:
    v = vertices @ FLIP_X.T  # a proper rotation (det=+1), so face winding stays correct
    v = v - (v.min(axis=0) + v.max(axis=0)) / 2
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=MESH_COLOR, metallicFactor=0.0, roughnessFactor=0.6
    )
    mesh = trimesh.Trimesh(
        vertices=v,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(material=material),
        process=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path, file_type="glb")
    return path
