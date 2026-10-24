"""Milestone 2 check: samples/* -> GLB + overlay, plus a front-view render of each GLB
as a glTF viewer would show it (Y up, camera on +Z) to confirm upright and centered.

Usage: python -m scripts.make_glbs [--out output/m2]   (from the repo root)
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh

from server import config
from server.model import BodyModel, NoPersonError, decode_image

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def render_glb_front(glb: Path, size=512) -> np.ndarray:
    scene = pyrender.Scene.from_trimesh_scene(trimesh.load(glb), bg_color=(1, 1, 1, 1))
    bounds = scene.bounds
    height = bounds[1][1] - bounds[0][1]
    cam_pose = np.eye(4)
    cam_pose[2, 3] = height * 1.6  # on +Z looking toward -Z, glTF's default "front"
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=cam_pose)
    r = pyrender.OffscreenRenderer(size, size)
    color, _ = r.render(scene)
    r.delete()
    # mark the world origin so "centered" is visible at a glance
    color = color.copy()
    cv2.drawMarker(color, (size // 2, size // 2), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
    return color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"output/m2_{config.DETECTOR}")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config.MESH_DIR = out

    samples = sorted(p for p in (config.ROOT / "samples").iterdir() if p.suffix.lower() in EXTS)
    imgs = {p.stem: decode_image(p.read_bytes()) for p in samples}
    # "empty room": blank background strip from the first sample, no person in it
    first = next(iter(imgs.values()))
    imgs["_empty_room"] = cv2.resize(first[: first.shape[0] // 8], (960, 720))

    t = time.perf_counter()
    model = BodyModel()
    model.load()
    model.warmup(first)
    report = {
        "detector": config.DETECTOR,
        "fov": config.FOV,
        "load_ms": model.load_timings_ms,
        "load_total_ms": round((time.perf_counter() - t) * 1000),
        "images": {},
    }

    for name, img in imgs.items():
        t = time.perf_counter()
        try:
            res = model.reconstruct(img)
        except NoPersonError:
            report["images"][name] = {"people": 0}
            continue
        glb = res.mesh_path.rename(out / f"{name}.glb")
        (out / f"{name}_overlay.png").write_bytes(res.overlay_png)
        cv2.imwrite(str(out / f"{name}_glb_front.png"), cv2.cvtColor(render_glb_front(glb), cv2.COLOR_RGB2BGR))
        m = trimesh.load(glb, force="mesh")
        report["images"][name] = {
            "people": res.people,
            "size": [img.shape[1], img.shape[0]],
            "timings_ms": res.timings_ms,
            "total_ms": round((time.perf_counter() - t) * 1000),
            "glb_kb": round(glb.stat().st_size / 1024),
            "glb_bounds_center": np.round(m.bounds.mean(axis=0), 4).tolist(),
            "glb_extent_xyz": np.round(m.extents, 3).tolist(),
        }

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main()
