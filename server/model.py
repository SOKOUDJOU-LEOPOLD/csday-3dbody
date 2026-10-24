"""SAM 3D Body estimator: load once, warm up, reconstruct(img) -> result.

Loading mirrors upstream demo.py (local checkpoint + MHR asset) rather than the model
card's setup_sam_3d_body(), which downloads from HF and can't run on offline GPU nodes.
"""

import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config

sys.path.insert(0, str(config.UPSTREAM))

import torch  # noqa: E402
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402
from sam_3d_body.visualization.renderer import Renderer  # noqa: E402

from .mesh import MESH_COLOR, to_glb  # noqa: E402

log = logging.getLogger("csday.model")


class NoPersonError(Exception):
    pass


@dataclass
class Result:
    people: int
    overlay_png: bytes
    mesh_path: Path
    timings_ms: dict


def decode_image(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB uint8, long side capped at config.MAX_SIDE."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("not an image")
    h, w = bgr.shape[:2]
    scale = config.MAX_SIDE / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _keypoint_box_area(person: dict) -> float:
    kp = person["pred_keypoints_2d"]
    w, h = kp.max(axis=0) - kp.min(axis=0)
    return float(w * h)


class BodyModel:
    def __init__(self):
        self.estimator: SAM3DBodyEstimator | None = None
        self.load_timings_ms: dict = {}

    def load(self):
        device = torch.device("cuda")
        t = time.perf_counter()

        def lap(name):
            nonlocal t
            now = time.perf_counter()
            self.load_timings_ms[name] = round((now - t) * 1000)
            log.info("load %s: %d ms", name, self.load_timings_ms[name])
            t = now

        model, model_cfg = load_sam_3d_body(
            str(config.SAM3DB_CKPT), device=device, mhr_path=str(config.MHR_PATH)
        )
        lap("sam_3d_body")

        from tools.build_detector import HumanDetector

        detector = HumanDetector(
            name=config.DETECTOR, device=device, path=str(config.VITDET_DIR)
        )
        lap(f"detector_{config.DETECTOR}")

        fov_estimator = None
        if config.FOV:
            from tools.build_fov_estimator import FOVEstimator

            fov_estimator = FOVEstimator(
                name=config.FOV, device=device, path=str(config.MOGE_PATH)
            )
            lap(f"fov_{config.FOV}")

        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=detector,
            human_segmentor=None,
            fov_estimator=fov_estimator,
        )

    def warmup(self, img_rgb: np.ndarray):
        """One full pass so the first student doesn't pay for CUDA/cuDNN init."""
        t = time.perf_counter()
        self.estimator.process_one_image(img_rgb)
        self.load_timings_ms["warmup"] = round((time.perf_counter() - t) * 1000)
        log.info("warmup: %d ms", self.load_timings_ms["warmup"])

    def reconstruct(self, img_rgb: np.ndarray, all_people: bool = False) -> Result:
        timings = {}
        t = time.perf_counter()

        def lap(name):
            nonlocal t
            now = time.perf_counter()
            timings[name] = round((now - t) * 1000)
            t = now

        outputs = self.estimator.process_one_image(img_rgb)
        lap("inference")
        if not outputs:
            raise NoPersonError()

        people = outputs if all_people else [max(outputs, key=_keypoint_box_area)]
        faces = self.estimator.faces
        n_verts = len(people[0]["pred_vertices"])
        vertices = np.concatenate([p["pred_vertices"] + p["pred_cam_t"] for p in people])
        all_faces = np.concatenate([faces + i * n_verts for i in range(len(people))])

        mesh_path = to_glb(vertices, all_faces, config.MESH_DIR / f"{uuid.uuid4().hex}.glb")
        lap("glb")

        overlay_png = self._overlay(img_rgb, people, vertices, all_faces)
        lap("overlay")

        return Result(len(outputs), overlay_png, mesh_path, timings)

    def _overlay(self, img_rgb, people, vertices, faces) -> bytes:
        # same front-view render as upstream visualize_sample_together, without the
        # keypoint and side-view panels it concatenates alongside
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cam_t = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        renderer = Renderer(focal_length=people[0]["focal_length"], faces=faces)
        rendered = renderer(
            vertices - cam_t,
            cam_t,
            img_bgr.copy(),
            mesh_base_color=MESH_COLOR[:3],
            scene_bg_color=(1, 1, 1),
        )
        ok, png = cv2.imencode(".png", (rendered * 255).astype(np.uint8))
        assert ok
        return png.tobytes()
