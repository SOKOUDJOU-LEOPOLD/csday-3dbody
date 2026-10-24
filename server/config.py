"""All runtime settings, read from env vars so the same code runs on any GPU box."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "third_party" / "sam-3d-body"
CHECKPOINTS = Path(os.environ.get("CHECKPOINTS", ROOT / "checkpoints"))

DETECTOR = os.environ.get("DETECTOR", "vitdet")  # "vitdet" or "sam3"
FOV = os.environ.get("FOV", "moge2")  # "moge2" or "" (use the model's default FOV)
PORT = int(os.environ.get("PORT", 8000))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", 5))
MESH_TTL_S = int(os.environ.get("MESH_TTL_S", 600))
MAX_SIDE = int(os.environ.get("MAX_SIDE", 1280))
MESH_DIR = Path(os.environ.get("MESH_DIR", ROOT / "meshes"))

SAM3DB_CKPT = CHECKPOINTS / "sam-3d-body-dinov3" / "model.ckpt"
MHR_PATH = CHECKPOINTS / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
VITDET_DIR = CHECKPOINTS / "vitdet"
MOGE_PATH = CHECKPOINTS / "moge-2-vitl-normal" / "model.pt"

# GPU nodes have no internet: everything fetched at load time is pre-cached here
os.environ.setdefault("HF_HUB_CACHE", str(CHECKPOINTS / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(CHECKPOINTS / "torch_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
