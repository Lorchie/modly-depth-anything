"""
Depth Anything V2 extension for Modly.

Reference : https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf
GitHub    : https://github.com/DepthAnything/Depth-Anything-V2

Single node: image -> mesh
Full pipeline:
  1. Depth estimation (transformers)
  2. Back-projection -> 3D point grid (numpy)
  3. Grid triangulation with depth-discontinuity filtering (numpy + trimesh)
  4. GLB export (trimesh)
"""

import io
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from services.generators.base import BaseGenerator, GenerationCancelled

_HF_REPOS = {
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthAnythingGenerator(BaseGenerator):
    MODEL_ID     = "modly-depth-anything"
    DISPLAY_NAME = "Depth Anything V2"
    VRAM_GB      = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._processor       = None
        self._current_variant = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        return (self.model_dir / "config.json").exists()

    def load(self) -> None:
        # Deferred to _ensure_model() because variant is a runtime param.
        pass

    def unload(self) -> None:
        self._processor       = None
        self._current_variant = None
        super().unload()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch
        import trimesh

        model_type = params.get("model_type", "vitl")
        use_cuda   = params.get("use_cuda", "auto")
        focal      = float(params.get("focal_length", 500.0))
        max_faces  = int(params.get("max_faces", 50000))

        # -- Step 1: load depth model --
        self._ensure_model(model_type, use_cuda)
        self._check_cancelled(cancel_event)

        # -- Step 2: depth estimation --
        self._report(progress_cb, 5, "Loading image...")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        self._report(progress_cb, 15, "Running depth estimation...")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        depth = outputs.predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)

        self._check_cancelled(cancel_event)

        # -- Step 3: grid triangulation --
        self._report(progress_cb, 55, "Building mesh from depth map...")
        vertices, faces = _depth_to_grid_mesh(depth_norm, focal, max_faces)

        self._check_cancelled(cancel_event)

        # -- Step 4: export GLB --
        self._report(progress_cb, 90, "Exporting GLB...")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name     = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        out_path = self.outputs_dir / name
        mesh.export(str(out_path))

        self._report(progress_cb, 100, "Done")
        return out_path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ensure_model(self, model_type: str, use_cuda: str = "auto") -> None:
        """Load (or reload) the Depth Anything model for the requested variant."""
        if self._model is not None and self._current_variant == model_type:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        hf_repo = _HF_REPOS.get(model_type, _HF_REPOS["vitl"])

        if use_cuda == "true":
            device = "cuda"
        elif use_cuda == "false":
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_type == "vitl" and self.is_downloaded():
            model_path = str(self.model_dir)
        else:
            model_path = hf_repo

        print(f"[DepthAnythingGenerator] Loading {model_type} from {model_path}...")
        self._processor = AutoImageProcessor.from_pretrained(model_path)
        self._model     = AutoModelForDepthEstimation.from_pretrained(model_path).to(device)
        self._model.eval()
        self._device          = device
        self._current_variant = model_type
        print(f"[DepthAnythingGenerator] {model_type} loaded on {device}.")


# ------------------------------------------------------------------ #
# Grid mesh from depth map (pure numpy, no open3d needed)
# ------------------------------------------------------------------ #

def _depth_to_grid_mesh(depth_norm: np.ndarray, focal: float, max_faces: int):
    """
    Back-projects a normalised depth map into a 3D grid mesh.

    Uses a strided grid so the face count stays at or below max_faces.
    Filters out triangles that straddle depth discontinuities or background.

    Returns (vertices, faces) as numpy arrays.
    """
    h, w   = depth_norm.shape
    cx, cy = w / 2.0, h / 2.0

    # Choose stride so the resulting grid has at most max_faces triangles.
    # A grid of H x W samples produces 2*(H-1)*(W-1) triangles.
    if max_faces > 0:
        stride = max(1, int(np.sqrt(h * w / (max_faces / 2.0 + 1))))
    else:
        stride = 1

    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    H  = len(ys)
    W  = len(xs)

    yy, xx = np.meshgrid(ys, xs, indexing="ij")   # H x W
    zz     = depth_norm[yy, xx]                    # H x W

    X = (xx - cx) * zz / focal
    Y = -(yy - cy) * zz / focal
    Z = zz

    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Build two triangles per quad, fully vectorised
    row_idx = np.arange(H - 1)
    col_idx = np.arange(W - 1)
    rows, cols = np.meshgrid(row_idx, col_idx, indexing="ij")
    rows = rows.ravel()
    cols = cols.ravel()

    i00 = rows * W + cols
    i01 = rows * W + cols + 1
    i10 = (rows + 1) * W + cols
    i11 = (rows + 1) * W + cols + 1

    z_flat = Z.ravel()
    z00, z01, z10, z11 = z_flat[i00], z_flat[i01], z_flat[i10], z_flat[i11]

    # Reject triangles with large depth jumps (depth edges / occlusions)
    thr = 0.05
    ok1 = (
        (np.abs(z00 - z01) < thr) & (np.abs(z00 - z10) < thr) & (np.abs(z01 - z10) < thr)
        & (z00 > 0.01) & (z01 > 0.01) & (z10 > 0.01)
    )
    ok2 = (
        (np.abs(z01 - z11) < thr) & (np.abs(z01 - z10) < thr) & (np.abs(z11 - z10) < thr)
        & (z01 > 0.01) & (z11 > 0.01) & (z10 > 0.01)
    )

    faces = np.concatenate([
        np.column_stack([i00, i01, i10])[ok1],
        np.column_stack([i01, i11, i10])[ok2],
    ])

    return vertices, faces
