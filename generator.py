"""
Depth Anything V2 extension for Modly.

Reference : https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf
GitHub    : https://github.com/DepthAnything/Depth-Anything-V2

Pipeline:
  1. Depth estimation (transformers)
  2. Depth smoothing (numpy box blur)
  3. Back-projection -> textured 3D mesh (numpy + trimesh)
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

        model_type  = params.get("model_type", "vitl")
        use_cuda    = params.get("use_cuda", "auto")
        focal       = float(params.get("focal_length", 500.0))
        depth_scale = float(params.get("depth_scale", 1.5))
        max_faces   = int(params.get("max_faces", 100000))
        smooth      = str(params.get("smooth_depth", "true")).lower() == "true"

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

        # -- Step 3: depth smoothing (reduces noise, smoother mesh) --
        if smooth:
            self._report(progress_cb, 40, "Smoothing depth map...")
            depth_norm = _smooth_depth(depth_norm, radius=2)

        # -- Step 4: build mesh with vertex colours --
        self._report(progress_cb, 55, "Building mesh...")
        vertices, faces, vertex_colors = _depth_to_mesh(
            image, depth_norm, focal, depth_scale, max_faces
        )

        self._check_cancelled(cancel_event)

        # -- Step 5: export GLB --
        self._report(progress_cb, 90, "Exporting GLB...")
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False,
        )
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
# Depth smoothing (pure numpy, no scipy required)
# ------------------------------------------------------------------ #

def _smooth_depth(depth: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Box blur via sliding window average.
    Falls back silently if numpy version is too old.
    """
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        pad  = np.pad(depth, radius, mode="edge")
        size = 2 * radius + 1
        windows = sliding_window_view(pad, (size, size))
        return windows.mean(axis=(-1, -2)).astype(np.float32)
    except Exception:
        return depth.astype(np.float32)


# ------------------------------------------------------------------ #
# Mesh from depth map
# ------------------------------------------------------------------ #

def _depth_to_mesh(
    image: "Image.Image",
    depth_norm: np.ndarray,
    focal: float,
    depth_scale: float,
    max_faces: int,
):
    """
    Converts a normalised depth map into a textured 3D mesh.

    Coordinate system (Y-up, right-handed, viewer faces -Z):
      X  = image horizontal (left -> right)
      Y  = image vertical   (bottom -> top, row 0 = top)
      Z  = depth relief     (background = 0, foreground = depth_scale)

    Depth Anything V2 convention: higher predicted_depth = farther.
    We invert so near objects protrude toward the viewer (+Z).

    Returns (vertices, faces, vertex_colors_rgba).
    """
    h, w   = depth_norm.shape
    cx, cy = w / 2.0, h / 2.0

    # Stride so face count stays at or below max_faces
    if max_faces > 0:
        stride = max(1, int(np.sqrt(h * w / (max_faces / 2.0 + 1))))
    else:
        stride = 1

    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    H  = len(ys)
    W  = len(xs)

    yy, xx = np.meshgrid(ys, xs, indexing="ij")   # H x W
    zz     = depth_norm[yy, xx]                    # H x W, [0, 1]

    # Invert: near objects (low depth) -> high Z (pop toward viewer)
    z_relief = (1.0 - zz) * depth_scale

    # Orthographic projection in the image plane.
    # Dividing by focal gives image-coordinate units independent of resolution.
    X = (xx - cx) / focal
    Y = -(yy - cy) / focal    # flip Y: row 0 (top) -> positive Y
    Z = z_relief

    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Sample vertex colours from the original image
    img_arr = np.array(image)                       # h x w x 3, uint8
    colors  = img_arr[yy, xx]                       # H x W x 3
    rgba    = np.ones((H * W, 4), dtype=np.uint8) * 255
    rgba[:, :3] = colors.reshape(-1, 3)

    # Build two triangles per quad (fully vectorised)
    row_idx = np.arange(H - 1)
    col_idx = np.arange(W - 1)
    rows, cols = np.meshgrid(row_idx, col_idx, indexing="ij")
    rows = rows.ravel()
    cols = cols.ravel()

    i00 = rows * W + cols
    i01 = rows * W + cols + 1
    i10 = (rows + 1) * W + cols
    i11 = (rows + 1) * W + cols + 1

    # Use original (pre-inversion) depth for discontinuity detection
    z_flat = zz.ravel()
    z00, z01, z10, z11 = z_flat[i00], z_flat[i01], z_flat[i10], z_flat[i11]

    # Adaptive threshold: 90th percentile of depth gradients, clamped to [0.04, 0.20]
    grad_h = np.abs(np.diff(depth_norm, axis=1))
    grad_v = np.abs(np.diff(depth_norm, axis=0))
    thr    = float(np.percentile(
        np.concatenate([grad_h.ravel(), grad_v.ravel()]), 90
    )) * 2.0
    thr = float(np.clip(thr, 0.04, 0.20))

    ok1 = (
        (np.abs(z00 - z01) < thr) &
        (np.abs(z00 - z10) < thr) &
        (np.abs(z01 - z10) < thr)
    )
    ok2 = (
        (np.abs(z01 - z11) < thr) &
        (np.abs(z01 - z10) < thr) &
        (np.abs(z11 - z10) < thr)
    )

    faces = np.concatenate([
        np.column_stack([i00, i01, i10])[ok1],
        np.column_stack([i01, i11, i10])[ok2],
    ])

    return vertices, faces, rgba
