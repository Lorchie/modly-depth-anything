"""
Depth Anything V2 extension for Modly.

Reference : https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf
GitHub    : https://github.com/DepthAnything/Depth-Anything-V2

Pipeline:
  1. Depth estimation (transformers)
  2. Image pre-processing: auto-crop black borders, CLAHE brightness boost
  3. Median filter pass  -- removes depth outlier spikes (leaves, noise)
  4. Joint bilateral filter -- smooths depth guided by RGB edges (edge-preserving)
  5. Back-projection -> textured 3D mesh (numpy + trimesh)
  6. GLB export (trimesh) + optional depth map preview PNG
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

        model_type         = params.get("model_type", "vitl")
        use_cuda           = params.get("use_cuda", "auto")
        focal              = float(params.get("focal_length", 470.0))
        depth_scale        = float(params.get("depth_scale", 2.5))
        max_faces          = int(params.get("max_faces", 300000))
        smooth             = str(params.get("smooth_depth", "true")).lower() == "true"
        smooth_radius      = int(params.get("smooth_radius", 4))
        use_median         = str(params.get("median_filter", "true")).lower() == "true"
        auto_crop          = str(params.get("auto_crop", "true")).lower() == "true"
        auto_brightness    = str(params.get("auto_brightness", "true")).lower() == "true"
        disc_thr           = float(params.get("discontinuity_threshold", 0.10))
        save_depth_preview = str(params.get("save_depth_preview", "false")).lower() == "true"

        # -- Step 1: load depth model --
        self._ensure_model(model_type, use_cuda)
        self._check_cancelled(cancel_event)

        # -- Step 2: image pre-processing --
        self._report(progress_cb, 5, "Loading image...")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if auto_crop:
            image = _autocrop_black_borders(image)
        if auto_brightness:
            image = _auto_brightness(image)

        # -- Step 3: depth estimation --
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

        # -- Step 4: depth filtering (median + joint bilateral) --
        if use_median or smooth:
            self._report(progress_cb, 40, "Filtering depth map...")
            depth_norm = _filter_depth(
                depth_norm,
                image,
                sigma_space=smooth_radius,
                use_median=use_median,
                use_bilateral=smooth,
            )

        # -- Step 5: optional depth preview PNG --
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

        if save_depth_preview:
            preview     = (depth_norm * 255).astype(np.uint8)
            preview_img = Image.fromarray(preview, mode="L")
            preview_path = self.outputs_dir / f"{stem}.depth.png"
            preview_img.save(str(preview_path))
            print(f"[DepthAnythingGenerator] Depth preview saved: {preview_path}")

        # -- Step 6: build textured mesh --
        self._report(progress_cb, 55, "Building mesh...")
        vertices, faces, vertex_colors = _depth_to_mesh(
            image, depth_norm, focal, depth_scale, max_faces, disc_thr
        )

        self._check_cancelled(cancel_event)

        # -- Step 7: export GLB --
        self._report(progress_cb, 90, "Exporting GLB...")
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False,
        )
        out_path = self.outputs_dir / f"{stem}.glb"
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
# Image pre-processing
# ------------------------------------------------------------------ #

def _autocrop_black_borders(image: "Image.Image", threshold: int = 15) -> "Image.Image":
    """
    Detects and removes uniform black borders around the image.
    Scans each side inward until a row/column whose mean exceeds threshold is found.
    Returns the original image unchanged if no border is detected.
    """
    arr  = np.array(image)
    gray = arr.mean(axis=2)
    h, w = gray.shape
    top, bottom, left, right = 0, h, 0, w

    for i in range(h):
        if gray[i, :].mean() > threshold:
            top = i
            break

    for i in range(h - 1, -1, -1):
        if gray[i, :].mean() > threshold:
            bottom = i + 1
            break

    for j in range(w):
        if gray[:, j].mean() > threshold:
            left = j
            break

    for j in range(w - 1, -1, -1):
        if gray[:, j].mean() > threshold:
            right = j + 1
            break

    if top >= bottom or left >= right:
        return image

    return image.crop((left, top, right, bottom))


def _auto_brightness(image: "Image.Image") -> "Image.Image":
    """
    Boosts brightness for dark images (mean pixel value below 80).
    Uses CLAHE in LAB space for smart local contrast enhancement that
    preserves colour balance. Falls back to linear PIL boost if cv2 is
    unavailable.
    """
    arr  = np.array(image)
    mean = float(arr.mean())
    if mean >= 80.0:
        return image

    try:
        import cv2
        # CLAHE in LAB space: only the L (lightness) channel is equalised,
        # so hue and saturation are left intact.
        lab     = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq    = clahe.apply(l)
        lab_eq  = cv2.merge([l_eq, a, b])
        rgb     = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)
    except Exception:
        # Fallback: linear brightness scale toward mean=128
        from PIL import ImageEnhance
        factor = float(np.clip(128.0 / max(mean, 1.0), 1.0, 4.0))
        return ImageEnhance.Brightness(image).enhance(factor)


# ------------------------------------------------------------------ #
# Depth map filtering
# ------------------------------------------------------------------ #

def _filter_depth(
    depth: np.ndarray,
    guide_image: "Image.Image",
    sigma_space: int = 4,
    use_median: bool = True,
    use_bilateral: bool = True,
) -> np.ndarray:
    """
    Two-pass depth filter:

    Pass 1 -- Median filter (kernel 5):
        Removes outlier depth spikes (floating leaves, sensor noise) without
        blurring edges. Run before bilateral so the bilateral has clean input.

    Pass 2 -- Joint bilateral filter:
        Smooths depth only where RGB colours are similar, preserving sharp
        object boundaries. Uses cv2.ximgproc.jointBilateralFilter (contrib)
        when available, falls back to cv2.bilateralFilter, then to a numpy
        box blur if cv2 is not installed at all.

    sigma_space  -- spatial sigma passed directly to the bilateral filter.
    sigmaColor   -- fixed at 0.1 (depth is in [0,1]), tuned for edge preservation.
    """
    result = depth.astype(np.float32)

    try:
        import cv2

        if use_median:
            # medianBlur requires uint8; round-trip through [0,255]
            d8     = (result * 255.0).clip(0, 255).astype(np.uint8)
            d8     = cv2.medianBlur(d8, 5)
            result = d8.astype(np.float32) / 255.0

        if use_bilateral:
            guide = np.array(guide_image, dtype=np.uint8)
            dh, dw = result.shape
            if guide.shape[:2] != (dh, dw):
                guide = cv2.resize(guide, (dw, dh), interpolation=cv2.INTER_LINEAR)

            try:
                # Joint bilateral: RGB guide keeps depth edges aligned to colour edges
                result = cv2.ximgproc.jointBilateralFilter(
                    joint=guide,
                    src=result,
                    d=-1,
                    sigmaColor=0.1,
                    sigmaSpace=float(sigma_space),
                )
                print("[DepthAnythingGenerator] Joint bilateral filter applied.")
            except (AttributeError, cv2.error):
                # opencv-contrib not available -- standard bilateral filter
                result = cv2.bilateralFilter(
                    result,
                    d=-1,
                    sigmaColor=0.1,
                    sigmaSpace=float(sigma_space),
                )
                print("[DepthAnythingGenerator] Bilateral filter applied (ximgproc unavailable).")

    except ImportError:
        # cv2 not installed -- numpy box blur as last resort
        result = _numpy_box_blur(result, radius=sigma_space)

    return result.astype(np.float32)


def _numpy_box_blur(depth: np.ndarray, radius: int) -> np.ndarray:
    """Box blur fallback used when cv2 is not available."""
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        pad     = np.pad(depth, radius, mode="edge")
        size    = 2 * radius + 1
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
    discontinuity_threshold: float,
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

    thr = float(discontinuity_threshold)

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
