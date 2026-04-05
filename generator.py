"""
Depth Anything V2 extension for Modly.

Reference : https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf
GitHub    : https://github.com/DepthAnything/Depth-Anything-V2

Single node: image -> mesh
Full pipeline: depth estimation -> point cloud back-projection -> Poisson mesh.

Dependencies are installed by setup.py into venv/ at install time.
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

        model_type    = params.get("model_type", "vitl")
        use_cuda      = params.get("use_cuda", "auto")
        focal         = float(params.get("focal_length", 500.0))
        poisson_depth = int(params.get("poisson_depth", 8))
        max_faces     = int(params.get("max_faces", 50000))

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

        # -- Step 3: back-projection to point cloud --
        self._report(progress_cb, 50, "Building point cloud...")

        try:
            import open3d as o3d
        except ImportError:
            raise RuntimeError(
                "open3d is not installed. "
                "Click Repair on the Models page to re-run setup."
            )

        h, w   = depth_norm.shape
        cx, cy = w / 2.0, h / 2.0

        y_grid, x_grid = np.mgrid[0:h, 0:w]
        z = depth_norm
        X = (x_grid - cx) * z / focal
        Y = -(y_grid - cy) * z / focal

        mask   = z.ravel() > 0.01
        points = np.column_stack([X.ravel(), Y.ravel(), z.ravel()])[mask]

        pc        = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)

        self._check_cancelled(cancel_event)

        # -- Step 4: Poisson surface reconstruction --
        self._report(progress_cb, 65, "Reconstructing mesh (Poisson)...")
        pc.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pc, depth=poisson_depth
        )

        if max_faces > 0 and len(mesh.triangles) > max_faces:
            self._report(progress_cb, 85, "Simplifying mesh...")
            mesh = mesh.simplify_quadric_decimation(max_faces)

        self._check_cancelled(cancel_event)

        # -- Step 5: export GLB --
        self._report(progress_cb, 95, "Exporting GLB...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name     = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        out_path = self.outputs_dir / name
        o3d.io.write_triangle_mesh(str(out_path), mesh)

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

        # For vitl (default): load from Modly's managed model_dir when available.
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
