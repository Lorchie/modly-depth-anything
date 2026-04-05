"""
Depth Anything V2 extension for Modly.

Reference : https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf
GitHub    : https://github.com/DepthAnything/Depth-Anything-V2

Three nodes:
    image_to_depth      -- RGB image  -> normalised depth map (PNG)
    depth_to_pointcloud -- depth map  -> point cloud (PLY)
    pointcloud_to_mesh  -- point cloud -> triangulated mesh (GLB)

Dependencies are installed by setup.py into venv/ at install time.
"""

import io
import tempfile
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from services.generators.base import BaseGenerator, GenerationCancelled

_EXTENSION_DIR = Path(__file__).parent

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
        """True when the default (vitl) model has been downloaded by Modly."""
        return (self.model_dir / "config.json").exists()

    def load(self) -> None:
        # Actual model loading is deferred to _ensure_model() inside generate()
        # because the variant (vits / vitb / vitl) is a runtime param.
        pass

    def unload(self) -> None:
        self._processor       = None
        self._current_variant = None
        super().unload()

    # ------------------------------------------------------------------ #
    # Inference -- entry point
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        # Modly sets model_dir to MODELS_DIR/<ext_id>/<node_id> for each node.
        # runner.py injects MODEL_DIR env var so model_dir.name == node_id.
        # params.get("nodeId") is a fallback for standalone / future use.
        _known  = {"image_to_depth", "depth_to_pointcloud", "pointcloud_to_mesh"}
        node_id = self.model_dir.name
        if node_id not in _known:
            node_id = params.get("nodeId", "image_to_depth")

        if node_id == "image_to_depth":
            return self._node_image_to_depth(image_bytes, params, progress_cb, cancel_event)
        if node_id == "depth_to_pointcloud":
            return self._node_depth_to_pointcloud(image_bytes, params, progress_cb, cancel_event)
        if node_id == "pointcloud_to_mesh":
            return self._node_pointcloud_to_mesh(image_bytes, params, progress_cb, cancel_event)
        raise ValueError(f"[DepthAnythingGenerator] Unknown node: {node_id}")

    # ------------------------------------------------------------------ #
    # Node 1 -- Image -> Depth Map
    # ------------------------------------------------------------------ #

    def _node_image_to_depth(self, image_bytes, params, progress_cb, cancel_event):
        import torch

        model_type = params.get("model_type", "vitl")
        use_cuda   = params.get("use_cuda", "auto")
        self._ensure_model(model_type, use_cuda)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 10, "Loading image...")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        self._report(progress_cb, 30, "Running depth estimation...")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 75, "Saving depth map...")
        depth = outputs.predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_depth.png"
        out_path = self.outputs_dir / name
        Image.fromarray(depth_norm).save(str(out_path))

        self._report(progress_cb, 100, "Done")
        return out_path

    # ------------------------------------------------------------------ #
    # Node 2 -- Depth Map -> Point Cloud
    # ------------------------------------------------------------------ #

    def _node_depth_to_pointcloud(self, image_bytes, params, progress_cb, cancel_event):
        import open3d as o3d

        self._report(progress_cb, 10, "Loading depth map...")
        depth_img = Image.open(io.BytesIO(image_bytes)).convert("L")
        depth = np.array(depth_img, dtype=np.float32)
        h, w  = depth.shape

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 30, "Back-projecting to 3D...")

        focal = float(params.get("focal_length", 500.0))
        cx, cy = w / 2.0, h / 2.0

        # Vectorised back-projection -- avoids per-pixel Python loop
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        z = depth / 255.0
        X = (x_grid - cx) * z / focal
        Y = -(y_grid - cy) * z / focal

        # Discard near-zero depth (background / noise)
        mask   = z.ravel() > 0.01
        points = np.column_stack([X.ravel(), Y.ravel(), z.ravel()])[mask]

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 70, "Building point cloud...")

        pc        = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name     = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_pointcloud.ply"
        out_path = self.outputs_dir / name
        o3d.io.write_point_cloud(str(out_path), pc)

        self._report(progress_cb, 100, "Done")
        return out_path

    # ------------------------------------------------------------------ #
    # Node 3 -- Point Cloud -> Mesh
    # ------------------------------------------------------------------ #

    def _node_pointcloud_to_mesh(self, mesh_bytes, params, progress_cb, cancel_event):
        import open3d as o3d

        self._report(progress_cb, 5, "Loading point cloud...")
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp.write(mesh_bytes)
            tmp_path = Path(tmp.name)

        try:
            pc = o3d.io.read_point_cloud(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 25, "Estimating normals...")
        pc.estimate_normals()

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 45, "Reconstructing mesh (Poisson)...")
        poisson_depth = int(params.get("poisson_depth", 8))
        mesh, _       = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pc, depth=poisson_depth
        )

        max_faces = int(params.get("max_faces", 50000))
        if max_faces > 0 and len(mesh.triangles) > max_faces:
            self._report(progress_cb, 80, "Simplifying mesh...")
            mesh = mesh.simplify_quadric_decimation(max_faces)

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 92, "Exporting GLB...")
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

        # For vitl (default): use Modly's pre-downloaded files in model_dir if present,
        # otherwise fall back to HuggingFace Hub auto-download + caching.
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
