# Depth Anything V2 — Modly Extension

Converts any image into a textured 3D mesh (`.glb`) using
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) depth estimation
and orthographic back-projection.

**Extension ID:** `modly-depth-anything`  
**Version:** 2.2.0  
**Author:** Guillaume  
**VRAM:** ~4 GB (ViT-L)

---

## Pipeline

```
Image
  └─ 1. Auto-crop black borders (optional)
  └─ 2. CLAHE brightness boost (optional)
  └─ 3. Depth estimation  ──  Depth Anything V2 (ViT-S / ViT-B / ViT-L)
  └─ 4. Median filter  ──  removes outlier depth spikes (noise, leaves)
  └─ 5. Joint bilateral filter  ──  edge-preserving depth smoothing
  └─ 6. Back-projection  ──  depth map → vertices + faces
  └─ 7. GLB export via trimesh
```

---

## Model Variants

| Variant | HuggingFace Repo | Size | Speed |
|---------|-----------------|------|-------|
| ViT-S | `depth-anything/Depth-Anything-V2-Small-hf` | ~97 MB | Fastest |
| ViT-B | `depth-anything/Depth-Anything-V2-Base-hf` | ~390 MB | Balanced |
| ViT-L | `depth-anything/Depth-Anything-V2-Large-hf` | ~1.3 GB | Best quality |

ViT-L is downloaded locally when using Modly's built-in downloader.
The other variants are loaded directly from HuggingFace at runtime.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `vitl` | Model variant (vits / vitb / vitl) |
| `use_cuda` | `auto` | Device: auto-detect, force GPU, or force CPU |
| `depth_scale` | `2.5` | Foreground relief intensity (0.1 – 5.0) |
| `focal_length` | `470.0` | Horizontal/vertical mesh spread in px (470 = native) |
| `discontinuity_threshold` | `0.10` | Max depth jump before a face is removed (0.02 – 0.50) |
| `median_filter` | `true` | 5×5 median pass — removes depth spike noise |
| `smooth_depth` | `true` | Joint bilateral filter guided by RGB edges |
| `smooth_radius` | `4` | Bilateral spatial sigma (1 – 15) |
| `max_faces` | `300000` | Target face count; `-1` = no limit |
| `auto_crop` | `true` | Removes uniform black borders before estimation |
| `auto_brightness` | `true` | CLAHE equalisation for dark images |
| `save_depth_preview` | `false` | Saves a grayscale depth map `.png` alongside the GLB |

### Tuning tips

- **Vegetation / foliage:** raise `discontinuity_threshold` to 0.20 – 0.40
- **Clean indoor scenes:** keep `discontinuity_threshold` low (0.04 – 0.08)
- **More 3D pop:** increase `depth_scale` (2.5 → 4.0)
- **Faster export:** lower `max_faces` (100 000 – 150 000)

---

## Requirements

Dependencies are installed automatically by `setup.py` in an isolated venv.

| Package | Notes |
|---------|-------|
| `torch` + `torchvision` | CUDA build selected based on GPU architecture |
| `transformers >= 4.46.0` | HuggingFace model loading |
| `huggingface_hub` | Model download |
| `trimesh` | GLB export |
| `Pillow` | Image I/O |
| `numpy` | Depth map processing |
| `opencv-contrib-python` | Joint bilateral filter (`ximgproc`) — falls back to `opencv-python`, then numpy box blur |

### PyTorch selection matrix

| Python | GPU SM | CUDA | PyTorch |
|--------|--------|------|---------|
| ≥ 3.9 | SM ≥ 100 or CUDA ≥ 12.8 | 12.8 | 2.7.0 |
| ≥ 3.9 | SM ≥ 70 | 12.4 | 2.6.0 |
| ≥ 3.9 | SM < 70 (legacy) | 11.8 | 2.5.1 |
| 3.8 | any | 12.1 / 11.8 | 2.4.1 |

---

## Credits

| Resource | Link |
|----------|------|
| Depth Anything V2 paper & code | [github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| HuggingFace models | [huggingface.co/depth-anything](https://huggingface.co/depth-anything) |
| trimesh | [github.com/mikedh/trimesh](https://github.com/mikedh/trimesh) |
| OpenCV ximgproc | [docs.opencv.org](https://docs.opencv.org/4.x/df/d2d/group__ximgproc.html) |

**Depth Anything V2** — Yang, Lihe et al. (2024)  
*Depth Anything V2*, arXiv:2406.09414

```bibtex
@article{depth_anything_v2,
  title   = {Depth Anything V2},
  author  = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen
             and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal = {arXiv:2406.09414},
  year    = {2024}
}
```

---

## License

This extension is distributed as part of the Modly ecosystem.  
Depth Anything V2 model weights are released under the
[Apache 2.0 License](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE).
