"""
Depth Anything V2 extension setup -- run automatically by Modly at install time.

Creates an isolated venv and installs all required dependencies.
Called by Modly at extension install time with:

    python setup.py <json_args>

where json_args contains:
    python_exe    -- path to Modly's embedded Python (used to create the venv)
    ext_dir       -- absolute path to this extension directory
    gpu_sm        -- GPU compute capability as integer (e.g. 86 for Ampere, 0 = no GPU)
    cuda_version  -- CUDA driver version as integer (e.g. 124 = 12.4, optional)

Example (manual test):
    python setup.py '{"python_exe":"C:/path/python.exe","ext_dir":"C:/path/modly-depth-anything","gpu_sm":86}'
"""

import json
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv: Path, *args: str) -> None:
    is_win  = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv = ext_dir / "venv"

    print(f"[Depth Anything V2 setup] Creating venv at {venv} ...")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # PyTorch -- choose version based on GPU architecture
    if gpu_sm >= 100 or cuda_version >= 128:
        # Blackwell (RTX 50xx) -- requires cu128 + torch 2.7+
        torch_pkgs  = ["torch==2.7.0", "torchvision==0.22.0"]
        torch_index = "https://download.pytorch.org/whl/cu128"
        print(f"[Depth Anything V2 setup] GPU SM {gpu_sm}, CUDA {cuda_version} -> PyTorch 2.7 + CUDA 12.8")
    elif gpu_sm == 0 or gpu_sm >= 70:
        # Ampere / Ada Lovelace / Hopper -- cu124
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        torch_index = "https://download.pytorch.org/whl/cu124"
        print(f"[Depth Anything V2 setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
    else:
        # Pascal / Volta / Turing (sm_60-sm_75) -- cu118
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        torch_index = "https://download.pytorch.org/whl/cu118"
        print(f"[Depth Anything V2 setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")

    print("[Depth Anything V2 setup] Installing PyTorch ...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # Core dependencies
    print("[Depth Anything V2 setup] Installing core dependencies ...")
    pip(venv, "install",
        "Pillow",
        "numpy",
        "scipy",
        "open3d",
        "transformers>=4.46.0",
        "huggingface_hub",
    )

    print("[Depth Anything V2 setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    # Accepts either JSON (from Modly) or positional args (for manual testing)
    # Positional: python setup.py <python_exe> <ext_dir> <gpu_sm>
    # JSON:       python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86}'
    if len(sys.argv) >= 4:
        setup(
            python_exe   = sys.argv[1],
            ext_dir      = Path(sys.argv[2]),
            gpu_sm       = int(sys.argv[3]),
            cuda_version = int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe   = args["python_exe"],
            ext_dir      = Path(args["ext_dir"]),
            gpu_sm       = int(args.get("gpu_sm", 86)),
            cuda_version = int(args.get("cuda_version", 0)),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86}\'')
        sys.exit(1)
