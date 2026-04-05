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


def pip_try(venv: Path, *args: str) -> bool:
    """Same as pip() but returns False instead of raising on failure."""
    is_win  = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    result  = subprocess.run([str(pip_exe), *args], capture_output=True, text=True)
    if result.returncode != 0:
        last_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        print(f"[Depth Anything V2 setup]   pip error: {last_line}")
    return result.returncode == 0


def venv_python_version(venv: Path) -> tuple:
    """Returns (major, minor) of the venv's Python interpreter."""
    is_win = platform.system() == "Windows"
    exe    = venv / ("Scripts/python.exe" if is_win else "bin/python")
    out    = subprocess.check_output(
        [str(exe), "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
        text=True,
    ).strip()
    major, minor = out.split()
    return int(major), int(minor)


def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv = ext_dir / "venv"

    print(f"[Depth Anything V2 setup] Creating venv at {venv} ...")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    py_ver = venv_python_version(venv)
    print(f"[Depth Anything V2 setup] Venv Python: {py_ver[0]}.{py_ver[1]}")

    # PyTorch -- choose version based on GPU architecture and Python version
    # PyTorch 2.7 requires Python >= 3.9; fall back to 2.4 for Python 3.8
    if py_ver <= (3, 8):
        # Python 3.8 -- last supported PyTorch is 2.4.x
        if gpu_sm >= 100 or cuda_version >= 128:
            torch_pkgs  = ["torch==2.4.1", "torchvision==0.19.1"]
            torch_index = "https://download.pytorch.org/whl/cu121"
            print(f"[Depth Anything V2 setup] Python 3.8 + Blackwell -> PyTorch 2.4 + CUDA 12.1 (best available)")
        elif gpu_sm == 0 or gpu_sm >= 70:
            torch_pkgs  = ["torch==2.4.1", "torchvision==0.19.1"]
            torch_index = "https://download.pytorch.org/whl/cu121"
            print(f"[Depth Anything V2 setup] Python 3.8, GPU SM {gpu_sm} -> PyTorch 2.4 + CUDA 12.1")
        else:
            torch_pkgs  = ["torch==2.4.1", "torchvision==0.19.1"]
            torch_index = "https://download.pytorch.org/whl/cu118"
            print(f"[Depth Anything V2 setup] Python 3.8, GPU SM {gpu_sm} (legacy) -> PyTorch 2.4 + CUDA 11.8")
    elif gpu_sm >= 100 or cuda_version >= 128:
        torch_pkgs  = ["torch==2.7.0", "torchvision==0.22.0"]
        torch_index = "https://download.pytorch.org/whl/cu128"
        print(f"[Depth Anything V2 setup] GPU SM {gpu_sm}, CUDA {cuda_version} -> PyTorch 2.7 + CUDA 12.8")
    elif gpu_sm == 0 or gpu_sm >= 70:
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        torch_index = "https://download.pytorch.org/whl/cu124"
        print(f"[Depth Anything V2 setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
    else:
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        torch_index = "https://download.pytorch.org/whl/cu118"
        print(f"[Depth Anything V2 setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")

    print("[Depth Anything V2 setup] Installing PyTorch ...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # Core dependencies -- pin versions compatible with Python 3.8 when needed
    numpy_pkg        = "numpy<2.0"        if py_ver <= (3, 8) else "numpy"
    transformers_pkg = "transformers>=4.40.0,<4.50.0" if py_ver <= (3, 8) else "transformers>=4.46.0"

    print("[Depth Anything V2 setup] Installing core dependencies ...")
    pip(venv, "install",
        "Pillow",
        numpy_pkg,
        transformers_pkg,
        "huggingface_hub",
    )

    print("[Depth Anything V2 setup] Installing trimesh ...")
    pip(venv, "install", "trimesh")



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
