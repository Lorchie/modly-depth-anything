import subprocess, sys, os, json

def try_install(pip, package):
    try:
        subprocess.check_call([pip, "install", package])
        return True
    except:
        return False

def main(args):
    python_exe = args["python_exe"]
    ext_dir = args["ext_dir"]
    venv_dir = os.path.join(ext_dir, "venv")

    subprocess.check_call([python_exe, "-m", "venv", venv_dir])

    pip = os.path.join(venv_dir, "Scripts", "pip.exe") if sys.platform == "win32" \
        else os.path.join(venv_dir, "bin", "pip")

    # Try CUDA first
    if not try_install(pip, "torch --index-url https://download.pytorch.org/whl/cu121"):
        # Fallback CPU
        subprocess.check_call([pip, "install", "torch"])

    subprocess.check_call([pip, "install", "depth-anything", "opencv-python", "numpy", "open3d"])

if __name__ == "__main__":
    main(json.loads(sys.argv[1]))