#!/usr/bin/env python3
import subprocess
import sys
import platform
import argparse


def run(cmd):
    subprocess.run(cmd, check=False)


def detect_gpu():
    system = platform.system()

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            if "CUDA Version: 12" in result.stdout:
                return "cu121"
            elif "CUDA Version: 11" in result.stdout:
                return "cu118"
            return "cu121"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if system == "Darwin":
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if "Apple" in result.stdout:
                return "mps"
        except FileNotFoundError:
            pass

    if system == "Linux":
        try:
            result = subprocess.run(["rocm-smi"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return "rocm"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return "cpu"


def get_torch_cmd(backend):
    pkgs = ["torch", "torchvision", "torchaudio"]

    if backend.startswith("cu"):
        return ["uv", "pip", "install", *pkgs, "--index-url", f"https://download.pytorch.org/whl/{backend}"]
    elif backend == "rocm":
        return ["uv", "pip", "install", *pkgs, "--index-url", "https://download.pytorch.org/whl/rocm5.6"]
    elif backend == "mps":
        return ["uv", "pip", "install", *pkgs]
    else:
        return ["uv", "pip", "install", *pkgs, "--index-url", "https://download.pytorch.org/whl/cpu"]


def verify_installation(backend):
    code = '''import torch
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("MPS available")
else:
    print("CPU only")
    '''
    subprocess.run([sys.executable, "-c", code])


def main():
    parser = argparse.ArgumentParser(description="Setup GPU dependencies")
    parser.add_argument("--cuda", type=str, choices=["11.8", "12.1", "12.4", "12.8", "13.0"], help="Force CUDA version")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    args = parser.parse_args()

    if args.cpu:
        backend = "cpu"
    elif args.cuda:
        backend = f"cu{args.cuda.replace('.', '')}"
    else:
        backend = detect_gpu()

    print(f"Backend: {backend}")

    run(get_torch_cmd(backend))
    run(["uv", "pip", "install", "ultralytics", "--no-deps"])
    run(["uv", "pip", "install", "opencv-python"])
    run(["uv", "pip", "install", "-e", ".", "--no-deps"])

    if not args.skip_verify:
        verify_installation(backend)


if __name__ == "__main__":
    main()
