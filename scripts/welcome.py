#!/usr/bin/env python3
import os
import importlib.metadata
import importlib.util

def get_version(package_name):
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = "Package not installed"
    return version

# -----------------------------------------------------------------------------

slideflow_version = get_version("slideflow")
pytorch_version = get_version("torch")
tensorflow_version = get_version("tensorflow")
has_cucim = importlib.util.find_spec('cucim')
has_vips = importlib.util.find_spec('pyvips')
has_nc = importlib.util.find_spec('slideflow_noncommercial')
has_gpl = importlib.util.find_spec('slideflow_gpl')

# -----------------------------------------------------------------------------

def print_welcome():
    # Determine tensor backend
    tensor_backend = os.getenv("SF_BACKEND", "torch")
    if tensor_backend == "torch":
        tensor_backend = f"PyTorch {pytorch_version}"
    elif tensor_backend == "tensorflow":
        tensor_backend = f"TensorFlow {tensorflow_version}"

    # Determine slide backend
    slide_backend = os.getenv("SF_SLIDE_BACKEND", "cucim")
    vips_version = os.getenv("SF_VIPS_VERSION", "N/A")
    slide_backends = []
    if has_cucim and slide_backend == "cucim":
        slide_backends.append(f"CuCIM (default)")
    elif has_cucim:
        slide_backends.append("CuCIM")
    if has_vips and slide_backend == "libvips":
        slide_backends.append(f"Libvips {vips_version} (default)")
    elif has_vips:
        slide_backends.append(f"Libvips {vips_version}")
    slide_backends = ", ".join(slide_backends)

    # Determine Slideflow add-ons
    if has_nc and has_gpl:
        additional_modules = "Non-commercial & GPL-3.0 add-ons"
    elif has_nc:
        additional_modules = "Non-commercial add-on"
    elif has_gpl:
        additional_modules = "GPL-3.0 add-on"
    else:
        additional_modules = "None"

    # Use ANSI escape codes for colors
    BOLD = '\033[1m'
    END = '\033[0m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'

    # Print the welcome message with formatting
    print(f"{BOLD}Slideflow {slideflow_version}{END}")
    print(f"{GREEN}Tensor backend:{END} {tensor_backend}")
    print(f"{GREEN}Slide backends:{END} {slide_backends}")
    print(f"{BLUE}Additional modules:{END} {additional_modules}")

if __name__ == "__main__":
    print_welcome()
