from setuptools import setup, find_packages
import os
import shutil
import subprocess
import sys

def get_cuda_version():
    """Get CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Try to extract CUDA version from nvidia-smi output
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    return line.split('CUDA Version: ')[1].split()[0]
        return None
    except:
        return None

def get_cupy_package():
    """Get appropriate CuPy package based on CUDA version."""
    cuda_version = get_cuda_version()
    if not cuda_version:
        return None
    
    cuda_major = cuda_version.split('.')[0]
    if cuda_major == '12':
        return 'cupy-cuda12x>=12.0.0'
    elif cuda_major == '11':
        return 'cupy-cuda11x>=12.0.0'
    elif cuda_major == '10':
        return 'cupy-cuda10x>=12.0.0'
    return None

# Copy packages to root level for easier installation
if os.path.exists('src/VesselTracer') and not os.path.exists('VesselTracer'):
    shutil.copytree('src/VesselTracer', 'VesselTracer')

if os.path.exists('src/DataManager') and not os.path.exists('DataManager'):
    shutil.copytree('src/DataManager', 'DataManager')

# Get GPU dependencies
gpu_deps = []
cupy_package = get_cupy_package()
if cupy_package:
    gpu_deps.append(cupy_package)
    print(f"Detected CUDA version, adding GPU support with {cupy_package}")
else:
    print("No CUDA version detected. GPU support will not be available.")

setup(
    name="VascularAnalysis",
    version="0.1.0",
    packages=['VesselTracer', 'DataManager'],
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "tifffile>=2021.7.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "czifile>=2019.7.2",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "xmltodict",
        "skan",
    ],
    extras_require={
        'gpu': gpu_deps,
        'all': gpu_deps,  # Include GPU dependencies in 'all' extras
    },
    author="Matt",
    description="Tools for vascular analysis including DataManager and VesselTracer",
    python_requires=">=3.8",
    zip_safe=False,
    include_package_data=True,
)