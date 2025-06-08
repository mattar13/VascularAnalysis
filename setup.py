from setuptools import setup, find_packages
import os
import shutil

# Copy packages to root level for easier installation
if os.path.exists('src/VesselTracer') and not os.path.exists('VesselTracer'):
    shutil.copytree('src/VesselTracer', 'VesselTracer')

if os.path.exists('src/DataManager') and not os.path.exists('DataManager'):
    shutil.copytree('src/DataManager', 'DataManager')

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
        'gpu': [
            'cupy-cuda11x>=12.0.0',  # Replace 11x with your CUDA version
        ],
    },
    author="Matt",
    description="Tools for vascular analysis including DataManager and VesselTracer",
    python_requires=">=3.8",
    zip_safe=False,
    include_package_data=True,
)