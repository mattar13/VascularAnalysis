from setuptools import setup

setup(
    name="VascularAnalysis",
    version="0.1.0",
    packages=['VesselTracer', 'DataManager'],
    package_dir={
        'VesselTracer': 'src/VesselTracer',
        'DataManager': 'src/DataManager'
    },
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
    ],
    author="Matt",
    description="Tools for vascular analysis including DataManager and VesselTracer",
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "vessel-tracer=VesselTracer.cli:main",
            "data-manager=DataManager.cli:main",
        ],
    },
)