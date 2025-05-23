from setuptools import setup, find_namespace_packages

setup(
    name="VascularAnalysis",
    version="0.1.0",
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src'),
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
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "vessel-tracer=VesselTracer.cli:main",
            "data-manager=DataManager.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)