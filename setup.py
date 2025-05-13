from setuptools import setup, find_namespace_packages

setup(
    name="VascularAnalysis",
    version="0.1.0",
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src'),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "tifffile",
    ],
    author="Matt",
    description="Tools for vascular analysis including DataManager and VesselTracer",
    python_requires=">=3.6",
)