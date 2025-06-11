"""
VesselTracer Package

This package provides tools for analyzing vascular structures in 3D image data,
including preprocessing, segmentation, skeletonization, and analysis of vessel networks.

Main Components:
- VesselTracer: Main class for vessel analysis pipeline
- plotting: Visualization utilities
"""

# Import main components
# from .config import VesselTracerConfig
# from .image_model import ImageModel, ROI
# from .image_processor import ImageProcessor
# from .tracer import VesselTracer
# from .plotting import show_max_projection
from .tracer import VesselTracer

__version__ = '0.1.0'
__author__ = 'Matt'

__all__ = [
    'VesselTracer',
    'VesselTracerConfig', 
    'ImageModel',
    'ROI',
    'ImageProcessor',
    'show_max_projection',
]
