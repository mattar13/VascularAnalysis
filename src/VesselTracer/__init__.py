"""
VesselTracer Package

This package provides tools for analyzing vascular structures in 3D image data,
including preprocessing, segmentation, skeletonization, and analysis of vessel networks.

Main Components:
- VesselTracer: Main class for vessel analysis pipeline
- plotting: Visualization utilities
"""

from .tracer import VesselTracer
from .plotting import show_max_projection

__version__ = '0.1.0'
__author__ = 'Matt'

__all__ = [
    'VesselTracer',
    'show_max_projection',
    'plot_vertical_region_map',
]
