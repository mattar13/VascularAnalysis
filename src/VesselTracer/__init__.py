"""
VesselTracer Package

This package provides tools for analyzing vascular structures in 3D image data,
including preprocessing, segmentation, skeletonization, and analysis of vessel networks.

Main Components:
- VesselAnalysisController: Main pipeline orchestrator
- VesselTracer: Specialized vessel tracing and skeletonization 
- ImageProcessor: Image processing operations
- plotting: Visualization utilities
"""

# Import main components
from .config import VesselTracerConfig
from .image_model import ImageModel, ROI
from .image_processor import ImageProcessor
from .vessel_tracer import VesselTracer
from .controller import VesselAnalysisController
from .plotting import show_max_projection

# Keep the old tracer.py import for backward compatibility
# from .tracer import VesselTracer as LegacyVesselTracer

__version__ = '0.2.0'
__author__ = 'Matt'

__all__ = [
    'VesselAnalysisController',
    'VesselTracer',
    'VesselTracerConfig', 
    'ImageModel',
    'ROI',
    'ImageProcessor',
    # 'LegacyVesselTracer',
    'show_max_projection',
]
