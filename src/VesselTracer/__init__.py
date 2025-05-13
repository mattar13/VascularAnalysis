"""
VesselTracer Package

This package provides tools for analyzing vascular structures in 3D image data,
including preprocessing, segmentation, skeletonization, and analysis of vessel networks.

Main Components:
- preprocessing: Image preprocessing utilities
- segmentation: Binary segmentation tools
- skeletonize: Vessel skeleton extraction and analysis
- analysis: Vessel network analysis tools
- plotting: Visualization utilities
- config: Configuration settings
- io: Input/output operations
"""

from .preprocessing import smooth
from .segmentation import segment_binary
from .skeletonize import skeleton_stats
from .plotting import show_max_projection
from .config import PipelineConfig
from .analysis import (
    analyze_vessel_network,
    compute_vessel_metrics
)
from .io import (
    load_image_stack,
    save_results
)

__version__ = '0.1.0'
__author__ = 'Matt'

# Direct imports for commonly used functions
__all__ = [
    # Main functions
    'smooth',
    'segment_binary',
    'skeleton_stats',
    'show_max_projection',
    'PipelineConfig',
    'analyze_vessel_network',
    'compute_vessel_metrics',
    'load_image_stack',
    'save_results',
    
    # Module imports
    'preprocessing',
    'segmentation',
    'skeletonize',
    'analysis',
    'plotting',
    'config',
    'io'
]

# Import modules for backward compatibility
from . import (
    preprocessing,
    segmentation,
    skeletonize,
    analysis,
    plotting,
    config,
    io
)
