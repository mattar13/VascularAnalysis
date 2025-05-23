"""
VesselTracer - A tool for analyzing and visualizing vascular networks in 3D microscopy data.
"""

from .VesselTracer import VesselTracer
from .plotting import (
    plot_projections,
    plot_mean_zprofile,
    plot_path_projections,
    plot_layer_analysis,
    plot_vessel_analysis
)

__version__ = "0.1.0"
__author__ = 'Matt'

__all__ = [
    'VesselTracer',
    'plot_projections',
    'plot_mean_zprofile',
    'plot_path_projections',
    'plot_layer_analysis',
    'plot_vessel_analysis'
]
