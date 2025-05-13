"""
DataManager Package

This package provides tools for managing and analyzing vascular data, including loading,
storing, and processing multiple dataframes of vessel measurements.

Main Components:
- DataManager: Core class for handling vessel data
- Auxiliary Functions: Helper functions for data processing
- Plotting Functions: Visualization utilities
"""
import pandas as pd
import numpy as np
import tifffile as tiff #Opening .tif files
import re
import os

from .make_datafile import DataManager
from .auxillary_functions import (
    test_repo,
    insert_zero_rows_between_max_indices,
    apply_lut,
    adjust_to_4d,
    pad_columns
)

__version__ = '0.1.0'
__author__ = 'Matt'

__all__ = [
    'DataManager',
    'test_repo',
    'insert_zero_rows_between_max_indices',
    'apply_lut',
    'adjust_to_4d',
    'pad_columns'
]

from . import (
    make_datafile,
    auxillary_functions
)
