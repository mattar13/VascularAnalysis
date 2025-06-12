from typing import Optional, Dict, Any, Tuple
import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize
from skan import Skeleton, summarize
from scipy.signal import find_peaks, peak_widths
import pandas as pd
import time

from .config import VesselTracerConfig

class VesselTracer:
    """Vessel tracing class focused solely on skeletonization and path analysis.
    
    This class takes binary images and creates vessel skeletons and path dictionaries.
    It also handles region determination and analysis since these are closely related
    to vessel tracing operations.
    """
    
    def __init__(self, config: Optional[VesselTracerConfig] = None, verbose: int = 2):
        """Initialize VesselTracer.
        
        Args:
            config: Optional configuration object. If None, uses default config.
            verbose: Verbosity level for logging (0-3)
        """
        self.config = config if config is not None else VesselTracerConfig()
        self.verbose = verbose
        
    def _log(self, message: str, level: int = 1, timing: Optional[float] = None):
        """Log a message with appropriate verbosity level.
        
        Args:
            message: Message to log
            level: Minimum verbosity level required to show message (1-3)
            timing: Optional timing information to include
        """
        if self.verbose >= level:
            if timing is not None and self.verbose >= 3:
                print(f"{message} (took {timing:.2f}s)")
            else:
                print(message)
    
    #These functions are for the skeletonization and path tracing    
    def trace_paths(self, 
                    binary_volume: np.ndarray, split_paths: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        """Create vessel skeleton and trace paths.
        
        Args:
            split_paths: Whether to split paths at region boundaries
            
        Returns:
            Tuple containing:
            - paths: Dictionary of branch paths with coordinates
            - stats: DataFrame with branch statistics
        """
        start_time = time.time()    
        self._log("Tracing vessel paths...", level=1)
            
        self._log(f"Skeletonizing binary volume of shape {binary_volume.shape}", level=2)
        ske = sk_skeletonize(binary_volume)
        self._log(f"Skeletonized volume of shape {ske.shape}", level=2)
        
        # Create Skeleton object for path analysis
        skeleton_obj = Skeleton(ske)
        self._log("Created skeleton object", level=2)
        
        # Get detailed statistics using skan's summarize function
        stats = summarize(skeleton_obj, separator="-")
        n_paths = skeleton_obj.n_paths
        
        # Convert skeleton object to dictionary with region information
        paths_dict = {}
        path_id = 1
        
        self._log(f"Processing {n_paths} skeleton paths", level=2)
        
        for i in range(n_paths):
            # Get path coordinates from skeleton object
            path_coords = skeleton_obj.path_coordinates(i)
            # print(path_coords.shape)
            # print(path_coords[:, 0])
            if len(path_coords) == 0:
                continue
            
            # Calculate what fraction of path is in each region
            paths_dict[path_id] = {
                'coordinates': path_coords,
                'length': len(path_coords),
            }
            path_id += 1
        
        # Store the converted paths and update counts
        paths = paths_dict
        n_paths = len(paths_dict)
        
        self._log(f"Converted skeleton to dictionary with {n_paths} vessel paths", level=2)
        self._log("Path tracing complete", level=1, timing=time.time() - start_time)    
        return paths, stats, n_paths
    

    #These functions are for the region determination
    def _get_region_for_z(self, z_coord: float, 
                         region_bounds: Dict[str, Tuple[float, float, Tuple[float, float]]]) -> str:
        """Determine which region a z-coordinate belongs to.
        
        Args:
            z_coord: Z coordinate value
            region_bounds: Dictionary of region bounds
            
        Returns:
            Name of the region containing this z-coordinate
        """
        for region_name, (peak, sigma, (lower, upper)) in region_bounds.items():
            if lower <= z_coord <= upper:
                return region_name
        return 'Outside'
    
    #These functions are for the path dataframe creation
    def create_paths_dataframe(self, pixel_sizes: Optional[Tuple[float, float, float]] = None) -> pd.DataFrame:
        """Create a pandas DataFrame from the traced paths.
        
        Args:
            pixel_sizes: Optional tuple of (z,y,x) pixel sizes in microns for coordinate conversion
            
        Returns:
            DataFrame with detailed path information
        """
        if not self.paths:
            return pd.DataFrame()
            
        vessel_paths_data = []
        
        for path_id, path_info in self.paths.items():
            coords = path_info['coordinates']
            region = path_info.get('region', 'Unknown')
            original_id = path_info.get('original_path_id', path_id)
            length = path_info.get('length', len(coords))
            
            # Convert coordinates if pixel sizes provided
            if pixel_sizes:
                z_scale, y_scale, x_scale = pixel_sizes
            else:
                z_scale = y_scale = x_scale = 1.0
            
            # Convert path coordinates to DataFrame rows
            for i, coord in enumerate(coords):
                vessel_paths_data.append({
                    'Path_ID': path_id,
                    'Original_Path_ID': original_id,
                    'Primary_Region': region,
                    'Point_Index': i,
                    'Z': coord[0],  # Z coordinate in pixels
                    'X': coord[1],  # X coordinate in pixels  
                    'Y': coord[2],  # Y coordinate in pixels
                    'Z_microns': coord[0] * z_scale,  # Z in microns
                    'X_microns': coord[1] * x_scale,  # X in microns
                    'Y_microns': coord[2] * y_scale,  # Y in microns
                    'Path_Length': length,
                    'Is_Segment': path_info.get('is_segment', False)
                })
                
        return pd.DataFrame(vessel_paths_data)
    
    #These functions are for the path summary dataframe creation
    def create_path_summary_dataframe(self, pixel_sizes: Optional[Tuple[float, float, float]] = None) -> pd.DataFrame:
        """Create a summary DataFrame with one row per path.
        
        Args:
            pixel_sizes: Optional tuple of (z,y,x) pixel sizes in microns for coordinate conversion
            
        Returns:
            DataFrame with path summary information
        """
        if not self.paths:
            return pd.DataFrame()
            
        path_summary_data = []
        
        # Convert pixel sizes if provided
        if pixel_sizes:
            z_scale, y_scale, x_scale = pixel_sizes
        else:
            z_scale = y_scale = x_scale = 1.0
        
        for path_id, path_info in self.paths.items():
            coords = path_info['coordinates']
            region = path_info.get('region', 'Unknown')
            original_id = path_info.get('original_path_id', path_id)
            length = path_info.get('length', len(coords))
            
            # Create summary entry for this path
            summary_entry = {
                'Path_ID': path_id,
                'Original_Path_ID': original_id,
                'Primary_Region': region,
                'Path_Length_pixels': length,
                'Path_Length_microns': length * np.mean([x_scale, y_scale, z_scale]),  # Approximate length conversion
                'Start_Z': coords[0][0] if len(coords) > 0 else None,
                'End_Z': coords[-1][0] if len(coords) > 0 else None,
                'Start_Z_microns': coords[0][0] * z_scale if len(coords) > 0 else None,
                'End_Z_microns': coords[-1][0] * z_scale if len(coords) > 0 else None,
                'Is_Segment': path_info.get('is_segment', False)
            }
            
            # Add region fractions if available
            if 'region_fractions' in path_info:
                for region_name, fraction in path_info['region_fractions'].items():
                    summary_entry[f'{region_name}_Fraction'] = fraction
                    
            # Add z-range if available
            if 'z_range' in path_info:
                summary_entry['Z_Min'] = path_info['z_range'][0]
                summary_entry['Z_Max'] = path_info['z_range'][1]
                summary_entry['Z_Min_microns'] = path_info['z_range'][0] * z_scale
                summary_entry['Z_Max_microns'] = path_info['z_range'][1] * z_scale
                    
            path_summary_data.append(summary_entry)
            
        return pd.DataFrame(path_summary_data)
    
    def get_path_count(self) -> int:
        """Get the number of traced paths.
        
        Returns:
            Number of paths
        """
        return self.n_paths
    
    def get_path_by_id(self, path_id: int) -> Optional[Dict[str, Any]]:
        """Get path information by ID.
        
        Args:
            path_id: ID of the path to retrieve
            
        Returns:
            Dictionary with path information or None if not found
        """
        return self.paths.get(path_id)
    
    def clear_paths(self):
        """Clear all traced paths and statistics."""
        self.paths = {}
        self.stats = None
        self.n_paths = 0
        self.region_bounds = {} 