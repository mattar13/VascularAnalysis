from typing import Optional, Dict, Any, Tuple
import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize
from skan import Skeleton, summarize
import pandas as pd
import time

from .config import VesselTracerConfig

class VesselTracer:
    """Vessel tracing class focused solely on skeletonization and path analysis.
    
    This class takes binary images and creates vessel skeletons and path dictionaries.
    It does not handle image preprocessing, ROI extraction, or pipeline management.
    """
    
    def __init__(self, config: Optional[VesselTracerConfig] = None, verbose: int = 2):
        """Initialize VesselTracer.
        
        Args:
            config: Optional configuration object. If None, uses default config.
            verbose: Verbosity level for logging (0-3)
        """
        self.config = config if config is not None else VesselTracerConfig()
        self.verbose = verbose
        
        # Initialize path storage
        self.paths = {}
        self.stats = None
        self.n_paths = 0
        
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
    
    def trace_paths(self, binary_volume: np.ndarray, 
                   region_bounds: Optional[Dict[str, Tuple[float, float, Tuple[float, float]]]] = None,
                   split_paths: bool = False) -> Tuple[Dict[str, Any], Any]:
        """Create vessel skeleton and trace paths from binary volume.
        
        Args:
            binary_volume: 3D binary numpy array (boolean or 0/1)
            region_bounds: Optional dictionary of region bounds for path region assignment
            split_paths: Whether to split paths at region boundaries
            
        Returns:
            Tuple containing:
            - paths: Dictionary of branch paths with coordinates and metadata
            - stats: DataFrame with branch statistics from skan
        """
        start_time = time.time()
        self._log("Tracing vessel paths...", level=1)
        
        if binary_volume is None:
            raise ValueError("Binary volume cannot be None")
            
        if not isinstance(binary_volume, np.ndarray):
            raise ValueError("Binary volume must be a numpy array")
            
        if binary_volume.ndim != 3:
            raise ValueError("Binary volume must be 3D")
            
        # Ensure binary volume is boolean
        if binary_volume.dtype != bool:
            binary_volume = binary_volume.astype(bool)
            
        self._log(f"Skeletonizing binary volume of shape {binary_volume.shape}", level=2)
        ske = sk_skeletonize(binary_volume)
        self._log(f"Skeletonized volume of shape {ske.shape}", level=2)
        
        # Create Skeleton object for path analysis
        skeleton_obj = Skeleton(ske)
        self._log("Created skeleton object", level=2)
        
        # Get detailed statistics using skan's summarize function
        self.stats = summarize(skeleton_obj, separator="-")
        n_paths = skeleton_obj.n_paths
        
        # Convert skeleton object to dictionary with region information
        paths_dict = {}
        path_id = 1
        
        self._log(f"Processing {n_paths} skeleton paths", level=2)
        
        for i in range(n_paths):
            try:
                # Get path coordinates from skeleton object
                path_coords = skeleton_obj.path_coordinates(i)
                
                if len(path_coords) == 0:
                    continue
                
                if split_paths and region_bounds is not None:
                    # Split path into segments based on region boundaries
                    path_segments = self._split_path_at_region_boundaries(path_coords, region_bounds)
                    
                    for segment_region, segment_coords in path_segments:
                        # Create separate entry for each segment
                        paths_dict[path_id] = {
                            'original_path_id': i,
                            'coordinates': segment_coords,
                            'length': len(segment_coords),
                            'region': segment_region,
                            'z_range': (float(segment_coords[:, 0].min()), float(segment_coords[:, 0].max())) if len(segment_coords) > 0 else (0, 0),
                            'is_segment': True,
                            'parent_path_id': i
                        }
                        path_id += 1
                else:
                    # Store entire path with region information
                    primary_region = self._get_primary_region_for_path(path_coords, region_bounds) if region_bounds else 'Unknown'
                    region_fractions = self._calculate_region_fractions(path_coords, region_bounds) if region_bounds else {}
                    
                    paths_dict[path_id] = {
                        'original_path_id': i,
                        'coordinates': path_coords,
                        'length': len(path_coords),
                        'region': primary_region,
                        'region_fractions': region_fractions,
                        'z_range': (float(path_coords[:, 0].min()), float(path_coords[:, 0].max())) if len(path_coords) > 0 else (0, 0),
                        'is_segment': False
                    }
                    path_id += 1
                    
            except Exception as e:
                self._log(f"Warning: Error processing path {i}: {str(e)}", level=1)
                continue
        
        # Store the converted paths and update counts
        self.paths = paths_dict
        self.n_paths = len(paths_dict)
        
        self._log(f"Converted skeleton to dictionary with {self.n_paths} vessel paths", level=2)
        self._log("Path tracing complete", level=1, timing=time.time() - start_time)
        
        return self.paths, self.stats
    
    def _get_primary_region_for_path(self, path_coords: np.ndarray, 
                                   region_bounds: Dict[str, Tuple[float, float, Tuple[float, float]]]) -> str:
        """Determine the primary region for a path based on coordinate frequency.
        
        Args:
            path_coords: Array of path coordinates in ZXY format
            region_bounds: Dictionary of region bounds
            
        Returns:
            Name of the primary region for this path
        """
        if not region_bounds:
            return 'Unknown'
            
        # Count how many points fall in each region
        region_counts = {}
        for coord in path_coords:
            region = self._get_region_for_z(coord[0], region_bounds)
            region_counts[region] = region_counts.get(region, 0) + 1
            
        # Return region with most points
        if region_counts:
            return max(region_counts, key=region_counts.get)
        else:
            return 'Unknown'
    
    def _calculate_region_fractions(self, path_coords: np.ndarray,
                                  region_bounds: Dict[str, Tuple[float, float, Tuple[float, float]]]) -> Dict[str, float]:
        """Calculate the fraction of path points in each region.
        
        Args:
            path_coords: Array of path coordinates in ZXY format
            region_bounds: Dictionary of region bounds
            
        Returns:
            Dictionary mapping region names to fractions (0-1)
        """
        if not region_bounds:
            return {}
            
        # Count points in each region
        region_counts = {}
        total_points = len(path_coords)
        
        for coord in path_coords:
            region = self._get_region_for_z(coord[0], region_bounds)
            region_counts[region] = region_counts.get(region, 0) + 1
            
        # Convert counts to fractions
        region_fractions = {}
        for region in region_bounds.keys():
            count = region_counts.get(region, 0)
            region_fractions[region] = count / total_points if total_points > 0 else 0.0
            
        return region_fractions
    
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
    
    def _split_path_at_region_boundaries(self, path_coords: np.ndarray,
                                       region_bounds: Dict[str, Tuple[float, float, Tuple[float, float]]]) -> list:
        """Split a path into segments based on region boundaries.
        
        Args:
            path_coords: Array of coordinates for a single path in ZXY format
            region_bounds: Dictionary of region bounds
            
        Returns:
            List of tuples containing (region_name, path_segment_coords)
        """
        # Get regions for each point in the path
        regions = [self._get_region_for_z(coord[0], region_bounds) for coord in path_coords]
        
        # Initialize list to store path segments
        path_segments = []
        current_segment = []
        current_region = regions[0]
        
        for i, (coord, region) in enumerate(zip(path_coords, regions)):
            if region != current_region:
                # If we have points in the current segment, save it
                if current_segment:
                    # Convert list of coordinates to numpy array in ZXY format
                    segment_array = np.array(current_segment)
                    path_segments.append((current_region, segment_array))
                # Start new segment
                current_segment = [coord]
                current_region = region
            else:
                current_segment.append(coord)
                
        # Add the last segment if it exists
        if current_segment:
            # Convert list of coordinates to numpy array in ZXY format
            segment_array = np.array(current_segment)
            path_segments.append((current_region, segment_array))
            
        return path_segments
    
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