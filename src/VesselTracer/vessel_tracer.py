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
        
        # Initialize path storage
        self.paths = {}
        self.stats = None
        self.n_paths = 0
        
        # Initialize region storage
        self.region_bounds = {}
        
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
    
    def determine_regions(self, binary_volume: np.ndarray) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
        """Determine vessel regions based on the mean z-profile.
        
        Uses peak finding to identify distinct vessel layers and calculates
        their boundaries based on peak widths.
        
        Args:
            binary_volume: 3D binary numpy array containing vessel data
            
        Returns:
            Dictionary mapping region names to tuples of:
            (peak_position, sigma, (lower_bound, upper_bound))
        """
        self._log("Determining vessel regions...", level=1)
        
        if binary_volume is None:
            raise ValueError("Binary volume cannot be None")
            
        if not isinstance(binary_volume, np.ndarray):
            raise ValueError("Binary volume must be a numpy array")
            
        if binary_volume.ndim != 3:
            raise ValueError("Binary volume must be 3D")
            
        # Get mean z-profile (xy projection)
        mean_zprofile = np.mean(binary_volume, axis=(1,2))
        
        # Find peaks
        peaks, _ = find_peaks(mean_zprofile, distance=self.config.region_peak_distance)
        
        # Calculate peak widths
        widths_all, _, _, _ = peak_widths(
            mean_zprofile, peaks, rel_height=self.config.region_height_ratio)
        
        # Convert widths to sigmas
        sigmas = widths_all / (self.config.region_n_stds * np.sqrt(2 * np.log(2)))
        
        # Print peak information
        for i, pk in enumerate(peaks):
            self._log(f"Peak at z={pk:.1f}: σ ≈ {sigmas[i]:.2f}", level=2)
        
        # Create region bounds dictionary
        region_bounds = {
            region: (mu, sigma, (mu - sigma, mu + sigma))
            for region, mu, sigma in zip(self.config.regions, peaks, sigmas)
        }
        
        # Store region bounds
        self.region_bounds = region_bounds
        
        return region_bounds

    def create_region_map_volume(self, binary_volume: np.ndarray, 
                               region_bounds: Optional[Dict[str, Tuple[float, float, Tuple[float, float]]]] = None) -> np.ndarray:
        """Create a volume where each z-position is labeled with its region number.
        
        Creates a 3D array the same size as the volume where each voxel is assigned
        a region number based on its z-position:
        - 0: Unknown/outside regions
        - 1: Superficial
        - 2: Intermediate  
        - 3: Deep
        
        Args:
            binary_volume: 3D binary numpy array to create region map for
            region_bounds: Optional dictionary of region bounds. If None, uses stored region_bounds.
            
        Returns:
            np.ndarray: Volume with region labels (0-3) for each voxel
        """
        if binary_volume is None:
            raise ValueError("Binary volume cannot be None")
        
        if region_bounds is None:
            region_bounds = self.region_bounds
            
        if not region_bounds:
            raise ValueError("No region bounds available. Run determine_regions first.")
        
        # Create region map with same shape as volume
        Z, Y, X = binary_volume.shape
        region_map = np.zeros((Z, Y, X), dtype=np.uint8)
        
        # Define region number mapping
        region_numbers = {
            'superficial': 1,
            'intermediate': 2,
            'deep': 3
        }
        
        # Assign region numbers to each z-slice
        for z in range(Z):
            region_name = self._get_region_for_z(z, region_bounds)
            region_number = region_numbers.get(region_name, 0)  # 0 for unknown
            region_map[z, :, :] = region_number
        
        self._log(f"Created region map volume with shape {region_map.shape}", level=2)
        self._log(f"Region assignments: 0=unknown, 1=superficial, 2=intermediate, 3=deep", level=2)
        
        # Log region statistics
        unique_regions, counts = np.unique(region_map, return_counts=True)
        total_voxels = region_map.size
        for region_num, count in zip(unique_regions, counts):
            region_names = {0: 'unknown', 1: 'superficial', 2: 'intermediate', 3: 'deep'}
            region_name = region_names.get(region_num, f'region_{region_num}')
            percentage = (count / total_voxels) * 100
            self._log(f"  {region_name}: {count:,} voxels ({percentage:.1f}%)", level=2)
        
        return region_map
    
    def trace_paths(self, 
                    binary_volume: np.ndarray, region_bounds: Optional[Dict[str, Tuple[float, float, Tuple[float, float]]]] = None, split_paths: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
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
            try:
                # Get path coordinates from skeleton object
                path_coords = skeleton_obj.path_coordinates(i)
                
                if len(path_coords) == 0:
                    continue
                    
                if split_paths:
                    # Split path at region boundaries and create separate entries
                    path_segments = self.split_path_at_region_boundaries(path_coords)
                    
                    for region, segment_coords in path_segments:
                        if len(segment_coords) > 1:  # Only store segments with more than one point
                            paths_dict[path_id] = {
                                'original_path_id': i,
                                'region': region,
                                'coordinates': segment_coords,
                                'length': len(segment_coords)
                            }
                            path_id += 1
                else:
                    # Determine the primary region for this path using z-coordinates
                    z_coords = path_coords[:, 0]  # Extract z-coordinates
                    regions_in_path = [self.get_region_for_z(z) for z in z_coords]
                    
                    # Find the most common region in this path
                    from collections import Counter
                    region_counts = Counter(regions_in_path)
                    primary_region = region_counts.most_common(1)[0][0]
                    
                    # Calculate what fraction of path is in each region
                    region_fractions = {region: count/len(regions_in_path) 
                                      for region, count in region_counts.items()}
                    
                    paths_dict[path_id] = {
                        'original_path_id': i,
                        'region': primary_region,
                        'region_fractions': region_fractions,
                        'coordinates': path_coords,
                        'length': len(path_coords),
                        'z_range': (float(z_coords.min()), float(z_coords.max()))
                    }
                    path_id += 1
                    
            except Exception as e:
                self._log(f"Warning: Error processing path {i}: {str(e)}", level=1)
                continue
        
        # Store the converted paths and update counts
        paths = paths_dict
        n_paths = len(paths_dict)
        
        self._log(f"Converted skeleton to dictionary with {self.n_paths} vessel paths", level=2)
        if split_paths:
            self._log("Paths split at region boundaries", level=2)
        else:
            self._log("Paths labeled with primary regions", level=2)
            
        self._log("Path tracing complete", level=1, timing=time.time() - start_time)    
        return paths, stats, n_paths
    
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
        self.region_bounds = {} 