import numpy as np
import pandas as pd
import time
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from scipy.signal import find_peaks
from skimage.morphology import skeletonize_3d, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion

class VesselTracer:
    """Controller class for vessel tracing and region analysis operations.
    
    This class handles region determination, skeletonization, and vessel path tracing
    operations on vessel data models.
    """
    
    def __init__(self, verbose: int = 2):
        """Initialize the VesselTracer.
        
        Args:
            verbose: Verbosity level for logging
        """
        self.verbose = verbose
        
        # Analysis results storage
        self.skeleton = None
        self.paths = {}
        self.n_paths = 0
        self.region_bounds = {}
        self.regions_df = pd.DataFrame()
        self.paths_df = pd.DataFrame()
    
    def _log(self, message: str, level: int = 1, timing: Optional[float] = None):
        """Log a message with appropriate verbosity level."""
        if self.verbose >= level:
            if timing is not None and self.verbose >= 3:
                print(f"{message} (took {timing:.2f}s)")
            else:
                print(message)

    def _save_results(self, 
                     vessel_data, 
                     config, 
                     output_dir: Path,
                     save_volumes: bool,
                     save_original: bool,
                     save_smoothed: bool,
                     save_binary: bool,
                     save_separate: bool,
                     skip_dataframe: bool) -> None:
        """Save analysis results to output directory.
        
        Args:
            vessel_data: VesselData object
            config: ConfigManager object
            output_dir: Directory to save results
            save_volumes: Whether to save volume data
            save_original: Whether to save original volume
            save_smoothed: Whether to save smoothed volume
            save_binary: Whether to save binary volume
            save_separate: Whether to save volumes as separate files
            skip_dataframe: Whether to skip DataFrame saving
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save volume data
        if save_volumes:
            self._log("Saving volume data...", level=2)
            vessel_data.save_volume(
                output_dir=output_dir,
                save_original=save_original,
                save_smoothed=save_smoothed,
                save_binary=save_binary,
                save_separate=save_separate
            )
        
        # Save skeleton if available
        if self.skeleton is not None:
            self._log("Saving skeleton data...", level=2)
            import tifffile
            skeleton_file = output_dir / "skeleton.tif"
            tifffile.imwrite(str(skeleton_file), self.skeleton.astype(np.uint8))
        
        # Save analysis DataFrames
        if not skip_dataframe:
            if not self.regions_df.empty:
                self._log("Saving regions DataFrame...", level=2)
                regions_file = output_dir / "regions_analysis.csv"
                self.regions_df.to_csv(regions_file, index=False)
                
            if not self.paths_df.empty:
                self._log("Saving paths DataFrame...", level=2)
                paths_file = output_dir / "paths_analysis.csv"
                self.paths_df.to_csv(paths_file, index=False)
        
        # Save configuration
        config_file = output_dir / "analysis_config.yaml"
        config.save_config(config_file)
        self._log(f"Saved configuration to {config_file}", level=2)
        
        self._log(f"All results saved to {output_dir}", level=2)

    def determine_regions(self, vessel_data, config) -> Tuple[Dict[str, Tuple[int, int]], pd.DataFrame]:
        """Determine cortical layer regions based on vessel density profile.
        
        Args:
            vessel_data: VesselData object containing binary volume
            config: ConfigManager with region settings
            
        Returns:
            Tuple of (region_bounds dict, regions_df DataFrame)
        """
        start_time = time.time()
        self._log("Determining cortical regions...", level=1)
        
        if vessel_data.binary is None:
            raise ValueError("Binary volume not available. Run binarization first.")
        
        # Calculate vessel density profile along z-axis
        Z = vessel_data.binary.shape[0]
        vessel_density = np.zeros(Z)
        
        for z in range(Z):
            # Calculate vessel density as fraction of vessel pixels in each slice
            vessel_density[z] = np.sum(vessel_data.binary[z]) / vessel_data.binary[z].size
        
        self._log(f"Calculated vessel density profile (shape: {vessel_density.shape})", level=2)
        
        # Smooth the density profile to find major peaks
        from scipy.ndimage import gaussian_filter1d
        smoothed_density = gaussian_filter1d(vessel_density, sigma=2.0)
        
        # Find peaks in the smoothed density profile
        peaks, properties = find_peaks(
            smoothed_density,
            distance=config.region_peak_distance,
            height=config.region_height_ratio * smoothed_density.max()
        )
        
        self._log(f"Found {len(peaks)} density peaks at z-positions: {peaks}", level=2)
        
        # Define region boundaries based on peaks and expected regions
        regions = config.regions  # ['superficial', 'intermediate', 'deep']
        
        if len(peaks) >= 2:
            # Use peaks to define boundaries
            if len(peaks) == 2:
                # Two peaks: assume superficial and deep regions
                boundaries = [0, peaks[0], peaks[1], Z-1]
            else:
                # Multiple peaks: use first 3 or space them evenly
                if len(peaks) >= 3:
                    boundaries = [0, peaks[0], peaks[1], peaks[2], Z-1]
                else:
                    # Fallback: divide into equal thirds
                    boundaries = [0, Z//3, 2*Z//3, Z-1]
        else:
            # No clear peaks: divide into equal regions
            self._log("No clear density peaks found, dividing into equal regions", level=2)
            n_regions = len(regions)
            boundaries = [int(i * Z / n_regions) for i in range(n_regions + 1)]
            boundaries[-1] = Z - 1
        
        # Create region bounds dictionary
        self.region_bounds = {}
        region_data = []
        
        for i, region_name in enumerate(regions):
            if i < len(boundaries) - 1:
                start_z = boundaries[i]
                end_z = boundaries[i + 1]
                self.region_bounds[region_name] = (start_z, end_z)
                
                # Calculate region statistics
                region_density = vessel_density[start_z:end_z+1]
                region_data.append({
                    'Region': region_name,
                    'Start_Z': start_z,
                    'End_Z': end_z,
                    'Thickness_Slices': end_z - start_z + 1,
                    'Thickness_Microns': (end_z - start_z + 1) * vessel_data.pixel_size_z,
                    'Mean_Density': region_density.mean(),
                    'Max_Density': region_density.max(),
                    'Min_Density': region_density.min(),
                    'Std_Density': region_density.std()
                })
                
                self._log(f"{region_name}: z={start_z}-{end_z} "
                         f"({end_z-start_z+1} slices, "
                         f"{(end_z-start_z+1)*vessel_data.pixel_size_z:.1f}µm)", level=2)
        
        # Create DataFrame with region information
        self.regions_df = pd.DataFrame(region_data)
        
        self._log("Region determination complete", level=1, timing=time.time() - start_time)
        return self.region_bounds, self.regions_df
    
    def get_region_for_z(self, z_position: int) -> str:
        """Get the region name for a given z-position.
        
        Args:
            z_position: Z-coordinate to query
            
        Returns:
            str: Region name ('superficial', 'intermediate', 'deep', or 'unknown')
        """
        for region_name, (start_z, end_z) in self.region_bounds.items():
            if start_z <= z_position <= end_z:
                return region_name
        return 'unknown'
    
    def skeletonize(self, vessel_data, config) -> np.ndarray:
        """Create 3D skeleton of the binary vessel volume.
        
        Args:
            vessel_data: VesselData object containing binary volume
            config: ConfigManager with processing settings
            
        Returns:
            np.ndarray: 3D skeleton of vessels
        """
        start_time = time.time()
        self._log("Creating 3D skeleton...", level=1)
        
        if vessel_data.binary is None:
            raise ValueError("Binary volume not available. Run binarization first.")
        
        # Create 3D skeleton
        self.skeleton = skeletonize_3d(vessel_data.binary)
        
        # Prune short branches if specified
        if config.prune_length > 0:
            self._log(f"Pruning branches shorter than {config.prune_length} pixels", level=2)
            self.skeleton = self._prune_skeleton(self.skeleton, config.prune_length)
        
        self._log(f"Skeleton created. Total skeleton pixels: {np.sum(self.skeleton)}", level=2)
        self._log("Skeletonization complete", level=1, timing=time.time() - start_time)
        return self.skeleton
    
    def _prune_skeleton(self, skeleton: np.ndarray, prune_length: int) -> np.ndarray:
        """Prune short branches from skeleton.
        
        Args:
            skeleton: 3D binary skeleton
            prune_length: Maximum length of branches to remove
            
        Returns:
            np.ndarray: Pruned skeleton
        """
        # This is a simplified pruning algorithm
        # For more sophisticated pruning, consider using specialized libraries
        
        pruned = skeleton.copy()
        
        # Find endpoints (pixels with only one neighbor)
        for _ in range(prune_length):
            # Create a kernel to count neighbors
            from scipy.ndimage import convolve
            kernel = np.ones((3, 3, 3))
            kernel[1, 1, 1] = 0
            
            # Count neighbors for each skeleton pixel
            neighbor_count = convolve(pruned.astype(int), kernel, mode='constant')
            
            # Find endpoints (skeleton pixels with exactly 1 neighbor)
            endpoints = (pruned == 1) & (neighbor_count == 1)
            
            if not np.any(endpoints):
                break
                
            # Remove endpoints
            pruned[endpoints] = 0
        
        return pruned
    
    def trace_paths(self, vessel_data, config) -> Tuple[Dict, pd.DataFrame]:
        """Trace individual vessel paths from the skeleton.
        
        Args:
            vessel_data: VesselData object 
            config: ConfigManager with processing settings
            
        Returns:
            Tuple of (paths dict, paths_df DataFrame)
        """
        start_time = time.time()
        self._log("Tracing vessel paths...", level=1)
        
        if self.skeleton is None:
            raise ValueError("Skeleton not available. Run skeletonize first.")
        
        # Label connected components in skeleton
        labeled_skeleton = label(self.skeleton)
        regions = regionprops(labeled_skeleton)
        
        self.paths = {}
        path_data = []
        
        for i, region in enumerate(regions):
            path_id = i + 1
            
            # Get coordinates of this path
            coords = region.coords  # Returns (z, y, x) coordinates
            
            if len(coords) < 3:  # Skip very short paths
                continue
            
            # Calculate path properties
            path_length_pixels = len(coords)
            path_length_microns = path_length_pixels * vessel_data.pixel_size_z  # Approximate
            
            # Get z-range to determine regions
            z_coords = coords[:, 0]
            z_min, z_max = z_coords.min(), z_coords.max()
            
            # Determine which regions this path crosses
            regions_crossed = set()
            for z in range(z_min, z_max + 1):
                region_name = self.get_region_for_z(z)
                if region_name != 'unknown':
                    regions_crossed.add(region_name)
            
            # Classify path type
            if len(regions_crossed) > 1:
                path_type = 'diving'
            elif len(regions_crossed) == 1:
                path_type = list(regions_crossed)[0]
            else:
                path_type = 'unknown'
            
            # Store path information
            self.paths[path_id] = {
                'coordinates': coords,
                'length_pixels': path_length_pixels,
                'length_microns': path_length_microns,
                'z_min': z_min,
                'z_max': z_max,
                'regions': list(regions_crossed),
                'type': path_type
            }
            
            # Add to DataFrame data
            path_data.append({
                'Path_ID': path_id,
                'Length_Pixels': path_length_pixels,
                'Length_Microns': path_length_microns,
                'Z_Min': z_min,
                'Z_Max': z_max,
                'Z_Range': z_max - z_min,
                'Regions_Crossed': ','.join(sorted(regions_crossed)),
                'Path_Type': path_type,
                'Num_Regions': len(regions_crossed)
            })
        
        self.n_paths = len(self.paths) + 1  # +1 for background
        self.paths_df = pd.DataFrame(path_data)
        
        self._log(f"Traced {len(self.paths)} vessel paths", level=2)
        self._log(f"Path types: {self.paths_df['Path_Type'].value_counts().to_dict()}", level=2)
        self._log("Path tracing complete", level=1, timing=time.time() - start_time)
        
        return self.paths, self.paths_df 
    
    def run_analysis(self, vessel_data, config) -> None:
        """Run the vessel analysis workflow (skeletonization and path tracing).
        
        Args:
            vessel_data: VesselData object with binary volume
            config: ConfigManager with analysis parameters
        """
        analysis_start = time.time()
        self._log("Starting vessel analysis workflow...", level=1)
        
        # Skeletonization
        self.skeletonize(vessel_data, config)
        
        # Path tracing
        self.trace_paths(vessel_data, config)
        
        self._log("Vessel analysis workflow complete", level=1, 
            timing=time.time() - analysis_start)
        

    def run_pipeline(self, 
                    vessel_data, 
                    config, 
                    processor,
                    output_dir: Optional[Path] = None,
                    skip_smoothing: bool = False,
                    skip_binarization: bool = False,
                    skip_regions: bool = False,
                    skip_trace: bool = False,
                    skip_dataframe: bool = False,
                    save_volumes: bool = True,
                    save_original: bool = True,
                    save_smoothed: bool = True,
                    save_binary: bool = True,
                    save_separate: bool = False) -> None:
        """Run the complete vessel analysis pipeline.
        
        Args:
            vessel_data: VesselData object with loaded image data
            config: ConfigManager with analysis parameters
            processor: VesselProcessor for image processing operations
            output_dir: Directory to save results
            skip_smoothing: Skip Gaussian smoothing step
            skip_binarization: Skip binarization step
            skip_regions: Skip region determination step
            skip_trace: Skip vessel tracing step
            skip_dataframe: Skip DataFrame generation step
            save_volumes: Whether to save volume data
            save_original: Whether to save original volume
            save_smoothed: Whether to save smoothed volume
            save_binary: Whether to save binary volume
            save_separate: Whether to save volumes as separate files
        """
        pipeline_start = time.time()
        self._log("="*60, level=1)
        self._log("STARTING VESSEL ANALYSIS PIPELINE", level=1)
        self._log("="*60, level=1)
        
        # Step 1: ROI extraction (if enabled)
        if config.find_roi:
            self._log("Step 1: ROI Extraction", level=1)
            processor.segment_roi(vessel_data, config)
        else:
            self._log("Step 1: ROI Extraction - SKIPPED (using full volume)", level=1)
        
        # Step 2: Normalize image
        self._log("Step 2: Image Normalization", level=1)
        processor.normalize_image(vessel_data)
        
        # Step 3: Median filtering
        self._log("Step 3: Median Filtering", level=1)
        processor.median_filter(vessel_data, config)
        
        # Step 4: Background smoothing
        self._log("Step 4: Background Smoothing", level=1)
        processor.gaussian_filter(vessel_data, config, mode='background')
        
        # Step 5: Detrending
        self._log("Step 5: Detrending", level=1)
        processor.detrend(vessel_data)
        
        # Step 6: Regular smoothing
        if not skip_smoothing:
            self._log("Step 6: Regular Smoothing", level=1)
            processor.gaussian_filter(vessel_data, config, mode='smooth')
        else:
            self._log("Step 6: Regular Smoothing - SKIPPED", level=1)
        
        # Step 7: Binarization
        if not skip_binarization:
            self._log("Step 7: Binarization", level=1)
            processor.binarize(vessel_data, config)
        else:
            self._log("Step 7: Binarization - SKIPPED", level=1)
        
        # Step 8: Region determination
        if not skip_regions:
            self._log("Step 8: Region Determination", level=1)
            self.determine_regions(vessel_data, config)
        else:
            self._log("Step 8: Region Determination - SKIPPED", level=1)
        
        # Step 9: Vessel tracing
        if not skip_trace:
            self._log("Step 9: Vessel Tracing", level=1)
            self.run_analysis(vessel_data, config)
        else:
            self._log("Step 9: Vessel Tracing - SKIPPED", level=1)
        
        # Step 10: Save results
        if output_dir is not None:
            self._log("Step 10: Saving Results", level=1)
            self._save_results(vessel_data, config, output_dir, 
                             save_volumes, save_original, save_smoothed, 
                             save_binary, save_separate, skip_dataframe)
        else:
            self._log("Step 10: Saving Results - SKIPPED (no output directory)", level=1)
        
        total_time = time.time() - pipeline_start
        self._log("="*60, level=1)
        self._log(f"PIPELINE COMPLETE - Total time: {total_time:.2f}s", level=1)
        self._log("="*60, level=1)