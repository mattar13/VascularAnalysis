import numpy as np
import tifffile
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Union, List

# Import the components that the Controller will coordinate
from .image_processor import ImageProcessor
from .vessel_tracer import VesselTracer
from .config import VesselTracerConfig
from .image_model import ImageModel, ROI

class VesselAnalysisController:
    """Controller class that orchestrates the complete vessel analysis pipeline.
    
    This class manages the coordination between ImageProcessor, VesselTracer, and
    data models to provide a complete analysis workflow. It coordinates the pipeline
    but does not store any results - it only points to where results are stored in
    the component objects and delegates all operations.
    """
    
    def __init__(self, 
                 input_path: Union[str, Path],
                 config_path: Union[str, Path],
                 verbose: int = 2,
                 use_gpu: bool = False):
        """Initialize the controller.
        
        Args:
            input_path: Path to input image file
            config_path: Path to configuration file
            verbose: Verbosity level for logging (0-3)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.verbose = verbose
        self.use_gpu = use_gpu
        
        # Load configuration
        self.config = VesselTracerConfig(config_path)
        
        # Initialize image model
        self.image_model = ImageModel(filepath=input_path)
        self.roi_model = None  # Will be initialized during analysis
        
        # Initialize components
        self.processor = ImageProcessor(
            config=self.config,
            verbose=verbose,
            use_gpu=use_gpu
        )
        
        self.tracer = VesselTracer(
            config=self.config,  # Pass config directly
            verbose=True  # Can be made configurable
        )
        
    def _log(self, message: str, level: int = 1, timing: Optional[float] = None):
        """Log a message with appropriate verbosity level.
        
        Args:
            message: Message to log
            level: Minimum verbosity level required to show message (1-3)
            timing: Optional timing information to include
        """
        if self.config.verbose >= level:
            if timing is not None and self.config.verbose >= 3:
                print(f"{message} (took {timing:.2f}s)")
            else:
                print(message)
                
    def activate_gpu(self) -> bool:
        """Activate GPU mode for processing.
        
        Returns:
            bool: True if GPU mode was successfully activated, False otherwise.
        """
        # Delegate GPU activation to processor
        return self.processor.activate_gpu()

    def print_config(self):
        """Print current configuration."""
        pixel_sizes = self.processor.get_pixel_sizes() if self.processor.has_volume() else None
        self.config.print_config(pixel_sizes)

    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to a YAML file.
        
        Args:
            output_path: Path where to save the YAML configuration file
        """
        pixel_sizes = self.processor.get_pixel_sizes() if self.processor.has_volume() else None
        self.config.save_config(output_path, pixel_sizes)
    
    def save_metadata(self, output_path: Union[str, Path], format: str = 'excel') -> None:
        """Save metadata to file.
        
        Args:
            output_path: Path where to save the metadata
            format: Output format ('excel', 'csv', 'json')
        """
        pixel_sizes = self.processor.get_pixel_sizes() if self.processor.has_volume() else None
        input_type = 'Numpy Array' if isinstance(self.input_data, np.ndarray) else 'File'
        
        # Create processing status dictionary by checking component states
        processing_status = {
            'ROI Extraction': self.processor.has_roi(),
            'Background Subtraction': self.processor.has_background(),
            'Smoothing': True,  # Always performed if not skipped
            'Binarization': self.processor.has_binary(),
            'Region Detection': self.tracer.has_regions(),
            'Path Tracing': self.tracer.has_paths(),
            'Region Map Creation': self.tracer.has_region_map()
        }
        
        self.config.save_metadata(output_path, pixel_sizes, input_type, processing_status, format)

    def run_analysis(self,
                    remove_dead_frames: bool = True,
                    dead_frame_threshold: float = 3.0,
                    skip_smoothing: bool = False,
                    skip_binarization: bool = False,
                    skip_regions: bool = False, 
                    skip_trace: bool = False) -> None:
        """Run the analysis pipeline without saving any outputs.
        
        This executes all analysis steps in sequence by coordinating between components:
        1. Extract ROI
        2. Smooth volume (optional)
        3. Binarize vessels (optional)
        4. Determine regions (optional)
        5. Trace vessel paths (optional)
        
        Args:
            remove_dead_frames: Whether to remove low-intensity frames at start/end
            dead_frame_threshold: Number of standard deviations above minimum
            skip_smoothing: Whether to skip the smoothing step
            skip_binarization: Whether to skip the binarization step
            skip_regions: Whether to skip the region detection step
            skip_trace: Whether to skip the trace step
        """
        start_time = time.time()
        self._log("Starting analysis pipeline...", level=1)
        
        try:
           
            # 1. Extract ROI using ImageProcessor
            self._log("1. Extracting ROI...", level=1)
            if self.config.find_roi:
                self.roi_model = self.processor.segment_roi(self.image_model)
            else:
                self._log("Using full volume as ROI", level=1)
                self.roi_model = self.image_model
            
            #Normalize? 
            self.roi_model.volume = self.processor.normalize_image(self.roi_model)

            self._log("3. Detrending...", level=1)
            self.roi_model.volume = self.processor.detrend_volume(self.roi_model)
            
            print(f"\tVolume shape: {self.roi_model.volume.shape}")
            # 2. Remove dead frames if requested
            if self.config.remove_dead_frames:
                self._log("2. Removing dead frames...", level=1)
                self.roi_model.volume = self.processor.remove_dead_frames(self.roi_model)
            print(f"\tVolume shape: {self.roi_model.volume.shape}")

            # 3. Background estimation and subtraction using ImageProcessor
            self._log("4. Background estimation...", level=1)
            self.roi_model.background = self.processor.estimate_background(self.roi_model)
            
            # Perform background subtraction in controller
            self._log("4b. Background subtraction...", level=1)
            self.roi_model.volume = self.roi_model.volume - self.roi_model.background
            
            self._log(f"Background subtracted volume range: [{self.roi_model.volume.min():.3f}, {self.roi_model.volume.max():.3f}]", level=2)
            
            # 5. Smooth volume using ImageProcessor
            if not skip_smoothing:
                self._log("5. Smoothing volume...", level=1)
                self.roi_model.volume = self.processor.smooth_volume(self.roi_model)
            
            # 6. Binarize vessels using ImageProcessor
            if not skip_binarization:
                self._log("6. Binarizing vessels...", level=1)
                self.roi_model.binary = self.processor.binarize_volume(self.roi_model)
            
            #Lets try a 3D closing (some larger vessels are not closing properly)
            self.roi_model.binary = self.processor.morphological_closing(self.roi_model)

            # 7. Trace vessel paths using VesselTracer
            if not skip_trace:
                self._log("7. Tracing vessel paths...", level=1)

                # VesselTracer traces and stores paths internally
                self.roi_model.paths, self.roi_model.path_stats, self.roi_model.n_paths = self.tracer.trace_paths(
                    binary_volume=self.roi_model.binary,
                )


            # 8. Determine regions using VesselTracer
            if not skip_regions:
                self._log("8. Determining regions...", level=1)
                
                # Use ImageProcessor to determine regions
                self.roi_model.region_bounds = self.processor.determine_regions(self.roi_model)
                
                print(self.roi_model.region_bounds)
                for region, (peak, sigma, bounds) in self.roi_model.region_bounds.items():
                    self._log(f"\n{region}:", level=2)
                    self._log(f"  Peak position: {peak:.1f}", level=2)
                    self._log(f"  Width (sigma): {sigma:.1f}", level=2)
                    self._log(f"  Bounds: {bounds[0]:.1f} - {bounds[1]:.1f}", level=2)
                
                # Create region map volume using ImageProcessor
                self._log("8b. Creating region map...", level=1)
                self.roi_model.region = self.processor.create_region_map(self.roi_model, self.roi_model.region_bounds)
            
            self._log("Analysis complete", level=1, timing=time.time() - start_time)
            
        except Exception as e:
            self._log(f"Error in analysis pipeline: {str(e)}", level=1)
            raise 

    def update_roi_position(self, min_x: int, min_y: int, micron_roi: Optional[float] = None) -> None:
        """Update the ROI minimum position and optionally its size.
        
        This method updates the ROI parameters and clears any previously computed results
        so they will be recomputed with the new ROI on next access.
        
        Args:
            min_x: New X coordinate of ROI minimum in pixels
            min_y: New Y coordinate of ROI minimum in pixels
            micron_roi: Optional new ROI size in microns. If None, keeps current size.
        """
        print(f"\nUpdating ROI parameters:")
        print(f"  Min position: ({self.config.min_x}, {self.config.min_y}) -> ({min_x}, {min_y})")
        
        self.config.min_x = min_x
        self.config.min_y = min_y
        
        if micron_roi is not None:
            print(f"  Size: {self.config.micron_roi} -> {micron_roi} microns")
            self.config.micron_roi = micron_roi
        
        # Clear computed results in the components
        self.processor.clear_processed_data()
        self.tracer.clear_all_data()
                
        print("ROI updated. Next pipeline step will use new parameters.")

    def save_volume(self,
                     output_dir: str,
                     volume_type: str,
                     source: Optional[str] = 'roi',
                     filename: Optional[str] = None,
                     depth_coded: bool = False) -> None:
        """Save a specific volume type as a .tif file.
        
        Args:
            output_dir: Directory to save the volume to
            volume_type: Type of volume to save. Options:
                - 'volume': Original volume from the model
                - 'binary': Binary volume from ROI model
                - 'background': Background subtracted volume from ROI model
                - 'region': Region map from ROI model
            source: Source of the volume to save. Options:
                - 'roi': ROI volume from ROI model
                - 'image': Original volume from image model
            filename: Optional custom filename. If not provided, will use volume_type_volume.tif
            depth_coded: If True, creates depth-coded volume where intensity represents z-position
                        (only works with binary volume type)
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if source == 'roi':
            model = self.roi_model
        elif source == 'image':
            model = self.image_model
        else:
            raise ValueError(f"Invalid source: {source}. Must be one of: roi, image")

        # Get volume data based on type
        if volume_type == 'volume':
            if model is None or model.volume is None:
                raise ValueError("Model or volume not available")
            volume_data = model.volume
            
        elif volume_type == 'binary':
            if model is None or model.binary is None:
                raise ValueError("Model or binary volume not available")
            volume_data = model.binary.astype(np.uint8)
            
        elif volume_type == 'background':
            if model is None or model.background is None:
                raise ValueError("Model or background volume not available")
            volume_data = model.background
            
        elif volume_type == 'region':
            if model is None or model.region is None:
                raise ValueError("Model or region map not available")
            volume_data = model.region
            
        else:
            raise ValueError(f"Invalid volume type: {volume_type}. Must be one of: volume, roi, binary, background, region")
            
        # Handle depth coding if requested
        if depth_coded:
            if volume_type != 'binary':
                raise ValueError("Depth coding is only available for binary volumes")
            
            # Create depth-coded volume
            Z, Y, X = volume_data.shape
            depth_vol = np.zeros_like(volume_data, dtype=float)
            for z in range(Z):
                depth_vol[z] = volume_data[z] * z
                
            # Normalize depth values to [0,1]
            depth_vol = depth_vol / (Z-1) if depth_vol.max() > 0 else depth_vol
            volume_data = depth_vol
            
        # Set filename
        if filename is None:
            filename = f"{volume_type}_volume.tif"
            if depth_coded:
                filename = f"{volume_type}_depth_coded.tif"
        elif not filename.endswith('.tif'):
            filename = f"{filename}.tif"
            
        # Save volume
        output_file = output_path / filename
        print(f"Saving {volume_type} volume to {output_file}")
        tifffile.imwrite(str(output_file), volume_data)
        print(f"Volume saving complete! Shape: {volume_data.shape}, dtype: {volume_data.dtype}")

    def generate_analysis_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Generate pandas DataFrames containing analysis results.
        
        Creates DataFrames by delegating to the appropriate components.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all generated DataFrames
        """
        # Create metadata DataFrame using config method
        pixel_sizes = self.processor.get_pixel_sizes() if self.processor.has_volume() else None
        input_type = 'Numpy Array' if isinstance(self.input_data, np.ndarray) else 'File'
        
        # Create processing status dictionary by checking component states
        processing_status = {
            'ROI Extraction': self.processor.has_roi(),
            'Background Subtraction': self.processor.has_background(),
            'Smoothing': True,  # Assumed if not skipped
            'Binarization': self.processor.has_binary(),
            'Region Detection': self.tracer.has_regions(),
            'Path Tracing': self.tracer.has_paths(),
            'Region Map Creation': self.tracer.has_region_map()
        }
        
        # Add analysis-specific metadata
        if self.processor.has_roi():
            roi_shape = self.processor.get_roi_shape()
            processing_status['ROI Shape'] = str(roi_shape)
        
        metadata_df = self.config.generate_metadata_df(pixel_sizes, input_type, processing_status)
        
        # Create region bounds DataFrame if available (delegate to tracer)
        regions_df = self.tracer.get_regions_dataframe()
        
        # Create z-profile DataFrame (delegate to processor)
        z_profile_df = self.processor.get_z_profile_dataframe()
        
        # Create vessel paths DataFrames (delegate to tracer)
        if self.tracer.has_paths():
            pixel_sizes = self.processor.get_pixel_sizes()
            paths_df = self.tracer.create_paths_dataframe(pixel_sizes)
            path_summary_df = self.tracer.create_path_summary_dataframe(pixel_sizes)
        else:
            paths_df = pd.DataFrame()
            path_summary_df = pd.DataFrame()
        
        # Return DataFrames dictionary (don't store in Controller)
        analysis_dfs = {
            'metadata': metadata_df,
            'regions': regions_df,
            'z_profile': z_profile_df,
            'paths': paths_df,
            'path_summary': path_summary_df
        }
        
        return analysis_dfs

    def save_analysis_to_excel(self, output_path: Union[str, Path]) -> None:
        """Save all analysis DataFrames to an Excel file.
        
        Args:
            output_path: Path where to save the Excel file
        """
        output_path = Path(output_path)
        
        # Generate DataFrames on-demand
        analysis_dfs = self.generate_analysis_dataframes()
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Save each DataFrame to a different sheet
            analysis_dfs['metadata'].to_excel(writer, sheet_name='Metadata', index=False)
            if not analysis_dfs['regions'].empty:
                analysis_dfs['regions'].to_excel(writer, sheet_name='Region Analysis', index=False)
            if not analysis_dfs['z_profile'].empty:
                analysis_dfs['z_profile'].to_excel(writer, sheet_name='Z Profile', index=False)
            if not analysis_dfs['paths'].empty:
                analysis_dfs['paths'].to_excel(writer, sheet_name='Vessel Paths Detailed', index=False)
            if not analysis_dfs['path_summary'].empty:
                analysis_dfs['path_summary'].to_excel(writer, sheet_name='Vessel Paths Summary', index=False)

    def create_paths_dataframe(self, pixel_sizes: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Create a pandas DataFrame from the traced paths.
        
        Args:
            pixel_sizes: Optional dictionary of pixel sizes in microns for coordinate conversion
            
        Returns:
            DataFrame with detailed path information
        """
        if self.roi_model is None or self.roi_model.paths is None:
            return pd.DataFrame()
            
        vessel_paths_data = []
        
        for path_id, path_info in self.roi_model.paths.items():
            coords = path_info['coordinates']
            length = path_info['length']
            
            # Get region for each point
            regions = [self.processor._get_region_for_z(z, self.roi_model.region_bounds) 
                      for z in coords[:, 0]]
            
            # Convert coordinates if pixel sizes provided
            if pixel_sizes:
                z_scale = pixel_sizes.get('z', 1.0)
                y_scale = pixel_sizes.get('y', 1.0)
                x_scale = pixel_sizes.get('x', 1.0)
            else:
                z_scale = y_scale = x_scale = 1.0
            
            # Convert path coordinates to DataFrame rows
            for i, coord in enumerate(coords):
                vessel_paths_data.append({
                    'Path_ID': path_id,
                    'Point_Index': i,
                    'Primary_Region': regions[i],
                    'Z': coord[0],  # Z coordinate in pixels
                    'Y': coord[1],  # Y coordinate in pixels  
                    'X': coord[2],  # X coordinate in pixels
                    'Z_microns': coord[0] * z_scale,  # Z in microns
                    'Y_microns': coord[1] * y_scale,  # Y in microns
                    'X_microns': coord[2] * x_scale,  # X in microns,
                    'Path_Length': length
                })
                
        return pd.DataFrame(vessel_paths_data)
    
    def create_path_summary_dataframe(self, pixel_sizes: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Create a summary DataFrame with one row per path.
        
        Args:
            pixel_sizes: Optional dictionary of pixel sizes in microns for coordinate conversion
            
        Returns:
            DataFrame with path summary information
        """
        if self.roi_model is None or self.roi_model.paths is None:
            return pd.DataFrame()
            
        path_summary_data = []
        
        # Convert pixel sizes if provided
        if pixel_sizes:
            z_scale = pixel_sizes.get('z', 1.0)
            y_scale = pixel_sizes.get('y', 1.0)
            x_scale = pixel_sizes.get('x', 1.0)
        else:
            z_scale = y_scale = x_scale = 1.0
        
        for path_id, path_info in self.roi_model.paths.items():
            coords = path_info['coordinates']
            length = path_info['length']
            
            # Get regions for start and end points
            start_region = self.processor._get_region_for_z(coords[0, 0], self.roi_model.region_bounds)
            end_region = self.processor._get_region_for_z(coords[-1, 0], self.roi_model.region_bounds)
            
            # Get all regions traversed
            regions = [self.processor._get_region_for_z(z, self.roi_model.region_bounds) 
                      for z in coords[:, 0]]
            unique_regions = list(set(regions))
            
            # Calculate path length in microns if pixel sizes provided
            if pixel_sizes:
                length_microns = length * np.mean([x_scale, y_scale, z_scale])  # Approximate length conversion
            else:
                length_microns = length
            
            # Create summary entry for this path
            summary_entry = {
                'Path_ID': path_id,
                'Primary_Region': start_region,  # Use start region as primary
                'Path_Length_pixels': length,
                'Path_Length_microns': length_microns,
                'Start_Z': coords[0][0],
                'End_Z': coords[-1][0],
                'Start_Z_microns': coords[0][0] * z_scale,
                'End_Z_microns': coords[-1][0] * z_scale,
                'Num_Points': len(coords),
                'Start_Region': start_region,
                'End_Region': end_region,
                'Regions_Traversed': unique_regions
            }
            
            path_summary_data.append(summary_entry)
            
        return pd.DataFrame(path_summary_data)
    
    def multiscan(self, 
                  skip_smoothing: bool = False, 
                  skip_binarization: bool = False, 
                  skip_regions: bool = False, 
                  skip_trace: bool = False
                ) -> List[Dict[str, Any]]:
        """Scan multiple ROIs across the volume.
        
        Performs systematic scanning across the volume using the configured ROI size.
        For each ROI position, runs the full analysis pipeline and stores results.
        
        Args:
            skip_smoothing: Whether to skip the smoothing step
            skip_binarization: Whether to skip the binarization step
            skip_regions: Whether to skip the region detection step
            skip_trace: Whether to skip the trace step
            
        Returns:
            List of dictionaries containing ROI results
        """
        roi_results = []
        micron_roi = self.config.micron_roi
        
        # Calculate scan ranges based on volume dimensions and ROI size
        volume_shape = self.processor.get_volume_shape()
        pixel_sizes = self.processor.get_pixel_sizes()
        avg_pixel_size = (pixel_sizes[1] + pixel_sizes[2]) / 2  # Y and X pixel sizes
        pixel_roi = int(micron_roi / avg_pixel_size)
        
        # Get volume dimensions
        _, Y, X = volume_shape
        
        # Create scan ranges with ROI-sized steps
        xscan_rng = range(0, X - pixel_roi + 1, pixel_roi)
        yscan_rng = range(0, Y - pixel_roi + 1, pixel_roi)
        
        self._log(f"Setting up multiscan with {len(xscan_rng)}x{len(yscan_rng)} ROIs", level=2)
        self._log(f"ROI size: {pixel_roi} pixels ({micron_roi} microns)", level=2)

        # Iterate through ROI positions
        for y in yscan_rng:
            for x in xscan_rng:
                self._log(f"Processing ROI at position ({x}, {y})", level=2)
                
                # Update ROI position (this will clear previous results)
                self.update_roi_position(x, y)

                # Run analysis on this ROI
                try:
                    self.run_analysis(
                        skip_smoothing=skip_smoothing,
                        skip_binarization=skip_binarization, 
                        skip_regions=skip_regions,
                        skip_trace=skip_trace
                    )
                    
                    # Skip if no valid ROI was found
                    if not self.processor.has_roi():
                        self._log(f"Skipping ROI at ({x}, {y}) - no valid frames found", level=1)
                        continue

                    # Convert local path coordinates to global coordinates
                    paths_copy = {}
                    for path_id, path_info in self.tracer.get_paths().items():
                        if 'coordinates' in path_info:
                            coords = path_info['coordinates'].copy()
                            # Add ROI offset to convert local to global coordinates
                            coords[:, 1] += x  # X coordinate
                            coords[:, 2] += y  # Y coordinate
                            
                            # Create updated path info
                            path_copy = path_info.copy()
                            path_copy['coordinates'] = coords
                            paths_copy[path_id] = path_copy

                    # Store results with position info
                    roi_data = {
                        'center_x': x + pixel_roi//2,
                        'min_x': x,
                        'max_x': x + pixel_roi,
                        'center_y': y + pixel_roi//2,
                        'min_y': y,
                        'max_y': y + pixel_roi,
                        'paths': paths_copy
                    }
                    roi_results.append(roi_data)
                    
                    self._log(f"Completed ROI analysis at ({x}, {y})", level=2)
                except Exception as e:
                    self._log(f"Error processing ROI at ({x}, {y}): {str(e)}", level=1)
                    continue
        
        self._log(f"Completed multiscan analysis of {len(roi_results)} ROIs", level=1)
        return roi_results