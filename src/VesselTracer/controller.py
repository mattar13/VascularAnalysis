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
    
    def __init__(self, input_data: Union[str, Path, np.ndarray],
                 config_path: Optional[Union[str, Path]] = None,
                 pixel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """Initialize VesselAnalysisController with input data and configuration.
        
        Args:
            input_data: Either a path to a CZI/TIF file (as string or Path object) or a 3D numpy array
            config_path: Optional path to YAML config file. If None, uses default config
            pixel_sizes: Tuple of (z,y,x) pixel sizes in microns. Defaults to (1.0, 1.0, 1.0)
        """
        # Store input parameters for reference
        self.input_data = input_data
        self.pixel_sizes = pixel_sizes
        
        # Initialize configuration first
        self.config = VesselTracerConfig(config_path=config_path)
        
        # Create ImageModel based on input type
        if isinstance(input_data, np.ndarray):
            # Handle numpy array input
            self.image_model = ImageModel(
                input_data=input_data,
                pixel_sizes=pixel_sizes
            )
        else:
            # Handle filepath input (string or Path)
            self.image_model = ImageModel(
                filepath=input_data,
                pixel_sizes=pixel_sizes
            )
        self.roi_model = None #Load this if we decide to use segmentation

        # Initialize processing components with config only
        self.processor = ImageProcessor(
            config=self.config,  # Pass config directly
            verbose=True,  # Can be made configurable
            use_gpu=False  # Will be activated separately if needed
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

    def get_projection(self, 
                      axis: Union[int, List[int]], 
                      operation: str = 'mean', 
                      volume_type: str = 'binary',
                      z_range: Optional[Tuple[int, int]] = None,
                      y_range: Optional[Tuple[int, int]] = None,
                      x_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Generate a projection of the specified volume along specified axis/axes.
        
        Args:
            axis: Dimension(s) to project along (0=z, 1=y, 2=x, [1,2]=xy)
            operation: Type of projection ('max', 'min', 'mean', 'std')
            volume_type: Type of volume to project ('binary', 'background', 'volume', 'region_map')
            z_range: Optional tuple of (start, end) for z dimension
            y_range: Optional tuple of (start, end) for y dimension  
            x_range: Optional tuple of (start, end) for x dimension
                
        Returns:
            np.ndarray: Projected image
        """
        # Delegate projection generation to appropriate component
        if volume_type in ['binary', 'volume', 'background']:
            # Delegate to processor for volume-based projections
            return self.processor.get_projection(
                axis=axis, 
                operation=operation, 
                volume_type=volume_type,
                z_range=z_range, 
                y_range=y_range, 
                x_range=x_range
            )
        elif volume_type in ['region_map', 'region']:
            # Delegate to tracer for region-based projections
            return self.tracer.get_projection(
                axis=axis, 
                operation=operation, 
                volume_type='region_map',
                z_range=z_range, 
                y_range=y_range, 
                x_range=x_range
            )
        else:
            raise ValueError(f"volume_type must be one of ['binary', 'background', 'volume', 'region_map']")

    def run_analysis(self,
                    remove_dead_frames: bool = True,
                    dead_frame_threshold: float = 1.5,
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
                self.roi_model = self.processor.segment_roi(
                    image_model=self.image_model,
                    remove_dead_frames=remove_dead_frames, 
                    dead_frame_threshold=dead_frame_threshold
                )
            else:
                self._log("Using full volume as ROI", level=1)
                self.roi_model = self.image_model
                
            # 2. Background subtraction using ImageProcessor
            self._log("2. Background subtraction...", level=1)
            self.processor.median_filter_background_subtraction(self.roi_model)

            # 3. Detrend using ImageProcessor
            self._log("3. Detrending...", level=1)
            self.processor.detrend_volume(self.roi_model)
            
            # 4. Smooth volume using ImageProcessor
            if not skip_smoothing:
                self._log("4. Smoothing volume...", level=1)
                self.processor.smooth_volume(self.roi_model)
            
            # 5. Binarize vessels using ImageProcessor
            if not skip_binarization:
                self._log("5. Binarizing vessels...", level=1)
                self.processor.binarize_volume(self.roi_model)
            
            # 6. Determine regions using VesselTracer
            if not skip_regions:
                self._log("6. Determining regions...", level=1)
                
                # Get binary volume from processor and pass to tracer
                binary_volume = self.processor.binarize_volume(self.roi_model)
                if binary_volume is None:
                    raise ValueError("No binary volume available for region determination")
                
                # VesselTracer determines and stores regions internally
                region_bounds = self.tracer.determine_regions(binary_volume)
                for region, (peak, sigma, bounds) in region_bounds.items():
                    self._log(f"\n{region}:", level=2)
                    self._log(f"  Peak position: {peak:.1f}", level=2)
                    self._log(f"  Width (sigma): {sigma:.1f}", level=2)
                    self._log(f"  Bounds: {bounds[0]:.1f} - {bounds[1]:.1f}", level=2)
                
                # Create region map volume using VesselTracer
                self._log("6b. Creating region map volume...", level=1)
                self.tracer.create_region_map_volume(binary_volume, region_bounds)
            
            # 7. Trace vessel paths using VesselTracer
            if not skip_trace:
                self._log("7. Tracing vessel paths...", level=1)
                
                # Get binary volume from processor and pass to tracer
                binary_volume = self.processor.binarize_volume(self.roi_model)
                if binary_volume is None:
                    raise ValueError("No binary volume available for path tracing")
                
                # VesselTracer traces and stores paths internally
                paths, stats = self.tracer.trace_paths(
                    binary_volume=binary_volume,
                    region_bounds=None,  # VesselTracer will use its stored region_bounds
                    split_paths=False  # Can be made configurable
                )
            
            self._log("Analysis complete", level=1, timing=time.time() - start_time)
            
        except Exception as e:
            self._log(f"Error in analysis pipeline: {str(e)}", level=1)
            raise 
      
    def run_pipeline(self,
                    output_dir: Union[str, Path],
                    # Analysis steps
                    skip_smoothing: bool = False,
                    skip_binarization: bool = False,
                    skip_regions: bool = False,
                    skip_trace: bool = False,
                    # DataFrame generation
                    skip_dataframe: bool = False,
                    # Volume saving options
                    save_volumes: bool = True,
                    save_original: bool = True,
                    save_binary: bool = True,
                    save_region_map: bool = True,
                    save_separate: bool = False,
                    # ROI options
                    remove_dead_frames: bool = True,
                    dead_frame_threshold: float = 1.5) -> None:
        """Run the complete analysis pipeline and save all outputs.
        
        Args:
            output_dir: Directory to save outputs
            skip_smoothing: Whether to skip the smoothing step
            skip_binarization: Whether to skip the binarization step
            skip_regions: Whether to skip the region detection step
            skip_trace: Whether to skip the trace step
            skip_dataframe: Whether to skip generating and saving DataFrames
            save_volumes: Whether to save any volumes
            save_original: Whether to save the original volume
            save_binary: Whether to save the binary volume
            save_region_map: Whether to save the region map volume
            save_separate: Whether to save volumes separately
            remove_dead_frames: Whether to remove low-intensity frames at start/end
            dead_frame_threshold: Number of standard deviations above minimum
        """
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._log(f"Starting analysis pipeline...", level=1)
        self._log(f"Output directory: {output_dir}", level=2)
        
        # Run the analysis
        self.run_analysis(
            remove_dead_frames=remove_dead_frames,
            dead_frame_threshold=dead_frame_threshold,
            skip_smoothing=skip_smoothing,
            skip_binarization=skip_binarization,
            skip_regions=skip_regions,
            skip_trace=skip_trace,
        )
       
        # Generate and save DataFrames if not skipped
        if not skip_dataframe:
            self._log("8. Generating analysis DataFrames...", level=1)
            analysis_dfs = self.generate_analysis_dataframes()
            excel_path = output_dir / 'analysis_results.xlsx'
            self._log(f"Saving DataFrames to {excel_path}...", level=2)
            self.save_analysis_to_excel(excel_path)
        
        # Save volumes if requested
        if save_volumes:
            self._log("9. Saving volumes...", level=1)
            self.save_volume(
                output_dir,
                save_original=save_original,
                save_binary=save_binary,
                save_region_map=save_region_map,
                save_separate=save_separate
            )
            
        self._log("Pipeline complete!", level=1, timing=time.time() - start_time)

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
                   save_original: bool = True,
                   save_binary: bool = True,
                   save_region_map: bool = True,
                   save_separate: bool = False) -> None:
        """Save volume data as .tif files.
        
        Args:
            output_dir: Directory to save the .tif files
            save_original: Whether to save the original ROI volume
            save_binary: Whether to save the binary volume
            save_region_map: Whether to save the region map volume
            save_separate: If True, saves each volume type as a separate file.
                          If False, saves all volumes in a single multi-channel file.
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if we have the volumes we want to save (delegate to components)
        if save_binary and not self.processor.has_binary():
            raise ValueError("Binary volume not available. Run analysis pipeline first.")
        if save_region_map and not self.tracer.has_region_map():
            raise ValueError("Region map volume not available. Run analysis pipeline with regions enabled first.")
            
        # Prepare volumes for saving (get from components)
        volumes = []
        volume_names = []
        
        if save_original:
            # Get volume from processor
            original_data = self.processor.get_current_volume()
            volumes.append(original_data)
            volume_names.append('original')
        if save_binary:
            binary_data = self.processor.get_binary_volume()
            volumes.append(binary_data.astype(np.uint8))  # Convert boolean to uint8
            volume_names.append('binary')
        if save_region_map:
            region_data = self.tracer.get_region_map_volume()
            volumes.append(region_data)  # Already uint8
            volume_names.append('region_map')
            
        if not volumes:
            print("No volumes selected for saving!")
            return
            
        if save_separate:
            # Save each volume as a separate file
            for vol, name in zip(volumes, volume_names):
                output_file = output_path / f"{name}_volume.tif"
                print(f"Saving {name} volume to {output_file}")
                tifffile.imwrite(str(output_file), vol)
        else:
            # Save all volumes in a single multi-channel file
            # Stack volumes along a new channel dimension
            stacked_volumes = np.stack(volumes, axis=0)
            output_file = output_path / "all_volumes.tif"
            print(f"Saving all volumes to {output_file}")
            tifffile.imwrite(str(output_file), stacked_volumes)
            
        print("Volume saving complete!")

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

    # Properties that delegate to component objects (no storage in Controller)
    @property
    def volume(self) -> Optional[np.ndarray]:
        """Get the current processed volume from processor."""
        return self.processor.get_current_volume()
    
    @property
    def binary(self) -> Optional[np.ndarray]:
        """Get the binary volume from processor."""
        return self.processor.get_binary_volume()
    
    @property
    def region_map(self) -> Optional[np.ndarray]:
        """Get the region map volume from tracer."""
        return self.tracer.get_region_map_volume()
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Get the traced paths from tracer."""
        return self.tracer.get_paths()
    
    @property
    def path_count(self) -> int:
        """Get the number of traced paths from tracer."""
        return self.tracer.get_path_count()
    
    @property
    def region_bounds(self) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
        """Get the region bounds from tracer."""
        return self.tracer.get_region_bounds()
    
    @property
    def background(self) -> Optional[np.ndarray]:
        """Get the background volume from processor."""
        return self.processor.get_background_volume() 