from pathlib import Path
import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize
from skan import Skeleton, summarize
import yaml
from typing import Optional, Dict, Any, Tuple, Union, List
import tifffile
import time
from datetime import datetime
import pandas as pd

# Try to import CuPy
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    cp_ndimage = None

# Import the ImageProcessor for delegating processing tasks
from .image_processor import ImageProcessor
from .config import VesselTracerConfig
from .image_model import ImageModel

class VesselTracer:
    """Main class for vessel tracing pipeline.
    
    Attributes:
        input_data: Either a path to a CZI/TIF file (as string or Path object) or a 3D numpy array
        config_path: Optional path to YAML config file. If None, uses default config.
        pixel_sizes: Optional tuple of (z,y,x) pixel sizes in microns. Defaults to (1.0, 1.0, 1.0).
    """
      
    #Initialize the class and load image and config
    def __init__(self, input_data: Union[str, Path, np.ndarray],
                 config_path: Optional[Union[str, Path]] = None,
                 pixel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """Initialize VesselTracer with input data and configuration.
        
        Args:
            input_data: Either a path to a CZI/TIF file (as string or Path object) or a 3D numpy array
            config_path: Optional path to YAML config file. If None, uses default config
            pixel_sizes: Tuple of (z,y,x) pixel sizes in microns. Defaults to (1.0, 1.0, 1.0)
        """
        self.input_data = input_data
        self.pixel_sizes = pixel_sizes
        
        # Convert string paths to Path objects
        if isinstance(self.input_data, str):
            self.input_data = Path(self.input_data)
            
        # Initialize GPU mode as disabled
        self.gpu_available = GPU_AVAILABLE
        self.use_gpu = False
        
        # Initialize config
        self.config = VesselTracerConfig()
        self.config.load_config(config_path)
        
        # Initialize ImageModel for storing full volume data
        self.image_model = ImageModel()
        
        # Initialize ROI for region-specific processing
        self.roi_model = None
        
        # Initialize ImageProcessor for all processing operations
        self.processor = ImageProcessor(
            config=self.config,
            verbose=self.config.verbose,
            use_gpu=False  # Will be activated separately if needed
        )
        
        # Load image and convert config parameters to pixels
        self._load_image()
        self.image_model.volume = self.image_model.volume.astype("float32")
        self._convert_to_pixels()

        # Initialize other processing volumes
        self.background = None
        
    def _convert_to_pixels(self):
        """Convert micron-based parameters to pixel values using image resolution."""
        if self.image_model.pixel_size_x == 0 or self.image_model.pixel_size_y == 0 or self.image_model.pixel_size_z == 0:
            raise ValueError("Pixel sizes not set. Call _load_image first.")
            
        # Use config method to convert parameters
        pixel_sizes = self.image_model.get_pixel_sizes()
        pixel_conversions = self.config.convert_to_pixels(pixel_sizes)
        
        # Store converted values as instance attributes for easy access
        self.gauss_sigma_x = pixel_conversions['gauss_sigma_x']
        self.gauss_sigma_y = pixel_conversions['gauss_sigma_y']
        self.gauss_sigma_z = pixel_conversions['gauss_sigma_z']
        self.gauss_sigma = pixel_conversions['gauss_sigma']
        self.background_sigma_x = pixel_conversions['background_sigma_x']
        self.background_sigma_y = pixel_conversions['background_sigma_y']
        self.background_sigma_z = pixel_conversions['background_sigma_z']
        self.background_sigma = pixel_conversions['background_sigma']
        self.median_filter_size = pixel_conversions['median_filter_size']
        self.close_radius = pixel_conversions['close_radius']
        
        self._log(f"Converted parameters to pixels:", level=2)
        self._log(f"  Regular smoothing sigma (µm): {self.config.micron_gauss_sigma:.1f}", level=2)
        self._log(f"  Regular smoothing sigma (pixels):", level=2)
        self._log(f"    X: {self.gauss_sigma_x:.1f}", level=2)
        self._log(f"    Y: {self.gauss_sigma_y:.1f}", level=2)
        self._log(f"    Z: {self.gauss_sigma_z:.1f}", level=2)
        self._log(f"  Background smoothing sigma (µm): {self.config.micron_background_sigma:.1f}", level=2)
        self._log(f"  Background smoothing sigma (pixels):", level=2)
        self._log(f"    X: {self.background_sigma_x:.1f}", level=2)
        self._log(f"    Y: {self.background_sigma_y:.1f}", level=2)
        self._log(f"    Z: {self.background_sigma_z:.1f}", level=2)
        self._log(f"  Median filter size: {self.config.micron_median_filter_size:.1f} µm -> {self.median_filter_size} pixels", level=2)
        self._log(f"  Closing radius: {self.config.micron_close_radius:.1f} µm -> {self.close_radius} pixels", level=2)

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

    def _load_image(self):
        """Load image data from file or array."""
        start_time = time.time()
        self._log("Loading image data...", level=1)
        
        self._log(f"Input data type: {type(self.input_data)}", level=2)

        if isinstance(self.input_data, (str, Path)):
            # Handle file input
            file_path = Path(self.input_data)
            if file_path.suffix.lower() == '.czi':
                self.image_model.load_from_czi(file_path)
            elif file_path.suffix.lower() in ['.tif', '.tiff']:
                self.image_model.load_from_tiff(file_path, self.pixel_sizes)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        elif isinstance(self.input_data, np.ndarray):
            # Handle array input
            if self.pixel_sizes is None:
                raise ValueError("pixel_sizes must be provided when input_data is an array")
            self.image_model.load_from_array(self.input_data, self.pixel_sizes)
        else:
            raise ValueError("input_data must be either a file path (string or Path) or a numpy array")
        
        # Convert micron parameters to pixels
        self._convert_to_pixels()
        
        # Normalize volume
        self._log(f"Image loaded. Shape: {self.image_model.volume.shape}", level=2)
        self._log(f"Pixel sizes (µm): X={self.image_model.pixel_size_x:.3f}, Y={self.image_model.pixel_size_y:.3f}, Z={self.image_model.pixel_size_z:.3f}", level=2)
        self._log("Image loading complete", level=1, timing=time.time() - start_time)

    def activate_gpu(self) -> bool:
        """Activate GPU mode for processing.
        
        Returns:
            bool: True if GPU mode was successfully activated, False otherwise.
        """
        # Delegate GPU activation to the processor
        success = self.processor.activate_gpu()
        self.use_gpu = success
        return success

    def print_config(self):
        """Print current configuration."""
        pixel_sizes = self.image_model.get_pixel_sizes() if self.image_model.volume is not None else None
        self.config.print_config(pixel_sizes)

    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to a YAML file.
        
        Args:
            output_path: Path where to save the YAML configuration file
        """
        pixel_sizes = self.image_model.get_pixel_sizes() if self.image_model.volume is not None else None
        self.config.save_config(output_path, pixel_sizes)
    
    def save_metadata(self, output_path: Union[str, Path], format: str = 'excel') -> None:
        """Save metadata to file.
        
        Args:
            output_path: Path where to save the metadata
            format: Output format ('excel', 'csv', 'json')
        """
        pixel_sizes = self.image_model.get_pixel_sizes() if self.image_model.volume is not None else None
        input_type = 'Numpy Array' if isinstance(self.input_data, np.ndarray) else 'File'
        
        # Create processing status dictionary
        processing_status = {
            'ROI Extraction': self.roi_model is not None and self.roi_model.volume is not None,
            'Background Subtraction': hasattr(self, 'background'),
            'Smoothing': hasattr(self, 'gauss_sigma'),
            'Binarization': self.image_model.binary is not None,
            'Region Detection': hasattr(self, 'region_bounds'),
            'Path Tracing': len(self.image_model.paths) > 0,
            'Region Map Creation': self.image_model.region is not None
        }
        
        self.config.save_metadata(output_path, pixel_sizes, input_type, processing_status, format)
    
    #Now we can start the functions used to process the images




    #These functions are now used to trace the paths
    
    def trace_paths(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create vessel skeleton and trace paths.
            
        Returns:
            Tuple containing:
            - paths: Dictionary of branch paths with coordinates
            - stats: DataFrame with branch statistics
        """
        start_time = time.time()
        self._log("Tracing vessel paths...", level=1)
        
        if self.image_model.binary is None:
            self.binarize()
            
        self._log(f"Skeletonizing binary volume of shape {self.image_model.binary.shape}", level=2)
        ske = sk_skeletonize(self.image_model.binary)
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
                    
                # Store path without region information for now
                # Region analysis will be handled by ImageProcessor if needed
                paths_dict[path_id] = {
                    'original_path_id': i,
                    'coordinates': path_coords,
                    'length': len(path_coords),
                    'z_range': (float(path_coords[:, 0].min()), float(path_coords[:, 0].max())) if len(path_coords) > 0 else (0, 0)
                }
                path_id += 1
                    
            except Exception as e:
                self._log(f"Warning: Error processing path {i}: {str(e)}", level=1)
                continue
        
        # Store the converted paths and update counts
        self.image_model.paths = paths_dict
        self.roi_model.paths = paths_dict.copy() if self.roi_model else {}
        self.n_paths = len(paths_dict)
        
        self._log(f"Converted skeleton to dictionary with {self.n_paths} vessel paths", level=2)
            
        self._log("Path tracing complete", level=1, timing=time.time() - start_time)    
        return self.image_model.paths, self.stats

    #These functions are used for convenience functions to help with plotting

    def get_projection(self, 
                      axis: Union[int, List[int]], 
                      operation: str = 'mean', 
                      volume_type: str = 'binary',
                      z_range: Optional[Tuple[int, int]] = None,
                      y_range: Optional[Tuple[int, int]] = None,
                      x_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Generate a projection of the specified volume along specified axis/axes.
        
        This is a wrapper method that delegates to the appropriate ImageModel's get_projection method.
        
        Args:
            axis: Dimension(s) to project along. Options:
                - 0: z-projection (xy view)
                - 1: y-projection (xz view)
                - 2: x-projection (yz view)
                - [1,2]: xy-projection (z view)
            operation: Type of projection. Options:
                - 'max': Maximum intensity projection
                - 'min': Minimum intensity projection
                - 'mean': Average intensity projection
                - 'std': Standard deviation projection
            volume_type: Type of volume to project. Options:
                - 'binary': Binary volume (default)
                - 'background': Background volume from median filtering
                - 'volume': Current processed volume
                - 'region_map': Region map volume with region labels
            z_range: Optional tuple of (start, end) for z dimension
            y_range: Optional tuple of (start, end) for y dimension
            x_range: Optional tuple of (start, end) for x dimension
                
        Returns:
            np.ndarray: Projected image
            
        Raises:
            ValueError: If invalid axis, operation, or volume_type specified
        """
        # Determine which model to use based on volume_type and context
        if volume_type in ['binary', 'region_map', 'region']:
            # Use image_model for binary and region data
            model = self.image_model
            
            # Ensure binary data is available
            if volume_type == 'binary' and model.binary is None:
                raise ValueError("Binary volume not available. Run analysis pipeline first.")
            
            # Ensure region data is available  
            if volume_type in ['region_map', 'region'] and model.region is None:
                raise ValueError("Region map volume not available. Run analysis pipeline with regions enabled first.")
                
            # Map volume_type names for consistency
            model_volume_type = 'region' if volume_type == 'region_map' else volume_type
            
            return model.get_projection(
                axis=axis, 
                operation=operation, 
                volume_type=model_volume_type,
                z_range=z_range, 
                y_range=y_range, 
                x_range=x_range
            )
            
        elif volume_type == 'volume':
            # Use ROI model if available, otherwise use image model
            model = self.roi_model if self.roi_model is not None else self.image_model
            
            return model.get_projection(
                axis=axis, 
                operation=operation, 
                volume_type='volume',
                z_range=z_range, 
                y_range=y_range, 
                x_range=x_range
            )
            
        elif volume_type == 'background':
            # Background volume is stored in VesselTracer, use image_model with background parameter
            if not hasattr(self, 'background'):
                raise ValueError("Background volume not available. Run analysis pipeline first.")
                
            model = self.roi_model if self.roi_model is not None else self.image_model
            
            return model.get_projection(
                axis=axis, 
                operation=operation, 
                volume_type='background',
                background_volume=self.background,
                z_range=z_range, 
                y_range=y_range, 
                x_range=x_range
            )
            
        else:
            raise ValueError(f"volume_type must be one of ['binary', 'background', 'volume', 'region_map']")



    #These are pipeline functions used to run the analysis
    def run_analysis(self,
                    remove_dead_frames: bool = True,
                    dead_frame_threshold: float = 1.5,
                    skip_smoothing: bool = False,
                    skip_binarization: bool = False,
                    skip_regions: bool = False, 
                    skip_trace: bool = False) -> None:
        """Run the analysis pipeline without saving any outputs.
        
        This executes all analysis steps in sequence but skips visualization and saving:
        1. Extract ROI
        2. Smooth volume (optional)
        3. Binarize vessels (optional)
        4. Create skeleton
        5. Determine regions (optional)
        
        Args:
            skip_smoothing: Whether to skip the smoothing step
            skip_binarization: Whether to skip the binarization step
            skip_regions: Whether to skip the region detection step
            skip_trace: Whether to skip the trace step
        """
        start_time = time.time()
        self._log("Starting analysis pipeline...", level=1)
        
        try:
            # Extract ROI using ImageProcessor
            self._log("1. Extracting ROI...", level=1)
            self.roi_model = self.processor.segment_roi(
                self.image_model,
                remove_dead_frames=remove_dead_frames, 
                dead_frame_threshold=dead_frame_threshold
            )
            
            if self.roi_model is None:
                self._log("No valid ROI found, stopping analysis", level=1)
                return
                
            # Store frame range for backward compatibility
            if hasattr(self.roi_model, 'valid_frame_range'):
                self.valid_frame_range = self.roi_model.valid_frame_range
            else:
                self.valid_frame_range = (0, self.roi_model.volume.shape[0]-1)
            
            # Background subtraction using ImageProcessor
            self._log("2. Background subtraction...", level=1)
            corrected_volume, background_image = self.processor.median_filter_background_subtraction(self.roi_model)
            self.background = background_image  # Store for compatibility
            
            # Detrend using ImageProcessor
            self._log("3. Detrending...", level=1)
            self.processor.detrend_volume(self.roi_model)
            
            # Smooth volume using ImageProcessor
            if not skip_smoothing:
                self._log("4. Smoothing volume...", level=1)
                self.processor.smooth_volume(self.roi_model)
            
            # Binarize vessels using ImageProcessor
            if not skip_binarization:
                self._log("5. Binarizing vessels...", level=1)
                binary_volume = self.processor.binarize_volume(self.roi_model)
                # Store in image_model for compatibility
                self.image_model.binary = binary_volume
            
            # Determine regions using ImageProcessor
            if not skip_regions:
                self._log("6. Determining regions...", level=1)
                self.region_bounds = self.processor.determine_regions(self.roi_model)
                for region, (peak, sigma, bounds) in self.region_bounds.items():
                    self._log(f"\n{region}:", level=2)
                    self._log(f"  Peak position: {peak:.1f}", level=2)
                    self._log(f"  Width (sigma): {sigma:.1f}", level=2)
                    self._log(f"  Bounds: {bounds[0]:.1f} - {bounds[1]:.1f}", level=2)
                
                # Create region map volume using ImageProcessor
                self._log("6b. Creating region map volume...", level=1)
                region_map = self.processor.create_region_map_volume(self.roi_model, self.region_bounds)
                # Store in both models for compatibility
                self.image_model.region = region_map
            
            # Trace vessel paths (VesselTracer responsibility)
            if not skip_trace:
                self._log("7. Tracing vessel paths...", level=1)
                self.trace_paths()
            
            
            self._log("Analysis complete", level=1, timing=time.time() - start_time)
            
        except Exception as e:
            self._log(f"Error in analysis pipeline: {str(e)}", level=1)
            raise 
      
    def run_pipeline(self,
                    output_dir: Union[str, Path],
                    #Do we conduct any part of the analysis? 
                    skip_smoothing: bool = False,
                    skip_binarization: bool = False,
                    skip_regions: bool = False,
                    skip_trace: bool = False,
                    #Do we generate the DataFrames?

                    skip_dataframe: bool = False,
                    #Options to save the volumes
                    save_volumes: bool = True,
                    save_original: bool = True,
                    save_binary: bool = True,
                    save_region_map: bool = True,
                    save_separate: bool = False,


                ) -> None:
        """Run the complete analysis pipeline and save all outputs.
        
        Args:
            output_dir: Directory to save outputs
            save_original: Whether to save the original volume
            save_binary: Whether to save the binary volume
            save_region_map: Whether to save the region map volume
            save_volumes: Whether to save any volumes
            save_separate: Whether to save volumes separately
            skip_smoothing: Whether to skip the smoothing step
            skip_binarization: Whether to skip the binarization step
            skip_regions: Whether to skip the region detection step
            skip_trace: Whether to skip the trace step
            skip_dataframe: Whether to skip generating and saving DataFrames
        """
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._log(f"Starting analysis pipeline...", level=1)
        self._log(f"Output directory: {output_dir}", level=2)
        
        self.run_analysis(
            skip_smoothing=skip_smoothing,
            skip_binarization=skip_binarization,
            skip_regions=skip_regions,
            skip_trace=skip_trace,
        )
       
        # Generate and save DataFrames if not skipped
        if not skip_dataframe:
            self._log("8. Generating analysis DataFrames...", level=1)
            self.generate_analysis_dataframes()
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
        
        # Clear ROI-specific computed results since they're no longer valid
        # Note: main image volume is preserved, only ROI-specific results are cleared
        self.roi_model = None
        self.image_model.binary = None
        self.image_model.region = None
        self.image_model.paths = {}
                
        print("ROI updated. Next pipeline step will use new parameters.")

    #These functions save the volumes and datasheets for outputting
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
        
        # Check if we have the volumes we want to save
        if save_binary and self.image_model.binary is None:
            raise ValueError("Binary volume not available. Run analysis pipeline first.")
        if save_region_map and self.image_model.region is None:
            raise ValueError("Region map volume not available. Run analysis pipeline with regions enabled first.")
            
        # Prepare volumes for saving
        volumes = []
        volume_names = []
        
        if save_original:
            # Save ROI if available, otherwise save original volume
            original_data = self.roi_model.volume if self.roi_model is not None else self.image_model.volume
            volumes.append(original_data)
            volume_names.append('original')
        if save_binary:
            volumes.append(self.image_model.binary.astype(np.uint8))  # Convert boolean to uint8
            volume_names.append('binary')
        if save_region_map:
            volumes.append(self.image_model.region)  # Already uint8
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
        
        Creates and stores DataFrames for:
        - Metadata about the analysis
        - Region bounds and statistics
        - Z-profile data
        - Vessel paths
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all generated DataFrames
        """
        # Create metadata DataFrame using config method
        pixel_sizes = (self.pixel_size_z, self.pixel_size_y, self.pixel_size_x) if hasattr(self, 'pixel_size_x') else None
        input_type = 'Numpy Array' if isinstance(self.input_data, np.ndarray) else 'File'
        
        # Create processing status dictionary
        processing_status = {
            'ROI Extraction': hasattr(self, 'roi') and self.roi is not None,
            'Background Subtraction': hasattr(self, 'background'),
            'Smoothing': hasattr(self, 'gauss_sigma'),
            'Binarization': hasattr(self, 'binary'),
            'Region Detection': hasattr(self, 'region_bounds'),
            'Path Tracing': hasattr(self, 'paths'),
            'Region Map Creation': hasattr(self, 'region_map')
        }
        
        # Add analysis-specific metadata
        if hasattr(self, 'roi') and self.roi is not None:
            processing_status['ROI Shape'] = str(self.roi.shape)
        
        self.metadata_df = self.config.generate_metadata_df(pixel_sizes, input_type, processing_status)
        
        # Create region bounds DataFrame if available
        if hasattr(self, 'region_bounds'):
            region_data = []
            for region, (peak, sigma, bounds) in self.region_bounds.items():
                region_data.append({
                    'Region': region,
                    'Peak Position': peak,
                    'Sigma': sigma,
                    'Lower Bound': bounds[0],
                    'Upper Bound': bounds[1]
                })
            self.regions_df = pd.DataFrame(region_data)
        else:
            self.regions_df = pd.DataFrame()
        
        # Create z-profile DataFrame
        if hasattr(self, 'get_projection'):
            z_profile = self.get_projection([1, 2], operation='mean')
            self.z_profile_df = pd.DataFrame({
                'Z Position': np.arange(len(z_profile)),
                'Mean Intensity': z_profile
            })
        else:
            self.z_profile_df = pd.DataFrame()
        
        # Create vessel paths DataFrame if available
        if len(self.image_model.paths) > 0:
            vessel_paths_data = []
            path_summary_data = []
            
            for path_id, path_info in self.image_model.paths.items():
                coords = path_info['coordinates']  # Already in ZXY format
                region = path_info['region']
                original_id = path_info.get('original_path_id', path_id)
                length = path_info.get('length', len(coords))
                
                # Create summary entry for this path
                summary_entry = {
                    'Path_ID': path_id,
                    'Original_Path_ID': original_id,
                    'Primary_Region': region,
                    'Path_Length': length,
                    'Start_Z': coords[0][0] if len(coords) > 0 else None,
                    'End_Z': coords[-1][0] if len(coords) > 0 else None,
                }
                
                # Add region fractions if available (for non-split paths)
                if 'region_fractions' in path_info:
                    for region_name, fraction in path_info['region_fractions'].items():
                        summary_entry[f'{region_name}_Fraction'] = fraction
                        
                # Add z-range if available
                if 'z_range' in path_info:
                    summary_entry['Z_Min'] = path_info['z_range'][0]
                    summary_entry['Z_Max'] = path_info['z_range'][1]
                    
                path_summary_data.append(summary_entry)
                
                # Convert path coordinates to DataFrame rows (detailed view)
                for i, coord in enumerate(coords):
                    vessel_paths_data.append({
                        'Path_ID': path_id,
                        'Original_Path_ID': original_id,
                        'Primary_Region': region,
                        'Point_Index': i,
                        'Z': coord[0],  # Z coordinate
                        'X': coord[1],  # X coordinate
                        'Y': coord[2],  # Y coordinate
                        'Point_Region': self.get_region_for_z(coord[0])  # Region for this specific point
                    })
                    
            self.paths_df = pd.DataFrame(vessel_paths_data)
            self.path_summary_df = pd.DataFrame(path_summary_data)
        else:
            self.paths_df = pd.DataFrame()
            self.path_summary_df = pd.DataFrame()
        
        # Store all DataFrames in a dictionary
        self.analysis_dfs = {
            'metadata': self.metadata_df,
            'regions': self.regions_df,
            'z_profile': self.z_profile_df,
            'paths': self.paths_df,
            'path_summary': self.path_summary_df
        }
        
        return self.analysis_dfs

    def save_analysis_to_excel(self, output_path: Union[str, Path]) -> None:
        """Save all analysis DataFrames to an Excel file.
        
        Args:
            output_path: Path where to save the Excel file
        """
        output_path = Path(output_path)
        
        # Generate DataFrames if they don't exist
        if not hasattr(self, 'analysis_dfs'):
            self.generate_analysis_dataframes()
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Save each DataFrame to a different sheet
            self.metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            if not self.regions_df.empty:
                self.regions_df.to_excel(writer, sheet_name='Region Analysis', index=False)
            if not self.z_profile_df.empty:
                self.z_profile_df.to_excel(writer, sheet_name='Z Profile', index=False)
            if not self.paths_df.empty:
                self.paths_df.to_excel(writer, sheet_name='Vessel Paths Detailed', index=False)
            if hasattr(self, 'path_summary_df') and not self.path_summary_df.empty:
                self.path_summary_df.to_excel(writer, sheet_name='Vessel Paths Summary', index=False)

    def multiscan(self, 
                  skip_smoothing: bool = False, 
                  skip_binarization: bool = False, 
                  skip_regions: bool = False, 
                  skip_trace: bool = False
                ) -> None:
        """Scan multiple ROIs across the volume.
        
        Performs systematic scanning across the volume using the configured ROI size.
        For each ROI position, runs the full analysis pipeline and stores results.
        Paths are preserved across ROI changes, and global coordinates are maintained.
        """
        self.roi_results = []
        micron_roi = self.config.micron_roi
        # Calculate scan ranges based on volume dimensions and ROI size
        # Use average of X and Y pixel sizes for ROI conversion
        avg_pixel_size = (self.image_model.pixel_size_x + self.image_model.pixel_size_y) / 2
        pixel_roi = int(micron_roi / avg_pixel_size)
        
        # Get volume dimensions
        _, Y, X = self.image_model.volume.shape
        
        # Create scan ranges with ROI-sized steps
        xscan_rng = range(0, X - pixel_roi + 1, pixel_roi)
        yscan_rng = range(0, Y - pixel_roi + 1, pixel_roi)
        
        self._log(f"Setting up multiscan with {len(xscan_rng)}x{len(yscan_rng)} ROIs", level=2)
        self._log(f"ROI size: {pixel_roi} pixels ({micron_roi} microns)", level=2)

        
        # Iterate through ROI positions
        for y in yscan_rng:
            for x in xscan_rng:
                self._log(f"Processing ROI at position ({x}, {y})", level=2)
                
                # Update ROI position (this will clear roi and binary, but preserve volume and paths)
                self.update_roi_position(x, y)

                # Run analysis on this ROI
                try:
                    self.run_analysis(
                        skip_smoothing=skip_smoothing,
                        skip_binarization=skip_binarization, 
                        skip_regions=skip_regions,
                        skip_trace=skip_trace
                    )
                    
                    # Skip if ROI is None (no valid frames)
                    if self.roi is None:
                        self._log(f"Skipping ROI at ({x}, {y}) - no valid frames found", level=1)
                        continue

                    # Generate analysis data
                    self.generate_analysis_dataframes()

                    # Convert local path coordinates to global coordinates
                    if hasattr(self, 'paths') and self.paths:
                        for path_id, path_info in self.paths.items():
                            if 'coordinates' in path_info:
                                coords = path_info['coordinates']
                                # Add ROI offset to convert local to global coordinates
                                # ROI start position is already x, y (min coordinates)
                                
                                # Add offset to x and y coordinates (z remains the same)
                                coords[:, 1] += x  # X coordinate
                                coords[:, 2] += y  # Y coordinate

                    # Store results with position info
                    roi_data = {
                        'center_x': x + pixel_roi//2,
                        'min_x': x,
                        'max_x': x + pixel_roi,
                        'center_y': y + pixel_roi//2,
                        'min_y': y,
                        'max_y': y + pixel_roi,
                        'paths': self.paths.copy() if hasattr(self, 'paths') and self.paths else {}
                    }
                    self.roi_results.append(roi_data)
                    
                    self._log(f"Completed ROI analysis at ({x}, {y})", level=2)
                except Exception as e:
                    self._log(f"Error processing ROI at ({x}, {y}): {str(e)}", level=1)
                    continue
        
        # Store the multi-ROI results
        self._log(f"Completed multiscan analysis of {len(self.roi_results)} ROIs", level=1)
        return self.roi_results