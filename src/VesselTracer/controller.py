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
from .config import (
    VesselTracerConfig,
    StackConfig,
    load_stack_config,
    config_fingerprint,
    apply_stack_overrides,
)
from .image_model import ImageModel, ROI
from .logging_utils import Log, resolve_log_level

class VesselAnalysisController:
    """Controller class that orchestrates the complete vessel analysis pipeline.
    
    This class manages the coordination between ImageProcessor, VesselTracer, and
    data models to provide a complete analysis workflow. It coordinates the pipeline
    but does not store any results - it only points to where results are stored in
    the component objects and delegates all operations.
    """
    
    def __init__(self, 
                 input_path: Union[str, Path],
                 config_path: Optional[Union[str, Path]],
                 verbose: int = 2,
                 use_gpu: bool = False,
                 stack_config: Optional[StackConfig] = None,
                 stack_overrides: Optional[Dict[str, Any]] = None):
        """Initialize the controller.
        
        Args:
            input_path: Path to input image file
            config_path: Path to configuration file
            verbose: Verbosity level for logging (0-3)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.input_data = None
        
        # Load configuration
        self.config = VesselTracerConfig(config_path)
        self.stack_config = stack_config or load_stack_config(config_path)
        if stack_overrides:
            self.stack_config = apply_stack_overrides(self.stack_config, stack_overrides)
        self.active_stack_config = self.stack_config
        self.stack_fingerprint = config_fingerprint(self.stack_config)
        
        # Initialize image model
        self.image_model = ImageModel(filepath=input_path)
        self.image_model.stack_config = self.stack_config
        self.image_model.stack_config_fingerprint = self.stack_fingerprint
        self.roi_model = None  # Will be initialized during analysis

        # Logger
        log_level = resolve_log_level(self.config.verbose, self.stack_config.notes)
        self.logger = Log(enabled=True, level=log_level)
        
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
        """Log a message with appropriate verbosity level."""
        if timing is not None:
            message = f"{message} (took {timing:.2f}s)"
        if hasattr(self, 'logger') and self.logger:
            self.logger(message, level=level)

    def _resolve_flag(self, override: Optional[bool], config_value: bool) -> bool:
        """Resolve boolean overrides against config defaults."""
        if override is None:
            return config_value
        return override

    def _prepare_stack_config(self,
                              runtime_stack: Optional[StackConfig],
                              overrides: Optional[Dict[str, Any]]) -> StackConfig:
        """Resolve the active stack configuration for this run."""
        cfg = runtime_stack or self.stack_config
        cfg = apply_stack_overrides(cfg, overrides)
        fingerprint = config_fingerprint(cfg)

        self.image_model.stack_config = cfg
        self.image_model.stack_config_fingerprint = fingerprint
        self.stack_fingerprint = fingerprint
        self.active_stack_config = cfg

        # Refresh logger level
        self.logger.level = resolve_log_level(self.config.verbose, cfg.notes)
        return cfg

    def _run_global_preprocessing(self,
                                  stack_cfg: StackConfig,
                                  remove_dead_frames: bool,
                                  dead_frame_threshold: float,
                                  background_mode: str) -> None:
        """Apply normalization, detrending, dead-frame removal, and background subtraction."""
        if stack_cfg.normalize:
            self._log("Normalizing ROI volume...", level=1)
            self.roi_model.volume = self.processor.normalize_image(self.roi_model)

        if stack_cfg.detrend:
            self._log("Detrending ROI volume...", level=1)
            self.roi_model.volume = self.processor.detrend_volume(self.roi_model)

        if remove_dead_frames and self.config.remove_dead_frames:
            original_threshold = self.processor.config.dead_frame_threshold
            self.processor.config.dead_frame_threshold = dead_frame_threshold
            self._log("Removing dead frames...", level=1)
            self.roi_model.volume = self.processor.remove_dead_frames(self.roi_model)
            self.processor.config.dead_frame_threshold = original_threshold

        background_mode = (background_mode or "rolling_ball").lower()
        if background_mode != "none":
            self._log(f"Estimating background ({background_mode})...", level=1)
            self.roi_model.background = self.processor.estimate_background(
                self.roi_model,
                mode=background_mode
            )
            self._log("Subtracting background...", level=1)
            self.roi_model.volume = self.roi_model.volume - self.roi_model.background
            self._log(
                f"Background-subtracted range: "
                f"[{self.roi_model.volume.min():.3f}, {self.roi_model.volume.max():.3f}]",
                level=2
            )

    def _run_monolithic_pipeline(self,
                                 skip_binarization: bool,
                                 skip_closing: bool,
                                 threshold_method: str) -> None:
        """Run the legacy single-volume binarize/closing pipeline."""
        if skip_binarization:
            self._log("Skipping binarization per configuration.", level=2)
            return

        self._log("Binarizing vessels...", level=1)
        self.roi_model.binary = self.processor.binarize_volume(
            self.roi_model,
            method=threshold_method
        )

        if skip_closing:
            return

        self._log("Closing vessels (monolithic)...", level=1)
        self.roi_model.binary = self.processor.morphological_closing(self.roi_model)

    def _process_tile(self,
                      tile_roi: ROI,
                      tile_info: Dict[str, Any],
                      stack_cfg: StackConfig,
                      skip_binarization: bool,
                      skip_closing: bool) -> Dict[str, Any]:
        """Process a single tile and return intermediate products."""
        tile_cfg = stack_cfg.tiling
        result: Dict[str, Any] = {}

        vesselness, radius = self.processor.multiscale_vesselness(
            tile_roi.volume,
            tile_roi.get_pixel_sizes(),
            tile_cfg.vessel_scales,
            anisotropic_z=tile_cfg.anisotropic_z
        )
        result['vesselness'] = vesselness
        result['radius'] = radius

        if skip_binarization:
            result['binary'] = None
        else:
            method = tile_cfg.thresh_mode or self.config.binarization_method
            tile_roi.volume = vesselness if method.lower() in ("sauvola", "phansalkar") else tile_roi.volume
            tile_roi.binary = self.processor.binarize_volume(tile_roi, method=method)
            binary_volume = tile_roi.binary

            if not skip_closing:
                if tile_cfg.adaptive_closing:
                    binary_volume = self.processor.adaptive_geodesic_closing(
                        binary_volume,
                        radius,
                        tile_cfg.closing_rmin,
                        tile_cfg.closing_rmax
                    )
                else:
                    tile_roi.binary = binary_volume
                    binary_volume = self.processor.morphological_closing(tile_roi)

            result['binary'] = binary_volume

        if tile_cfg.cache_intermediates:
            self._cache_tile_products(tile_info['id'], result)

        return result

    def _cache_tile_products(self, tile_id: int, data: Dict[str, Any]) -> None:
        """Persist tile intermediates for debugging when requested."""
        if not self.image_model.filepath:
            return

        cache_dir = Path(self.image_model.filepath).parent / "tile_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"tile_{tile_id}_{self.stack_fingerprint}.npz"

        payload = {
            key: value
            for key, value in data.items()
            if isinstance(value, np.ndarray) and value is not None
        }
        if payload:
            np.savez_compressed(cache_path, **payload)

    def _run_tiled_pipeline(self,
                            stack_cfg: StackConfig,
                            skip_binarization: bool,
                            skip_closing: bool) -> Dict[str, np.ndarray]:
        """Process the ROI volume using overlapping XY tiles."""
        tile_cfg = stack_cfg.tiling
        tiles = self.processor.generate_tiles(
            self.roi_model.volume,
            tile_xy=tile_cfg.tile_xy,
            overlap=tile_cfg.tile_overlap,
            halo_xy=tile_cfg.halo_xy
        )

        tile_results: List[Dict[str, Any]] = []
        for tile in tiles:
            tile_volume = self.roi_model.volume[tile['full_slices']].copy()
            tile_roi = ROI(
                volume=tile_volume,
                pixel_size_x=self.roi_model.pixel_size_x,
                pixel_size_y=self.roi_model.pixel_size_y,
                pixel_size_z=self.roi_model.pixel_size_z
            )
            processed = self._process_tile(
                tile_roi,
                tile,
                stack_cfg,
                skip_binarization=skip_binarization,
                skip_closing=skip_closing
            )
            tile_results.append({**tile, **processed})

        stitched = self.processor.stitch_tiles(
            tile_results,
            volume_shape=self.roi_model.volume.shape,
            blend_mode=tile_cfg.blend_mode
        )

        self.roi_model.binary = stitched['binary']
        self.roi_model.vesselness = stitched.get('vesselness')
        self.roi_model.radius = stitched.get('radius')
        return stitched
                
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
                     remove_dead_frames: Optional[bool] = None,
                     dead_frame_threshold: Optional[float] = None,
                     skip_smoothing: Optional[bool] = None,
                     skip_binarization: Optional[bool] = None,
                     skip_regions: Optional[bool] = None, 
                     skip_closing: Optional[bool] = None,
                     skip_trace: Optional[bool] = None,
                     stack_config: Optional[StackConfig] = None,
                     stack_overrides: Optional[Dict[str, Any]] = None) -> None:
        """Run the analysis pipeline for the active stack.
        
        Args:
            remove_dead_frames: Override whether to drop dead frames (None -> config)
            dead_frame_threshold: Override threshold multiplier
            skip_smoothing: Skip smoothing stage
            skip_binarization: Skip binarization stage
            skip_regions: Skip region detection
            skip_closing: Skip closing/adaptive morphology
            skip_trace: Skip path tracing
            stack_config: Optional alternate StackConfig for this run
            stack_overrides: Dict overrides applied to the stack config
        """
        start_time = time.time()
        stack_cfg = self._prepare_stack_config(stack_config, stack_overrides)

        smoothing_disabled = (
            stack_cfg.smoothing is None or
            (isinstance(stack_cfg.smoothing, str) and stack_cfg.smoothing.lower() == "none")
        )

        flags = {
            'remove_dead_frames': self._resolve_flag(remove_dead_frames, stack_cfg.remove_dead_frames),
            'dead_frame_threshold': dead_frame_threshold if dead_frame_threshold is not None else stack_cfg.dead_frame_threshold,
            'skip_smoothing': self._resolve_flag(skip_smoothing, smoothing_disabled),
            'skip_binarization': self._resolve_flag(skip_binarization, stack_cfg.skip_binarization),
            'skip_regions': self._resolve_flag(skip_regions, stack_cfg.skip_regions),
            'skip_closing': self._resolve_flag(skip_closing, stack_cfg.skip_closing),
            'skip_trace': self._resolve_flag(skip_trace, stack_cfg.skip_trace),
        }

        self._log(f"Starting analysis pipeline (cfg={self.stack_fingerprint[:8]})...", level=1)
        
        try:
            with self.logger.section("ROI Extraction", level=1):
                if self.config.find_roi:
                    self.roi_model = self.processor.segment_roi(self.image_model)
                else:
                    self._log("Using full volume as ROI", level=2)
                    self.roi_model = ROI(
                        volume=self.image_model.volume.copy(),
                        pixel_size_x=self.image_model.pixel_size_x,
                        pixel_size_y=self.image_model.pixel_size_y,
                        pixel_size_z=self.image_model.pixel_size_z,
                        min_x=0,
                        min_y=0,
                        max_x=self.image_model.volume.shape[2],
                        max_y=self.image_model.volume.shape[1]
                    )

            with self.logger.section("Global preprocessing", level=1):
                self._run_global_preprocessing(
                    stack_cfg,
                    remove_dead_frames=flags['remove_dead_frames'],
                    dead_frame_threshold=flags['dead_frame_threshold'],
                    background_mode=stack_cfg.background_mode
                )

            if not flags['skip_smoothing']:
                with self.logger.section("Smoothing", level=1):
                    self.roi_model.volume = self.processor.smooth_volume(self.roi_model)

            if not flags['skip_regions']:
                with self.logger.section("Region detection", level=1):
                    self.roi_model.region_bounds = self.processor.determine_regions_with_splines(self.roi_model)
                    for region, (peak, sigma, bounds) in self.roi_model.region_bounds.items():
                        self._log(f"{region}: peak={peak:.1f}, sigma={sigma:.1f}, bounds=({bounds[0]:.1f}, {bounds[1]:.1f})", level=2)

            threshold_method = stack_cfg.tiling.thresh_mode or self.config.binarization_method
            if stack_cfg.tiling.use_tiled_pipeline:
                with self.logger.section("Tiled vessel extraction", level=1):
                    self._run_tiled_pipeline(
                        stack_cfg,
                        skip_binarization=flags['skip_binarization'],
                        skip_closing=flags['skip_closing']
                    )
            else:
                with self.logger.section("Monolithic vessel extraction", level=1):
                    self._run_monolithic_pipeline(
                        skip_binarization=flags['skip_binarization'],
                        skip_closing=flags['skip_closing'],
                        threshold_method=threshold_method
                    )

            if not flags['skip_trace']:
                with self.logger.section("Tracing", level=1):
                    if self.roi_model.binary is None:
                        raise ValueError("Tracing requested but binary volume is missing")
                    self.roi_model.paths, self.roi_model.path_stats, self.roi_model.n_paths = self.tracer.trace_paths(
                        binary_volume=self.roi_model.binary,
                    )
                    self._log("Smoothing traced paths...", level=2)
                    self.roi_model.paths = self.tracer.smooth_paths(self.roi_model.paths)
                    try:
                        self.roi_model.labeled_paths = self.processor.label_paths_by_surfaces(self.roi_model)
                    except Exception as exc:
                        self._log(f"Error in labeling paths by surfaces: {exc}", level=1)
                        self.roi_model.labeled_paths = None

                    self.roi_model.region_projections = self.processor.create_region_projections(self.roi_model)

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

        # Clear ROI model data
        if self.roi_model:
            self.roi_model.volume = None
            self.roi_model.binary = None
            self.roi_model.background = None
            self.roi_model.region = None
            self.roi_model.paths = None
            self.roi_model.region_bounds = None
                
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

    def create_paths_dataframe(self, pixel_sizes: Optional[Dict[str, float]] = None, source: str = 'roi') -> pd.DataFrame:
        """Create a pandas DataFrame from the traced paths.
        
        Args:
            pixel_sizes: Optional dictionary of pixel sizes in microns for coordinate conversion
            source: Source of paths to use ('roi' or 'image')
            
        Returns:
            DataFrame with detailed path information
        """
        if source == 'roi':
            model = self.roi_model
        elif source == 'image':
            model = self.image_model
        else:
            raise ValueError(f"Invalid source: {source}. Must be one of: roi, image")
            
        if model is None or model.paths is None:
            return pd.DataFrame()
            
        vessel_paths_data = []
        
        for path_id, path_info in model.paths.items():
            coords = path_info['coordinates']
            length = path_info['length']
            
            # Get region for each point
            regions = [self.processor._get_region_for_z(z, model.region_bounds) 
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
    
    def create_path_summary_dataframe(self, pixel_sizes: Optional[Dict[str, float]] = None, source: str = 'roi') -> pd.DataFrame:
        """Create a summary DataFrame with one row per path.
        
        Args:
            pixel_sizes: Optional dictionary of pixel sizes in microns for coordinate conversion
            source: Source of paths to use ('roi' or 'image')
            
        Returns:
            DataFrame with path summary information
        """
        if source == 'roi':
            model = self.roi_model
        elif source == 'image':
            model = self.image_model
        else:
            raise ValueError(f"Invalid source: {source}. Must be one of: roi, image")
            
        if model is None or model.paths is None:
            return pd.DataFrame()
            
        path_summary_data = []
        
        # Convert pixel sizes if provided
        if pixel_sizes:
            z_scale = pixel_sizes.get('z', 1.0)
            y_scale = pixel_sizes.get('y', 1.0)
            x_scale = pixel_sizes.get('x', 1.0)
        else:
            z_scale = y_scale = x_scale = 1.0
        
        for path_id, path_info in model.paths.items():
            coords = path_info['coordinates']
            length = path_info['length']
            
            # Get regions for start and end points
            start_region = self.processor._get_region_for_z(coords[0, 0], model.region_bounds)
            end_region = self.processor._get_region_for_z(coords[-1, 0], model.region_bounds)
            
            # Get all regions traversed
            regions = [self.processor._get_region_for_z(z, model.region_bounds) 
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
        Results are mapped back to the main image model coordinates.
        
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
        volume_shape = self.image_model.volume.shape  # Use main image model for dimensions
        pixel_sizes = (self.image_model.pixel_size_z, 
                      self.image_model.pixel_size_y, 
                      self.image_model.pixel_size_x)
        avg_pixel_size = (pixel_sizes[1] + pixel_sizes[2]) / 2  # Y and X pixel sizes
        pixel_roi = int(micron_roi / avg_pixel_size)
        
        # Get volume dimensions
        _, Y, X = volume_shape
        
        # Create scan ranges with ROI-sized steps
        xscan_rng = range(0, X - pixel_roi + 1, pixel_roi)
        yscan_rng = range(0, Y - pixel_roi + 1, pixel_roi)
        
        self._log(f"\nMultiscan Configuration:", level=1)
        self._log(f"Volume dimensions: {volume_shape}", level=1)
        self._log(f"ROI size: {pixel_roi} pixels ({micron_roi} microns)", level=1)
        self._log(f"X scan range: {min(xscan_rng)} to {max(xscan_rng)}", level=1)
        self._log(f"Y scan range: {min(yscan_rng)} to {max(yscan_rng)}", level=1)
        self._log(f"Total ROIs: {len(xscan_rng)}x{len(yscan_rng)} = {len(xscan_rng)*len(yscan_rng)}", level=1)

        # Initialize image model arrays if they don't exist
        if self.image_model.background is None:
            self.image_model.background = np.zeros_like(self.image_model.volume)
        if self.image_model.binary is None:
            self.image_model.binary = np.zeros_like(self.image_model.volume, dtype=np.uint8)
        if self.image_model.region is None:
            self.image_model.region = np.zeros_like(self.image_model.volume, dtype=np.uint8)

        # Store all paths from all ROIs
        all_paths = {}
        path_id_counter = 0

        # Iterate through ROI positions
        for y in yscan_rng:
            for x in xscan_rng:
                roi_min_x = x
                roi_max_x = x + pixel_roi
                roi_min_y = y
                roi_max_y = y + pixel_roi
                
                self._log(f"\nProcessing ROI at:", level=2)
                self._log(f"  Position: ({x}, {y})", level=2)
                self._log(f"  X range: {roi_min_x} to {roi_max_x}", level=2)
                self._log(f"  Y range: {roi_min_y} to {roi_max_y}", level=2)
                
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
                        self._log(f"  Skipping ROI - no valid frames found", level=2)
                        continue

                    # Get paths from current ROI
                    current_paths = self.roi_model.paths if self.roi_model and self.roi_model.paths else {}
                    
                    # Convert local path coordinates to global coordinates and store
                    for path_id, path_info in current_paths.items():
                        if 'coordinates' in path_info:
                            coords = path_info['coordinates'].copy()
                            # Add ROI offset to convert local to global coordinates
                            coords[:, 1] += y  # Y coordinate
                            coords[:, 2] += x  # X coordinate
                            
                            # Create new path with global coordinates
                            new_path_id = f"path_{path_id_counter}"
                            all_paths[new_path_id] = {
                                'coordinates': coords,
                                'length': path_info['length']
                            }
                            path_id_counter += 1

                    # Store ROI results in image model
                    if self.roi_model.background is not None:
                        self.image_model.background[:, roi_min_y:roi_max_y, roi_min_x:roi_max_x] = self.roi_model.background
                    if self.roi_model.binary is not None:
                        self.image_model.binary[:, roi_min_y:roi_max_y, roi_min_x:roi_max_x] = self.roi_model.binary
                    if self.roi_model.region is not None:
                        self.image_model.region[:, roi_min_y:roi_max_y, roi_min_x:roi_max_x] = self.roi_model.region

                    # Store ROI results with position info
                    roi_data = {
                        'center_x': x + pixel_roi//2,
                        'min_x': roi_min_x,
                        'max_x': roi_max_x,
                        'center_y': y + pixel_roi//2,
                        'min_y': roi_min_y,
                        'max_y': roi_max_y,
                        'paths': current_paths
                    }
                    roi_results.append(roi_data)
                    
                    self._log(f"  Completed ROI analysis - Found {len(current_paths)} paths", level=2)
                    
                except Exception as e:
                    self._log(f"  Error processing ROI: {str(e)}", level=2)
                    continue
        
        # Store all paths in the main image model
        if self.image_model:
            self.image_model.paths = all_paths
        
        self._log(f"\nMultiscan Summary:", level=1)
        self._log(f"Total ROIs processed: {len(roi_results)}", level=1)
        self._log(f"Total paths found: {len(all_paths)}", level=1)
        
        return roi_results
