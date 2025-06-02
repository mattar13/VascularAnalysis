from dataclasses import dataclass
from pathlib import Path
import numpy as np
import xmltodict
from czifile import CziFile
from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, ball, skeletonize as sk_skeletonize
from skimage.filters import threshold_otsu, threshold_triangle
from skan import Skeleton, summarize
import yaml
from typing import Optional, Dict, Any, Tuple, Union, List
from scipy.signal import find_peaks, peak_widths
import tifffile
import time
from datetime import datetime

@dataclass
class VesselTracer:
    """Main class for vessel tracing pipeline.
    
    Attributes:
        czi_path: Path to CZI file to analyze
        config_path: Optional path to YAML config file. If None, uses default config.
    """
    czi_path: str
    config_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._load_config()
        self._load_image()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path is None:
            self.config_path = Path(__file__).parent.parent.parent / 'config' / 'default_vessel_config.yaml'
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # ROI settings
        self.micron_roi = config['roi']['micron_roi']
        self.center_x = config['roi']['center_x']
        self.center_y = config['roi']['center_y']
        
        # Scale bar settings
        self.scalebar_length = config['scalebar']['length']
        self.scalebar_x = config['scalebar']['x']
        self.scalebar_y = config['scalebar']['y']
        
        # Pre-processing settings
        self.gauss_sigma = config['preprocessing']['gauss_sigma']
        self.min_object_size = config['preprocessing']['min_object_size']
        self.close_radius = config['preprocessing']['close_radius']
        self.prune_length = config['preprocessing']['prune_length']
        self.median_filter_size = config['preprocessing']['median_filter_size']
        self.binarization_method = config['preprocessing']['binarization_method']

        # Region settings
        self.regions = config.get('regions', ['superficial', 'intermediate', 'deep'])
        self.region_peak_distance = config.get('region_peak_distance', 2)
        self.region_height_ratio = config.get('region_height_ratio', 0.80)
        self.region_n_stds = config.get('region_n_stds', 2)
        
        # Verbose settings
        self.verbose = config.get('verbose', 2)  # Default to level 1
        
        # Image properties (set by load_image)
        self.pixel_size_x = 0.0
        self.pixel_size_y = 0.0
        self.pixel_size_z = 0.0
        
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

    def _load_image(self):
        """Load CZI file and extract pixel sizes."""
        start_time = time.time()
        self._log("Loading CZI file...", level=1)
        
        with CziFile(self.czi_path) as czi:
            self.volume = czi.asarray()[0, 0, 0, 0, ::-1, :, :, 0]
            meta = xmltodict.parse(czi.metadata())

        def _px_um(axis_id: str) -> float:
            for d in meta["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]:
                if d["@Id"] == axis_id:
                    return 1 / float(d["Value"]) / 1e6
            raise KeyError(axis_id)

        # Set pixel sizes
        self.pixel_size_x = _px_um("X")
        self.pixel_size_y = _px_um("Y")
        self.pixel_size_z = _px_um("Z")
        
        # Normalize volume
        self.volume = self._normalize_image(self.volume.astype("float32"))
        
        self._log(f"Image loaded. Shape: {self.volume.shape}", level=2)
        self._log(f"Pixel sizes (µm): X={self.pixel_size_x:.3f}, Y={self.pixel_size_y:.3f}, Z={self.pixel_size_z:.3f}", level=2)
        self._log("Image loading complete", level=1, timing=time.time() - start_time)
        
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0,1] range."""
        img = (img - img.min()) / (img.max() - img.min())
        return img
        
    def segment_roi(self, remove_dead_frames: bool = True, dead_frame_threshold: float = 1.5) -> np.ndarray:
        """Extract and segment region of interest from volume.
        
        Extracts a region of interest based on configured center position
        and size in microns. Optionally removes dead frames at start/end.
        
        Args:
            remove_dead_frames: Whether to remove low-intensity frames at start/end
            dead_frame_threshold: Number of standard deviations above minimum
                                intensity to use as threshold for dead frames
        
        Returns:
            np.ndarray: ROI volume extracted from main volume
        """
        start_time = time.time()
        self._log("Extracting ROI...", level=1)
        
        # Convert ROI size from microns to pixels
        h_x = round(self.micron_roi/2 * 1/self.pixel_size_x)
        h_y = round(self.micron_roi/2 * 1/self.pixel_size_y)
        
        self._log(f"ROI size: {h_x*2}x{h_y*2} pixels", level=2)
        
        # Extract initial ROI
        roi = self.volume[:, 
                        self.center_y-h_y:self.center_y+h_y,
                        self.center_x-h_x:self.center_x+h_x]
        
        if remove_dead_frames:
            self._log("Removing dead frames...", level=2)
            # Calculate mean intensity profile along z
            z_profile = np.mean(roi, axis=(1,2))
            
            # Find frames above threshold
            threshold = z_profile.min() + dead_frame_threshold * z_profile.std()
            valid_frames = np.where(z_profile > threshold)[0]
            
            if len(valid_frames) > 0:
                frame_start = valid_frames[0]
                frame_end = valid_frames[-1]
                
                self._log(f"Original z-range: 0-{len(z_profile)}", level=2)
                self._log(f"Valid frame range: {frame_start}-{frame_end}", level=2)
                self._log(f"Removed {frame_start} frames from start", level=2)
                self._log(f"Removed {len(z_profile)-frame_end-1} frames from end", level=2)
                
                # Update ROI to exclude dead frames
                roi = roi[frame_start:frame_end+1]
                
                # Store frame range for reference
                self.valid_frame_range = (frame_start, frame_end)
            else:
                self._log("Warning: No frames found above threshold!", level=1)
                self.valid_frame_range = (0, len(z_profile)-1)
        else:
            self.valid_frame_range = (0, roi.shape[0]-1)
        
        self.roi_volume = self._normalize_image(roi)
        self._log(f"ROI extraction complete. Final shape: {roi.shape}", level=2)
        self._log("ROI extraction complete", level=1, timing=time.time() - start_time)
        return roi
    
    def median_filter(self) -> np.ndarray:
        """Apply median filter to ROI volume."""
        start_time = time.time()
        self._log("Applying median filter...", level=1)
        
        if not hasattr(self, 'roi_volume'): 
            self.segment_roi()
            
        self.roi_volume = median_filter(self.roi_volume, size=self.median_filter_size)
        
        self._log("Median filter complete", level=1, timing=time.time() - start_time)
        return self.roi_volume
    
    def detrend(self) -> np.ndarray:
        """Remove linear trend from ROI along z-axis.
        
        Corrects for linear attenuation in depth by:
        1. Computing mean intensity profile along z
        2. Fitting a linear trend
        3. Removing trend from each xy slice
        
        Must be called after segment_roi().
        
        Returns:
            np.ndarray: Detrended ROI volume
        """
        start_time = time.time()
        self._log("Detrending volume...", level=1)
        
        if not hasattr(self, 'roi_volume'):
            self.segment_roi()
            
        Z, Y, X = self.roi_volume.shape
        
        # Calculate mean intensity profile
        z_profile = np.mean(self.roi_volume, axis=(1,2))
        z_positions = np.arange(Z)
        
        # Fit linear trend
        coeffs = np.polyfit(z_positions, z_profile, deg=1)
        trend = np.polyval(coeffs, z_positions)
        
        self._log(f"Linear trend coefficients: {coeffs}", level=2)
        
        # Calculate correction factors
        correction = trend / np.mean(trend)
        
        # Apply correction to each z-slice
        detrended = np.zeros_like(self.roi_volume)
        for z in range(Z):
            detrended[z] = self.roi_volume[z] / correction[z]
            
        # Normalize to [0,1] range
        self.roi_volume = self._normalize_image(detrended)
        
        self._log("Detrending complete", level=1, timing=time.time() - start_time)
        return self.roi_volume
    
    def background_smoothing(self, epsilon = 1e-6) -> np.ndarray:
        """Apply super smoothing to ROI volume."""
        start_time = time.time()
        self._log("Super smoothing volume...", level=1)
                    
        background_smooth = gaussian_filter(self.roi_volume, sigma=self.super_smooth_sigma)
        
        self._log("Super smoothing complete", level=1, timing=time.time() - start_time)
        self.roi_volume = (self.roi_volume - background_smooth) / (epsilon + background_smooth)
        
    def smooth(self) -> np.ndarray:
        """Apply Gaussian smoothing to ROI volume."""
        start_time = time.time()
        self._log("Smoothing volume...", level=1)
        
        if not hasattr(self, 'roi_volume'):
            self.detrend()
            
        self._log(f"Using Gaussian sigma: {self.gauss_sigma}", level=2)
        self.smoothed = gaussian_filter(self.roi_volume, sigma=self.gauss_sigma)
        
        #Do we want to add this here, or do background smoothing to the entire ROI

        self._log("Smoothing complete", level=1, timing=time.time() - start_time)
        return self.smoothed
        
    def binarize(self) -> np.ndarray:
        """Binarize the smoothed volume using triangle thresholding.
        
        This method applies triangle thresholding to the entire 3D volume at once,
        removes small objects, and performs morphological operations to clean up
        the binary volume.
        
        Returns:
            np.ndarray: Binary volume after thresholding and cleaning
        """
        start_time = time.time()
        self._log("Binarizing volume...", level=1)
        
        if not hasattr(self, 'smoothed'):
            self.smooth()
            
        # Calculate triangle threshold using entire volume
        thresh = threshold_triangle(self.smoothed.ravel())
        self._log(f"Triangle threshold: {thresh:.3f}", level=2)
        
        # Apply threshold to entire volume at once
        bw_vol = self.smoothed > thresh
        
        # Remove small objects in 3D
        bw_vol = remove_small_objects(bw_vol, min_size=self.min_object_size)
        
        # Apply morphological operations
        if self.close_radius > 0:
            # # First do opening to remove small noise spots
            # self._log(f"Performing 3D opening with radius {self.close_radius}", level=2)
            # bw_vol = binary_opening(bw_vol, ball(self.close_radius))
            
            # Then do closing to connect nearby vessel segments
            self._log(f"Performing 3D closing with radius {self.close_radius}", level=2)
            bw_vol = binary_closing(bw_vol, ball(self.close_radius))
            
            # # Do one more opening to clean up any remaining noise
            # self._log(f"Performing final 3D opening with radius {self.close_radius}", level=2)
            # bw_vol = binary_opening(bw_vol, ball(self.close_radius))
            
        self.binary = bw_vol
        self._log("Binarization complete", level=1, timing=time.time() - start_time)
        return bw_vol
        
    def skeletonize(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create and analyze vessel skeleton.
        
        Returns:
            Tuple containing:
            - paths: Dictionary of branch paths with coordinates
            - stats: DataFrame with branch statistics
        """
        start_time = time.time()
        self._log("Skeletonizing volume...", level=1)
        
        if not hasattr(self, 'binary'):
            self.binarize()
            
        self._log(f"Skeletonizing binary volume of shape {self.binary.shape}", level=2)
        ske = sk_skeletonize(self.binary)
        self._log(f"Skeletonized volume of shape {ske.shape}", level=2)
        
        # Create Skeleton object for path analysis
        self.skeleton = Skeleton(ske)
        self._log("Created skeleton object", level=2)
        
        # Extract paths from skeleton
        self.paths = {}
        coords = self.skeleton.coordinates
        total_paths = self.skeleton.paths.shape[0]
        #This is a very slow operation, lets see what we get until then
        # for i, path in enumerate(self.skeleton.paths):
        #     self._log(f"Processing path {i+1} out of {total_paths}", level=2)
        #     # Convert sparse matrix path to dense array of indices
        #     path_indices = path.toarray().flatten().nonzero()[0]
        #     # Get coordinates for these indices
        #     path_coords = coords[path_indices]
        #     self.paths[i] = path_coords  # Store the coordinates array
        
        # Get detailed statistics using skan's summarize function
        self.stats = summarize(self.skeleton, separator="-")
        
        self._log(f"Found {len(self.paths)} vessel paths", level=2)
        self._log("Skeletonization complete", level=1, timing=time.time() - start_time)
        return self.paths, self.stats

    def get_depth_volume(self) -> np.ndarray:
        """Create a volume where each vessel is labeled by its z-depth.
        
        Returns:
            np.ndarray: Volume with z-depth information for each vessel
        """
        if not hasattr(self, 'binary'):
            self.binarize()
            
        Z, Y, X = self.binary.shape
        depth_vol = np.zeros((Z, Y, X), dtype=float)
        
        for z in range(Z):
            depth_vol[z] = self.binary[z] * z
            
        return depth_vol
        
    def get_projection(self, axis: Union[int, List[int]], operation: str = 'mean') -> np.ndarray:
        """Generate a projection of the binary volume along specified axis/axes.
        
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
                
        Returns:
            np.ndarray: Projected image
            
        Raises:
            ValueError: If invalid axis or operation specified
        """
        if not hasattr(self, 'binary'):
            self.binarize()
            
        # Validate operation
        valid_ops = {
            'max': np.max,
            'min': np.min,
            'mean': np.mean,
            'std': np.std
        }
        
        if operation not in valid_ops:
            raise ValueError(f"Operation must be one of {list(valid_ops.keys())}")
            
        # Validate axis
        valid_axes = [0, 1, 2, [1,2]]
        if not (axis in valid_axes or (isinstance(axis, list) and axis == [1,2])):
            raise ValueError("Axis must be 0 (z), 1 (y), 2 (x), or [1,2] (xy)")
            
        # Get projection function
        proj_func = valid_ops[operation]
        
        # Calculate projection
        if isinstance(axis, list):
            # Special case for xy projection (along z)
            projection = proj_func(self.binary, axis=tuple(axis))
        else:
            projection = proj_func(self.binary, axis=axis)
            
        return projection
        
    def determine_regions(self) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
        """Determine vessel regions based on the mean z-profile.
        
        Uses peak finding to identify distinct vessel layers and calculates
        their boundaries based on peak widths.
        
        Returns:
            Dictionary mapping region names to tuples of:
            (peak_position, sigma, (lower_bound, upper_bound))
        """
        # Get mean z-profile (xy projection)
        mean_zprofile = self.get_projection([1, 2], operation='mean')
        
        # Find peaks
        peaks, _ = find_peaks(mean_zprofile, distance=self.region_peak_distance)
        
        # Calculate peak widths
        widths_all, _, _, _ = peak_widths(
            mean_zprofile, peaks, rel_height=self.region_height_ratio)
        
        # Convert widths to sigmas
        sigmas = widths_all / (self.region_n_stds * np.sqrt(2 * np.log(2)))
        
        # Print peak information
        for i, pk in enumerate(peaks):
            self._log(f"Peak at z={pk:.1f}: σ ≈ {sigmas[i]:.2f}")
        
        # Create region bounds dictionary
        self.region_bounds = {
            region: (mu, sigma, (mu - sigma, mu + sigma))
            for region, mu, sigma in zip(self.regions, peaks, sigmas)
        }
        
        return self.region_bounds
        
    def analyze_layers(self) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
        """Analyze vessel layers in the binary segmentation."""
        if not hasattr(self, 'binary'):
            self.binarize()
            
        if not hasattr(self, 'region_bounds'):
            self.determine_regions()
            
        return self.region_bounds
        
    def print_config(self):
        """Print current configuration."""
        print("\nVesselTracer Configuration")
        print("=========================")
        
        print("\nROI Settings:")
        print(f"    micron_roi  -> {self.micron_roi:8} [Size of region of interest in microns]")
        print(f"    center_x    -> {self.center_x:8} [X coordinate of ROI center]")
        print(f"    center_y    -> {self.center_y:8} [Y coordinate of ROI center]")
        
        print("\nScale Bar Settings:")
        print(f"    scalebar_length  -> {self.scalebar_length:8} [Length of scale bar in plot units]")
        print(f"    scalebar_x       -> {self.scalebar_x:8} [X position of scale bar in plot]")
        print(f"    scalebar_y       -> {self.scalebar_y:8} [Y position of scale bar in plot]")
        
        print("\nPre-processing Parameters:")
        print(f"    gauss_sigma      -> {self.gauss_sigma:8} [Gaussian smoothing sigma]")
        print(f"    min_object_size  -> {self.min_object_size:8} [Minimum object size to keep]")
        print(f"    close_radius     -> {self.close_radius:8} [Closing operation radius in voxels]")
        print(f"    prune_length     -> {self.prune_length:8} [Length to prune skeleton branches]")
        
        print("\nRegion Settings:")
        print(f"    regions          -> {self.regions}")
        print(f"    region_peak_distance -> {self.region_peak_distance}")
        print(f"    region_height_ratio -> {self.region_height_ratio}")
        print(f"    region_n_stds      -> {self.region_n_stds}")
        
        print("\nImage Properties:")
        print(f"    pixel_size_x     -> {self.pixel_size_x:8.3f} [Pixel size in x direction (µm/pixel)]")
        print(f"    pixel_size_y     -> {self.pixel_size_y:8.3f} [Pixel size in y direction (µm/pixel)]")
        print(f"    pixel_size_z     -> {self.pixel_size_z:8.3f} [Pixel size in z direction (µm/pixel)]")
        print("\n")

    def run_analysis(self,
                    skip_smoothing: bool = False,
                    skip_binarization: bool = False,
                    skip_regions: bool = False) -> None:
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
        """
        start_time = time.time()
        self._log("Starting analysis pipeline...", level=1)
        
        try:
            # Extract ROI
            self._log("1. Extracting ROI...", level=1)
            self.segment_roi(remove_dead_frames=True, dead_frame_threshold=1.5)
            self.median_filter()
            self.background_smoothing()
            self.detrend()
            
            # Smooth volume
            if not skip_smoothing:
                self._log("2. Smoothing volume...", level=1)
                self.smooth()
            
            # Binarize vessels
            if not skip_binarization:
                self._log("3. Binarizing vessels...", level=1)
                self.binarize()
            
            # Create skeleton
            self._log("4. Creating skeleton...", level=1)
            self.skeletonize()
            
            # Determine regions
            if not skip_regions:
                self._log("5. Determining regions...", level=1)
                regions = self.determine_regions()
                for region, (peak, sigma, bounds) in regions.items():
                    self._log(f"\n{region}:", level=2)
                    self._log(f"  Peak position: {peak:.1f}", level=2)
                    self._log(f"  Width (sigma): {sigma:.1f}", level=2)
                    self._log(f"  Bounds: {bounds[0]:.1f} - {bounds[1]:.1f}", level=2)
            
            self._log("Analysis complete", level=1, timing=time.time() - start_time)
            
        except Exception as e:
            self._log(f"Error in analysis pipeline: {str(e)}", level=1)
            raise 
      
    def run_pipeline(self, 
                    output_dir: Optional[str] = None,
                    skip_smoothing: bool = False,
                    skip_binarization: bool = False,
                    skip_regions: bool = False) -> None:
        """Run the complete vessel analysis pipeline with visualization and data saving.
        
        This executes all analysis steps in sequence and saves results:
        1. Extract ROI
        2. Smooth volume (optional)
        3. Binarize vessels (optional)
        4. Create skeleton
        5. Determine regions (optional)
        6. Generate visualizations
        7. Save analysis results
        
        Args:
            output_dir: Directory to save results. If None, creates timestamped directory.
            skip_smoothing: Whether to skip the smoothing step
            skip_binarization: Whether to skip the binarization step
            skip_regions: Whether to skip the region detection step
        """
        from datetime import datetime
        import matplotlib.pyplot as plt
        import pandas as pd
        from .plotting import plot_projections, plot_mean_zprofile, plot_path_projections
        
        start_time = time.time()
        self._log("Starting pipeline with visualization and saving...", level=1)
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"results_{Path(self.czi_path).stem}_{timestamp}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        self._log(f"Results will be saved to: {output_path}", level=1)
        
        try:
            # Run the analysis steps
            self.run_analysis(
                skip_smoothing=skip_smoothing,
                skip_binarization=skip_binarization,
                skip_regions=skip_regions
            )
            
            # Save visualizations
            self._log("Generating visualizations...", level=1)
            
            # Projections
            self._log("Creating smoothed projections...", level=2)
            fig1, axes1 = plot_projections(self, mode='smoothed')
            fig1.savefig(output_path / 'projections_smoothed.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            self._log("Creating binary projections...", level=2)
            fig2, axes2 = plot_projections(self, mode='binary')
            fig2.savefig(output_path / 'projections_binary.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            self._log("Creating depth-coded projections...", level=2)
            fig3, axes3 = plot_projections(self, mode='binary', depth_coded=True)
            fig3.savefig(output_path / 'projections_depth.png', dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            # Z-profile
            self._log("Creating vessel distribution plot...", level=2)
            fig4, axes4 = plot_mean_zprofile(self)
            fig4.savefig(output_path / 'vessel_distribution.png', dpi=300, bbox_inches='tight')
            plt.close(fig4)
            
            # Path projections
            if hasattr(self, 'paths'):
                self._log("Creating path projections...", level=2)
                fig5, axes5 = plot_path_projections(self)
                fig5.savefig(output_path / 'path_projections.png', dpi=300, bbox_inches='tight')
                plt.close(fig5)
            
            # Save analysis results
            self._log("Saving analysis results...", level=1)
            
            # Create Excel writer
            excel_path = output_path / 'analysis_results.xlsx'
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Save metadata
                self._log("Saving metadata...", level=2)
                metadata = pd.DataFrame({
                    'Parameter': ['Input File', 'Timestamp', 'Config Used', 
                                'Smoothing Applied', 'Binarization Applied', 
                                'Region Detection Applied'],
                    'Value': [
                        str(self.czi_path),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        str(self.config_path) if self.config_path else 'Default',
                        not skip_smoothing,
                        not skip_binarization,
                        not skip_regions
                    ]
                })
                metadata.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Save region bounds if available
                if hasattr(self, 'region_bounds'):
                    self._log("Saving region analysis...", level=2)
                    region_data = []
                    for region, (peak, sigma, bounds) in self.region_bounds.items():
                        region_data.append({
                            'Region': region,
                            'Peak Position': peak,
                            'Sigma': sigma,
                            'Lower Bound': bounds[0],
                            'Upper Bound': bounds[1]
                        })
                    regions_df = pd.DataFrame(region_data)
                    regions_df.to_excel(writer, sheet_name='Region Analysis', index=False)
                
                # Save mean z-profile
                self._log("Saving z-profile data...", level=2)
                z_profile = self.get_projection([1, 2], operation='mean')
                z_profile_df = pd.DataFrame({
                    'Z Position': np.arange(len(z_profile)),
                    'Mean Intensity': z_profile
                })
                z_profile_df.to_excel(writer, sheet_name='Z Profile', index=False)
                
                # Save vessel paths
                if hasattr(self, 'paths'):
                    self._log("Extracting and saving vessel paths...", level=2)
                    # Get coordinates of vessel paths with their labels
                    vessel_paths_df = pd.DataFrame()
                    
                    # Process each path
                    for path_id, path_coords in self.paths.items():
                        # Convert path coordinates to DataFrame
                        path_df = pd.DataFrame({
                            'Path_ID': path_id,
                            'X': path_coords[:, 2],  # Note: CZI files are typically ZYX order
                            'Y': path_coords[:, 1],
                            'Z': path_coords[:, 0]
                        })
                        vessel_paths_df = pd.concat([vessel_paths_df, path_df], ignore_index=True)
                    
                    # Sort by Path_ID, then Z, Y, X for better organization
                    vessel_paths_df = vessel_paths_df.sort_values(['Path_ID', 'Z', 'Y', 'X'])
                    
                    # Save to Excel
                    vessel_paths_df.to_excel(writer, sheet_name='Vessel Paths', index=False)
                    
                    # Add summary statistics
                    self._log("Calculating path statistics...", level=2)
                    path_stats = pd.DataFrame({
                        'Metric': ['Total Points', 'Number of Paths', 'Unique X', 'Unique Y', 'Unique Z', 
                                 'Min X', 'Max X', 'Min Y', 'Max Y', 'Min Z', 'Max Z'],
                        'Value': [
                            len(vessel_paths_df),
                            len(self.paths),
                            vessel_paths_df['X'].nunique(),
                            vessel_paths_df['Y'].nunique(),
                            vessel_paths_df['Z'].nunique(),
                            vessel_paths_df['X'].min(),
                            vessel_paths_df['X'].max(),
                            vessel_paths_df['Y'].min(),
                            vessel_paths_df['Y'].max(),
                            vessel_paths_df['Z'].min(),
                            vessel_paths_df['Z'].max()
                        ]
                    })
                    path_stats.to_excel(writer, sheet_name='Path Statistics', index=False)
                    
                    # Add per-path statistics
                    self._log("Calculating per-path statistics...", level=2)
                    path_details = []
                    for path_id, path_coords in self.paths.items():
                        path_data = vessel_paths_df[vessel_paths_df['Path_ID'] == path_id]
                        path_details.append({
                            'Path_ID': path_id,
                            'Number_of_Points': len(path_data),
                            'Z_Range': path_data['Z'].max() - path_data['Z'].min(),
                            'Y_Range': path_data['Y'].max() - path_data['Y'].min(),
                            'X_Range': path_data['X'].max() - path_data['X'].min(),
                            'Min_Z': path_data['Z'].min(),
                            'Max_Z': path_data['Z'].max(),
                            'Min_Y': path_data['Y'].min(),
                            'Max_Y': path_data['Y'].max(),
                            'Min_X': path_data['X'].min(),
                            'Max_X': path_data['X'].max()
                        })
                    
                    path_details_df = pd.DataFrame(path_details)
                    path_details_df.to_excel(writer, sheet_name='Path Details', index=False)
            
            self._log(f"Analysis complete! Results saved to: {output_path}", level=1, timing=time.time() - start_time)
            
        except Exception as e:
            self._log(f"Error in analysis pipeline: {str(e)}", level=1)
            # Save error log
            with open(output_path / 'error_log.txt', 'w') as f:
                f.write(f"Error in analysis pipeline:\n{str(e)}")
            raise
        
    def update_roi_position(self, center_x: int, center_y: int, micron_roi: Optional[float] = None) -> None:
        """Update the ROI center position and optionally its size.
        
        This method updates the ROI parameters and clears any previously computed results
        so they will be recomputed with the new ROI on next access.
        
        Args:
            center_x: New X coordinate of ROI center in pixels
            center_y: New Y coordinate of ROI center in pixels
            micron_roi: Optional new ROI size in microns. If None, keeps current size.
        """
        print(f"\nUpdating ROI parameters:")
        print(f"  Center: ({self.center_x}, {self.center_y}) -> ({center_x}, {center_y})")
        
        self.center_x = center_x
        self.center_y = center_y
        
        if micron_roi is not None:
            print(f"  Size: {self.micron_roi} -> {micron_roi} microns")
            self.micron_roi = micron_roi
        
        # Clear computed results since they're no longer valid
        for attr in ['roi_volume', 'smoothed', 'binary', 'skeleton', 'stats', 'region_bounds']:
            if hasattr(self, attr):
                delattr(self, attr)
                
        print("ROI updated. Next pipeline step will use new parameters.")

    def save_volume(self, 
                   output_dir: str,
                   save_original: bool = True,
                   save_smoothed: bool = True,
                   save_binary: bool = True,
                   save_separate: bool = False) -> None:
        """Save volume data as .tif files.
        
        Args:
            output_dir: Directory to save the .tif files
            save_original: Whether to save the original ROI volume
            save_smoothed: Whether to save the smoothed volume
            save_binary: Whether to save the binary volume
            save_separate: If True, saves each volume type as a separate file.
                          If False, saves all volumes in a single multi-channel file.
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure we have the volumes we want to save
        if not hasattr(self, 'roi_volume'):
            self.segment_roi()
        if save_smoothed and not hasattr(self, 'smoothed'):
            self.smooth()
        if save_binary and not hasattr(self, 'binary'):
            self.binarize()
            
        # Prepare volumes for saving
        volumes = []
        volume_names = []
        
        if save_original:
            volumes.append(self.roi_volume)
            volume_names.append('original')
        if save_smoothed:
            volumes.append(self.smoothed)
            volume_names.append('smoothed')
        if save_binary:
            volumes.append(self.binary.astype(np.uint8))  # Convert boolean to uint8
            volume_names.append('binary')
            
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