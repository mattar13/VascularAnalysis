from dataclasses import dataclass
from pathlib import Path
import numpy as np
import xmltodict
from czifile import CziFile
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, binary_closing, ball, skeletonize as sk_skeletonize
from skimage.filters import threshold_otsu
from skan import Skeleton, summarize
import yaml
from typing import Optional, Dict, Any, Tuple, Union, List
from scipy.signal import find_peaks, peak_widths

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
        self.micron_roi = config['roi_volume']['micron_roi']
        self.center_x = config['roi_volume']['center_x']
        self.center_y = config['roi_volume']['center_y']
        
        # Scale bar settings
        self.scalebar_length = config['scalebar']['length']
        self.scalebar_x = config['scalebar']['x']
        self.scalebar_y = config['scalebar']['y']
        
        # Pre-processing settings
        self.gauss_sigma = config['preprocessing']['gauss_sigma']
        self.min_object_size = config['preprocessing']['min_object_size']
        self.close_radius = config['preprocessing']['close_radius']
        self.prune_length = config['preprocessing']['prune_length']
        
        # Region settings
        self.regions = config.get('regions', ['superficial', 'intermediate', 'deep'])
        self.region_peak_distance = config.get('region_peak_distance', 2)
        self.region_height_ratio = config.get('region_height_ratio', 0.80)
        self.region_n_stds = config.get('region_n_stds', 2)
        
        # Image properties (set by load_image)
        self.pixel_size_x = 0.0
        self.pixel_size_y = 0.0
        self.pixel_size_z = 0.0
        
    def _load_image(self):
        """Load CZI file and extract pixel sizes."""
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
        
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0,1] range."""
        img = (img - img.min()) / (img.max() - img.min())
        return img
        
    def segment_roi(self, remove_dead_frames: bool = False, dead_frame_threshold: float = 1.5) -> np.ndarray:
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
        # Convert ROI size from microns to pixels
        h_x = round(self.micron_roi/2 * 1/self.pixel_size_x)
        h_y = round(self.micron_roi/2 * 1/self.pixel_size_y)
        
        # Extract initial ROI
        roi = self.volume[:, 
                        self.center_y-h_y:self.center_y+h_y,
                        self.center_x-h_x:self.center_x+h_x]
        
        if remove_dead_frames:
            # Calculate mean intensity profile along z
            z_profile = np.mean(roi, axis=(1,2))
            
            # Find frames above threshold
            threshold = z_profile.min() + dead_frame_threshold * z_profile.std()
            valid_frames = np.where(z_profile > threshold)[0]
            
            if len(valid_frames) > 0:
                frame_start = valid_frames[0]
                frame_end = valid_frames[-1]
                
                print(f"Removing dead frames:")
                print(f"  Original z-range: 0-{len(z_profile)}")
                print(f"  Valid frame range: {frame_start}-{frame_end}")
                print(f"  Removed {frame_start} frames from start")
                print(f"  Removed {len(z_profile)-frame_end-1} frames from end")
                
                # Update ROI to exclude dead frames
                roi = roi[frame_start:frame_end+1]
                
                # Store frame range for reference
                self.valid_frame_range = (frame_start, frame_end)
            else:
                print("Warning: No frames found above threshold!")
                self.valid_frame_range = (0, len(z_profile)-1)
        else:
            self.valid_frame_range = (0, roi.shape[0]-1)
        
        self.roi_volume = roi
        return roi
        
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
        if not hasattr(self, 'roi_volume'):
            self.segment_roi()
            
        Z, Y, X = self.roi_volume.shape
        
        # Calculate mean intensity profile
        z_profile = np.mean(self.roi_volume, axis=(1,2))
        z_positions = np.arange(Z)
        
        # Fit linear trend
        coeffs = np.polyfit(z_positions, z_profile, deg=1)
        trend = np.polyval(coeffs, z_positions)
        
        # Calculate correction factors
        correction = trend / np.mean(trend)
        
        # Apply correction to each z-slice
        detrended = np.zeros_like(self.roi_volume)
        for z in range(Z):
            detrended[z] = self.roi_volume[z] / correction[z]
            
        # Normalize to [0,1] range
        self.roi_volume = self._normalize_image(detrended)
        
        return self.roi_volume
    
    def smooth(self) -> np.ndarray:
        """Apply Gaussian smoothing to ROI volume."""
        if not hasattr(self, 'roi_volume'):
            self.detrend()
        self.smoothed = gaussian_filter(self.roi_volume, sigma=self.gauss_sigma)
        return self.smoothed
        
    def binarize(self) -> np.ndarray:
        """Binarize the smoothed volume using Otsu thresholding.
        
        This method processes the volume slice by slice using a global threshold,
        removes small objects, and performs 3D closing.
        
        Returns:
            np.ndarray: Binary volume after thresholding and cleaning
        """
        if not hasattr(self, 'smoothed'):
            self.smooth()
            
        Z, Y, X = self.smoothed.shape
        bw_vol = np.zeros((Z, Y, X), dtype=bool)
        
        # Calculate global threshold using all slices
        thresh = threshold_otsu(self.smoothed.ravel())
        print(f"Global Otsu threshold: {thresh:.3f}")
        
        # Process each slice
        for z in range(Z):
            img = self.smoothed[z]
            bw = img > thresh
            
            # Remove small objects
            bw_removed = remove_small_objects(bw, min_size=self.min_object_size)
            bw_vol[z] = bw_removed
            
        # 3D closing
        if self.close_radius > 0:
            bw_vol = binary_closing(bw_vol, ball(self.close_radius))
            
        self.binary = bw_vol
        return bw_vol
        
    def skeletonize(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create and analyze vessel skeleton.
        
        Returns:
            Tuple containing:
            - paths: Dictionary of branch paths with coordinates
            - stats: DataFrame with branch statistics
        """
        if not hasattr(self, 'binary'):
            self.binarize()
            
        print(f"Skeletonizing binary volume of shape {self.binary.shape}")
        ske = sk_skeletonize(self.binary)
        print(f"Skeletonized volume of shape {ske.shape}")
        
        skeleton = Skeleton(ske)
        print(f"Created skeleton object")
        
        # Extract paths from skeleton
        self.paths = {}
        coords = skeleton.coordinates
        for i, path in enumerate(skeleton.paths):
            print(f"Path {i}: {path}")
            path_coords = coords[path]
            self.paths[i] = {
                'coords': path_coords,  # Full path coordinates
                'start': path_coords[0],  # Start point
                'end': path_coords[-1],   # End point
                'length': len(path_coords) # Path length
            }
        
        self.stats = summarize(skeleton, separator="-")
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
            print(f"Peak at z={pk:.1f}: σ ≈ {sigmas[i]:.2f}")
        
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
        
    def run_pipeline(self) -> None:
        """Run the complete vessel analysis pipeline.
        
        This executes all analysis steps in sequence:
        1. Extract ROI
        2. Smooth volume
        3. Binarize vessels
        4. Create skeleton
        5. Determine regions
        6. Analyze layers
        """
        print("\nRunning vessel analysis pipeline...")
        print("1. Extracting ROI...")
        self.segment_roi()
        
        print("2. Smoothing volume...")
        self.smooth()
        
        print("3. Binarizing vessels...")
        self.binarize()
        
        print("4. Creating skeleton...")
        self.skeletonize()
        
        print("5. Determining regions...")
        self.determine_regions()
        
        print("6. Analyzing layers...")
        self.analyze_layers()
        
        print("Pipeline complete!\n")
        
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