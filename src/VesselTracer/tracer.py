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

@dataclass
class VesselTracer:
    """Main class for vessel tracing pipeline.
    
    Attributes:
        input_data: Either a path to a CZI/TIF file (as string or Path object) or a 3D numpy array
        config_path: Optional path to YAML config file. If None, uses default config.
        pixel_sizes: Optional tuple of (z,y,x) pixel sizes in microns. Defaults to (1.0, 1.0, 1.0).
    """
    input_data: Union[str, Path, np.ndarray]
    config_path: Optional[Union[str, Path]] = None
    pixel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        # Convert string paths to Path objects
        if isinstance(self.input_data, str):
            self.input_data = Path(self.input_data)
        if isinstance(self.config_path, str):
            self.config_path = Path(self.config_path)
            
        # Initialize GPU mode as disabled
        self.gpu_available = GPU_AVAILABLE
        self.use_gpu = False
            
        self._load_config()
        self._load_image()
        self.volume = self.volume.astype("float32")
        
    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path is None:
            # Point to the config directory in the project root
            self.config_path = Path(__file__).parent.parent.parent / 'config' / 'default_vessel_config.yaml'
            if not self.config_path.exists():
                raise FileNotFoundError(f"Default config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # ROI settings
        self.find_roi = config['roi']['find_roi']
        self.micron_roi = config['roi']['micron_roi']
        self.center_x = config['roi']['center_x']
        self.center_y = config['roi']['center_y']
        
        # Scale bar settings
        self.scalebar_length = config['scalebar']['length']
        self.scalebar_x = config['scalebar']['x']
        self.scalebar_y = config['scalebar']['y']
        
        # Store micron values for reference
        self.micron_gauss_sigma = config['preprocessing']['gauss_sigma']
        self.micron_background_sigma = config['preprocessing']['background_sigma']
        self.micron_median_filter_size = config['preprocessing']['median_filter_size']
        self.micron_close_radius = config['preprocessing']['close_radius']
        
        # Pre-processing settings (will be converted to pixels after loading image)
        self.gauss_sigma_x = None  # Will be set in _convert_to_pixels
        self.gauss_sigma_y = None  # Will be set in _convert_to_pixels
        self.gauss_sigma_z = None  # Will be set in _convert_to_pixels
        self.background_sigma_x = None  # Will be set in _convert_to_pixels
        self.background_sigma_y = None  # Will be set in _convert_to_pixels
        self.background_sigma_z = None  # Will be set in _convert_to_pixels
        self.median_filter_size = None  # Will be set in _convert_to_pixels
        self.close_radius = None  # Will be set in _convert_to_pixels
        self.min_object_size = config['preprocessing']['min_object_size']
        self.prune_length = config['preprocessing']['prune_length']
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
        
    def _convert_to_pixels(self):
        """Convert micron-based parameters to pixel values using image resolution."""
        if self.pixel_size_x == 0 or self.pixel_size_y == 0 or self.pixel_size_z == 0:
            raise ValueError("Pixel sizes not set. Call _load_image first.")
            
        # Convert Gaussian sigma for each dimension (for regular smoothing)
        self.gauss_sigma_x = self.micron_gauss_sigma / self.pixel_size_x
        self.gauss_sigma_y = self.micron_gauss_sigma / self.pixel_size_y
        self.gauss_sigma_z = self.micron_gauss_sigma / self.pixel_size_z
        self.gauss_sigma = (self.gauss_sigma_z, self.gauss_sigma_y, self.gauss_sigma_x)
        
        # Convert background smoothing sigma for each dimension
        self.background_sigma_x = self.micron_background_sigma / self.pixel_size_x
        self.background_sigma_y = self.micron_background_sigma / self.pixel_size_y
        self.background_sigma_z = self.micron_background_sigma / self.pixel_size_z
        self.background_sigma = (self.background_sigma_z, self.background_sigma_y, self.background_sigma_x)
        
        # Convert median filter size (use average of x and y pixel sizes)
        avg_xy_pixel_size = (self.pixel_size_x + self.pixel_size_y) / 2
        self.median_filter_size = int(round(self.micron_median_filter_size / avg_xy_pixel_size))
        # Ensure odd size for median filter
        if self.median_filter_size % 2 == 0:
            self.median_filter_size += 1
            
        # Convert closing radius (use average of x and y pixel sizes)
        self.close_radius = int(round(self.micron_close_radius / avg_xy_pixel_size))
        
        self._log(f"Converted parameters to pixels:", level=2)
        self._log(f"  Regular smoothing sigma (µm): {self.micron_gauss_sigma:.1f}", level=2)
        self._log(f"  Regular smoothing sigma (pixels):", level=2)
        self._log(f"    X: {self.gauss_sigma_x:.1f}", level=2)
        self._log(f"    Y: {self.gauss_sigma_y:.1f}", level=2)
        self._log(f"    Z: {self.gauss_sigma_z:.1f}", level=2)
        self._log(f"  Background smoothing sigma (µm): {self.micron_background_sigma:.1f}", level=2)
        self._log(f"  Background smoothing sigma (pixels):", level=2)
        self._log(f"    X: {self.background_sigma_x:.1f}", level=2)
        self._log(f"    Y: {self.background_sigma_y:.1f}", level=2)
        self._log(f"    Z: {self.background_sigma_z:.1f}", level=2)
        self._log(f"  Median filter size: {self.micron_median_filter_size:.1f} µm -> {self.median_filter_size} pixels", level=2)
        self._log(f"  Closing radius: {self.micron_close_radius:.1f} µm -> {self.close_radius} pixels", level=2)

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
        """Load image data from file or array."""
        start_time = time.time()
        self._log("Loading image data...", level=1)
        
        self._log(f"Input data type: {type(self.input_data)}", level=2)

        if isinstance(self.input_data, (str, Path)):
            # Handle file input
            file_path = Path(self.input_data)
            if file_path.suffix.lower() == '.czi':
                self._load_czi(file_path)
            elif file_path.suffix.lower() in ['.tif', '.tiff']:
                self._load_tiff(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        elif isinstance(self.input_data, np.ndarray):
            # Handle array input
            if self.pixel_sizes is None:
                raise ValueError("pixel_sizes must be provided when input_data is an array")
            self._load_array(self.input_data)
        else:
            raise ValueError("input_data must be either a file path (string or Path) or a numpy array")
        
        # Convert micron parameters to pixels
        self._convert_to_pixels()
        
        # Normalize volume
        self._log(f"Image loaded. Shape: {self.volume.shape}", level=2)
        self._log(f"Pixel sizes (µm): X={self.pixel_size_x:.3f}, Y={self.pixel_size_y:.3f}, Z={self.pixel_size_z:.3f}", level=2)
        self._log("Image loading complete", level=1, timing=time.time() - start_time)

    def _load_czi(self, file_path: Path):
        """Load CZI file and extract pixel sizes."""
        with CziFile(file_path) as czi:
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

    def _load_tiff(self, file_path: Path):
        """Load TIFF file."""
        self.volume = tifffile.imread(file_path)
        
        # For TIFF files, use the provided pixel sizes or defaults
        self.pixel_size_z, self.pixel_size_y, self.pixel_size_x = self.pixel_sizes

    def _load_array(self, array: np.ndarray):
        """Load data from numpy array."""
        if array.ndim != 3:
            raise ValueError("Input array must be 3D")
        
        self.volume = array
        self.pixel_size_z, self.pixel_size_y, self.pixel_size_x = self.pixel_sizes

    def activate_gpu(self) -> bool:
        """Activate GPU mode for processing.
        
        Returns:
            bool: True if GPU mode was successfully activated, False otherwise.
        """
        if not self.gpu_available:
            print("Warning: Cannot activate GPU mode - CuPy is not available.")
            print("Please install CuPy with the appropriate CUDA version for your system.")
            return False
            
        try:
            # Test GPU functionality with a simple operation
            with cp.cuda.Device(0):
                test_array = cp.array([1, 2, 3])
                result = test_array + test_array
                if not isinstance(result, cp.ndarray):
                    raise RuntimeError("GPU test operation failed")
            
            self.use_gpu = True
            print("GPU mode activated successfully.")
            print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to activate GPU mode - {str(e)}")
            print("Falling back to CPU mode.")
            self.use_gpu = False
            return False
        
    def normalize_image(self) -> np.ndarray:
        """Normalize image to [0,1] range."""
        print("volume.shape: ", self.volume.shape)
        print("volume.min(): ", self.volume.min())
        print("volume.max(): ", self.volume.max())
        self.previous_volume = self.volume.copy()
        self.volume = (self.volume - self.volume.min()) / (self.volume.max() - self.volume.min())
        
    def segment_roi(self, remove_dead_frames: bool = True, dead_frame_threshold: float = 1.5) -> np.ndarray:
        """Extract and segment region of interest from volume.
        
        If find_roi is False, uses the entire volume. Otherwise, extracts a region
        of interest based on configured center position and size in microns.
        Optionally removes dead frames at start/end.
        
        Args:
            remove_dead_frames: Whether to remove low-intensity frames at start/end
            dead_frame_threshold: Number of standard deviations above minimum
                                intensity to use as threshold for dead frames
        
        Returns:
            np.ndarray: ROI volume extracted from main volume
        """
        start_time = time.time()
        self._log("Processing volume...", level=1)
        
        if not self.find_roi:
            self._log("Using entire volume (ROI finding disabled)", level=2)
            self.valid_frame_range = (0, self.volume.shape[0]-1)
            return self.volume
        else:
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
            self.volume = roi
            self._log(f"ROI extraction complete. Final shape: {roi.shape}", level=2)
            self._log("ROI extraction complete", level=1, timing=time.time() - start_time)
            return self.volume
    
    def median_filter(self) -> np.ndarray:
        """Apply median filter to volume."""
        start_time = time.time()
        self._log("Applying median filter...", level=1)
        
        if self.find_roi and not hasattr(self, 'valid_frame_range'): 
            self.segment_roi()
            
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for median filtering", level=2)
            # Convert to GPU array
            gpu_volume = cp.asarray(self.volume)
            
            # Apply median filter on GPU
            gpu_filtered = cp_ndimage.median_filter(gpu_volume, size=self.median_filter_size)
            
            # Convert back to CPU
            self.volume = cp.asnumpy(gpu_filtered)
        else:
            self.volume = median_filter(self.volume, size=self.median_filter_size)
        
        self._log("Median filter complete", level=1, timing=time.time() - start_time)
        return self.volume
    
    def background_smoothing(self, epsilon = 1e-6, mode = 'gaussian') -> np.ndarray:
        """Apply background smoothing to volume with proper handling of anisotropic voxels.
        
        This is used to estimate and remove the background intensity variation.
        Uses a larger smoothing kernel than regular smoothing.
        """
        start_time = time.time()
        self._log(f"Applying background smoothing with {mode}...", level=1)
        self._log(f"Using background sigma (pixels):", level=2)
        self._log(f"  X: {self.background_sigma_x:.1f}", level=2)
        self._log(f"  Y: {self.background_sigma_y:.1f}", level=2)
        self._log(f"  Z: {self.background_sigma_z:.1f}", level=2)
        self.unfiltered_volume = self.volume.copy()
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for background smoothing", level=2)
            # Convert to GPU array
            gpu_volume = cp.asarray(self.volume)
            
            if mode == 'gaussian':
                background_smooth = cp_ndimage.gaussian_filter(gpu_volume, sigma=self.background_sigma)
            elif mode == 'median':
                # For median filter, we still use the same size in all dimensions
                # since it's a discrete operation
                background_smooth = cp_ndimage.median_filter(gpu_volume, size=self.median_filter_size)
            
            # Convert back to CPU
            background_smooth = cp.asnumpy(background_smooth)
        else:
            if mode == 'gaussian':
                background_smooth = gaussian_filter(self.volume, sigma=self.background_sigma)
            elif mode == 'median':
                # For median filter, we still use the same size in all dimensions
                # since it's a discrete operation
                background_smooth = median_filter(self.volume, size=self.median_filter_size)
        
        self._log("Background smoothing complete", level=1, timing=time.time() - start_time)
        self.volume = self.volume / (epsilon + background_smooth)
        return self.volume

    def detrend(self) -> np.ndarray:
        """Remove linear trend from volume along z-axis.
        
        Corrects for linear attenuation in depth by:
        1. Computing mean intensity profile along z
        2. Fitting a linear trend
        3. Removing trend from each xy slice
        
        Returns:
            np.ndarray: Detrended volume
        """
        start_time = time.time()
        self._log("Detrending volume...", level=1)
        
        if self.find_roi and not hasattr(self, 'valid_frame_range'):
            self.segment_roi()
            
        Z, Y, X = self.volume.shape
        
        # Calculate mean intensity profile
        z_profile = np.mean(self.volume, axis=(1,2))
        z_positions = np.arange(Z)
        
        # Fit linear trend
        coeffs = np.polyfit(z_positions, z_profile, deg=1)
        trend = np.polyval(coeffs, z_positions)
        
        self._log(f"Linear trend coefficients: {coeffs}", level=2)
        
        # Calculate correction factors
        correction = trend / np.mean(trend)
        
        # Apply correction to each z-slice
        detrended = np.zeros_like(self.volume)
        for z in range(Z):
            detrended[z] = self.volume[z] / correction[z]
            
        # Normalize to [0,1] range
        self.unfiltered_volume = self.volume.copy()
        self.volume = detrended
        
        self._log("Detrending complete", level=1, timing=time.time() - start_time)
        return self.volume

    def smooth(self) -> np.ndarray:
        """Apply regular Gaussian smoothing to volume with proper handling of anisotropic voxels.
        
        This is used for noise reduction and vessel enhancement.
        Uses a smaller smoothing kernel than background smoothing.
        """
        start_time = time.time()
        self._log("Applying regular smoothing...", level=1)
        
        if self.find_roi and not hasattr(self, 'valid_frame_range'):
            self.segment_roi()
            
        self._log(f"Using regular smoothing sigma (pixels):", level=2)
        self._log(f"  X: {self.gauss_sigma_x:.1f}", level=2)
        self._log(f"  Y: {self.gauss_sigma_y:.1f}", level=2)
        self._log(f"  Z: {self.gauss_sigma_z:.1f}", level=2)
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for smoothing", level=2)
            # Convert to GPU array
            gpu_volume = cp.asarray(self.volume)
            
            # Apply 3D Gaussian filter with different sigma for each dimension
            gpu_smoothed = cp_ndimage.gaussian_filter(gpu_volume, sigma=self.gauss_sigma)
            
            # Convert back to CPU
            self.volume = self.smoothed = cp.asnumpy(gpu_smoothed)
        else:
            # Apply 3D Gaussian filter with different sigma for each dimension
            self.volume = self.smoothed = gaussian_filter(self.volume, sigma=self.gauss_sigma)
        
        self._log("Regular smoothing complete", level=1, timing=time.time() - start_time)
        return self.volume

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
            
        # Calculate triangle threshold using entire volume
        thresh = threshold_triangle(self.smoothed.ravel())
        self._log(f"Triangle threshold: {thresh:.3f}", level=2)
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for binarization", level=2)
            # Convert to GPU array
            gpu_volume = cp.asarray(self.volume)
            
            # Apply threshold on GPU
            bw_vol = gpu_volume > thresh
            
            # Convert back to CPU for morphological operations
            bw_vol = cp.asnumpy(bw_vol)
        else:
            # Apply threshold to entire volume at once
            bw_vol = self.volume > thresh
        
        # Remove small objects in 3D
        bw_vol = remove_small_objects(bw_vol, min_size=self.min_object_size)
        
        # Apply morphological operations
        if self.close_radius > 0:
            # Then do closing to connect nearby vessel segments
            self._log(f"Performing 3D closing with radius {self.close_radius}", level=2)
            bw_vol = binary_closing(bw_vol, ball(self.close_radius))
            
        self.binary = bw_vol
        self._log("Binarization complete", level=1, timing=time.time() - start_time)
        return bw_vol
        
    def trace_paths(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create vessel skeleton and trace paths.
        
        Returns:
            Tuple containing:
            - paths: Dictionary of branch paths with coordinates
            - stats: DataFrame with branch statistics
        """
        start_time = time.time()
        self._log("Tracing vessel paths...", level=1)
        
        if not hasattr(self, 'binary'):
            self.binarize()
            
        self._log(f"Skeletonizing binary volume of shape {self.binary.shape}", level=2)
        ske = sk_skeletonize(self.binary)
        self._log(f"Skeletonized volume of shape {ske.shape}", level=2)
        
        # Create Skeleton object for path analysis
        self.paths = Skeleton(ske)
        self._log("Created skeleton object", level=2)
        
        # Extract paths from skeleton
        # self.paths = {}
        coords = self.paths.coordinates
        # for i in range(1,self.paths.n_paths):
        #     print(i)
        #This is a very slow operation, lets see what we get until then
        # for i, path in enumerate(self.skeleton.paths):
        #     self._log(f"Processing path {i+1} out of {total_paths}", level=2)
        #     # Convert sparse matrix path to dense array of indices
        #     path_indices = path.toarray().flatten().nonzero()[0]
        #     # Get coordinates for these indices
        #     path_coords = coords[path_indices]
        #     self.paths[i] = path_coords  # Store the coordinates array
        
        # Get detailed statistics using skan's summarize function
        self.stats = summarize(self.paths, separator="-")
        
        self._log(f"Found {self.paths.paths.shape[0]} vessel paths", level=2)
        self._log("Path tracing complete", level=1, timing=time.time() - start_time)
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
        print(f"    find_roi       -> {self.find_roi}")
        if self.find_roi:
            print(f"    micron_roi     -> {self.micron_roi:8} [Size of region of interest in microns]")
            print(f"    center_x       -> {self.center_x:8} [X coordinate of ROI center]")
            print(f"    center_y       -> {self.center_y:8} [Y coordinate of ROI center]")
        
        print("\nScale Bar Settings:")
        print(f"    scalebar_length  -> {self.scalebar_length:8} [Length of scale bar in plot units]")
        print(f"    scalebar_x       -> {self.scalebar_x:8} [X position of scale bar in plot]")
        print(f"    scalebar_y       -> {self.scalebar_y:8} [Y position of scale bar in plot]")
        
        print("\nPre-processing Parameters:")
        print(f"    gauss_sigma      -> {self.micron_gauss_sigma:.1f} µm -> {self.gauss_sigma_x:.1f}, {self.gauss_sigma_y:.1f}, {self.gauss_sigma_z:.1f} pixels")
        print(f"    min_object_size  -> {self.min_object_size:8} [Minimum object size to keep]")
        print(f"    close_radius     -> {self.micron_close_radius:.1f} µm -> {self.close_radius} pixels")
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
            # Extract ROI
            # Normalize image before analysis
            self._log("0. Normalizing image...", level=1)
            self.normalize_image()
            
            self._log("1. Extracting ROI...", level=1)
            if self.find_roi:
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
            
            # Trace vessel paths
            self._log("4. Tracing vessel paths...", level=1)
            if not skip_trace:
                self.trace_paths()
            
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
                    save_smoothed: bool = True,
                    save_binary: bool = True,
                    save_separate: bool = False,

                    plot_projections: bool = True,
                    plot_regions: bool = True,
                    plot_paths: bool = True,

                ) -> None:
        """Run the complete analysis pipeline and save all outputs.
        
        Args:
            output_dir: Directory to save outputs
            save_original: Whether to save the original volume
            save_smoothed: Whether to save the smoothed volume
            save_binary: Whether to save the binary volume
            save_skeleton: Whether to save the skeleton volume
            save_volumes: Whether to save any volumes
            save_projections: Whether to save projections
            save_regions: Whether to save region analysis
            save_paths: Whether to save vessel paths
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
                save_smoothed=save_smoothed,
                save_binary=save_binary,
                save_separate=save_separate
            )
            
            # # Save projections if requested
            # if save_projections:
            #     self._log("10. Saving projections...", level=1)
            #     self.save_projections(output_dir)
            
            # # Save region analysis if requested and available
            # if save_regions and hasattr(self, 'region_bounds'):
            #     self._log("11. Saving region analysis...", level=1)
            #     self.save_region_analysis(output_dir)
            
            # # Save vessel paths if requested and available
            # if save_paths and hasattr(self, 'paths'):
            #     self._log("12. Saving vessel paths...", level=1)
            #     self.save_vessel_paths(output_dir)
            
            self._log("Pipeline complete!", level=1, timing=time.time() - start_time)

        if plot_projections:
            fig1, ax = plot_projections(self)   
        if plot_regions:
            fig2, ax = plot_regions(self)
        if plot_paths:
            fig3, ax = plot_paths(self)

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
        for attr in ['volume', 'smoothed', 'binary', 'skeleton', 'stats', 'region_bounds']:
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
        if save_smoothed and not hasattr(self, 'smoothed'):
            self.smooth()
        if save_binary and not hasattr(self, 'binary'):
            self.binarize()
            
        # Prepare volumes for saving
        volumes = []
        volume_names = []
        
        if save_original:
            volumes.append(self.volume)
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
        from datetime import datetime
        
        # Create metadata DataFrame
        self.metadata_df = pd.DataFrame({
            'Parameter': [
                'Input Type',
                'Timestamp',
                'Config Used',
                'ROI Finding',
                'Smoothing Applied',
                'Binarization Applied',
                'Region Detection Applied',
                'Volume Shape',
                'Pixel Sizes (Z,Y,X)'
            ],
            'Value': [
                'Numpy Array' if isinstance(self.input_data, np.ndarray) else 'File',
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(self.config_path),
                self.find_roi,
                hasattr(self, 'smoothed'),
                hasattr(self, 'binary'),
                hasattr(self, 'region_bounds'),
                str(self.volume.shape),
                str(self.pixel_sizes)
            ]
        })
        
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
        if hasattr(self, 'paths'):
            vessel_paths_data = []
            print(self.paths.paths.shape[0])
            for path_id in range(1,self.paths.paths.shape[0]):
                path_coords = self.paths.path_coordinates(path_id)
                # Convert path coordinates to DataFrame rows
                for i, coords in enumerate(path_coords):
                    vessel_paths_data.append({
                        'Path_ID': path_id,
                        'Point_Index': i,
                        'X': coords[2],  # Note: CZI files are typically ZYX order
                        'Y': coords[1],
                        'Z': coords[0]
                    })
            self.paths_df = pd.DataFrame(vessel_paths_data)
        else:
            self.paths_df = pd.DataFrame()
        
        # Store all DataFrames in a dictionary
        self.analysis_dfs = {
            'metadata': self.metadata_df,
            'regions': self.regions_df,
            'z_profile': self.z_profile_df,
            'paths': self.paths_df
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
                self.paths_df.to_excel(writer, sheet_name='Vessel Paths', index=False)