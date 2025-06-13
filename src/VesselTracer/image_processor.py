from typing import Optional, Dict, Any, Tuple, Union, List
import numpy as np
import time
from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, ball
from skimage.filters import threshold_otsu, threshold_triangle
import concurrent.futures
from scipy.signal import find_peaks, peak_widths
from tqdm import tqdm
import multiprocessing
from scipy.interpolate import BSpline, make_smoothing_spline

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    cp_ndimage = None

# Import the data models from their respective modules
from .image_model import ImageModel, ROI
from .config import VesselTracerConfig

class ImageProcessor:
    """Image processing functionality for vessel analysis.
    
    This class contains core image processing methods that operate on ImageModel
    and ROI objects. It handles GPU acceleration, logging, and maintains 
    separation between data models and processing logic.
    
    Core operations:
    - Normalization
    - Median filtering for background subtraction
    - Gaussian smoothing
    - Binarization (thresholding)
    - Morphological operations (opening, closing)
    """
    
    def __init__(self, 
                 config: VesselTracerConfig,
                 verbose: int = 2,
                 use_gpu: bool = False):
        """Initialize the ImageProcessor.
        
        Args:
            config: VesselTracerConfig object with processing parameters
            verbose: Verbosity level for logging (0-3)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.config = config
        self.verbose = verbose
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu_available = GPU_AVAILABLE
        
        # Cache for converted pixel parameters
        self._pixel_conversions = {}

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

    def _get_pixel_conversions(self, image_like: Union[ImageModel, ROI]) -> Dict[str, Any]:
        """Get pixel conversions for the given image-like object, using cache if available."""
        pixel_sizes = image_like.get_pixel_sizes()
        cache_key = tuple(pixel_sizes)
        
        if cache_key not in self._pixel_conversions:
            self._pixel_conversions[cache_key] = self.config.convert_to_pixels(pixel_sizes)
            
        return self._pixel_conversions[cache_key]

    def normalize_image(self, image_like: Union[ImageModel, ROI]) -> np.ndarray:
        """Normalize image to [0,1] range.
        
        Args:
            image_like: ImageModel or ROI object to normalize
            
        Returns:
            np.ndarray: Normalized volume
        """
        if image_like.volume is None:
            raise ValueError("No volume data available for normalization")
            
        self._log("Normalizing image to [0,1] range...", level=2)
        self._log(f"Original range: [{image_like.volume.min():.3f}, {image_like.volume.max():.3f}]", level=2)
        
        # Store original volume as backup
        original_volume = image_like.volume.copy()
        
        # Normalize to [0,1]
        vol_min = image_like.volume.min()
        vol_max = image_like.volume.max()
        
        if vol_max > vol_min:
            image_like.volume = (image_like.volume - vol_min) / (vol_max - vol_min)
        else:
            self._log("Warning: Volume has constant intensity, cannot normalize", level=1)
            
        self._log(f"Normalized range: [{image_like.volume.min():.3f}, {image_like.volume.max():.3f}]", level=2)
        return image_like.volume

    def remove_dead_frames(self, image_like: Union[ImageModel, ROI], operation: str = 'median') -> np.ndarray:
        """Remove frames with unusually low intensity (dead frames).
        
        Supports two methods for dead frame removal:
        1. 'threshold': Uses statistical thresholding based on mean and std
        2. 'peaks': Uses peak finding to determine valid frame range
        
        Args:
            image_like: ImageModel or ROI object containing the volume to process
            method: Method to use for dead frame removal ('threshold' or 'peaks')
            frames_from_edge: Number of frames to remove from the edges of the volume
        Returns:
            np.ndarray: Volume with dead frames removed
        """
        method = self.config.dead_frame_method
        frames_from_edge = self.config.dead_frame_frames_from_edge
        
        start_time = time.time()
        self._log(f"Removing dead frames using {method} method...", level=1)
        
        if image_like.volume is None:
            raise ValueError("No volume data available for dead frame removal")
            
        if operation == 'mean':
            # Calculate mean intensity for each z-slice
            mean_intensities = np.mean(image_like.volume, axis=(1,2))
        elif operation == 'median':
            mean_intensities = np.median(image_like.volume, axis=(1,2))
        else:
            raise ValueError(f"Unknown operation: {operation}. Must be either 'mean' or 'median'")  
        
        if method == 'threshold':
            # Use statistical thresholding method
            threshold = self.config.dead_frame_threshold
            self._log(f"Using threshold method with threshold: {threshold}", level=2)
            
            # Calculate statistics
            mean_intensity = np.mean(mean_intensities)
            std_intensity = np.std(mean_intensities)
            
            # Calculate threshold value
            threshold_value = mean_intensity - (threshold * std_intensity)
            
            # Find dead frames
            dead_frames = mean_intensities < threshold_value
            n_dead = np.sum(dead_frames)
            
            if n_dead > 0:
                self._log(f"Found {n_dead} dead frames", level=2)
                self._log(f"Mean intensity: {mean_intensity:.2f}", level=2)
                self._log(f"Std intensity: {std_intensity:.2f}", level=2)
                self._log(f"Threshold value: {threshold_value:.2f}", level=2)
                
                # Remove dead frames by keeping only valid frames
                volume = image_like.volume[~dead_frames]
                
                # Update the valid frame range in the ROI model
                if hasattr(image_like, 'valid_frame_range'):
                    valid_frames = np.where(~dead_frames)[0]
                    image_like.valid_frame_range = (valid_frames[0], valid_frames[-1])
                
                self._log(f"Removed {n_dead} dead frames. New volume shape: {volume.shape}", level=2)
            else:
                self._log("No dead frames found", level=2)
                volume = image_like.volume
                
        elif method == 'peaks':
            # Use peak finding method
            self._log("Using peak finding method", level=2)
            
            # Find peaks in mean intensity profile
            peaks, _ = find_peaks(mean_intensities, distance=self.config.region_peak_distance)
            
            if len(peaks) < 2:
                raise ValueError("Not enough peaks found to determine valid frame range")
                
            # Get first and last peaks
            first_peak = peaks[0]
            last_peak = peaks[-1]
            
            # Calculate peak widths
            widths_all, _, _, _ = peak_widths(
                mean_intensities, peaks, rel_height=self.config.region_height_ratio)
            
            # Get widths for first and last peaks
            if frames_from_edge < 0:
                first_width = widths_all[0]
                last_width = widths_all[-1]
            else:
                first_width = frames_from_edge
                last_width = frames_from_edge
            
            # Calculate valid frame range
            start_frame = max(0, int(first_peak - first_width))
            end_frame = min(len(mean_intensities) - 1, int(last_peak + last_width))
            
            self._log(f"First peak at frame {first_peak} with width {first_width:.1f}", level=2)
            self._log(f"Last peak at frame {last_peak} with width {last_width:.1f}", level=2)
            self._log(f"Valid frame range: {start_frame} to {end_frame}", level=2)
            
            # Create mask for valid frames
            valid_frames = np.zeros(len(mean_intensities), dtype=bool)
            valid_frames[start_frame:end_frame + 1] = True
            
            # Remove frames outside valid range
            volume = image_like.volume[valid_frames]
            
            # Update the valid frame range in the ROI model
            if hasattr(image_like, 'valid_frame_range'):
                image_like.valid_frame_range = (start_frame, end_frame)
            
            n_removed = len(mean_intensities) - len(volume)
            self._log(f"Removed {n_removed} frames outside valid range. New volume shape: {volume.shape}", level=2)
            
        else:
            raise ValueError(f"Unknown method: {method}. Must be either 'threshold' or 'peaks'")
        
        self._log("Dead frame removal complete", level=1, timing=time.time() - start_time)
        return volume

    def segment_roi(self, image_model: ImageModel) -> Optional[ROI]:
        """Extract and segment region of interest from volume.
        
        Args:
            image_model: Source ImageModel containing the full volume
        
        Returns:
            ROI: ROI object containing the extracted region, or None if no valid frames found
        """
        start_time = time.time()
        self._log("Extracting ROI...", level=1)
        
        if image_model.volume is None:
            raise ValueError("No volume data available for ROI extraction")
        
        if not self.config.find_roi:
            self._log("Using entire volume (ROI finding disabled)", level=2)
            # Create ROI model using the entire volume
            roi_model = ROI(
                volume=image_model.volume.copy(),
                pixel_size_x=image_model.pixel_size_x,
                pixel_size_y=image_model.pixel_size_y,
                pixel_size_z=image_model.pixel_size_z,
                min_x=0,
                min_y=0,
                max_x=image_model.volume.shape[2],
                max_y=image_model.volume.shape[1]
            )
            return roi_model
        else:
            # Convert ROI size from microns to pixels
            roi_x = round(self.config.micron_roi * 1/image_model.pixel_size_x)
            roi_y = round(self.config.micron_roi * 1/image_model.pixel_size_y)
            
            self._log(f"ROI size: {roi_x}x{roi_y} pixels", level=2)
            
            # Extract initial ROI using min coordinates
            roi = image_model.volume[:, 
                            self.config.min_y:self.config.min_y+roi_y,
                            self.config.min_x:self.config.min_x+roi_x]
            
            valid_frame_range = (0, roi.shape[0]-1)
                    
            # Create ROI model
            roi_model = ROI(
                volume=roi,
                pixel_size_x=image_model.pixel_size_x,
                pixel_size_y=image_model.pixel_size_y,
                pixel_size_z=image_model.pixel_size_z,
                min_x=self.config.min_x,
                min_y=self.config.min_y,
                dx=roi_x,
                dy=roi_y
            )
            
            # Store the valid frame range for reference
            roi_model.valid_frame_range = valid_frame_range
            
            self._log(f"ROI extraction complete. Final shape: {roi.shape}", level=2)
            self._log("ROI extraction complete", level=1, timing=time.time() - start_time)
            return roi_model

    def _process_z_slice(self, z_slice: np.ndarray, median_filter_size: int) -> np.ndarray:
        """Process a single z-slice for median filtering to estimate background.
        
        Args:
            z_slice: 2D array representing a single z-slice
            median_filter_size: Size of median filter
            
        Returns:
            Background slice estimated using median filter
        """
        # Create background image using median filter
        background_slice = median_filter(z_slice, size=median_filter_size)
        
        return background_slice

    def estimate_background(self, image_like: Union[ImageModel, ROI]) -> np.ndarray:
        """Apply median filter for background subtraction using multithreading.
        
        Creates a background image using median filtering. This technique is commonly 
        used for background estimation in microscopy images.
        
        Args:
            image_like: ImageModel or ROI object containing the volume to process
            
        Returns:
            np.ndarray: Estimated background volume
        """
        start_time = time.time()
        self._log("Estimating background using median filter...", level=1)
        
        if image_like.volume is None:
            raise ValueError("No volume data available for median filtering")
            
        # Get pixel conversions
        pixel_conversions = self._get_pixel_conversions(image_like)
        median_filter_size = pixel_conversions['median_filter_size']
        
        self._log(f"Median filter size: {median_filter_size} pixels", level=2)
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for median filtering", level=2)
            # Convert to GPU array
            gpu_volume = cp.asarray(image_like.volume)
            
            # Create background image using median filter
            gpu_background = cp_ndimage.median_filter(gpu_volume, size=median_filter_size)
            
            # Convert back to CPU
            background_image = cp.asnumpy(gpu_background)
        else:
            self._log("Using CPU multithreading for median filtering", level=1)
            
            # Get number of z-slices
            n_slices = image_like.volume.shape[0]
            
            # Create empty array for background
            background_image = np.zeros_like(image_like.volume)
            
            # Get max_workers from config
            max_workers = self.config.max_workers
            
            # Get number of available CPU cores
            cpu_count = multiprocessing.cpu_count()
            self._log(f"Available CPU cores: {cpu_count}", level=2)
            
            # Process slices in parallel with progress bar
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Log number of workers being used
                if max_workers is None:
                    self._log(f"Using default number of worker threads ({cpu_count})", level=2)
                else:
                    self._log(f"Using {max_workers} worker threads (out of {cpu_count} available cores)", level=2)
                
                # Submit all z-slices for processing
                future_to_slice = {
                    executor.submit(self._process_z_slice, image_like.volume[z], median_filter_size): z 
                    for z in range(n_slices)
                }
                
                # Process results as they complete with progress bar
                with tqdm(total=n_slices, desc="Processing slices", unit="slice") as pbar:
                    for future in concurrent.futures.as_completed(future_to_slice):
                        z = future_to_slice[future]
                        try:
                            background_slice = future.result()
                            background_image[z] = background_slice
                            pbar.update(1)
                        except Exception as e:
                            self._log(f"Error processing slice {z}: {str(e)}", level=1)
        
        self._log("Background estimation complete", level=1, timing=time.time() - start_time)
        self._log(f"Background volume range: [{background_image.min():.3f}, {background_image.max():.3f}]", level=2)
        
        return background_image

    def detrend_volume(self, image_like: Union[ImageModel, ROI]) -> np.ndarray:
        """Remove linear trend from volume along z-axis.
        
        Corrects for linear attenuation in depth by:
        1. Computing mean intensity profile along z
        2. Fitting a linear trend
        3. Removing trend from each xy slice
        
        Args:
            image_like: ImageModel or ROI object containing the volume to detrend
            
        Returns:
            np.ndarray: Detrended volume
        """
        start_time = time.time()
        self._log("Detrending volume...", level=1)
        
        if image_like.volume is None:
            raise ValueError("No volume data available for detrending")
            
        Z, Y, X = image_like.volume.shape
        
        # Calculate mean intensity profile
        z_profile = np.mean(image_like.volume, axis=(1,2))
        z_positions = np.arange(Z)
        
        # Fit linear trend
        coeffs = np.polyfit(z_positions, z_profile, deg=1)
        trend = np.polyval(coeffs, z_positions)
        
        self._log(f"Linear trend coefficients: {coeffs}", level=2)
        
        # Calculate correction factors
        correction = trend / np.mean(trend)
        
        # Apply correction to each z-slice
        detrended = np.zeros_like(image_like.volume)
        for z in range(Z):
            detrended[z] = image_like.volume[z] / correction[z]
        
        self._log("Detrending complete", level=1, timing=time.time() - start_time)
        return detrended

    def smooth_volume(self, image_like: Union[ImageModel, ROI]) -> np.ndarray:
        """Apply regular Gaussian smoothing to volume with proper handling of anisotropic voxels.
        
        This is used for noise reduction and vessel enhancement.
        Uses a smaller smoothing kernel than background smoothing.
        
        Args:
            image_like: ImageModel or ROI object containing the volume to smooth
            
        Returns:
            np.ndarray: Smoothed volume
        """
        start_time = time.time()
        self._log("Applying regular smoothing...", level=1)
        
        if image_like.volume is None:
            raise ValueError("No volume data available for smoothing")
            
        # Get pixel conversions
        pixel_conversions = self._get_pixel_conversions(image_like)
        gauss_sigma = pixel_conversions['gauss_sigma']
        
        self._log(f"Using regular smoothing sigma (pixels):", level=2)
        self._log(f"  X: {pixel_conversions['gauss_sigma_x']:.1f}", level=2)
        self._log(f"  Y: {pixel_conversions['gauss_sigma_y']:.1f}", level=2)
        self._log(f"  Z: {pixel_conversions['gauss_sigma_z']:.1f}", level=2)
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for smoothing", level=2)
            # Convert to GPU array
            gpu_volume = cp.asarray(image_like.volume)
            
            # Apply 3D Gaussian filter with different sigma for each dimension
            gpu_smoothed = cp_ndimage.gaussian_filter(gpu_volume, sigma=gauss_sigma)
            
            # Convert back to CPU
            smoothed_volume = cp.asnumpy(gpu_smoothed)
        else:
            # Apply 3D Gaussian filter with different sigma for each dimension
            smoothed_volume = gaussian_filter(image_like.volume, sigma=gauss_sigma)
        
        self._log("Regular smoothing complete", level=1, timing=time.time() - start_time)
        return smoothed_volume

    def binarize_volume(self, image_like: Union[ImageModel, ROI], method: str = None) -> np.ndarray:
        """Binarize the smoothed volume using triangle thresholding.
        
        This method applies triangle thresholding to the entire 3D volume at once,
        removes small objects, and performs morphological operations to clean up
        the binary volume.
        
        Args:
            image_like: ImageModel or ROI object containing the volume to binarize
            method: Binarization method ('triangle' or 'otsu'). If None, uses config setting.
            
        Returns:
            np.ndarray: Binary volume after thresholding and cleaning
        """
        start_time = time.time()
        self._log("Binarizing volume...", level=1)
        
        if image_like.volume is None:
            raise ValueError("No volume data available for binarization")
            
        # Use method from config if not specified
        if method is None:
            method = self.config.binarization_method
            
        # Get pixel conversions
        pixel_conversions = self._get_pixel_conversions(image_like)
        close_radius = pixel_conversions['close_radius']
        
        # Calculate threshold using entire volume
        if method.lower() == 'triangle':
            thresh = threshold_triangle(image_like.volume.ravel())
        elif method.lower() == 'otsu':
            thresh = threshold_otsu(image_like.volume.ravel())
        else:
            raise ValueError(f"Unknown binarization method: {method}")
            
        self._log(f"{method} threshold: {thresh:.3f}", level=2)
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for binarization", level=2)
            # Convert to GPU array
            gpu_volume = cp.asarray(image_like.volume)
            
            # Apply threshold on GPU
            bw_vol = gpu_volume > thresh
            
            # Convert back to CPU for morphological operations
            bw_vol = cp.asnumpy(bw_vol)
        else:
            # Apply threshold to entire volume at once
            bw_vol = image_like.volume > thresh
        
        # Remove small objects in 3D
        bw_vol = remove_small_objects(bw_vol, min_size=self.config.min_object_size)

        self._log("Binarization complete", level=1, timing=time.time() - start_time)
        return bw_vol

    def morphological_opening(self, image_like: Union[ImageModel, ROI], radius: int = None) -> np.ndarray:
        """Apply morphological opening to binary volume.
        
        Opening removes small objects and separates touching objects.
        
        Args:
            image_like: ImageModel or ROI object containing binary volume
            radius: Radius of structuring element. If None, uses config value.
            
        Returns:
            np.ndarray: Binary volume after opening
        """
        start_time = time.time()
        self._log("Applying morphological opening...", level=1)
        
        if image_like.binary is None:
            raise ValueError("No binary volume available for morphological opening")
            
        if radius is None:
            pixel_conversions = self._get_pixel_conversions(image_like)
            radius = max(1, int(pixel_conversions.get('open_radius', 1)))
        
        self._log(f"Opening radius: {radius} pixels", level=2)
        
        # Apply opening
        opened_volume = binary_opening(image_like.binary, ball(radius))
        
        # Update the binary volume
        image_like.binary = opened_volume
        
        self._log("Morphological opening complete", level=1, timing=time.time() - start_time)
        return opened_volume

    def morphological_closing(self, image_like: Union[ImageModel, ROI], radius: int = None) -> np.ndarray:
        """Apply morphological closing to binary volume.
        
        Closing fills small gaps and connects nearby objects.
        
        Args:
            image_like: ImageModel or ROI object containing binary volume
            radius: Radius of structuring element. If None, uses config value.
            
        Returns:
            np.ndarray: Binary volume after closing
        """
        close_radius = self.config.close_radius
        start_time = time.time()
        self._log("Applying morphological closing...", level=1)
        
        if image_like.binary is None:
            raise ValueError("No binary volume available for morphological closing")
            
        self._log(f"Closing radius: {close_radius} pixels", level=2)
        
        # Apply closing
        closed_volume = binary_closing(image_like.binary, ball(close_radius))
        
        # Update the binary volume
        image_like.binary = closed_volume
        
        self._log("Morphological closing complete", level=1, timing=time.time() - start_time)
        return closed_volume

    def determine_regions(self, image_like: Union[ImageModel, ROI]) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
        """Determine vessel regions based on the mean z-profile.
        
        Uses peak finding to identify distinct vessel layers and calculates
        their boundaries based on peak widths.
        
        Args:
            image_like: ImageModel or ROI object containing the volume to analyze
            
        Returns:
            Dictionary mapping region names to tuples of:
            (peak_position, sigma, (lower_bound, upper_bound))
        """
        start_time = time.time()
        self._log("Determining vessel regions...", level=1)
        
        if image_like.volume is None:
            raise ValueError("No volume data available for region determination")
            
        # Get mean z-profile (xy projection)
        mean_zprofile = image_like.get_projection([1, 2], operation='mean')
        
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
        
        # Store the region bounds in the image object
        image_like.region_bounds = region_bounds
        
        self._log("Region determination complete", level=1, timing=time.time() - start_time)
        return region_bounds

    def create_region_map(self, image_like: Union[ImageModel, ROI], 
                         region_bounds: Optional[Dict[str, Tuple[float, float, Tuple[float, float]]]] = None) -> np.ndarray:
        """Create a volume where each z-position is labeled with its region number.
        
        Creates a 3D array the same size as the volume where each voxel is assigned
        a region number based on its z-position:
        - 0: Unknown/outside regions
        - 1: Superficial
        - 2: Intermediate  
        - 3: Deep
        
        Args:
            image_like: ImageModel or ROI object containing the volume to create region map for
            region_bounds: Optional dictionary of region bounds. If None, uses stored region_bounds.
            
        Returns:
            np.ndarray: Volume with region labels (0-3) for each voxel
        """
        start_time = time.time()
        self._log("Creating region map...", level=1)
        
        if image_like.volume is None:
            raise ValueError("No volume data available for region map creation")
        
        if region_bounds is None:
            region_bounds = image_like.region_bounds
            
        if not region_bounds:
            raise ValueError("No region bounds available. Run determine_regions first.")
        
        # Create region map with same shape as volume
        Z, Y, X = image_like.volume.shape
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
        
        self._log(f"Created region map with shape {region_map.shape}", level=2)
        self._log(f"Region assignments: 0=unknown, 1=superficial, 2=intermediate, 3=deep", level=2)
        
        # Log region statistics
        unique_regions, counts = np.unique(region_map, return_counts=True)
        total_voxels = region_map.size
        for region_num, count in zip(unique_regions, counts):
            region_names = {0: 'unknown', 1: 'superficial', 2: 'intermediate', 3: 'deep'}
            region_name = region_names.get(region_num, f'region_{region_num}')
            percentage = (count / total_voxels) * 100
            self._log(f"  {region_name}: {count:,} voxels ({percentage:.1f}%)", level=2)
        
        self._log("Region map creation complete", level=1, timing=time.time() - start_time)
        return region_map
        
    def _get_region_for_z(self, z_coord: float, region_bounds: Dict[str, Tuple[float, float, Tuple[float, float]]]) -> str:
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

    def get_xy_points_at_z(self, rbf, z_level: float, grid_size: int = 100) -> np.ndarray:
        """Find x,y points where the spline surface intersects a given z-level.
        
        Args:
            rbf: RBF interpolator for the surface
            z_level: Z-level to find points at
            grid_size: Size of the grid to sample points from
            
        Returns:
            np.ndarray: Array of (x,y) points where the surface intersects the z-level
        """
        # Create a grid of x,y points
        x_coords = np.linspace(0, self.config.micron_roi, grid_size)
        y_coords = np.linspace(0, self.config.micron_roi, grid_size)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Evaluate the surface at all grid points
        points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        z_values = rbf(points).reshape(X_grid.shape)
        
        # Find points where z_value is close to z_level
        tolerance = 0.1  # Adjust this based on your needs
        mask = np.abs(z_values - z_level) < tolerance
        
        # Get the x,y coordinates of these points
        significant_points = np.column_stack([
            X_grid[mask],
            Y_grid[mask]
        ])
        
        return significant_points

    def determine_regions_with_splines(self, image_like: Union[ImageModel, ROI], d_subroi: int = 25) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
        """Determine vessel regions by fitting B-spline surfaces to layer peaks.
        
        This method:
        1. Creates a pooled/binned version of the volume (x,y pooled, z preserved)
        2. Finds peaks in each pooled region's z-profile
        3. Fits a B-spline surface for each layer using the 3D peak coordinates
        4. Creates a region map based on the B-spline surfaces
        
        Args:
            image_like: ImageModel or ROI object containing the volume to analyze
            d_subroi: Size of sub-ROIs in pixels (both width and height)
            
        Returns:
            Dictionary mapping region names to tuples of:
            (peak_position, sigma, (lower_bound, upper_bound))
        """
        d_subroi = self.config.subroi_segmenting_size
        print("Sub-ROI Segmenting Size: ", d_subroi)
        start_time = time.time()
        self._log("Determining vessel regions using B-splines...", level=1)
        
        if image_like.volume is None:
            raise ValueError("No volume data available for region determination")
            
        Z, Y, X = image_like.volume.shape
        
        # First, get global mean z-profile to establish reference peak positions
        global_mean_zprofile = np.mean(image_like.volume, axis=(1,2))
        global_peaks, _ = find_peaks(global_mean_zprofile, distance=self.config.region_peak_distance)
            
        # Sort global peaks by z-position (ascending)
        global_peaks = np.sort(global_peaks[:len(self.config.regions)])
        
        # Calculate global peak widths
        global_widths, _, _, _ = peak_widths(
            global_mean_zprofile, global_peaks, 
            rel_height=self.config.region_height_ratio)
        
        # Convert global widths to sigmas
        global_sigmas = global_widths / (self.config.region_n_stds * np.sqrt(2 * np.log(2)))
        print("Global Peaks: ", global_peaks)
        print("Global Sigmas: ", global_sigmas)
        print("Global Widths: ", global_widths)
        
        # Calculate number of pooled regions in each dimension
        n_y = max(1, Y // d_subroi)
        n_x = max(1, X // d_subroi)
        print("Number of pooled regions: ", n_y, n_x)
        
        # Initialize arrays to store peak positions for each pooled region
        print("Initializing peak positions and layers...")
        image_like.peak_positions = []
        image_like.peak_layers = []
        for i in range(n_y):
            for j in range(n_x):
                y_start = i * d_subroi
                y_end = min((i + 1) * d_subroi, Y)
                x_start = j * d_subroi
                x_end = min((j + 1) * d_subroi, X)
                
                # Average the region
                z_profile = np.mean(
                    image_like.volume[:, y_start:y_end, x_start:x_end],
                    axis=(1,2)
                )
                
                # Find peaks
                peaks, _ = find_peaks(z_profile, distance=self.config.region_peak_distance)
                
                if len(peaks) > 0:
                    # For each global peak position, find the closest local peak
                    for k, global_peak in enumerate(global_peaks):
                        # Calculate distances to all local peaks
                        distances = np.abs(peaks - global_peak)
                        closest_idx = np.argmin(distances)
                        
                        # Store the peak position and layer
                        image_like.peak_positions.append((peaks[closest_idx], y_start + d_subroi/2, x_start + d_subroi/2))
                        image_like.peak_layers.append(k)
        
        # Convert to numpy arrays for easier filtering
        image_like.peak_positions = np.array(image_like.peak_positions)
        image_like.peak_layers = np.array(image_like.peak_layers)
        
        # Initialize arrays to store B-spline surfaces
        print("Fitting B-spline surfaces...")
        spline_surfaces = []
        
        # Fit B-spline surface for each layer
        for layer_idx in range(len(self.config.regions)):
            print(f"Fitting B-spline surface for layer {layer_idx}...")
            # Get points for this layer
            layer_mask = image_like.peak_layers == layer_idx
            
            if not np.any(layer_mask):
                continue
                
            layer_points = image_like.peak_positions[layer_mask]
            # print(f"Layer {layer_idx} points: {layer_points}")
            
            # Create B-spline surface
            from scipy.interpolate import RBFInterpolator
            x_points = layer_points[:, 1]
            y_points = layer_points[:, 2]
            z_points = layer_points[:, 0]

            # Create the interpolator - now using all 3 coordinates
            rbf = RBFInterpolator(
                np.column_stack((x_points, y_points)),  # Use all 3D coordinates
                z_points,  # Target is 0 for the surface
                kernel='thin_plate_spline',
                neighbors=20
            )
            
            spline_surfaces.append(rbf)
        
        # Store the spline surfaces
        image_like.spline_surfaces = spline_surfaces
        
        # Create region map using the spline surfaces
        region_map = np.zeros((Z, Y, X), dtype=np.uint8)
        region_numbers = {
            'superficial': 1,
            'intermediate': 2,
            'deep': 3
        }
        
        # Create coordinate grids for efficient evaluation
        print("Generating roi mask...")
        y_coords = np.arange(Y)
        x_coords = np.arange(X)
        Y_grid, X_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # For each z-slice, evaluate all surfaces at once
        for z in range(Z):
            print(f"Evaluating z-slice {z}...")
            # Create points for this slice
            points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
            
            # Evaluate each surface at this z-level
            surface_values = []
            for rbf in spline_surfaces:
                # Get predicted z values for all x,y points
                predicted_z = rbf(points).reshape(X_grid.shape)
                # Points where predicted z is less than current z are above the surface
                surface_values.append(predicted_z < z)
            
            # Create masks for each region
            if len(surface_values) >= 3:
                # Points above first surface are superficial
                superficial_mask = surface_values[0]
                # Points between first and second surface are intermediate
                intermediate_mask = (~surface_values[0]) & surface_values[1]
                # Points between second and third surface are deep
                deep_mask = (~surface_values[1]) & surface_values[2]
                
                # Assign region numbers
                region_map[z][superficial_mask] = region_numbers['superficial']
                region_map[z][intermediate_mask] = region_numbers['intermediate']
                region_map[z][deep_mask] = region_numbers['deep']
        
        # Store the region map
        image_like.region = region_map
        
        # Create region bounds dictionary using mean positions
        mean_peaks = np.array([np.mean(image_like.peak_positions[image_like.peak_layers == i, 0]) for i in range(len(self.config.regions))])
        mean_sigmas = global_sigmas  # Use global sigmas for now
        
        region_bounds = {
            region: (mu, sigma, (mu - sigma, mu + sigma))
            for region, mu, sigma in zip(self.config.regions, mean_peaks, mean_sigmas)
        }
        
        # Store the region bounds
        image_like.region_bounds = region_bounds
        
        self._log("Region determination complete", level=1, timing=time.time() - start_time)
        return region_bounds
