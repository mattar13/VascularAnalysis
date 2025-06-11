from typing import Optional, Dict, Any, Tuple, Union, List
import numpy as np
import time
from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, ball
from skimage.filters import threshold_otsu, threshold_triangle
import concurrent.futures

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

    def segment_roi(self, 
                   image_model: ImageModel,
                   remove_dead_frames: bool = True, 
                   dead_frame_threshold: float = 1.5) -> Optional[ROI]:
        """Extract and segment region of interest from volume.
        
        Args:
            image_model: Source ImageModel containing the full volume
            remove_dead_frames: Whether to remove low-intensity frames at start/end
            dead_frame_threshold: Number of standard deviations above minimum
                                intensity to use as threshold for dead frames
        
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
                    
                    self._log(f"Original z-range: 0-{len(z_profile)-1}", level=2)
                    self._log(f"Valid frame range: {frame_start}-{frame_end}", level=2)
                    self._log(f"Removed {frame_start} frames from start", level=2)
                    self._log(f"Removed {len(z_profile)-frame_end-1} frames from end", level=2)
                    
                    # Update ROI to exclude dead frames
                    roi = roi[frame_start:frame_end+1]
                    valid_frame_range = (frame_start, frame_end)
                else:
                    self._log("Warning: No frames found above threshold! Skipping this ROI.", level=1)
                    return None
                    
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
            
            # Process slices in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit all z-slices for processing
                future_to_slice = {
                    executor.submit(self._process_z_slice, image_like.volume[z], median_filter_size): z 
                    for z in range(n_slices)
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_slice):
                    z = future_to_slice[future]
                    try:
                        background_slice = future.result()
                        background_image[z] = background_slice
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
        
        # Apply morphological operations
        if close_radius > 0:
            # Then do closing to connect nearby vessel segments
            self._log(f"Performing 3D closing with radius {close_radius}", level=2)
            bw_vol = binary_closing(bw_vol, ball(close_radius))
            

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
        start_time = time.time()
        self._log("Applying morphological closing...", level=1)
        
        if image_like.binary is None:
            raise ValueError("No binary volume available for morphological closing")
            
        if radius is None:
            pixel_conversions = self._get_pixel_conversions(image_like)
            radius = pixel_conversions.get('close_radius', 1)
        
        self._log(f"Closing radius: {radius} pixels", level=2)
        
        # Apply closing
        closed_volume = binary_closing(image_like.binary, ball(radius))
        
        # Update the binary volume
        image_like.binary = closed_volume
        
        self._log("Morphological closing complete", level=1, timing=time.time() - start_time)
        return closed_volume 