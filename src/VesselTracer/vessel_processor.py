#This is the controller class for the vessel processor

import numpy as np
import time
from typing import Optional
from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import remove_small_objects, binary_closing, ball
from skimage.filters import threshold_otsu, threshold_triangle

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    cp_ndimage = None

class VesselProcessor:
    """Controller class for vessel image processing operations.
    
    This class handles filtering, smoothing, and binarization operations
    on vessel data models.
    """
    
    def __init__(self, use_gpu: bool = False, verbose: int = 2):
        """Initialize the VesselProcessor.
        
        Args:
            use_gpu: Whether to attempt GPU acceleration
            verbose: Verbosity level for logging
        """
        self.verbose = verbose
        self.use_gpu = False
        self.gpu_available = GPU_AVAILABLE
        
        if use_gpu:
            self.activate_gpu()
    
    def _log(self, message: str, level: int = 1, timing: Optional[float] = None):
        """Log a message with appropriate verbosity level."""
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
            return False
            
        try:
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
            self.use_gpu = False
            return False
    
    def median_filter(self, vessel_data, config) -> np.ndarray:
        """Apply median filter to volume.
        
        Args:
            vessel_data: VesselData object containing the volume
            config: ConfigManager with filter settings
            
        Returns:
            np.ndarray: Filtered volume
        """
        start_time = time.time()
        self._log("Applying median filter...", level=1)
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for median filtering", level=2)
            gpu_volume = cp.asarray(vessel_data.volume)
            gpu_filtered = cp_ndimage.median_filter(gpu_volume, size=config.median_filter_size)
            vessel_data.volume = cp.asnumpy(gpu_filtered)
        else:
            vessel_data.volume = median_filter(vessel_data.volume, size=config.median_filter_size)
        
        self._log("Median filter complete", level=1, timing=time.time() - start_time)
        return vessel_data.volume
    
    def gaussian_filter(self, vessel_data, config, mode: str = 'smooth', epsilon: float = 1e-6) -> np.ndarray:
        """Apply Gaussian filtering to volume.
        
        Args:
            vessel_data: VesselData object containing the volume
            config: ConfigManager with smoothing settings
            mode: Filtering mode ('smooth' for regular smoothing, 'background' for background correction)
            epsilon: Small value to prevent division by zero (used in background mode)
            
        Returns:
            np.ndarray: Filtered volume
        """
        start_time = time.time()
        
        if mode == 'smooth':
            self._log("Applying regular Gaussian smoothing...", level=1)
            sigma = config.gauss_sigma
        elif mode == 'background':
            self._log("Applying background Gaussian smoothing...", level=1)
            sigma = config.background_sigma
            vessel_data.unfiltered_volume = vessel_data.volume.copy()
        else:
            raise ValueError("Mode must be 'smooth' or 'background'")
        
        self._log(f"Using sigma (pixels): {sigma}", level=2)
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for Gaussian filtering", level=2)
            gpu_volume = cp.asarray(vessel_data.volume)
            gpu_filtered = cp_ndimage.gaussian_filter(gpu_volume, sigma=sigma)
            filtered_volume = cp.asnumpy(gpu_filtered)
        else:
            filtered_volume = gaussian_filter(vessel_data.volume, sigma=sigma)
        
        if mode == 'smooth':
            vessel_data.volume = vessel_data.smoothed = filtered_volume
        elif mode == 'background':
            # Background correction: divide original by smoothed background
            vessel_data.volume = vessel_data.volume / (epsilon + filtered_volume)
        
        self._log(f"Gaussian {mode} filtering complete", level=1, timing=time.time() - start_time)
        return vessel_data.volume
    
    def detrend(self, vessel_data) -> np.ndarray:
        """Remove linear trend from volume along z-axis.
        
        Args:
            vessel_data: VesselData object containing the volume
        
        Returns:
            np.ndarray: Detrended volume
        """
        start_time = time.time()
        self._log("Detrending volume...", level=1)
        
        Z, Y, X = vessel_data.volume.shape
        
        # Calculate mean intensity profile along z
        z_profile = np.mean(vessel_data.volume, axis=(1,2))
        z_positions = np.arange(Z)
        
        # Fit linear trend
        coeffs = np.polyfit(z_positions, z_profile, deg=1)
        trend = np.polyval(coeffs, z_positions)
        
        self._log(f"Linear trend coefficients: {coeffs}", level=2)
        
        # Calculate correction factors
        correction = trend / np.mean(trend)
        
        # Apply correction to each z-slice
        detrended = np.zeros_like(vessel_data.volume)
        for z in range(Z):
            detrended[z] = vessel_data.volume[z] / correction[z]
            
        # Store unfiltered volume
        vessel_data.unfiltered_volume = vessel_data.volume.copy()
        vessel_data.volume = detrended
        
        self._log("Detrending complete", level=1, timing=time.time() - start_time)
        return vessel_data.volume
    
    def binarize(self, vessel_data, config) -> np.ndarray:
        """Binarize the volume using threshold-based segmentation.
        
        Args:
            vessel_data: VesselData object containing the volume
            config: ConfigManager with binarization settings
            
        Returns:
            np.ndarray: Binary volume after thresholding and cleaning
        """
        start_time = time.time()
        self._log("Binarizing volume...", level=1)
        
        # Use smoothed volume if available, otherwise current volume
        volume_to_binarize = vessel_data.smoothed if vessel_data.smoothed is not None else vessel_data.volume
            
        # Calculate threshold based on method
        if config.binarization_method == 'triangle':
            thresh = threshold_triangle(volume_to_binarize.ravel())
        elif config.binarization_method == 'otsu':
            thresh = threshold_otsu(volume_to_binarize.ravel())
        else:
            raise ValueError(f"Unknown binarization method: {config.binarization_method}")
            
        self._log(f"{config.binarization_method.title()} threshold: {thresh:.3f}", level=2)
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for binarization", level=2)
            gpu_volume = cp.asarray(volume_to_binarize)
            bw_vol = gpu_volume > thresh
            bw_vol = cp.asnumpy(bw_vol)
        else:
            bw_vol = volume_to_binarize > thresh
        
        # Remove small objects in 3D
        bw_vol = remove_small_objects(bw_vol, min_size=config.min_object_size)
        
        # Apply morphological operations
        if config.close_radius > 0:
            self._log(f"Performing 3D closing with radius {config.close_radius}", level=2)
            bw_vol = binary_closing(bw_vol, ball(config.close_radius))
            
        vessel_data.binary = bw_vol
        self._log("Binarization complete", level=1, timing=time.time() - start_time)
        return bw_vol