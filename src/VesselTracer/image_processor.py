from typing import Optional, Dict, Any, Tuple, Union, List
import numpy as np
import time
from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, ball, skeletonize as sk_skeletonize
from skimage.filters import threshold_otsu, threshold_triangle
from skan import Skeleton, summarize
from scipy.signal import find_peaks, peak_widths
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
    
    This class contains all image processing methods that operate on ImageModel
    and ROI objects. It handles GPU acceleration, logging, and maintains 
    separation between data models and processing logic.
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

    def _get_pixel_conversions(self, image_model: ImageModel) -> Dict[str, Any]:
        """Get pixel conversions for the given image model, using cache if available."""
        pixel_sizes = image_model.get_pixel_sizes()
        cache_key = tuple(pixel_sizes)
        
        if cache_key not in self._pixel_conversions:
            self._pixel_conversions[cache_key] = self.config.convert_to_pixels(pixel_sizes)
            
        return self._pixel_conversions[cache_key]

    def normalize_image(self, image_model: ImageModel) -> np.ndarray:
        """Normalize image to [0,1] range.
        
        Args:
            image_model: ImageModel object to normalize
            
        Returns:
            np.ndarray: Normalized volume
        """
        if image_model.volume is None:
            raise ValueError("No volume data available for normalization")
            
        self._log("Normalizing image to [0,1] range...", level=2)
        self._log(f"Original range: [{image_model.volume.min():.3f}, {image_model.volume.max():.3f}]", level=2)
        
        # Store original volume as backup
        original_volume = image_model.volume.copy()
        
        # Normalize to [0,1]
        vol_min = image_model.volume.min()
        vol_max = image_model.volume.max()
        
        if vol_max > vol_min:
            image_model.volume = (image_model.volume - vol_min) / (vol_max - vol_min)
        else:
            self._log("Warning: Volume has constant intensity, cannot normalize", level=1)
            
        self._log(f"Normalized range: [{image_model.volume.min():.3f}, {image_model.volume.max():.3f}]", level=2)
        return image_model.volume

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

    def _process_z_slice(self, z_slice: np.ndarray, median_filter_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single z-slice for median filtering.
        
        Args:
            z_slice: 2D array representing a single z-slice
            median_filter_size: Size of median filter
            
        Returns:
            Tuple containing (background_slice, corrected_slice)
        """
        # Create background image using median filter
        background_slice = median_filter(z_slice, size=median_filter_size)
        
        # Subtract background from original
        corrected_slice = z_slice - background_slice
        
        return background_slice, corrected_slice

    def median_filter_background_subtraction(self, image_model: ROI) -> Tuple[np.ndarray, np.ndarray]:
        """Apply median filter for background subtraction using multithreading.
        
        Creates a background image using median filtering and subtracts it from 
        the original image. This technique is commonly used for background 
        correction in microscopy images.
        
        Args:
            roi_model: ROI object containing the volume to process
            
        Returns:
            Tuple of (corrected_volume, background_volume)
        """
        start_time = time.time()
        self._log("Applying median filter for background subtraction...", level=1)
        
        if image_model.volume is None:
            raise ValueError("No volume data available for median filtering")
            
        # Get pixel conversions
        pixel_conversions = self._get_pixel_conversions(image_model)
        median_filter_size = pixel_conversions['median_filter_size']
        
        self._log(f"Median filter size: {median_filter_size} pixels", level=2)
        
        # Store original ROI before processing
        original_roi = image_model.volume.copy()
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for median filtering", level=2)
            # Convert to GPU array
            gpu_roi = cp.asarray(image_model.volume)
            
            # Create background image using median filter
            gpu_background = cp_ndimage.median_filter(gpu_roi, size=median_filter_size)
            
            # Subtract background from original
            gpu_corrected = gpu_roi - gpu_background
            
            # Convert back to CPU
            background_image = cp.asnumpy(gpu_background)
            corrected_volume = cp.asnumpy(gpu_corrected)
        else:
            self._log("Using CPU multithreading for median filtering", level=2)
            
            # Get number of z-slices
            n_slices = image_model.volume.shape[0]
            
            # Create empty arrays for results
            background_image = np.zeros_like(image_model.volume)
            corrected_volume = np.zeros_like(image_model.volume)
            
            # Process slices in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit all z-slices for processing
                future_to_slice = {
                    executor.submit(self._process_z_slice, image_model.volume[z], median_filter_size): z 
                    for z in range(n_slices)
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_slice):
                    z = future_to_slice[future]
                    try:
                        background_slice, corrected_slice = future.result()
                        background_image[z] = background_slice
                        corrected_volume[z] = corrected_slice
                    except Exception as e:
                        self._log(f"Error processing slice {z}: {str(e)}", level=1)
        
        # Update the ROI model
            image_model.volume = corrected_volume
        image_model.background = background_image
        self._log("Median filter background subtraction complete", level=1, timing=time.time() - start_time)
        self._log(f"Background subtracted ROI range: [{image_model.volume.min():.3f}, {image_model.volume.max():.3f}]", level=2)
        
        return corrected_volume, background_image

    def detrend_volume(self, image_model: ROI) -> np.ndarray:
        """Remove linear trend from volume along z-axis.
        
        Corrects for linear attenuation in depth by:
        1. Computing mean intensity profile along z
        2. Fitting a linear trend
        3. Removing trend from each xy slice
        
        Args:
            roi_model: ROI object containing the volume to detrend
            
        Returns:
            np.ndarray: Detrended volume
        """
        start_time = time.time()
        self._log("Detrending volume...", level=1)
        
        if image_model.volume is None:
            raise ValueError("No volume data available for detrending")
            
        Z, Y, X = image_model.volume.shape
        
        # Calculate mean intensity profile
        z_profile = np.mean(image_model.volume, axis=(1,2))
        z_positions = np.arange(Z)
        
        # Fit linear trend
        coeffs = np.polyfit(z_positions, z_profile, deg=1)
        trend = np.polyval(coeffs, z_positions)
        
        self._log(f"Linear trend coefficients: {coeffs}", level=2)
        
        # Calculate correction factors
        correction = trend / np.mean(trend)
        
        # Apply correction to each z-slice
        detrended = np.zeros_like(image_model.volume)
        for z in range(Z):
            detrended[z] = image_model.volume[z] / correction[z]
            
        # Update the ROI model
        image_model.volume = detrended
        
        self._log("Detrending complete", level=1, timing=time.time() - start_time)
        return detrended

    def smooth_volume(self, image_model: ROI) -> np.ndarray:
        """Apply regular Gaussian smoothing to volume with proper handling of anisotropic voxels.
        
        This is used for noise reduction and vessel enhancement.
        Uses a smaller smoothing kernel than background smoothing.
        
        Args:
            roi_model: ROI object containing the volume to smooth
            
        Returns:
            np.ndarray: Smoothed volume
        """
        start_time = time.time()
        self._log("Applying regular smoothing...", level=1)
        
        if image_model.volume is None:
            raise ValueError("No volume data available for smoothing")
            
        # Get pixel conversions
        pixel_conversions = self._get_pixel_conversions(image_model)
        gauss_sigma = pixel_conversions['gauss_sigma']
        
        self._log(f"Using regular smoothing sigma (pixels):", level=2)
        self._log(f"  X: {pixel_conversions['gauss_sigma_x']:.1f}", level=2)
        self._log(f"  Y: {pixel_conversions['gauss_sigma_y']:.1f}", level=2)
        self._log(f"  Z: {pixel_conversions['gauss_sigma_z']:.1f}", level=2)
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for smoothing", level=2)
            # Convert to GPU array
            gpu_roi = cp.asarray(image_model.volume)
            
            # Apply 3D Gaussian filter with different sigma for each dimension
            gpu_smoothed = cp_ndimage.gaussian_filter(gpu_roi, sigma=gauss_sigma)
            
            # Convert back to CPU
            smoothed_volume = cp.asnumpy(gpu_smoothed)
        else:
            # Apply 3D Gaussian filter with different sigma for each dimension
            smoothed_volume = gaussian_filter(image_model.volume, sigma=gauss_sigma)
        
        # Update the ROI model
        image_model.volume = smoothed_volume
        
        self._log("Regular smoothing complete", level=1, timing=time.time() - start_time)
        return smoothed_volume

    def binarize_volume(self, image_model: ROI, method: str = None) -> np.ndarray:
        """Binarize the smoothed volume using triangle thresholding.
        
        This method applies triangle thresholding to the entire 3D volume at once,
        removes small objects, and performs morphological operations to clean up
        the binary volume.
        
        Args:
            roi_model: ROI object containing the volume to binarize
            method: Binarization method ('triangle' or 'otsu'). If None, uses config setting.
            
        Returns:
            np.ndarray: Binary volume after thresholding and cleaning
        """
        start_time = time.time()
        self._log("Binarizing volume...", level=1)
        
        if image_model.volume is None:
            raise ValueError("No volume data available for binarization")
            
        # Use method from config if not specified
        if method is None:
            method = self.config.binarization_method
            
        # Get pixel conversions
        pixel_conversions = self._get_pixel_conversions(image_model)
        close_radius = pixel_conversions['close_radius']
        
        # Calculate threshold using entire ROI
        if method.lower() == 'triangle':
            thresh = threshold_triangle(image_model.volume.ravel())
        elif method.lower() == 'otsu':
            thresh = threshold_otsu(image_model.volume.ravel())
        else:
            raise ValueError(f"Unknown binarization method: {method}")
            
        self._log(f"{method} threshold: {thresh:.3f}", level=2)
        
        if self.use_gpu and GPU_AVAILABLE:
            self._log("Using GPU acceleration for binarization", level=2)
            # Convert to GPU array
            gpu_roi = cp.asarray(image_model.volume)
            
            # Apply threshold on GPU
            bw_vol = gpu_roi > thresh
            
            # Convert back to CPU for morphological operations
            bw_vol = cp.asnumpy(bw_vol)
        else:
            # Apply threshold to entire ROI at once
            bw_vol = image_model.volume > thresh
        
        # Remove small objects in 3D
        bw_vol = remove_small_objects(bw_vol, min_size=self.config.min_object_size)
        
        # Apply morphological operations
        if close_radius > 0:
            # Then do closing to connect nearby vessel segments
            self._log(f"Performing 3D closing with radius {close_radius}", level=2)
            bw_vol = binary_closing(bw_vol, ball(close_radius))
            
        # Store in ROI model
        image_model.binary = bw_vol
        self._log("Binarization complete", level=1, timing=time.time() - start_time)
        return bw_vol

    def determine_regions(self, image_model: ROI) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
        """Determine vessel regions based on the mean z-profile.
        
        Uses peak finding to identify distinct vessel layers and calculates
        their boundaries based on peak widths.
        
        Args:
            roi_model: ROI object containing binary data
            
        Returns:
            Dictionary mapping region names to tuples of:
            (peak_position, sigma, (lower_bound, upper_bound))
        """
        self._log("Determining vessel regions...", level=1)
        
        if image_model.binary is None:
            raise ValueError("No binary data available for region analysis")
            
        # Get mean z-profile (xy projection)
        mean_zprofile = np.mean(image_model.binary, axis=(1,2))
        
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
        
        return region_bounds

    def create_region_map_volume(self, image_model: ROI, region_bounds: Dict[str, Tuple[float, float, Tuple[float, float]]]) -> np.ndarray:
        """Create a volume where each z-position is labeled with its region number.
        
        Creates a 3D array the same size as the volume where each voxel is assigned
        a region number based on its z-position:
        - 0: Unknown/diving regions
        - 1: Superficial
        - 2: Intermediate  
        - 3: Deep
        
        Args:
            roi_model: ROI object to create region map for
            region_bounds: Dictionary of region bounds from determine_regions()
            
        Returns:
            np.ndarray: Volume with region labels (0-3) for each voxel
        """
        if image_model.volume is None:
            raise ValueError("No volume data available for region map creation")
        
        # Create region map with same shape as ROI
        Z, Y, X = image_model.volume.shape
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
        
        # Store in ROI model
        image_model.region = region_map
        self._log(f"Created region map volume with shape {region_map.shape}", level=2)
        self._log(f"Region assignments: 0=unknown, 1=superficial, 2=intermediate, 3=deep", level=2)
        
        # Log region statistics
        unique_regions, counts = np.unique(region_map, return_counts=True)
        total_voxels = region_map.size
        for region_num, count in zip(unique_regions, counts):
            region_names = {0: 'unknown', 1: 'superficial', 2: 'intermediate', 3: 'deep'}
            region_name = region_names.get(region_num, f'region_{region_num}')
            percentage = (count / total_voxels) * 100
            self._log(f"  {region_name}: {count:,} voxels ({percentage:.1f}%)", level=2)
        
        return region_map

    def _get_region_for_z(self, z_coord: float, region_bounds: Dict[str, Tuple[float, float, Tuple[float, float]]]) -> str:
        """Determine which region a z-coordinate falls into.
        
        Args:
            z_coord: Z-coordinate to check
            region_bounds: Dictionary of region bounds
            
        Returns:
            String indicating the region name ('superficial', 'intermediate', 'deep', or 'unknown')
        """
        for region, (_, _, (lower, upper)) in region_bounds.items():
            if lower <= z_coord <= upper:
                return region
        return 'unknown'


    def trace_vessel_paths(self, image_model: ROI, region_bounds: Dict[str, Tuple[float, float, Tuple[float, float]]], split_paths: bool = False) -> Tuple[Dict[str, Any], Any]:
        """Create vessel skeleton and trace paths.
        
        Args:
            roi_model: ROI object containing binary data
            region_bounds: Dictionary of region bounds
            split_paths: Whether to split paths at region boundaries
            
        Returns:
            Tuple containing:
            - paths: Dictionary of branch paths with coordinates
            - stats: DataFrame with branch statistics
        """
        start_time = time.time()
        self._log("Tracing vessel paths...", level=1)
        
        if image_model.binary is None:
            raise ValueError("No binary data available for path tracing")
            
        self._log(f"Skeletonizing binary volume of shape {image_model.binary.shape}", level=2)
        ske = sk_skeletonize(image_model.binary)
        self._log(f"Skeletonized volume of shape {ske.shape}", level=2)
        
        # Create Skeleton object for path analysis
        skeleton_obj = Skeleton(ske)
        self._log("Created skeleton object", level=2)
        
        # Get detailed statistics using skan's summarize function
        stats = summarize(skeleton_obj, separator="-")
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
                    
                if split_paths:
                    # Split path at region boundaries and create separate entries
                    path_segments = self._split_path_at_region_boundaries(path_coords, region_bounds)
                    
                    for region, segment_coords in path_segments:
                        if len(segment_coords) > 1:  # Only store segments with more than one point
                            paths_dict[path_id] = {
                                'original_path_id': i,
                                'region': region,
                                'coordinates': segment_coords,
                                'length': len(segment_coords)
                            }
                            path_id += 1
                else:
                    # Determine the primary region for this path using z-coordinates
                    z_coords = path_coords[:, 0]  # Extract z-coordinates
                    regions_in_path = [self._get_region_for_z(z, region_bounds) for z in z_coords]
                    
                    # Find the most common region in this path
                    from collections import Counter
                    region_counts = Counter(regions_in_path)
                    primary_region = region_counts.most_common(1)[0][0]
                    
                    # Calculate what fraction of path is in each region
                    region_fractions = {region: count/len(regions_in_path) 
                                      for region, count in region_counts.items()}
                    
                    paths_dict[path_id] = {
                        'original_path_id': i,
                        'region': primary_region,
                        'region_fractions': region_fractions,
                        'coordinates': path_coords,
                        'length': len(path_coords),
                        'z_range': (float(z_coords.min()), float(z_coords.max()))
                    }
                    path_id += 1
                    
            except Exception as e:
                self._log(f"Warning: Error processing path {i}: {str(e)}", level=1)
                continue
        
        # Store the converted paths
        roi_model.paths = paths_dict
        n_paths_final = len(paths_dict)
        
        self._log(f"Converted skeleton to dictionary with {n_paths_final} vessel paths", level=2)
        if split_paths:
            self._log("Paths split at region boundaries", level=2)
        else:
            self._log("Paths labeled with primary regions", level=2)
            
        self._log("Path tracing complete", level=1, timing=time.time() - start_time)    
        return paths_dict, stats

    def _split_path_at_region_boundaries(self, path_coords: np.ndarray, region_bounds: Dict[str, Tuple[float, float, Tuple[float, float]]]) -> List[Tuple[str, np.ndarray]]:
        """Split a path into segments based on region boundaries.
        
        Args:
            path_coords: Array of coordinates for a single path
            region_bounds: Dictionary of region bounds
            
        Returns:
            List of tuples containing (region_name, path_segment_coords)
            where path_segment_coords is a 2D array with:
            - [:,0] = z-coordinates
            - [:,1] = x-coordinates
            - [:,2] = y-coordinates
        """
        # Get regions for each point in the path
        regions = [self._get_region_for_z(coord[0], region_bounds) for coord in path_coords]
        
        # Initialize list to store path segments
        path_segments = []
        current_segment = []
        current_region = regions[0]
        
        for i, (coord, region) in enumerate(zip(path_coords, regions)):
            if region != current_region:
                # If we have points in the current segment, save it
                if current_segment:
                    # Convert list of coordinates to numpy array in ZXY format
                    segment_array = np.array(current_segment)
                    path_segments.append((current_region, segment_array))
                # Start new segment
                current_segment = [coord]
                current_region = region
            else:
                current_segment.append(coord)
                
        # Add the last segment if it exists
        if current_segment:
            # Convert list of coordinates to numpy array in ZXY format
            segment_array = np.array(current_segment)
            path_segments.append((current_region, segment_array))
            
        return path_segments

    def get_depth_volume(self, roi_model: ROI) -> np.ndarray:
        """Create a volume where each vessel is labeled by its z-depth.
        
        Args:
            roi_model: ROI object containing binary data
            
        Returns:
            np.ndarray: Volume with z-depth information for each vessel
        """
        if roi_model.binary is None:
            raise ValueError("No binary data available for depth volume creation")
            
        Z, Y, X = roi_model.binary.shape
        depth_vol = np.zeros((Z, Y, X), dtype=float)
        
        for z in range(Z):
            depth_vol[z] = roi_model.binary[z] * z
            
        return depth_vol 