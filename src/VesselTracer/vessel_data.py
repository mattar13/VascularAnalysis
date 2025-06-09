from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, Optional, List
import numpy as np
import xmltodict
from czifile import CziFile
import tifffile
import time

@dataclass
class VesselData:
    """Handles vessel image data storage and basic operations.
    
    This class manages the loading, storage, and basic manipulation of vessel image data
    from various file formats or numpy arrays.
    """

    def __init__(self, input_data: Union[str, Path, np.ndarray], pixel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """Initialize after dataclass creation."""
        # Convert string paths to Path objects
        if isinstance(self.input_data, str):
            self.input_data = Path(self.input_data)
            
        # Initialize data containers
        self.volume = None
        self.original_volume = None
        self.previous_volume = None
        self.unfiltered_volume = None
        self.smoothed = None
        self.binary = None
        
        # Image properties
        self.pixel_size_x = 0.0
        self.pixel_size_y = 0.0
        self.pixel_size_z = 0.0
        self.valid_frame_range = None
        
        self._load_image()
        
    def _log(self, message: str, level: int = 1, timing: Optional[float] = None, verbose: int = 2):
        """Log a message with appropriate verbosity level.
        
        Args:
            message: Message to log
            level: Minimum verbosity level required to show message (1-3)
            timing: Optional timing information to include
            verbose: Current verbosity setting
        """
        if verbose >= level:
            if timing is not None and verbose >= 3:
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
        
        # Store original volume
        self.original_volume = self.volume.copy()
        self.volume = self.volume.astype("float32")
        
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

    def normalize_image(self) -> np.ndarray:
        """Normalize image to [0,1] range."""
        print("volume.shape: ", self.volume.shape)
        print("volume.min(): ", self.volume.min())
        print("volume.max(): ", self.volume.max())
        self.previous_volume = self.volume.copy()
        self.volume = (self.volume - self.volume.min()) / (self.volume.max() - self.volume.min())
        return self.volume
        
    def segment_roi(self, center_x: int, center_y: int, micron_roi: float, 
                   remove_dead_frames: bool = True, dead_frame_threshold: float = 1.5) -> np.ndarray:
        """Extract and segment region of interest from volume.
        
        Args:
            center_x: X coordinate of ROI center
            center_y: Y coordinate of ROI center  
            micron_roi: ROI size in microns
            remove_dead_frames: Whether to remove low-intensity frames at start/end
            dead_frame_threshold: Number of standard deviations above minimum
                                intensity to use as threshold for dead frames
        
        Returns:
            np.ndarray: ROI volume extracted from main volume
        """
        start_time = time.time()
        self._log("Processing volume...", level=1)
        
        # Convert ROI size from microns to pixels
        h_x = round(micron_roi/2 * 1/self.pixel_size_x)
        h_y = round(micron_roi/2 * 1/self.pixel_size_y)
        
        self._log(f"ROI size: {h_x*2}x{h_y*2} pixels", level=2)
        
        # Extract initial ROI
        roi = self.volume[:, 
                        center_y-h_y:center_y+h_y,
                        center_x-h_x:center_x+h_x]
        
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
        # Use binary if available, otherwise use current volume
        data = self.binary if self.binary is not None else self.volume
            
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
            projection = proj_func(data, axis=tuple(axis))
        else:
            projection = proj_func(data, axis=axis)
            
        return projection

    def get_depth_volume(self) -> np.ndarray:
        """Create a volume where each vessel is labeled by its z-depth.
        
        Returns:
            np.ndarray: Volume with z-depth information for each vessel
        """
        if self.binary is None:
            raise ValueError("Binary volume not available. Run binarization first.")
            
        Z, Y, X = self.binary.shape
        depth_vol = np.zeros((Z, Y, X), dtype=float)
        
        for z in range(Z):
            depth_vol[z] = self.binary[z] * z
            
        return depth_vol

    def save_volume(self, 
                   output_dir: Union[str, Path],
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
        
        # Prepare volumes for saving
        volumes = []
        volume_names = []
        
        if save_original and self.volume is not None:
            volumes.append(self.volume)
            volume_names.append('original')
        if save_smoothed and self.smoothed is not None:
            volumes.append(self.smoothed)
            volume_names.append('smoothed')
        if save_binary and self.binary is not None:
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

    def get_pixel_sizes(self) -> Tuple[float, float, float]:
        """Get the pixel sizes in (x, y, z) order.
        
        Returns:
            Tuple of pixel sizes (x, y, z) in microns per pixel
        """
        return (self.pixel_size_x, self.pixel_size_y, self.pixel_size_z) 