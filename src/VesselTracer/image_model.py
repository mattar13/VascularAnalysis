from dataclasses import dataclass
from pathlib import Path
import numpy as np
import xmltodict
from czifile import CziFile
import tifffile
from typing import Optional, Dict, Any, Tuple, Union, List

@dataclass
class ImageModel:
    """Data model for storing image volumes and associated data."""
    input_data: Optional[np.ndarray] = None
    volume: Optional[np.ndarray] = None
    binary: Optional[np.ndarray] = None
    background: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None
    paths: Optional[Dict[str, Any]] = None
    filepath: Optional[Union[str, Path]] = None
    pixel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Image properties
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    pixel_size_z: float = 1.0
    shape: Optional[Tuple[int, int, int]] = None
    
    def __post_init__(self):
        """Initialize paths and handle automatic loading based on filepath or input_data."""
        if self.paths is None:
            self.paths = {}
            
        # Handle automatic loading
        if self.filepath is not None:
            # Convert to Path object if string
            if isinstance(self.filepath, str):
                self.filepath = Path(self.filepath)
                
            # Determine file type and load
            if self.filepath.suffix.lower() == '.czi':
                self.load_from_czi(self.filepath)
            elif self.filepath.suffix.lower() in ['.tif', '.tiff']:
                self.load_from_tiff(self.filepath, self.pixel_sizes)
            else:
                raise ValueError(f"Unsupported file format: {self.filepath.suffix}")
        elif self.input_data is not None:
            # Load from numpy array
            self.load_from_array(self.input_data, self.pixel_sizes)
    
    def load_from_czi(self, file_path: Path) -> None:
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
        self.shape = self.volume.shape
    def load_from_tiff(self, file_path: Path, pixel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        """Load TIFF file."""
        self.volume = tifffile.imread(file_path)
        self.shape = self.volume.shape
        # For TIFF files, use the provided pixel sizes
        self.pixel_size_z, self.pixel_size_y, self.pixel_size_x = pixel_sizes

    def load_from_array(self, array: np.ndarray, pixel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        """Load data from numpy array."""
        if array.ndim != 3:
            raise ValueError("Input array must be 3D")
        
        self.volume = array
        self.pixel_size_z, self.pixel_size_y, self.pixel_size_x = pixel_sizes
        self.shape = self.volume.shape
        
    def get_pixel_sizes(self) -> Tuple[float, float, float]:
        """Get pixel sizes as tuple (z, y, x)."""
        return (self.pixel_size_z, self.pixel_size_y, self.pixel_size_x)

    def get_projection(self, 
                      axis: Union[int, List[int]], 
                      operation: str = 'mean', 
                      volume_type: str = 'volume',
                      background_volume: Optional[np.ndarray] = None,
                      z_range: Optional[Tuple[int, int]] = None,
                      y_range: Optional[Tuple[int, int]] = None,
                      x_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Generate a projection of the specified volume along specified axis/axes.
        
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
                - 'binary': Binary volume
                - 'background': Background volume from median filtering
                - 'volume': Current processed volume (default)
                - 'region': Region map volume with region labels
            background_volume: Optional background volume for 'background' type
            z_range: Optional tuple of (start, end) for z dimension
            y_range: Optional tuple of (start, end) for y dimension
            x_range: Optional tuple of (start, end) for x dimension
                
        Returns:
            np.ndarray: Projected image
            
        Raises:
            ValueError: If invalid axis, operation, or volume_type specified
        """
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
            
        # Validate and get the specified volume
        if volume_type == 'binary':
            if self.binary is None:
                raise ValueError("Binary volume not available.")
            volume = self.binary
        elif volume_type == 'background':
            if hasattr(self, 'background') and self.background is not None:
                volume = self.background
            elif background_volume is not None:
                volume = background_volume
            else:
                raise ValueError("Background volume not available.")
        elif volume_type == 'volume':
            if self.volume is None:
                raise ValueError("Volume not available.")
            volume = self.volume
        elif volume_type == 'region':
            if self.region is None:
                raise ValueError("Region map volume not available.")
            volume = self.region
        else:
            raise ValueError(f"volume_type must be one of ['binary', 'background', 'volume', 'region']")
            
        # Create slice objects for each dimension
        slices = [slice(None)] * 3  # Default to full range for all dimensions
        
        # Update slices based on provided ranges
        if z_range is not None:
            slices[0] = slice(z_range[0], z_range[1])
        if y_range is not None:
            slices[1] = slice(y_range[0], y_range[1])
        if x_range is not None:
            slices[2] = slice(x_range[0], x_range[1])
            
        # Apply slices to volume
        volume = volume[tuple(slices)]
            
        # Get projection function
        proj_func = valid_ops[operation]
        
        # Calculate projection
        if isinstance(axis, list):
            # Special case for xy projection (along z)
            projection = proj_func(volume, axis=tuple(axis))
        else:
            projection = proj_func(volume, axis=axis)
            
        return projection

@dataclass  
class ROI(ImageModel):
    """Data model for Region of Interest with coordinate information."""
    
    min_x: int = 0
    min_y: int = 0
    max_x: Optional[int] = None
    max_y: Optional[int] = None
    dx: Optional[int] = None
    dy: Optional[int] = None
    
    def __post_init__(self):
        """Initialize derived coordinates."""
        super().__post_init__()
        if self.volume is not None and (self.max_x is None or self.max_y is None):
            _, height, width = self.volume.shape
            if self.max_x is None:
                self.max_x = width
            if self.max_y is None:
                self.max_y = height
        
        if self.max_x is not None and self.dx is None:
            self.dx = self.max_x - self.min_x
        if self.max_y is not None and self.dy is None:
            self.dy = self.max_y - self.min_y
    
    def update_coordinates(self, min_x: int, min_y: int, dx: Optional[int] = None, dy: Optional[int] = None) -> None:
        """Update ROI coordinates."""
        self.min_x = min_x
        self.min_y = min_y
        
        if dx is not None:
            self.dx = dx
            self.max_x = min_x + dx
        if dy is not None:
            self.dy = dy
            self.max_y = min_y + dy

    def get_center(self) -> Tuple[int, int]:
        """Get center coordinates of ROI."""
        center_x = self.min_x + (self.dx // 2) if self.dx else self.min_x
        center_y = self.min_y + (self.dy // 2) if self.dy else self.min_y
        return (center_x, center_y)
