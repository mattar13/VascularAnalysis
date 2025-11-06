from dataclasses import dataclass
from pathlib import Path
import numpy as np
import xmltodict
from czifile import CziFile
import tifffile
from typing import Optional, Dict, Any, Tuple, Union, List
from .config import VesselTracerConfig, DEFAULT_DIVING_COLOR

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
    region_bounds: Optional[Dict[str, Tuple[float, float, Tuple[float, float]]]] = None
    
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
                      x_range: Optional[Tuple[int, int]] = None,
                      depth_coded: bool = False) -> np.ndarray:
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
            depth_coded: If True, creates depth-coded projections where intensity
                        represents z-position (only works with binary mode)
                
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

        # Handle depth coding if requested
        if depth_coded and volume_type == 'binary':
            Z = volume.shape[0]
            depth_vol = np.zeros_like(volume, dtype=float)
            for z in range(Z):
                depth_vol[z] = volume[z] * z
            
            # Calculate projection
            if isinstance(axis, list):
                # Special case for xy projection (along z)
                projection = valid_ops[operation](depth_vol, axis=tuple(axis))
            else:
                projection = valid_ops[operation](depth_vol, axis=axis)
            
            # Normalize depth projections to [0,1]
            projection = projection / (Z-1) if projection.max() > 0 else projection
        else:
            # Calculate regular projection
            if isinstance(axis, list):
                # Special case for xy projection (along z)
                projection = valid_ops[operation](volume, axis=tuple(axis))
            else:
                projection = valid_ops[operation](volume, axis=axis)
            
        return projection
    
    def truncate(self, axis: int, range_values: Tuple[int, int], volume_type: str = 'volume') -> np.ndarray:
        """Truncate the volume along a specified axis within a given range.
        
        Args:
            axis: Dimension to truncate along (0: z, 1: y, 2: x)
            range_values: Tuple of (start, end) indices for truncation
            volume_type: Type of volume to truncate ('volume', 'binary', 'background', 'region')
            
        Returns:
            np.ndarray: Truncated volume
            
        Raises:
            ValueError: If invalid axis or volume_type specified
        """
        # Validate axis
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be 0 (z), 1 (y), or 2 (x)")
            
        # Validate range values
        start, end = range_values
        if start < 0 or end < start:
            raise ValueError("Invalid range values. Start must be >= 0 and end must be >= start")
            
        # Get the appropriate volume
        if volume_type == 'binary':
            if self.binary is None:
                raise ValueError("Binary volume not available.")
            volume = self.binary
        elif volume_type == 'background':
            if self.background is None:
                raise ValueError("Background volume not available.")
            volume = self.background
        elif volume_type == 'volume':
            if self.volume is None:
                raise ValueError("Volume not available.")
            volume = self.volume
        elif volume_type == 'region':
            if self.region is None:
                raise ValueError("Region map volume not available.")
            volume = self.region
        else:
            raise ValueError(f"volume_type must be one of ['volume', 'binary', 'background', 'region']")
            
        # Create slice objects for each dimension
        slices = [slice(None)] * 3  # Default to full range for all dimensions
        slices[axis] = slice(start, end)
        
        # Apply truncation
        self.volume = volume[tuple(slices)]

        return self.volume

    def get_path_coordinates(self, 
                     region_colorcode: bool = False,
                     region_bounds: Optional[Dict[str, Tuple[float, float, Tuple[float, float]]]] = None,
                     region_color_map: Optional[Dict[str, str]] = None,
                     diving_color: str = DEFAULT_DIVING_COLOR,
                     x_range: Optional[Tuple[float, float]] = None,
                     y_range: Optional[Tuple[float, float]] = None,
                     z_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get coordinates for all vessel paths.
        
        Args:
            region_colorcode: If True, color-code paths based on their region
            region_bounds: Optional dictionary of region boundaries for color coding
            region_color_map: Optional mapping from region name to color
            diving_color: Color used when path spans multiple regions
            x_range: Optional tuple of (min_x, max_x) to filter paths
            y_range: Optional tuple of (min_y, max_y) to filter paths
            z_range: Optional tuple of (min_z, max_z) to filter paths
            
        Returns:
            Tuple of (x_coords, y_coords, z_coords, colors) where:
            - x_coords: Array of x coordinates for each path point
            - y_coords: Array of y coordinates for each path point
            - z_coords: Array of z coordinates for each path point
            - colors: Array of colors for each path point (if region_colorcode=True)
            
        Raises:
            ValueError: If paths are not available
        """
        if self.paths is None:
            raise ValueError("No paths found. Run trace_paths() first.")
            
        # Define colors for regions
        region_colors = region_color_map or {}
        multi_region_color = diving_color or DEFAULT_DIVING_COLOR
        
        # Initialize lists to store coordinates and colors
        all_x_paths = []
        all_y_paths = []
        all_z_paths = []
        all_colors_paths = []
        
        # Process each path
        for path_id, path in self.paths.items():
            path_coords = path['coordinates']
            if len(path_coords) > 0:
                # Extract x, y, z coordinates
                z_coords = path_coords[:, 0]  # z is first coordinate
                y_coords = path_coords[:, 1]  # y is second coordinate
                x_coords = path_coords[:, 2]  # x is third coordinate
                
                # Check if path has any points within the specified ranges
                if x_range is None:
                    all_x_paths.append(x_coords)
                
                if y_range is None:
                    all_y_paths.append(y_coords)

                if z_range is None:
                    all_z_paths.append(z_coords)
                
                #I don't know yet how to handle the case where x_range, y_range, and z_range are not None. Lets work on it later
                # Handle region color coding
                if region_colorcode and region_bounds is not None:
                    # Get the region for each point in the path
                    regions = [self._get_region_for_z(z, region_bounds) for z in z_coords]
                    unique_regions = np.unique(regions)
                    
                    if len(unique_regions) > 1 or unique_regions[0] == 'Outside':
                        # Plot as diving vessel if path crosses multiple regions
                        all_colors_paths.append(multi_region_color)
                    else:
                        # Plot in the color of its single region
                        region = unique_regions[0]
                        all_colors_paths.append(region_colors.get(region, multi_region_color))
                else:
                    all_colors_paths.append('red')
                
        return all_x_paths, all_y_paths, all_z_paths, all_colors_paths
    
    def _get_region_for_z(self, z: float, region_bounds: Dict[str, Tuple[float, float, Tuple[float, float]]]) -> str:
        """Helper method to determine which region a z-coordinate belongs to."""
        for region, (peak, sigma, bounds) in region_bounds.items():
            if bounds[0] <= z <= bounds[1]:
                return region
        return 'Outside'

    def get_center(self) -> Tuple[int, int]:
        """Get center coordinates of ROI."""
        center_x = self.min_x + (self.dx // 2) if self.dx else self.min_x
        center_y = self.min_y + (self.dy // 2) if self.dy else self.min_y
        return (center_x, center_y)

    def create_roi(self, config: 'VesselTracerConfig') -> 'ROI':
        """Create a Region of Interest from the current image model.
        
        Args:
            config: Configuration object containing ROI parameters
            
        Returns:
            ROI: New ROI object containing the extracted region
        """
        if self.volume is None:
            raise ValueError("No volume data available for ROI extraction")
            
        if not config.find_roi:
            # Create ROI model using the entire volume
            return ROI(
                volume=self.volume.copy(),
                pixel_size_x=self.pixel_size_x,
                pixel_size_y=self.pixel_size_y,
                pixel_size_z=self.pixel_size_z,
                min_x=0,
                min_y=0,
                max_x=self.volume.shape[2],
                max_y=self.volume.shape[1]
            )
        else:
            # Convert ROI size from microns to pixels
            roi_x = round(config.micron_roi * 1/self.pixel_size_x)
            roi_y = round(config.micron_roi * 1/self.pixel_size_y)
            
            # Extract initial ROI using min coordinates
            roi = self.volume[:, 
                        config.min_y:config.min_y+roi_y,
                        config.min_x:config.min_x+roi_x]
            
            # Create ROI model
            roi_model = ROI(
                volume=roi,
                pixel_size_x=self.pixel_size_x,
                pixel_size_y=self.pixel_size_y,
                pixel_size_z=self.pixel_size_z,
                min_x=config.min_x,
                min_y=config.min_y,
                dx=roi_x,
                dy=roi_y
            )
            
            # Store the valid frame range
            roi_model.valid_frame_range = (0, roi.shape[0]-1)
            
            return roi_model

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
