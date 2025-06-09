from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union, Optional, List
import yaml

@dataclass
class ConfigManager:
    """Manages configuration settings for vessel tracing.
    
    This class handles loading, storing, and converting configuration parameters
    including ROI settings, preprocessing parameters, and region definitions.
    """
    config_path: Optional[Union[str, Path]] = None
    pixel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    def __post_init__(self):
        """Initialize configuration after dataclass creation."""
        if isinstance(self.config_path, str):
            self.config_path = Path(self.config_path)
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file."""
        if self.config_path is None:
            # Point to the config directory in the project root
            self.config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'default_vessel_config.yaml'
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
        
        # Other preprocessing settings
        self.min_object_size = config['preprocessing']['min_object_size']
        self.prune_length = config['preprocessing']['prune_length']
        self.binarization_method = config['preprocessing']['binarization_method']
        
        # Region settings
        self.regions = config.get('regions', ['superficial', 'intermediate', 'deep'])
        self.region_peak_distance = config.get('region_peak_distance', 2)
        self.region_height_ratio = config.get('region_height_ratio', 0.80)
        self.region_n_stds = config.get('region_n_stds', 2)
        
        # Verbose settings
        self.verbose = config.get('verbose', 2)
        
        # Initialize pixel-based parameters (will be set when pixel sizes are available)
        self.pixel_size_x = 0.0
        self.pixel_size_y = 0.0
        self.pixel_size_z = 0.0
        
        # Pixel-based parameters (calculated in convert_to_pixels)
        self.gauss_sigma_x = None
        self.gauss_sigma_y = None
        self.gauss_sigma_z = None
        self.gauss_sigma = None
        self.background_sigma_x = None
        self.background_sigma_y = None
        self.background_sigma_z = None
        self.background_sigma = None
        self.median_filter_size = None
        self.close_radius = None
    
    def set_pixel_sizes(self, pixel_size_x: float, pixel_size_y: float, pixel_size_z: float):
        """Set pixel sizes and convert micron parameters to pixels.
        
        Args:
            pixel_size_x: Pixel size in X dimension (microns/pixel)
            pixel_size_y: Pixel size in Y dimension (microns/pixel)  
            pixel_size_z: Pixel size in Z dimension (microns/pixel)
        """
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.pixel_size_z = pixel_size_z
        self.convert_to_pixels()
    
    def convert_to_pixels(self):
        """Convert micron-based parameters to pixel values using image resolution."""
        if self.pixel_size_x == 0 or self.pixel_size_y == 0 or self.pixel_size_z == 0:
            raise ValueError("Pixel sizes not set. Call set_pixel_sizes first.")
            
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
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to a YAML file.
        
        Args:
            output_path: Path where to save the YAML configuration file
        """
        output_path = Path(output_path)
        
        # Create configuration dictionary
        config = {
            'roi': {
                'find_roi': self.find_roi,
                'micron_roi': self.micron_roi,
                'center_x': self.center_x,
                'center_y': self.center_y
            },
            'scalebar': {
                'length': self.scalebar_length,
                'x': self.scalebar_x,
                'y': self.scalebar_y
            },
            'preprocessing': {
                'gauss_sigma': self.micron_gauss_sigma,
                'background_sigma': self.micron_background_sigma,
                'median_filter_size': self.micron_median_filter_size,
                'close_radius': self.micron_close_radius,
                'min_object_size': self.min_object_size,
                'prune_length': self.prune_length,
                'binarization_method': self.binarization_method
            },
            'regions': self.regions,
            'region_peak_distance': self.region_peak_distance,
            'region_height_ratio': self.region_height_ratio,
            'region_n_stds': self.region_n_stds,
            'verbose': self.verbose
        }
        
        # Add pixel sizes if they've been set
        if hasattr(self, 'pixel_size_x') and self.pixel_size_x > 0:
            config['pixel_sizes'] = {
                'x': float(self.pixel_size_x),
                'y': float(self.pixel_size_y),
                'z': float(self.pixel_size_z)
            }
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to YAML file
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
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
        if self.gauss_sigma_x is not None:
            print(f"    gauss_sigma      -> {self.micron_gauss_sigma:.1f} µm -> {self.gauss_sigma_x:.1f}, {self.gauss_sigma_y:.1f}, {self.gauss_sigma_z:.1f} pixels")
        else:
            print(f"    gauss_sigma      -> {self.micron_gauss_sigma:.1f} µm (not converted to pixels yet)")
        print(f"    min_object_size  -> {self.min_object_size:8} [Minimum object size to keep]")
        if self.close_radius is not None:
            print(f"    close_radius     -> {self.micron_close_radius:.1f} µm -> {self.close_radius} pixels")
        else:
            print(f"    close_radius     -> {self.micron_close_radius:.1f} µm (not converted to pixels yet)")
        print(f"    prune_length     -> {self.prune_length:8} [Length to prune skeleton branches]")
        
        print("\nRegion Settings:")
        print(f"    regions          -> {self.regions}")
        print(f"    region_peak_distance -> {self.region_peak_distance}")
        print(f"    region_height_ratio -> {self.region_height_ratio}")
        print(f"    region_n_stds      -> {self.region_n_stds}")
        
        if hasattr(self, 'pixel_size_x') and self.pixel_size_x > 0:
            print("\nImage Properties:")
            print(f"    pixel_size_x     -> {self.pixel_size_x:8.3f} [Pixel size in x direction (µm/pixel)]")
            print(f"    pixel_size_y     -> {self.pixel_size_y:8.3f} [Pixel size in y direction (µm/pixel)]")
            print(f"    pixel_size_z     -> {self.pixel_size_z:8.3f} [Pixel size in z direction (µm/pixel)]")
        print("\n")
    
    def update_roi_position(self, center_x: int, center_y: int, micron_roi: Optional[float] = None) -> None:
        """Update the ROI center position and optionally its size.
        
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
        
        print("ROI updated.") 