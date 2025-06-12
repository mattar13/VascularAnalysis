from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Any
import yaml


@dataclass
class VesselTracerConfig:
    """Configuration class for VesselTracer parameters."""
    
    # Configuration file path
    config_path: Optional[Union[str, Path]] = None
    
    # ROI settings
    find_roi: bool = True
    micron_roi: float = 500.0
    min_x: int = 500
    min_y: int = 500
    
    # Dead frames settings
    remove_dead_frames: bool = True
    dead_frame_threshold: float = 0.75
    
    # Median filter settings
    median_filter_size: float = 45.0
    max_workers: Optional[int] = None
    
    # Gaussian filter settings
    gaussian_sigma: float = 1.5
    
    # Closing settings
    close_radius: float = 2.0
    min_object_size: int = 64
    prune_length: int = 5
    
    # Binarization settings
    binarization_method: str = 'triangle'
    
    # Region settings
    regions: List[str] = None
    region_peak_distance: int = 2
    region_height_ratio: float = 0.80
    region_n_stds: float = 3.0
    
    # Scale bar settings
    scalebar_length: float = 25.0
    scalebar_x: float = 15.0
    scalebar_y: float = 200.0
    
    # Verbose settings
    verbose: int = 2
    
    def __post_init__(self):
        """Initialize default regions and automatically load config if path provided."""
        if self.regions is None:
            self.regions = ['superficial', 'intermediate', 'deep']
            
        # Automatically load config if path is provided
        if self.config_path is not None:
            self.load_config(self.config_path)

    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Point to the config directory in the project root
            config_path = Path(__file__).parent.parent.parent / 'config' / 'default_vessel_config.yaml'
            if not config_path.exists():
                raise FileNotFoundError(f"Default config file not found at {config_path}")
        
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # ROI settings
        self.find_roi = config['roi']['find_roi']
        self.micron_roi = config['roi']['micron_roi']
        self.min_x = config['roi']['min_x']
        self.min_y = config['roi']['min_y']
        
        # Dead frames settings
        self.remove_dead_frames = config['dead_frames']['remove']
        self.dead_frame_threshold = config['dead_frames']['threshold']
        
        # Median filter settings
        self.median_filter_size = config['median_filter']['size']
        self.max_workers = config['median_filter']['max_workers']
        
        # Gaussian filter settings
        self.gaussian_sigma = config['gaussian_filter']['sigma']
        
        # Closing settings
        self.close_radius = config['closing']['close_radius']
        self.min_object_size = config['closing']['min_object_size']
        self.prune_length = config['closing']['prune_length']
        
        # Binarization settings
        self.binarization_method = config['binarization']['method']
        
        # Region settings
        self.regions = config.get('regions', ['superficial', 'intermediate', 'deep'])
        self.region_peak_distance = config['region']['peak_distance']
        self.region_height_ratio = config['region']['height_ratio']
        self.region_n_stds = config['region']['n_stds']
        
        # Scale bar settings
        self.scalebar_length = config['scalebar']['length']
        self.scalebar_x = config['scalebar']['x']
        self.scalebar_y = config['scalebar']['y']
        
        # Verbose settings
        self.verbose = config.get('verbose', 2)

    def save_config(self, output_path: Union[str, Path], pixel_sizes: Optional[Tuple[float, float, float]] = None) -> None:
        """Save current configuration to a YAML file.
        
        Args:
            output_path: Path where to save the YAML configuration file
            pixel_sizes: Optional tuple of (z,y,x) pixel sizes to include
        """
        output_path = Path(output_path)
        
        # Create configuration dictionary
        config = {
            'roi': {
                'find_roi': self.find_roi,
                'micron_roi': self.micron_roi,
                'min_x': self.min_x,
                'min_y': self.min_y
            },
            'dead_frames': {
                'remove': self.remove_dead_frames,
                'threshold': self.dead_frame_threshold
            },
            'median_filter': {
                'size': self.median_filter_size,
                'max_workers': self.max_workers
            },
            'gaussian_filter': {
                'sigma': self.gaussian_sigma
            },
            'closing': {
                'close_radius': self.close_radius,
                'min_object_size': self.min_object_size,
                'prune_length': self.prune_length
            },
            'binarization': {
                'method': self.binarization_method
            },
            'region': {
                'peak_distance': self.region_peak_distance,
                'height_ratio': self.region_height_ratio,
                'n_stds': self.region_n_stds
            },
            'regions': self.regions,
            'scalebar': {
                'length': self.scalebar_length,
                'x': self.scalebar_x,
                'y': self.scalebar_y
            },
            'verbose': self.verbose
        }
        
        # Add pixel sizes if provided
        if pixel_sizes is not None:
            config['pixel_sizes'] = {
                'z': float(pixel_sizes[0]),
                'y': float(pixel_sizes[1]),
                'x': float(pixel_sizes[2])
            }
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to YAML file
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        print(f"Configuration saved to {output_path}")

    def convert_to_pixels(self, pixel_sizes: Tuple[float, float, float]) -> Dict[str, Any]:
        """Convert micron-based parameters to pixel values using image resolution.
        
        Args:
            pixel_sizes: Tuple of (pixel_size_z, pixel_size_y, pixel_size_x) in microns
            
        Returns:
            Dictionary containing converted pixel values
        """
        pixel_size_z, pixel_size_y, pixel_size_x = pixel_sizes
        
        # Convert Gaussian sigma for each dimension (for regular smoothing)
        gauss_sigma_x = self.gaussian_sigma / pixel_size_x
        gauss_sigma_y = self.gaussian_sigma / pixel_size_y
        gauss_sigma_z = self.gaussian_sigma / pixel_size_z
        gauss_sigma = (gauss_sigma_z, gauss_sigma_y, gauss_sigma_x)
        
        # Convert median filter size (use average of x and y pixel sizes)
        avg_xy_pixel_size = (pixel_size_x + pixel_size_y) / 2
        median_filter_size = int(round(self.median_filter_size / avg_xy_pixel_size))
        # Ensure odd size for median filter
        if median_filter_size % 2 == 0:
            median_filter_size += 1
            
        return {
            'gauss_sigma_x': gauss_sigma_x,
            'gauss_sigma_y': gauss_sigma_y,
            'gauss_sigma_z': gauss_sigma_z,
            'gauss_sigma': gauss_sigma,
            'median_filter_size': median_filter_size,
            'close_radius': self.close_radius
        }

    def generate_metadata_df(self, 
                            pixel_sizes: Optional[Tuple[float, float, float]] = None,
                            input_type: str = 'Unknown',
                            processing_status: Optional[Dict[str, bool]] = None) -> 'pd.DataFrame':
        """Generate a pandas DataFrame containing configuration metadata.
        
        Args:
            pixel_sizes: Optional tuple of (z,y,x) pixel sizes in microns
            input_type: Type of input data ('File', 'Numpy Array', etc.)
            processing_status: Dictionary of processing steps completed
            
        Returns:
            pd.DataFrame: Metadata DataFrame
        """
        from datetime import datetime
        import pandas as pd
        
        # Basic configuration metadata
        metadata = [
            ('Input Type', input_type),
            ('Timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ('ROI Finding Enabled', self.find_roi),
            ('ROI Size (microns)', self.micron_roi),
            ('ROI Min X', self.min_x),
            ('ROI Min Y', self.min_y),
            ('Gaussian Sigma (microns)', self.gaussian_sigma),
            ('Median Filter Size (microns)', self.median_filter_size),
            ('Close Radius (microns)', self.close_radius),
            ('Min Object Size', self.min_object_size),
            ('Prune Length', self.prune_length),
            ('Binarization Method', self.binarization_method),
            ('Regions', ', '.join(self.regions)),
            ('Region Peak Distance', self.region_peak_distance),
            ('Region Height Ratio', self.region_height_ratio),
            ('Region N Stds', self.region_n_stds),
            ('Verbose Level', self.verbose)
        ]
        
        # Add pixel size information if available
        if pixel_sizes is not None:
            pixel_size_z, pixel_size_y, pixel_size_x = pixel_sizes
            metadata.extend([
                ('Pixel Size X (microns)', pixel_size_x),
                ('Pixel Size Y (microns)', pixel_size_y),
                ('Pixel Size Z (microns)', pixel_size_z)
            ])
            
            # Add converted pixel values
            pixel_conversions = self.convert_to_pixels(pixel_sizes)
            metadata.extend([
                ('Gauss Sigma X (pixels)', pixel_conversions['gauss_sigma_x']),
                ('Gauss Sigma Y (pixels)', pixel_conversions['gauss_sigma_y']),
                ('Gauss Sigma Z (pixels)', pixel_conversions['gauss_sigma_z']),
                ('Median Filter Size (pixels)', pixel_conversions['median_filter_size']),
                ('Close Radius (pixels)', pixel_conversions['close_radius'])
            ])
        
        # Add processing status if available
        if processing_status is not None:
            for step, completed in processing_status.items():
                metadata.append((f'{step} Completed', completed))
        
        return pd.DataFrame(metadata, columns=['Parameter', 'Value'])

    def save_metadata(self, 
                     output_path: Union[str, Path],
                     pixel_sizes: Optional[Tuple[float, float, float]] = None,
                     input_type: str = 'Unknown',
                     processing_status: Optional[Dict[str, bool]] = None,
                     format: str = 'excel') -> None:
        """Save metadata to file.
        
        Args:
            output_path: Path where to save the metadata
            pixel_sizes: Optional tuple of (z,y,x) pixel sizes in microns
            input_type: Type of input data
            processing_status: Dictionary of processing steps completed
            format: Output format ('excel', 'csv', 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_df = self.generate_metadata_df(pixel_sizes, input_type, processing_status)
        
        if format.lower() == 'excel':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.xlsx')
            metadata_df.to_excel(output_path, sheet_name='Metadata', index=False)
        elif format.lower() == 'csv':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.csv')
            metadata_df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.json')
            # Convert to dictionary for JSON
            metadata_dict = dict(zip(metadata_df['Parameter'], metadata_df['Value']))
            import json
            with open(output_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'excel', 'csv', or 'json'.")
            
        print(f"Metadata saved to {output_path}")

    def print_config(self, pixel_sizes: Optional[Tuple[float, float, float]] = None) -> None:
        """Print current configuration."""
        print("\nVesselTracer Configuration")
        print("=========================")
        
        print("\nROI Settings:")
        print(f"    find_roi       -> {self.find_roi}")
        if self.find_roi:
            print(f"    micron_roi     -> {self.micron_roi:8} [Size of region of interest in microns]")
            print(f"    min_x          -> {self.min_x:8} [X coordinate of ROI minimum]")
            print(f"    min_y          -> {self.min_y:8} [Y coordinate of ROI minimum]")
        
        print("\nDead Frames Settings:")
        print(f"    remove_dead_frames -> {self.remove_dead_frames}")
        print(f"    dead_frame_threshold -> {self.dead_frame_threshold:.2f}")
        
        print("\nScale Bar Settings:")
        print(f"    scalebar_length  -> {self.scalebar_length:8} [Length of scale bar in plot units]")
        print(f"    scalebar_x       -> {self.scalebar_x:8} [X position of scale bar in plot]")
        print(f"    scalebar_y       -> {self.scalebar_y:8} [Y position of scale bar in plot]")
        
        print("\nPre-processing Parameters:")
        if pixel_sizes:
            pixel_conversions = self.convert_to_pixels(pixel_sizes)
            print(f"    gauss_sigma      -> {self.gaussian_sigma:.1f} µm -> {pixel_conversions['gauss_sigma_x']:.1f}, {pixel_conversions['gauss_sigma_y']:.1f}, {pixel_conversions['gauss_sigma_z']:.1f} pixels")
        else:
            print(f"    gauss_sigma      -> {self.gaussian_sigma:.1f} µm")
        print(f"    min_object_size  -> {self.min_object_size:8} [Minimum object size to keep]")
        print(f"    close_radius     -> {self.close_radius:.1f} µm")
        print(f"    prune_length     -> {self.prune_length:8} [Length to prune skeleton branches]")
        
        print("\nRegion Settings:")
        print(f"    regions          -> {self.regions}")
        print(f"    region_peak_distance -> {self.region_peak_distance}")
        print(f"    region_height_ratio -> {self.region_height_ratio}")
        print(f"    region_n_stds      -> {self.region_n_stds}")
        
        if pixel_sizes:
            print("\nImage Properties:")
            print(f"    pixel_size_x     -> {pixel_sizes[2]:8.3f} [Pixel size in x direction (µm/pixel)]")
            print(f"    pixel_size_y     -> {pixel_sizes[1]:8.3f} [Pixel size in y direction (µm/pixel)]")
            print(f"    pixel_size_z     -> {pixel_sizes[0]:8.3f} [Pixel size in z direction (µm/pixel)]")
        print("\n")
