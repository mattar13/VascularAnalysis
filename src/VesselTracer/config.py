from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Optional

@dataclass
class PipelineConfig:
    # ROI
    micron_roi: float
    center_x: int
    center_y: int

    # Scale bar
    scalebar_length: float
    scalebar_x: float
    scalebar_y: float

    # Pre-processing
    gauss_sigma: float
    min_object_size: int
    close_radius: int
    prune_length: int
    
    # Image properties (set by load_vessel_image)
    pixel_size_x: float = 0.0
    pixel_size_y: float = 0.0
    pixel_size_z: float = 0.0

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> 'PipelineConfig':
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'default_vessel_config.yaml'
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            micron_roi=config_dict['roi']['micron_roi'],
            center_x=config_dict['roi']['center_x'],
            center_y=config_dict['roi']['center_y'],
            scalebar_length=config_dict['scalebar']['length'],
            scalebar_x=config_dict['scalebar']['x'],
            scalebar_y=config_dict['scalebar']['y'],
            gauss_sigma=config_dict['preprocessing']['gauss_sigma'],
            min_object_size=config_dict['preprocessing']['min_object_size'],
            close_radius=config_dict['preprocessing']['close_radius'],
            prune_length=config_dict['preprocessing']['prune_length']
        )

    def to_yaml(self, config_path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = {
            'roi': {
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
                'gauss_sigma': self.gauss_sigma,
                'min_object_size': self.min_object_size,
                'close_radius': self.close_radius,
                'prune_length': self.prune_length
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

    def print_config(self):
        """Print the configuration in a formatted, readable way with descriptions."""
        print("\nVesselTracer Pipeline Configuration")
        print("==================================")
        
        print("\nROI Settings:")
        print(f"    micron_roi  -> {self.micron_roi:8} [Size of region of interest in microns]")
        print(f"    center_x    -> {self.center_x:8} [X coordinate of ROI center]")
        print(f"    center_y    -> {self.center_y:8} [Y coordinate of ROI center]")
        
        print("\nScale Bar Settings:")
        print(f"    scalebar_length  -> {self.scalebar_length:8} [Length of scale bar in plot units]")
        print(f"    scalebar_x       -> {self.scalebar_x:8} [X position of scale bar in plot]")
        print(f"    scalebar_y       -> {self.scalebar_y:8} [Y position of scale bar in plot]")
        
        print("\nPre-processing Parameters:")
        print(f"    gauss_sigma      -> {self.gauss_sigma:8} [Gaussian smoothing sigma before Frangi/threshold]")
        print(f"    min_object_size  -> {self.min_object_size:8} [Minimum object size to keep after segmentation]")
        print(f"    close_radius     -> {self.close_radius:8} [Closing operation radius in voxels]")
        print(f"    prune_length     -> {self.prune_length:8} [Length to prune skeleton branches]")
        
        print("\nImage Properties:")
        print(f"    pixel_size_x     -> {self.pixel_size_x:8.3f} [Pixel size in x direction (µm/pixel)]")
        print(f"    pixel_size_y     -> {self.pixel_size_y:8.3f} [Pixel size in y direction (µm/pixel)]")
        print(f"    pixel_size_z     -> {self.pixel_size_z:8.3f} [Pixel size in z direction (µm/pixel)]")
        print("\n")
