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

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> 'PipelineConfig':
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'default_vessel_config.yaml'
        
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
        """Print the configuration in a formatted, readable way."""
        print("\nVesselTracer Pipeline Configuration")
        print("==================================")
        
        print("\nROI Settings:")
        print(f"  Micron ROI: {self.micron_roi}")
        print(f"  Center X: {self.center_x}")
        print(f"  Center Y: {self.center_y}")
        
        print("\nScale Bar Settings:")
        print(f"  Length: {self.scalebar_length}")
        print(f"  Position X: {self.scalebar_x}")
        print(f"  Position Y: {self.scalebar_y}")
        
        print("\nPre-processing Parameters:")
        print(f"  Gaussian Sigma: {self.gauss_sigma}")
        print(f"  Minimum Object Size: {self.min_object_size}")
        print(f"  Closing Radius: {self.close_radius} voxels")
        print(f"  Pruning Length: {self.prune_length} units")
        print("\n")
