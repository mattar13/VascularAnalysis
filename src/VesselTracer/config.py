from dataclasses import dataclass

@dataclass
class PipelineConfig:
    # ROI
    micron_roi: float = 200.0
    center_x: int = 1200
    center_y: int = 1250

    # Scale bar (kept for plotting)
    scalebar_length: float = 25.0
    scalebar_x: float = 15.0
    scalebar_y: float = 200.0

    # Pre-processing
    gauss_sigma: float = 1.0          # before Frangi / threshold
    min_object_size: int = 64
    close_radius: int = 1             # voxels
    prune_length: int = 5             # skeleton branch prune
