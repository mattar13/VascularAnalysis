from dataclasses import dataclass
from pathlib import Path
import numpy as np
import xmltodict
from czifile import CziFile
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, binary_closing, ball, skeletonize as sk_skeletonize
from skan import Skeleton, summarize
import yaml
from typing import Optional, Dict, Any, Tuple

@dataclass
class VesselTracer:
    """Main class for vessel tracing pipeline.
    
    Attributes:
        czi_path: Path to CZI file to analyze
        config_path: Optional path to YAML config file. If None, uses default config.
    """
    czi_path: str
    config_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._load_config()
        self._load_image()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path is None:
            self.config_path = Path(__file__).parent.parent.parent / 'config' / 'default_vessel_config.yaml'
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # ROI settings
        self.micron_roi = config['roi']['micron_roi']
        self.center_x = config['roi']['center_x']
        self.center_y = config['roi']['center_y']
        
        # Scale bar settings
        self.scalebar_length = config['scalebar']['length']
        self.scalebar_x = config['scalebar']['x']
        self.scalebar_y = config['scalebar']['y']
        
        # Pre-processing settings
        self.gauss_sigma = config['preprocessing']['gauss_sigma']
        self.min_object_size = config['preprocessing']['min_object_size']
        self.close_radius = config['preprocessing']['close_radius']
        self.prune_length = config['preprocessing']['prune_length']
        
        # Image properties (set by load_image)
        self.pixel_size_x = 0.0
        self.pixel_size_y = 0.0
        self.pixel_size_z = 0.0
        
    def _load_image(self):
        """Load CZI file and extract pixel sizes."""
        with CziFile(self.czi_path) as czi:
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
        
        # Normalize volume
        self.volume = self._normalize_image(self.volume.astype("float32"))
        
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0,1] range."""
        img = (img - img.min()) / (img.max() - img.min())
        return img
        
    def extract_roi(self) -> np.ndarray:
        """Extract region of interest from volume."""
        # Convert ROI size from microns to pixels
        h_x = round(self.micron_roi/2 * 1/self.pixel_size_x)
        h_y = round(self.micron_roi/2 * 1/self.pixel_size_y)
        
        # Extract ROI
        self.roi = self.volume[:, 
                             self.center_y-h_y:self.center_y+h_y,
                             self.center_x-h_x:self.center_x+h_x]
        return self.roi
        
    def smooth(self) -> np.ndarray:
        """Apply Gaussian smoothing to ROI."""
        if not hasattr(self, 'roi'):
            self.extract_roi()
        self.smoothed = gaussian_filter(self.roi, sigma=self.gauss_sigma)
        return self.smoothed
        
    def segment(self) -> np.ndarray:
        """Segment vessels using binary thresholding."""
        if not hasattr(self, 'smoothed'):
            self.smooth()
            
        # Threshold at mean + 1 std
        thresh = self.smoothed.mean() + self.smoothed.std()
        binary = self.smoothed > thresh
        
        # Remove small objects
        binary = remove_small_objects(binary, min_size=self.min_object_size)
        
        # Binary closing to fill small gaps
        if self.close_radius > 0:
            binary = binary_closing(binary, ball(self.close_radius))
            
        self.binary = binary
        return self.binary
        
    def skeletonize(self) -> Tuple[Skeleton, Dict[str, Any]]:
        """Create and analyze vessel skeleton."""
        if not hasattr(self, 'binary'):
            self.segment()
            
        print(f"Skeletonizing binary volume of shape {self.binary.shape}")
        ske = sk_skeletonize(self.binary)
        print(f"Skeletonized volume of shape {ske.shape}")
        
        self.skeleton = Skeleton(ske)
        print(f"Created skeleton object")
        
        self.stats = summarize(self.skeleton, separator="-")
        return self.skeleton, self.stats
        
    def analyze_layers(self) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
        """Analyze vessel layers in the binary segmentation."""
        if not hasattr(self, 'binary'):
            self.segment()
            
        mean_z = np.mean(self.binary, axis=(1,2))
        from scipy.signal import find_peaks, peak_widths
        
        peaks, _ = find_peaks(mean_z, distance=2)
        widths, *_ = peak_widths(mean_z, peaks, rel_height=0.80)
        sigmas = widths / (2 * np.sqrt(2 * np.log(2)))
        
        regions = ["superficial", "intermediate", "deep"]
        self.layer_info = {
            name: (pk, σ, (pk - σ, pk + σ))
            for name, pk, σ in zip(regions, peaks, sigmas)
        }
        return self.layer_info
        
    def print_config(self):
        """Print current configuration."""
        print("\nVesselTracer Configuration")
        print("=========================")
        
        print("\nROI Settings:")
        print(f"    micron_roi  -> {self.micron_roi:8} [Size of region of interest in microns]")
        print(f"    center_x    -> {self.center_x:8} [X coordinate of ROI center]")
        print(f"    center_y    -> {self.center_y:8} [Y coordinate of ROI center]")
        
        print("\nScale Bar Settings:")
        print(f"    scalebar_length  -> {self.scalebar_length:8} [Length of scale bar in plot units]")
        print(f"    scalebar_x       -> {self.scalebar_x:8} [X position of scale bar in plot]")
        print(f"    scalebar_y       -> {self.scalebar_y:8} [Y position of scale bar in plot]")
        
        print("\nPre-processing Parameters:")
        print(f"    gauss_sigma      -> {self.gauss_sigma:8} [Gaussian smoothing sigma]")
        print(f"    min_object_size  -> {self.min_object_size:8} [Minimum object size to keep]")
        print(f"    close_radius     -> {self.close_radius:8} [Closing operation radius in voxels]")
        print(f"    prune_length     -> {self.prune_length:8} [Length to prune skeleton branches]")
        
        print("\nImage Properties:")
        print(f"    pixel_size_x     -> {self.pixel_size_x:8.3f} [Pixel size in x direction (µm/pixel)]")
        print(f"    pixel_size_y     -> {self.pixel_size_y:8.3f} [Pixel size in y direction (µm/pixel)]")
        print(f"    pixel_size_z     -> {self.pixel_size_z:8.3f} [Pixel size in z direction (µm/pixel)]")
        print("\n")
        
    def run_pipeline(self) -> None:
        """Run the complete vessel analysis pipeline.
        
        This executes all analysis steps in sequence:
        1. Extract ROI
        2. Smooth volume
        3. Segment vessels
        4. Create skeleton
        5. Analyze layers
        """
        print("\nRunning vessel analysis pipeline...")
        print("1. Extracting ROI...")
        self.extract_roi()
        
        print("2. Smoothing volume...")
        self.smooth()
        
        print("3. Segmenting vessels...")
        self.segment()
        
        print("4. Creating skeleton...")
        self.skeletonize()
        
        print("5. Analyzing layers...")
        self.analyze_layers()
        
        print("Pipeline complete!\n") 