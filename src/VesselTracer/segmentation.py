import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, ball, binary_closing
from .config import PipelineConfig

def extract_roi(volume: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Extract a region of interest (ROI) from a 3D volume.
    
    Args:
        volume: 3D numpy array (z, y, x)
        config: PipelineConfig object containing ROI parameters and pixel sizes
        
    Returns:
        3D numpy array containing the extracted ROI
    """
    # Convert ROI size from microns to pixels
    h_x = round(config.micron_roi/2 * 1/config.pixel_size_x)
    h_y = round(config.micron_roi/2 * 1/config.pixel_size_y)
    
    # Extract ROI
    roi = volume[:, 
                 config.center_y-h_y:config.center_y+h_y,
                 config.center_x-h_x:config.center_x+h_x]
    
    return roi

def segment_binary(vol: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Segment vessels using binary thresholding.
    
    Args:
        vol: Input volume
        config: PipelineConfig object containing segmentation parameters
        
    Returns:
        Binary segmentation mask
    """
    # Threshold at mean + 1 std
    thresh = vol.mean() + vol.std()
    binary = vol > thresh
    
    # Remove small objects
    binary = remove_small_objects(binary, min_size=config.min_object_size)
    
    # Binary closing to fill small gaps
    if config.close_radius > 0:
        binary = binary_closing(binary, ball(config.close_radius))
        
    return binary
