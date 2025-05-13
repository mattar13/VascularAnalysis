import numpy as np
from scipy.ndimage import gaussian_filter
from .config import PipelineConfig

def crop_and_detrend(vol: np.ndarray,
                     center: tuple[int, int],
                     micron_roi: float,
                     px_x: float,
                     px_y: float) -> np.ndarray:
    """Crop a square ROI and subtract linear attenuation along Z."""
    h_x = int(micron_roi / px_x / 2)
    h_y = int(micron_roi / px_y / 2)
    yc, xc = center
    roi = vol[:, yc-h_y:yc+h_y, xc-h_x:xc+h_x]

    # Remove empty slices
    zprofile = roi.mean(axis=(1, 2))
    keep = zprofile > zprofile.min() + 1.5 * zprofile.std()
    roi = roi[keep]

    # Linear attenuation correction
    z = np.arange(roi.shape[0])
    m, b = np.polyfit(z, roi.mean(axis=(1, 2)), 1)
    roi = roi - (m * z + b)[:, None, None]
    return roi

def smooth(vol: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Apply Gaussian smoothing to volume.
    
    Args:
        vol: Input volume
        config: PipelineConfig object containing smoothing parameters
        
    Returns:
        Smoothed volume
    """
    return gaussian_filter(vol, sigma=config.gauss_sigma)
