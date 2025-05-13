import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, ball

def segment_binary(vol: np.ndarray,
                   min_size: int,
                   close_radius: int) -> np.ndarray:
    """Global Otsu threshold + 3-D clean-up."""
    thresh = threshold_otsu(vol.ravel())
    bw = vol > thresh
    bw = remove_small_objects(bw, min_size)
    bw = closing(bw, ball(close_radius))
    return bw
