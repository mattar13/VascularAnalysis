import numpy as np
from scipy.signal import find_peaks, peak_widths
from collections import defaultdict

def analyze_vessel_layers(mean_zprofile: np.ndarray, n_stds: float = 2):
    """Analyze vessel profile to determine layer boundaries.
    
    Args:
        mean_zprofile: Mean intensity profile along Z-axis
        n_stds: Number of standard deviations for layer width
        
    Returns:
        dict: Layer information with peaks, widths, and boundaries
    """
    peaks, _ = find_peaks(mean_zprofile, distance=2)
    widths, *_ = peak_widths(mean_zprofile, peaks, rel_height=0.80)
    sigmas = widths / (n_stds * np.sqrt(2 * np.log(2)))
    regions = ["superficial", "intermediate", "deep"]  # editable
    return {
        name: (pk, σ, (pk - σ, pk + σ))
        for name, pk, σ in zip(regions, peaks, sigmas)
    }

def segment_vessel_paths(paths: dict, layer_labels: np.ndarray):
    """Segment vessel paths by layer.
    
    Args:
        paths: Dictionary of vessel paths with coordinates
        layer_labels: Array of layer labels for each z-position
        
    Returns:
        dict: Segmented paths with layer information
    """
    new_paths = {}
    for bid, (zs, ys, xs) in paths.items():
        layer_ids = [layer_labels[z] for z in zs]
        start = 0
        for i in range(1, len(zs)):
            if layer_ids[i] != layer_ids[i-1]:
                new_paths[(bid, start)] = (
                    zs[start:i], ys[start:i], xs[start:i], layer_ids[i-1]
                )
                start = i
        new_paths[(bid, start)] = (
            zs[start:], ys[start:], xs[start:], layer_ids[start]
        )
    return new_paths
