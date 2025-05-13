import numpy as np
from scipy.signal import find_peaks, peak_widths
from collections import defaultdict

def determine_region(mean_zprofile: np.ndarray, n_stds=2):
    peaks, _ = find_peaks(mean_zprofile, distance=2)
    widths, *_ = peak_widths(mean_zprofile, peaks, rel_height=0.80)
    sigmas = widths / (n_stds * np.sqrt(2 * np.log(2)))
    regions = ["superficial", "intermediate", "deep"]  # editable
    return {
        name: (pk, σ, (pk - σ, pk + σ))
        for name, pk, σ in zip(regions, peaks, sigmas)
    }

def split_paths_by_layer(paths, labels):
    new_paths = {}
    for bid, (zs, ys, xs) in paths.items():
        layer_ids = [labels[z] for z in zs]
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
