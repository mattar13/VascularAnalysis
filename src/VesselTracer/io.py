from typing import Tuple
import numpy as np
import xmltodict
from czifile import CziFile
import tifffile as tiff

def load_vessel_image(path: str) -> Tuple[np.ndarray, float, float, float]:
    """Load a 3-D vessel image and voxel sizes (µm).
    
    Args:
        path: Path to CZI file
        
    Returns:
        tuple: (image_volume, pixel_size_x, pixel_size_y, pixel_size_z)
    """
    with CziFile(path) as czi:
        vol = czi.asarray()[0, 0, 0, 0, ::-1, :, :, 0]
        meta = xmltodict.parse(czi.metadata())

    def _px_um(axis_id: str) -> float:
        for d in meta["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]:
            if d["@Id"] == axis_id:
                return 1 / float(d["Value"]) / 1e6
        raise KeyError(axis_id)

    px_x, px_y, px_z = map(_px_um, ("X", "Y", "Z"))
    vol = normalize_image(vol.astype("float32"))
    return vol, px_x, px_y, px_z

def normalize_image(arr: np.ndarray) -> np.ndarray:
    """Normalize image array to [0,1] range.
    
    Args:
        arr: Input array
        
    Returns:
        np.ndarray: Normalized array
    """
    arr = arr.astype("float32")
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def save_vessel_stack(path: str, stack: np.ndarray) -> None:
    """Save vessel stack as TIFF file with proper metadata.
    
    Args:
        path: Output file path
        stack: 3D array in (Z,Y,X) order
    """
    tiff.imwrite(path, stack, metadata={"axes": "ZYX"})
