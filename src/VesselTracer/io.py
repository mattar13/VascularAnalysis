from typing import Tuple
import numpy as np
import xmltodict
from czifile import CziFile
import tifffile as tiff
from .config import PipelineConfig

def load_vessel_image(path: str, config: PipelineConfig) -> np.ndarray:
    """Load a 3-D vessel image and update config with voxel sizes (µm).
    
    Args:
        path: Path to CZI file
        config: PipelineConfig object to update with pixel sizes
        
    Returns:
        np.ndarray: Image volume
    """
    with CziFile(path) as czi:
        vol = czi.asarray()[0, 0, 0, 0, ::-1, :, :, 0]
        meta = xmltodict.parse(czi.metadata())

    def _px_um(axis_id: str) -> float:
        for d in meta["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]:
            if d["@Id"] == axis_id:
                return 1 / float(d["Value"]) / 1e6
        raise KeyError(axis_id)

    # Update config with pixel sizes
    config.pixel_size_x = _px_um("X")
    config.pixel_size_y = _px_um("Y")
    config.pixel_size_z = _px_um("Z")
    
    vol = normalize_image(vol.astype("float32"))
    return vol

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0,1] range."""
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def save_vessel_stack(path: str, stack: np.ndarray) -> None:
    """Save vessel stack as TIFF file with proper metadata.
    
    Args:
        path: Output file path
        stack: 3D array in (Z,Y,X) order
    """
    tiff.imwrite(path, stack, metadata={"axes": "ZYX"})
