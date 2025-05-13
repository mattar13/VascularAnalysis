from typing import Tuple
import numpy as np
import xmltodict
from czifile import CziFile
import tifffile as tiff

def load_czi(path: str) -> Tuple[np.ndarray, float, float, float]:
    """Load a 3-D volume and voxel sizes (µm)."""
    with CziFile(path) as czi:
        vol = czi.asarray()[0, 0, 0, 0, ::-1, :, :, 0]
        meta = xmltodict.parse(czi.metadata())

    def _px_um(axis_id: str) -> float:
        for d in meta["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]:
            if d["@Id"] == axis_id:
                return 1 / float(d["Value"]) / 1e6
        raise KeyError(axis_id)

    px_x, px_y, px_z = map(_px_um, ("X", "Y", "Z"))
    vol = norm_uint(vol.astype("float32"))
    return vol, px_x, px_y, px_z

def norm_uint(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def save_zstack_as_tif(path: str, stack: np.ndarray) -> None:
    """Save (Z, Y, X) array with axis metadata."""
    tiff.imwrite(path, stack, metadata={"axes": "ZYX"})
