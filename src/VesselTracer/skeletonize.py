import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

def skeleton_stats(binary_vol: np.ndarray):
    print(f"Skeletonizing binary volume of shape {binary_vol.shape}")
    ske = skeletonize(binary_vol)
    print(f"Skeletonized volume of shape {ske.shape}")
    skel = Skeleton(ske)
    print(f"Skeleton object of type {type(skel)}")
    df = summarize(skel, separator="-")
    return skel, df
