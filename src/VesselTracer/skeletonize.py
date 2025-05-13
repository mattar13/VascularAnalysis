import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

def skeleton_stats(binary_vol: np.ndarray):
    ske = skeletonize(binary_vol)
    skel = Skeleton(ske)
    df = summarize(skel, separator="-")
    return skel, df
