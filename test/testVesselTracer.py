import numpy as np
import matplotlib.pyplot as plt

# --- pipeline pieces ---------------------------------------------------------
from VesselTracer.preprocessing import smooth
from VesselTracer.segmentation  import segment_binary
from VesselTracer.skeletonize   import skeleton_stats
from VesselTracer.plotting      import show_max_projection

# ----------------------------------------------------------------------------- 
# 1) Make a 10×10×10 volume with one bright 'vessel' (a diagonal line)
# ----------------------------------------------------------------------------- 
vol = np.zeros((10, 10, 10), dtype=np.float32)
for i in range(10):
    vol[i, i, i] = 1.0            # set a single-voxel path
# (you could draw something thicker for a more realistic test)

# 2) Smooth a little so thresholding has something to grab onto
vol_smoothed = smooth(vol, sigma=1.0)

# 3) Binary segmentation (min_size=1 keeps even single voxels)
bw = segment_binary(vol_smoothed,
                    min_size   = 1,
                    close_radius = 0)

# 4) Skeleton statistics
skel, df = skeleton_stats(bw)

print(df)
# 5) Report + quick viz --------------------------------------------------------
print("\nSkeleton summary:")
print(df[["path_length", "voxel_count"]])

# Max-intensity projection just so we can *see* something
show_max_projection(vol_smoothed, title="10×10×10 synthetic volume")
plt.tight_layout()
plt.show()