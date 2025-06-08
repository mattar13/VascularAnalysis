# %% [markdown]
# # Vessel Tracing Pipeline Demo
# 
# This notebook demonstrates the complete vessel tracing pipeline with visualizations at each step.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from VesselTracer import VesselTracer
import tifffile
from mpl_toolkits.axes_grid1 import ImageGrid

# Set up plotting style
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [12, 8]

# %% [markdown]
# ## 1. Load and Display Input Data

# %%
# Initialize the VesselTracer with your input file
input_path = Path("../test/test_files/240207_002.czi")  # Update this path to your input file
tracer = VesselTracer(input_path)

# Display the original volume
def plot_volume_slices(volume, title="Original Volume", num_slices=5):
    fig = plt.figure(figsize=(15, 10))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(1, num_slices),
                    axes_pad=0.1,
                    share_all=True)
    
    # Select evenly spaced slices
    z_indices = np.linspace(0, volume.shape[0]-1, num_slices, dtype=int)
    
    for idx, z in enumerate(z_indices):
        im = grid[idx].imshow(volume[z], cmap='gray')
        grid[idx].set_title(f'Z-slice {z}')
        grid[idx].axis('off')
    
    plt.colorbar(im, ax=grid.axes_all)
    plt.suptitle(title, y=1.02)
    plt.show()

plot_volume_slices(tracer.volume)

# %% [markdown]
# ## 2. ROI Extraction and Preprocessing

# %%
# Extract ROI and apply preprocessing
tracer.segment_roi()
tracer.median_filter()
tracer.background_smoothing()
tracer.detrend()

# Plot the preprocessed volume
plot_volume_slices(tracer.volume, title="Preprocessed Volume")

# %% [markdown]
# ## 3. Smoothing

# %%
# Apply smoothing
tracer.smooth()

# Plot the smoothed volume
plot_volume_slices(tracer.smoothed, title="Smoothed Volume") 