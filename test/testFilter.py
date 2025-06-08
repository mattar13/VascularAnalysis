import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from VesselTracer import VesselTracer
from VesselTracer.plotting import plot_projections, plot_mean_zprofile

def create_sphere_volume(size=(100, 100, 100), radius=20):
    """
    Create a 3D volume with a sphere of ones in the middle.
    
    Parameters:
    -----------
    size : tuple
        Size of the volume (z, y, x)
    radius : float
        Radius of the sphere
        
    Returns:
    --------
    volume : ndarray
        3D volume with a sphere of ones
    """
    # Create coordinate arrays
    z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
    
    # Calculate center coordinates
    center = np.array([s//2 for s in size])
    
    # Calculate distance from center for each point
    dist = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    
    # Create sphere mask
    volume = (dist <= radius).astype(np.float32)
    
    return volume

def plot_volume_slices(volume, title="Volume Slices"):
    """
    Plot three orthogonal slices through the volume.
    
    Parameters:
    -----------
    volume : ndarray
        3D volume to plot
    title : str
        Title for the plot
    """
    # Get center indices
    center = np.array([s//2 for s in volume.shape])
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot three orthogonal slices
    ax1.imshow(volume[center[0], :, :], cmap='gray')
    ax1.set_title('XY slice')
    
    ax2.imshow(volume[:, center[1], :], cmap='gray')
    ax2.set_title('XZ slice')
    
    ax3.imshow(volume[:, :, center[2]], cmap='gray')
    ax3.set_title('YZ slice')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    # Create test volume
    volume = create_sphere_volume(size=(100, 100, 100), radius=20)
    #Here we want to run the tracer on that volume
    print("Volume shape: ", volume.shape)
    tracer = VesselTracer(volume)
    #tracer.segment_roi(remove_dead_frames=True, dead_frame_threshold=1.5)
    tracer.run_analysis(skip_trace=True)

    fig1, axes1 = plot_projections(tracer, mode='smoothed')

    fig2, axes2 = plot_projections(tracer, mode='binary')

    fig3, axes3 = plot_projections(tracer, mode='binary', depth_coded=True)

    fig4, axes4 = plot_mean_zprofile(tracer)

    # Plot the volume
    plot_volume_slices(volume, "Test Volume with Sphere")
    
    # Print volume statistics
    print(f"Volume shape: {volume.shape}")
    print(f"Number of ones: {np.sum(volume)}")
    print(f"Volume fraction: {np.sum(volume) / volume.size:.3f}")

if __name__ == "__main__":
    main()
