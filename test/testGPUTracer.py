#!/usr/bin/env python3
"""
GPU Tracer Test

This script creates a test volume with random noise and a sphere
to test GPU operations using CuPy.
"""

import numpy as np
import cupy as cp
import time
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from VesselTracer import VesselTracer

def check_gpu_availability():
    """
    Check GPU availability and print detailed information about available GPUs.
    """
    print("\nGPU Availability Check")
    print("=====================")
    
    try:
        # Check if CUDA is available
        cuda_available = cp.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            # Get number of devices
            n_devices = cp.cuda.runtime.getDeviceCount()
            print(f"Number of CUDA devices: {n_devices}")
            
            # Print information for each device
            for i in range(n_devices):
                print(f"\nDevice {i}:")
                device = cp.cuda.Device(i)
                
                # Get device properties
                props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"  Name: {props['name'].decode()}")
                print(f"  Compute Capability: {props['major']}.{props['minor']}")
                print(f"  Total Memory: {props['totalGlobalMem'] / (1024**3):.1f} GB")
                print(f"  Multi-Processor Count: {props['multiProcessorCount']}")
                
                # Get current memory info
                mem_info = device.mem_info
                print(f"  Free Memory: {mem_info[0] / (1024**3):.1f} GB")
                print(f"  Total Memory: {mem_info[1] / (1024**3):.1f} GB")
                
                # Try a simple GPU operation
                try:
                    with device:
                        x = cp.array([1, 2, 3])
                        y = cp.array([4, 5, 6])
                        z = x + y
                        print(f"  Test Operation: Successful")
                except Exception as e:
                    print(f"  Test Operation: Failed - {str(e)}")
        
        return cuda_available
        
    except Exception as e:
        print(f"Error checking GPU availability: {str(e)}")
        return False

def verify_gpu_memory(required_memory_gb):
    """
    Verify if there's enough GPU memory available.
    
    Args:
        required_memory_gb: Required memory in GB
    
    Returns:
        bool: True if enough memory is available
    """
    try:
        if not cp.cuda.is_available():
            return False
            
        device = cp.cuda.Device(0)
        free_memory = device.mem_info[0] / (1024**3)  # Convert to GB
        
        print(f"\nGPU Memory Check")
        print(f"Required Memory: {required_memory_gb:.1f} GB")
        print(f"Available Memory: {free_memory:.1f} GB")
        
        if free_memory >= required_memory_gb:
            print("✓ Sufficient GPU memory available")
            return True
        else:
            print("✗ Insufficient GPU memory")
            return False
            
    except Exception as e:
        print(f"Error checking GPU memory: {str(e)}")
        return False

def create_test_volume_cpu(shape=(100, 3000, 3000), tube_radius=50, noise_level=0.1, plane='xy'):
    """
    Create a test volume on CPU using NumPy.
    
    Args:
        shape: Tuple of (depth, height, width)
        tube_radius: Radius of the tube in voxels
        noise_level: Standard deviation of the random noise
        plane: Plane for tube projection ('xy' or 'xz')
    
    Returns:
        numpy.ndarray: The test volume
    """
    print("Creating test volume on CPU...")
    start_time = time.time()
    
    # Create base volume with random noise
    volume = np.random.normal(0.1, noise_level, shape).astype(np.float32)
    volume = np.clip(volume, 0, 1)  # Ensure all values are between 0 and 1
    
    # Calculate center coordinates
    center_z = shape[0] // 2
    center_y = shape[1] // 2
    center_x = shape[2] // 2
    
    if plane == 'xy':
        # Create coordinate grids for x and y
        y, x = np.ogrid[:shape[1], :shape[2]]
        # Calculate distance from center in x-y plane
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        # Create tube mask (same for all z-slices)
        tube_mask = dist_from_center <= tube_radius
        # Add tube to volume with higher intensity
        for z in range(shape[0]):
            volume[z, tube_mask] = 0.8  # Set tube intensity
        # Add some noise to the tube
        tube_noise = np.random.normal(0, noise_level/2, size=shape)
        for z in range(shape[0]):
            volume[z, tube_mask] += tube_noise[z, tube_mask]
    
    elif plane == 'xz':
        # Create coordinate grids for x and z
        z, x = np.ogrid[:shape[0], :shape[2]]
        # Calculate distance from center in x-z plane
        dist_from_center = np.sqrt((z - center_z)**2 + (x - center_x)**2)
        # Create tube mask (same for all y-slices)
        tube_mask = dist_from_center <= tube_radius
        # Add tube to volume with higher intensity
        for y in range(shape[1]):
            volume[:, y, tube_mask] = 0.8  # Set tube intensity
        # Add some noise to the tube
        tube_noise = np.random.normal(0, noise_level/2, size=shape)
        for y in range(shape[1]):
            volume[:, y, tube_mask] += tube_noise[:, y, tube_mask]
    
    else:
        raise ValueError("plane must be either 'xy' or 'xz'")
    
    volume = np.clip(volume, 0, 1)  # Ensure values stay in [0,1]
    
    end_time = time.time()
    print(f"CPU volume creation took {end_time - start_time:.2f} seconds")
    print(f"Volume shape: {volume.shape}")
    print(f"Memory usage: {volume.nbytes / (1024**3):.2f} GB")
    print(f"Tube projected in {plane} plane")
    
    return volume

def create_test_volume_gpu(shape=(100, 3000, 3000), tube_radius=50, noise_level=0.1, plane='xy'):
    """
    Create a test volume on CPU and convert to GPU using CuPy.
    
    Args:
        shape: Tuple of (depth, height, width)
        tube_radius: Radius of the tube in voxels
        noise_level: Standard deviation of the random noise
        plane: Plane for tube projection ('xy' or 'xz')
    
    Returns:
        cupy.ndarray: The test volume on GPU
    """
    print("Creating test volume on CPU and converting to GPU...")
    start_time = time.time()
    
    # Create volume on CPU first
    cpu_volume = create_test_volume_cpu(shape, tube_radius, noise_level, plane)
    
    # Convert to GPU
    gpu_volume = cp.asarray(cpu_volume)
    
    end_time = time.time()
    print(f"GPU conversion took {end_time - start_time:.2f} seconds")
    print(f"Volume shape: {gpu_volume.shape}")
    print(f"Memory usage: {gpu_volume.nbytes / (1024**3):.2f} GB")
    
    return gpu_volume

def plot_z_sections(volume, num_sections=5, save_path=None):
    """
    Plot multiple z-sections of the volume.
    
    Args:
        volume: 3D numpy array
        num_sections: Number of z-sections to plot
        save_path: Path to save the figure (optional)
    """
    # Create a custom colormap for better visualization
    colors = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    # Calculate z-indices for sections
    z_indices = np.linspace(0, volume.shape[0]-1, num_sections, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(1, num_sections, figsize=(20, 4))
    fig.suptitle('Z-Sections of Volume', fontsize=16)
    
    for idx, z in enumerate(z_indices):
        im = axes[idx].imshow(volume[z], cmap=cmap, vmin=0, vmax=1)
        axes[idx].set_title(f'Z = {z}')
        axes[idx].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes.ravel().tolist(), label='Intensity')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved z-sections plot to {save_path}")
    
    plt.show()

def plot_gaussian_splat(volume, sigma=2.0, save_path=None, downsample_factor=10):
    """
    Create a 3D Gaussian splat visualization of the volume.
    
    Args:
        volume: 3D numpy array
        sigma: Standard deviation for Gaussian blur
        save_path: Path to save the figure (optional)
        downsample_factor: Factor to downsample the volume for 3D plotting
    """
    from scipy.ndimage import gaussian_filter
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create a custom colormap
    colors = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    # Downsample the volume for 3D plotting
    volume_ds = volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    # Create coordinate grids
    z, y, x = np.meshgrid(np.arange(volume_ds.shape[0]),
                         np.arange(volume_ds.shape[1]),
                         np.arange(volume_ds.shape[2]),
                         indexing='ij')
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    scatter = ax.scatter(x, y, z, c=volume_ds.flatten(),
                        cmap=cmap, alpha=0.1, s=1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Gaussian Splat Visualization')
    
    # Add colorbar
    plt.colorbar(scatter, label='Intensity')
    
    # Set the viewing angle
    ax.view_init(elev=30, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D Gaussian splat plot to {save_path}")
    
    plt.show()
    
    # Also create a 2D projection for reference
    projection = np.mean(volume, axis=0)
    projection = gaussian_filter(projection, sigma=sigma)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(projection, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label='Intensity')
    plt.title('2D Gaussian Splat Projection (Reference)')
    plt.axis('off')
    
    if save_path:
        ref_path = str(save_path).replace('.png', '_2d.png')
        plt.savefig(ref_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D reference plot to {ref_path}")
    
    plt.show()

def save_volume_as_tiff(volume, output_path, dtype=np.uint16):
    """
    Save a 3D volume as a TIFF z-stack.
    
    Args:
        volume: 3D numpy array
        output_path: Path to save the TIFF file
        dtype: Data type for saving (default: uint16)
    """
    import tifffile
    
    # Convert to desired dtype and scale to full range
    if dtype == np.uint16:
        volume_save = (volume * 65535).astype(dtype)
    elif dtype == np.uint8:
        volume_save = (volume * 255).astype(dtype)
    else:
        volume_save = volume.astype(dtype)
    
    # Save as TIFF
    tifffile.imwrite(output_path, volume_save)
    print(f"Saved volume as TIFF z-stack to {output_path}")
    print(f"Volume shape: {volume_save.shape}")
    print(f"Data type: {volume_save.dtype}")
    print(f"File size: {volume_save.nbytes / (1024**3):.2f} GB")

def main():
    """Main function to demonstrate volume creation and visualization."""
    print("GPU Tracer Test")
    print("==============")
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Test volume dimensions
    shape = (100, 3000, 3000)
    
    # Calculate required memory (float32)
    required_memory = np.prod(shape) * 4 / (1024**3)  # in GB
    print(f"\nRequired memory for volume: {required_memory:.1f} GB")
    
    try:
        # Create volume on CPU
        volume = create_test_volume_cpu(shape)
        
        # Create VesselTracer in CPU mode
        print("\nInitializing VesselTracer in CPU mode...")
        config_path = Path(__file__).parent.parent / 'config' / 'default_vessel_config.yaml'
        
        print(f"Config path: {config_path}")
        tracer = VesselTracer(volume, config_path=config_path)
        
        print("Activating GPU...")
        tracer.activate_gpu()
        # Try to activate GPU if available
        # if gpu_available:
        #     print("\nAttempting to activate GPU mode...")
        #     if tracer.activate_gpu():
        #         print("Successfully activated GPU mode")
        #     else:
        #         print("Failed to activate GPU mode, continuing in CPU mode")
        
        # Create output directory if it doesn't exist
        output_dir = Path('test/output')
        output_dir.mkdir(exist_ok=True)
        
        # Save volume as TIFF
        print("\nSaving volume as TIFF...")
        save_volume_as_tiff(volume, 
                          output_dir / 'test_volume.tif',
                          dtype=np.uint16)
        
        # Plot z-sections
        print("\nPlotting z-sections...")
        plot_z_sections(volume, num_sections=5, 
                       save_path=output_dir / 'z_sections.png')
        plt.close('all')  # Close all figures
        
        # Plot Gaussian splat
        print("\nPlotting Gaussian splat...")
        plot_gaussian_splat(volume, sigma=2.0,
                           save_path=output_dir / 'gaussian_splat.png')
        plt.close('all')  # Close all figures
        
    except Exception as e:
        print(f"Error during operations: {e}")
        print("GPU mode not available")
        raise e
        # # Create volume on CPU
        # volume = create_test_volume_cpu(shape)
        
        # # Create output directory if it doesn't exist
        # output_dir = Path('test/output')
        # output_dir.mkdir(exist_ok=True)
        
        # # Save volume as TIFF
        # print("\nSaving volume as TIFF...")
        # save_volume_as_tiff(volume, 
        #                   output_dir / 'test_volume.tif',
        #                   dtype=np.uint16)
        
        # # Plot z-sections
        # print("\nPlotting z-sections...")
        # plot_z_sections(volume, num_sections=5,
        #                save_path=output_dir / 'z_sections.png')
        # plt.close('all')  # Close all figures
        
        # # Plot Gaussian splat
        # print("\nPlotting Gaussian splat...")
        # plot_gaussian_splat(volume, sigma=2.0,
        #                    save_path=output_dir / 'gaussian_splat.png')
        # plt.close('all')  # Close all figures

if __name__ == "__main__":
    main()
