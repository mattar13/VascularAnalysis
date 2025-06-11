#!/usr/bin/env python3
"""
Example usage of the ImageProcessor class.

This demonstrates how to use the new ImageProcessor class independently 
from the VesselTracer orchestrator class for more modular image processing.
"""

import numpy as np
from pathlib import Path
from VesselTracer import ImageModel, ROI, VesselTracerConfig, ImageProcessor

def example_basic_processing():
    """Example of basic image processing workflow using ImageProcessor."""
    
    # Create a synthetic 3D volume for demonstration
    print("Creating synthetic 3D volume...")
    Z, Y, X = 50, 100, 100
    volume = np.random.rand(Z, Y, X).astype(np.float32)
    
    # Add some vessel-like structures
    volume[20:30, 40:60, 40:60] += 0.5  # Bright region
    volume[25:35, 30:70, 30:70] += 0.3  # Another vessel
    
    # Create ImageModel
    image_model = ImageModel(
        volume=volume,
        pixel_size_x=0.5,  # microns per pixel
        pixel_size_y=0.5,
        pixel_size_z=1.0
    )
    
    # Create configuration
    config = VesselTracerConfig()
    config.find_roi = False  # Use entire volume
    config.micron_gauss_sigma = 1.0
    config.micron_median_filter_size = 10.0
    config.binarization_method = 'triangle'
    
    # Create ImageProcessor
    processor = ImageProcessor(
        config=config,
        verbose=2,  # Show detailed logging
        use_gpu=False  # Set to True if you have CuPy installed
    )
    
    print("\n" + "="*50)
    print("Starting Image Processing Pipeline")
    print("="*50)
    
    # Step 1: Extract ROI (in this case, the entire volume)
    print("\nStep 1: Extracting ROI...")
    roi = processor.segment_roi(image_model)
    
    # Step 2: Apply median filter for background subtraction
    print("\nStep 2: Background subtraction...")
    corrected_volume, background = processor.median_filter_background_subtraction(roi)
    
    # Step 3: Detrend the volume
    print("\nStep 3: Detrending...")
    detrended_volume = processor.detrend_volume(roi)
    
    # Step 4: Apply smoothing
    print("\nStep 4: Smoothing...")
    smoothed_volume = processor.smooth_volume(roi)
    
    # Step 5: Binarize
    print("\nStep 5: Binarization...")
    binary_volume = processor.binarize_volume(roi)
    
    # Step 6: Determine regions
    print("\nStep 6: Region analysis...")
    region_bounds = processor.determine_regions(roi)
    print(f"Found {len(region_bounds)} regions:")
    for region_name, (peak, sigma, bounds) in region_bounds.items():
        print(f"  {region_name}: peak={peak:.1f}, sigma={sigma:.1f}, bounds={bounds}")
    
    # Step 7: Create region map
    print("\nStep 7: Creating region map...")
    region_map = processor.create_region_map_volume(roi, region_bounds)
    
    # Step 8: Trace vessel paths
    print("\nStep 8: Tracing vessel paths...")
    paths, stats = processor.trace_vessel_paths(roi, region_bounds, split_paths=False)
    print(f"Found {len(paths)} vessel paths")
    
    print("\n" + "="*50)
    print("Processing Complete!")
    print("="*50)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Original volume shape: {volume.shape}")
    print(f"ROI shape: {roi.volume.shape}")
    print(f"Binary volume sum: {binary_volume.sum()}")
    print(f"Number of vessel paths: {len(paths)}")
    print(f"Region map unique values: {np.unique(region_map)}")
    
    return {
        'image_model': image_model,
        'roi': roi,
        'background': background,
        'binary': binary_volume,
        'region_map': region_map,
        'paths': paths,
        'region_bounds': region_bounds
    }

def example_gpu_processing():
    """Example showing GPU acceleration (if available)."""
    
    print("\n" + "="*50)
    print("GPU Processing Example")
    print("="*50)
    
    # Create configuration
    config = VesselTracerConfig()
    
    # Create processor with GPU enabled
    processor = ImageProcessor(config=config, verbose=2, use_gpu=True)
    
    # Try to activate GPU
    if processor.activate_gpu():
        print("GPU successfully activated!")
        print("All subsequent processing will use GPU acceleration.")
    else:
        print("GPU activation failed. Using CPU processing.")
    
    return processor

def example_custom_parameters():
    """Example showing custom processing parameters."""
    
    print("\n" + "="*50)
    print("Custom Parameters Example")
    print("="*50)
    
    # Create custom configuration
    config = VesselTracerConfig()
    
    # Customize parameters (in microns)
    config.micron_gauss_sigma = 2.0  # Heavier smoothing
    config.micron_median_filter_size = 20.0  # Larger background filter
    config.micron_close_radius = 3.0  # More aggressive morphological closing
    config.min_object_size = 100  # Remove smaller objects
    config.binarization_method = 'otsu'  # Use Otsu instead of triangle
    
    # Customize region finding
    config.regions = ['surface', 'middle', 'bottom']
    config.region_peak_distance = 5
    config.region_height_ratio = 0.7
    
    print("Custom configuration created:")
    print(f"  Gaussian sigma: {config.micron_gauss_sigma} µm")
    print(f"  Median filter size: {config.micron_median_filter_size} µm")
    print(f"  Close radius: {config.micron_close_radius} µm")
    print(f"  Min object size: {config.min_object_size} voxels")
    print(f"  Binarization method: {config.binarization_method}")
    print(f"  Region names: {config.regions}")
    
    # Create processor with custom config
    processor = ImageProcessor(config=config, verbose=2)
    
    return processor

def example_direct_usage():
    """Example showing direct usage without VesselTracer orchestrator."""
    
    print("\n" + "="*50)
    print("Direct ImageProcessor Usage")
    print("="*50)
    
    # You can use ImageProcessor directly for specific operations
    # without going through the full VesselTracer pipeline
    
    # Create a simple test volume
    volume = np.random.rand(30, 50, 50).astype(np.float32)
    
    # Create ImageModel and ROI directly
    roi = ROI(
        volume=volume,
        pixel_size_x=0.8,
        pixel_size_y=0.8,  
        pixel_size_z=1.2,
        min_x=0,
        min_y=0
    )
    
    # Create basic config
    config = VesselTracerConfig()
    processor = ImageProcessor(config, verbose=1)
    
    # Perform individual operations
    print("Applying just smoothing...")
    smoothed = processor.smooth_volume(roi)
    
    print("Applying just binarization...")
    binary = processor.binarize_volume(roi)
    
    print(f"Smoothed volume range: [{smoothed.min():.3f}, {smoothed.max():.3f}]")
    print(f"Binary volume sum: {binary.sum()}")
    
    return roi, smoothed, binary

if __name__ == "__main__":
    print("ImageProcessor Examples")
    print("="*60)
    
    # Run basic processing example
    results = example_basic_processing()
    
    # Show GPU example
    gpu_processor = example_gpu_processing()
    
    # Show custom parameters
    custom_processor = example_custom_parameters()
    
    # Show direct usage
    roi, smoothed, binary = example_direct_usage()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60) 