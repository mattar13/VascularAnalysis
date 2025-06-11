#!/usr/bin/env python3
"""
Demonstration of the new VesselTracer architecture.

This script shows how the responsibilities are now separated:
- VesselAnalysisController: Orchestrates the entire pipeline
- ImageProcessor: Handles all image processing operations
- VesselTracer: Focuses solely on vessel tracing from binary images
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import the new architecture components
from VesselTracer import (
    VesselAnalysisController, 
    VesselTracer, 
    ImageProcessor, 
    VesselTracerConfig,
    ImageModel,
    ROI
)

def demo_controller_pipeline():
    """Demonstrate the complete pipeline using the Controller."""
    print("=" * 60)
    print("DEMO 1: Complete Pipeline with VesselAnalysisController")
    print("=" * 60)
    
    # Create synthetic data for demonstration
    print("Creating synthetic 3D vessel data...")
    Z, Y, X = 50, 100, 100
    volume = np.random.rand(Z, Y, X).astype(np.float32) * 0.3
    
    # Add vessel-like structures
    volume[20:30, 40:60, 40:60] += 0.5  # Main vessel
    volume[25:35, 30:70, 30:70] += 0.3  # Branch vessel
    volume[15:25, 50:80, 20:50] += 0.4  # Another vessel
    
    # Initialize controller with synthetic data
    controller = VesselAnalysisController(
        input_data=volume,
        pixel_sizes=(1.0, 0.5, 0.5)  # z, y, x in microns
    )
    
    print(f"Loaded volume shape: {controller.image_model.volume.shape}")
    print("Running complete analysis pipeline...")
    
    # Run the complete analysis
    controller.run_analysis(
        skip_smoothing=False,
        skip_binarization=False,
        skip_regions=False,
        skip_trace=False
    )
    
    # Print results
    print(f"\nResults:")
    print(f"- Binary volume created: {controller.binary is not None}")
    print(f"- Region bounds found: {len(controller.region_bounds)}")
    print(f"- Vessel paths traced: {controller.path_count}")
    
    if controller.path_count > 0:
        print(f"- Total path points: {sum(len(path['coordinates']) for path in controller.paths.values())}")
    
    return controller

def demo_individual_components():
    """Demonstrate using individual components separately."""
    print("\n" + "=" * 60)
    print("DEMO 2: Using Individual Components Separately")
    print("=" * 60)
    
    # Create synthetic data
    Z, Y, X = 30, 80, 80
    volume = np.random.rand(Z, Y, X).astype(np.float32) * 0.2
    volume[10:20, 30:50, 30:50] += 0.6  # Bright vessel structure
    
    print("Step 1: Setting up data models...")
    # Create image model
    image_model = ImageModel()
    image_model.load_from_array(volume, pixel_sizes=(1.0, 0.5, 0.5))
    
    # Create configuration
    config = VesselTracerConfig()
    config.find_roi = False  # Use entire volume
    config.micron_gauss_sigma = 1.0
    config.binarization_method = 'triangle'
    
    print("Step 2: Using ImageProcessor for preprocessing...")
    # Create ImageProcessor
    processor = ImageProcessor(config=config, verbose=2)
    
    # Extract ROI (entire volume in this case)
    roi_model = processor.segment_roi(image_model)
    print(f"ROI extracted: {roi_model.volume.shape}")
    
    # Apply image processing steps
    processor.median_filter_background_subtraction(roi_model)
    processor.detrend_volume(roi_model)
    processor.smooth_volume(roi_model)
    binary_volume = processor.binarize_volume(roi_model)
    
    print(f"Binary volume created: {binary_volume.shape}, sum: {binary_volume.sum()}")
    
    print("Step 3: Using VesselTracer for path tracing...")
    # Create VesselTracer
    tracer = VesselTracer(config=config, verbose=2)
    
    # Trace paths from binary volume
    paths, stats = tracer.trace_paths(binary_volume)
    
    print(f"Paths traced: {len(paths)}")
    if len(paths) > 0:
        print(f"First path length: {len(paths[1]['coordinates'])}")
        
    # Create DataFrames
    paths_df = tracer.create_paths_dataframe(pixel_sizes=(1.0, 0.5, 0.5))
    summary_df = tracer.create_path_summary_dataframe(pixel_sizes=(1.0, 0.5, 0.5))
    
    print(f"Paths DataFrame shape: {paths_df.shape}")
    print(f"Summary DataFrame shape: {summary_df.shape}")
    
    return roi_model, tracer

def demo_comparison():
    """Compare the old and new approaches."""
    print("\n" + "=" * 60)
    print("DEMO 3: Architecture Comparison")
    print("=" * 60)
    
    print("NEW ARCHITECTURE:")
    print("- VesselAnalysisController: Orchestrates pipeline, manages data flow")
    print("- ImageProcessor: Handles all image processing (ROI, smoothing, binarization)")
    print("- VesselTracer: Focuses only on skeletonization and path analysis")
    print("- Data Models (ImageModel, ROI): Store data without processing logic")
    print("")
    print("BENEFITS:")
    print("- ✅ Clear separation of concerns")
    print("- ✅ VesselTracer can be used independently for just tracing")
    print("- ✅ ImageProcessor can be used for custom processing pipelines")
    print("- ✅ Controller manages the complete workflow")
    print("- ✅ Each component has a single responsibility")
    print("- ✅ Easier to test individual components")
    print("- ✅ More modular and extensible")
    
    print("\nOLD ARCHITECTURE:")
    print("- VesselTracer (in tracer.py): Did everything - loading, processing, tracing, saving")
    print("- Monolithic design with mixed responsibilities")

def demo_flexibility():
    """Demonstrate the flexibility of the new architecture."""
    print("\n" + "=" * 60)
    print("DEMO 4: Flexibility Examples")
    print("=" * 60)
    
    # Create test data
    volume = np.random.rand(20, 60, 60).astype(np.float32)
    volume[8:12, 20:40, 20:40] = 0.8  # Vessel structure
    
    print("Example 1: Using VesselTracer independently")
    print("-" * 45)
    
    # Create binary volume directly (simulating external preprocessing)
    binary_vol = volume > 0.5
    
    # Use VesselTracer directly
    tracer = VesselTracer(verbose=1)
    paths, stats = tracer.trace_paths(binary_vol)
    print(f"Traced {len(paths)} paths from external binary volume")
    
    print("\nExample 2: Custom processing with ImageProcessor")
    print("-" * 50)
    
    # Use ImageProcessor for custom workflow
    config = VesselTracerConfig()
    processor = ImageProcessor(config, verbose=1)
    
    image_model = ImageModel()
    image_model.load_from_array(volume, (1.0, 1.0, 1.0))
    
    roi = processor.segment_roi(image_model)
    processor.smooth_volume(roi)
    custom_binary = processor.binarize_volume(roi, method='otsu')
    print(f"Custom processing created binary volume: {custom_binary.shape}")
    
    print("\nExample 3: Controller with custom parameters")
    print("-" * 46)
    
    controller = VesselAnalysisController(volume, pixel_sizes=(2.0, 1.0, 1.0))
    controller.config.micron_gauss_sigma = 2.0  # Custom smoothing
    controller.run_analysis(skip_regions=True)  # Skip region detection
    print(f"Controller with custom config traced {controller.path_count} paths")

if __name__ == "__main__":
    print("VesselTracer New Architecture Demonstration")
    print("=" * 60)
    
    # Run all demonstrations
    controller = demo_controller_pipeline()
    roi_model, tracer = demo_individual_components()
    demo_comparison()
    demo_flexibility()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("The new architecture provides:")
    print("1. Better separation of concerns")
    print("2. More modular and testable code")
    print("3. Flexibility to use components independently")
    print("4. Cleaner interfaces and responsibilities")
    print("5. Easier maintenance and extension")
    print("\nThe Controller manages the full pipeline,")
    print("while individual components can be used as needed.") 