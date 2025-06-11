# ImageProcessor Class - Architecture Overview

## Introduction

The `ImageProcessor` class is a new functional class that extracts all image processing functionality from the `VesselTracer` class, providing better separation of concerns and more modular code architecture. This design allows you to use image processing methods independently or as part of the full `VesselTracer` pipeline.

## Architecture Benefits

### 1. **Separation of Concerns**
- **VesselTracer**: Orchestrates the pipeline, manages data flow, handles I/O, and coordinates analysis
- **ImageProcessor**: Focuses solely on image processing algorithms and operations
- **ImageModel/ROI**: Pure data models without processing logic

### 2. **Modularity**
- Use individual processing methods without running the entire pipeline
- Easy to test specific processing steps in isolation
- Better code reusability across different contexts

### 3. **Maintainability**
- Processing algorithms are centralized in one class
- Easier to add new processing methods
- Cleaner interfaces and dependencies

## Class Structure

```
VesselTracer/
├── image_model.py      # Data models (ImageModel, ROI)
├── image_processor.py  # Processing algorithms (ImageProcessor)
├── tracer.py          # Pipeline orchestration (VesselTracer, VesselTracerConfig)
└── plotting.py        # Visualization utilities
```

## Key Classes

### ImageProcessor

The core processing class that contains all image processing methods:

```python
from VesselTracer import ImageProcessor, VesselTracerConfig

# Create configuration
config = VesselTracerConfig()
config.micron_gauss_sigma = 2.0
config.micron_median_filter_size = 25.0

# Create processor
processor = ImageProcessor(
    config=config,
    verbose=2,
    use_gpu=False  # Set to True for GPU acceleration
)
```

### Available Processing Methods

1. **normalize_image(image_model)** - Normalize image to [0,1] range
2. **segment_roi(image_model, ...)** - Extract region of interest
3. **median_filter_background_subtraction(roi_model)** - Background subtraction
4. **detrend_volume(roi_model)** - Remove linear intensity trend
5. **smooth_volume(roi_model)** - Gaussian smoothing
6. **binarize_volume(roi_model, method)** - Threshold-based binarization
7. **determine_regions(roi_model)** - Find vessel depth regions
8. **create_region_map_volume(roi_model, region_bounds)** - Create region labels
9. **trace_vessel_paths(roi_model, region_bounds, ...)** - Skeletonize and trace paths

## Usage Examples

### 1. Using ImageProcessor Independently

```python
import numpy as np
from VesselTracer import ImageModel, ROI, VesselTracerConfig, ImageProcessor

# Create your data
volume = np.random.rand(50, 100, 100).astype(np.float32)

# Create data models
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

# Create processor
processor = ImageProcessor(config=config, verbose=2)

# Run individual processing steps
roi = processor.segment_roi(image_model)
smoothed_vol = processor.smooth_volume(roi)
binary_vol = processor.binarize_volume(roi)
```

### 2. Using with VesselTracer (Updated Architecture)

The `VesselTracer` class now uses `ImageProcessor` internally:

```python
from VesselTracer import VesselTracer

# Initialize tracer (now creates ImageProcessor internally)
tracer = VesselTracer("path/to/image.czi")

# Activate GPU acceleration (delegates to ImageProcessor)
tracer.activate_gpu()

# Run analysis (uses ImageProcessor methods internally)
tracer.run_analysis()
```

### 3. Custom Processing Pipeline

```python
# Create custom configuration
config = VesselTracerConfig()
config.micron_gauss_sigma = 2.0  # Heavier smoothing
config.binarization_method = 'otsu'  # Use Otsu instead of triangle
config.regions = ['surface', 'middle', 'deep']

# Create processor with custom settings
processor = ImageProcessor(config=config, verbose=2)

# Run custom pipeline
roi = processor.segment_roi(image_model)
corrected, background = processor.median_filter_background_subtraction(roi)
detrended = processor.detrend_volume(roi)
smoothed = processor.smooth_volume(roi)
binary = processor.binarize_volume(roi, method='otsu')

# Analyze regions and trace paths
region_bounds = processor.determine_regions(roi)
region_map = processor.create_region_map_volume(roi, region_bounds)
paths, stats = processor.trace_vessel_paths(roi, region_bounds, split_paths=True)
```

## GPU Acceleration

GPU acceleration is handled transparently by the `ImageProcessor`:

```python
# Create processor with GPU enabled
processor = ImageProcessor(config=config, use_gpu=True)

# Activate GPU (tests CUDA functionality)
if processor.activate_gpu():
    print("GPU activated successfully!")
else:
    print("Falling back to CPU processing")

# All subsequent operations will use GPU if available
smoothed = processor.smooth_volume(roi)  # Uses GPU automatically
binary = processor.binarize_volume(roi)  # Uses GPU automatically
```

## Configuration Management

The `VesselTracerConfig` class manages all processing parameters in microns:

```python
config = VesselTracerConfig()

# Set parameters in microns (automatically converted to pixels)
config.micron_gauss_sigma = 2.0           # Gaussian smoothing
config.micron_median_filter_size = 25.0   # Median filter size
config.micron_close_radius = 1.5          # Morphological closing

# Processing parameters
config.min_object_size = 64               # Minimum object size (voxels)
config.binarization_method = 'triangle'   # Thresholding method

# Region analysis parameters  
config.regions = ['superficial', 'intermediate', 'deep']
config.region_peak_distance = 2
config.region_height_ratio = 0.80
```

## Migration from Old Architecture

If you were using `VesselTracer` methods directly, the interface remains the same:

```python
# Old way (still works)
tracer = VesselTracer("image.czi")
tracer.smooth()
tracer.binarize()

# New way (more flexible)
tracer = VesselTracer("image.czi")
# ImageProcessor is created automatically and used internally
tracer.smooth()  # Delegates to processor.smooth_volume()
tracer.binarize()  # Delegates to processor.binarize_volume()

# Direct access to processor
tracer.processor.smooth_volume(tracer.roi_model)
```

## Error Handling

The `ImageProcessor` provides clear error messages:

```python
try:
    # This will raise ValueError if no volume data
    processor.smooth_volume(empty_roi)
except ValueError as e:
    print(f"Processing error: {e}")

try:
    # This will raise ValueError if CuPy not available
    processor.activate_gpu()
except Exception as e:
    print(f"GPU activation failed: {e}")
```

## Testing Individual Components

You can now test individual processing steps easily:

```python
import pytest
from VesselTracer import ImageProcessor, VesselTracerConfig, ROI

def test_smoothing():
    # Create test data
    volume = np.random.rand(10, 20, 20)
    roi = ROI(volume=volume, pixel_size_x=1.0, pixel_size_y=1.0, pixel_size_z=1.0)
    
    # Create processor
    config = VesselTracerConfig()
    processor = ImageProcessor(config)
    
    # Test smoothing
    smoothed = processor.smooth_volume(roi)
    
    # Verify results
    assert smoothed.shape == volume.shape
    assert smoothed.dtype == volume.dtype
```

## Performance Considerations

1. **GPU Acceleration**: Significant speedup for large volumes when CuPy is available
2. **Parallel Processing**: Median filtering uses multithreading on CPU
3. **Memory Efficiency**: Processing is done in-place where possible
4. **Caching**: Pixel conversions are cached to avoid recomputation

## Future Extensions

The modular architecture makes it easy to add new features:

1. **New Processing Methods**: Add methods to `ImageProcessor`
2. **Alternative Algorithms**: Easy to implement different binarization or smoothing methods
3. **Custom Pipelines**: Create specialized processors for different imaging modalities
4. **Batch Processing**: Process multiple images with the same `ImageProcessor` instance

## Example: Complete Workflow

See `example_usage.py` for a complete example showing:
- Basic processing pipeline
- GPU acceleration usage
- Custom parameter configuration
- Direct method usage
- Error handling

```bash
python example_usage.py
```

This will run through all the different ways to use the `ImageProcessor` class. 