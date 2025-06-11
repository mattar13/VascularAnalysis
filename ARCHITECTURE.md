# VesselTracer New Architecture

## Overview

The VesselTracer package has been restructured to provide better separation of concerns and more modular functionality. The new architecture separates responsibilities between different components:

## Components

### 1. VesselAnalysisController (`controller.py`)
**Role**: Pipeline orchestrator and main entry point
- Manages the complete vessel analysis workflow
- Coordinates between ImageProcessor and VesselTracer
- Handles file I/O and configuration
- Provides convenient access to results
- **Only class that should execute the full pipeline**

### 2. VesselTracer (`vessel_tracer.py`)
**Role**: Specialized vessel tracing and skeletonization
- Takes binary images and creates vessel skeletons
- Traces paths and creates path dictionaries
- Generates pandas DataFrames from traced paths
- **Focused solely on tracing operations**

### 3. ImageProcessor (`image_processor.py`)
**Role**: Image processing operations
- ROI extraction and segmentation
- Background subtraction and detrending
- Smoothing and filtering
- Binarization and thresholding
- Region detection and analysis

### 4. Data Models (`image_model.py`)
**Role**: Data storage without processing logic
- `ImageModel`: Stores full volume data and metadata
- `ROI`: Stores region-of-interest data and operations

## Usage Examples

### Complete Pipeline (Recommended)

```python
from VesselTracer import VesselAnalysisController

# Initialize controller
controller = VesselAnalysisController("path/to/image.czi")

# Optional: activate GPU acceleration
controller.activate_gpu()

# Run complete analysis
controller.run_analysis(
    skip_smoothing=False,
    skip_binarization=False,
    skip_regions=False,
    skip_trace=False
)

# Access results
print(f"Traced {controller.path_count} vessel paths")
print(f"Found {len(controller.region_bounds)} regions")

# Generate visualizations using plotting functions
from VesselTracer.plotting import plot_projections, plot_paths
fig, ax = plot_projections(controller, mode='binary')
fig, ax = plot_paths(controller, region_colorcode=True)
```

### Using VesselTracer Independently

```python
from VesselTracer import VesselTracer
import numpy as np

# Create or load a binary volume
binary_volume = np.load("binary_vessels.npy")

# Initialize tracer
tracer = VesselTracer(verbose=2)

# Trace paths from binary volume
paths, stats = tracer.trace_paths(binary_volume)

# Generate DataFrames
paths_df = tracer.create_paths_dataframe(pixel_sizes=(1.0, 0.5, 0.5))
summary_df = tracer.create_path_summary_dataframe(pixel_sizes=(1.0, 0.5, 0.5))

print(f"Traced {len(paths)} paths")
print(f"Detailed DataFrame shape: {paths_df.shape}")
```

### Using ImageProcessor for Custom Workflows

```python
from VesselTracer import ImageProcessor, ImageModel, VesselTracerConfig

# Load data
image_model = ImageModel()
image_model.load_from_czi("image.czi")

# Create custom configuration
config = VesselTracerConfig()
config.micron_gauss_sigma = 2.0
config.binarization_method = 'otsu'

# Create processor
processor = ImageProcessor(config=config, verbose=2)

# Apply custom processing steps
roi = processor.segment_roi(image_model)
processor.smooth_volume(roi)
binary = processor.binarize_volume(roi, method='triangle')

print(f"Custom processing complete: {binary.shape}")
```

## Migration from Old Architecture

### Old Way (tracer.py)
```python
from VesselTracer import VesselTracer  # Old monolithic class

tracer = VesselTracer("image.czi")
tracer.run_analysis()
```

### New Way
```python
from VesselTracer import VesselAnalysisController

controller = VesselAnalysisController("image.czi")
controller.run_analysis()
```

## Benefits of New Architecture

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Modularity**: Components can be used independently for custom workflows
3. **Testability**: Individual components are easier to test in isolation
4. **Flexibility**: Mix and match components as needed
5. **Maintainability**: Cleaner interfaces and reduced complexity
6. **Extensibility**: Easy to add new processing methods or tracers

## Backward Compatibility

The old `VesselTracer` class (from `tracer.py`) is still available as `LegacyVesselTracer`:

```python
from VesselTracer import LegacyVesselTracer

# Use old interface
tracer = LegacyVesselTracer("image.czi")
tracer.run_analysis()
```

## File Structure

```
src/VesselTracer/
├── controller.py           # VesselAnalysisController
├── vessel_tracer.py        # VesselTracer (tracing only)
├── image_processor.py      # ImageProcessor
├── image_model.py          # Data models (ImageModel, ROI)
├── config.py              # Configuration management
├── plotting.py            # Visualization utilities
├── tracer.py              # Legacy VesselTracer (backward compatibility)
└── __init__.py            # Package imports
```

## Running the Demo

To see the new architecture in action:

```bash
python test/demo_new_architecture.py
```

This demonstrates:
- Complete pipeline with Controller
- Individual component usage
- Architecture comparison
- Flexibility examples

## Testing

Update your test files to use the new architecture:

```python
# Old
from VesselTracer import VesselTracer
tracer = VesselTracer(input_path)

# New
from VesselTracer import VesselAnalysisController
controller = VesselAnalysisController(input_path)
```

The plotting functions remain compatible and will work with the controller object. 