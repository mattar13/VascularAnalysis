# VesselTracer

A Python package for automated analysis of 3D vascular networks from microscopy images.

## Overview

VesselTracer is a tool designed to automate the analysis of vascular networks in 3D microscopy images. It provides functionality for vessel segmentation, layer detection, and quantitative analysis of vascular structures across different tissue layers.

## Features

- **Automated Vessel Segmentation**: Process 3D microscopy images to identify vascular structures
- **Layer Detection**: Automatically detect and analyze different tissue layers
- **3D Visualization**: View projections with depth coding and layer highlighting
- **Quantitative Analysis**: Measure vessel distribution and characteristics across layers
- **Configurable Pipeline**: Customize analysis parameters through YAML configuration files
- **CZI Support**: Native support for Zeiss CZI microscopy file format

## Usage

```bash
python runVesselTracing.py input_file.czi
```

### Required arguments:
- `input_file`: Path to the input CZI file containing 3D microscopy data

### Example:
```bash
python runVesselTracing.py data/sample_vessels.czi
```

### What the script does:
1. Loads and preprocesses the 3D microscopy data
2. Detects and traces vessel paths
3. Classifies vessels into layers (superficial, intermediate, deep)
4. Generates visualizations including:
   - XY, XZ, and ZY projections with depth coding
   - 3D view of vessel paths
   - Layer analysis plots

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/VesselTracer.git
cd VesselTracer

# Install the package
pip install -e .
```

## Quick Start

```python
from VesselTracer import VesselTracer
from pathlib import Path

# Initialize with a CZI file
tracer = VesselTracer("path/to/your/image.czi")

# Run the analysis pipeline
tracer.segment_roi()
tracer.smooth()
tracer.binarize()
tracer.determine_regions()

# Visualize results
tracer.plot_projections(mode='binary', depth_coded=True)
tracer.plot_mean_zprofile()
```

## Configuration

VesselTracer uses YAML configuration files to customize the analysis pipeline. Example configuration:

```yaml
segmentation:
  threshold: 0.5
  min_size: 100

smoothing:
  sigma: 1.0

region_detection:
  min_peak_height: 0.1
  min_peak_distance: 10
```

## Documentation

For detailed documentation, please refer to the following sections:

- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Configuration Guide](docs/configuration.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](license.txt) file for details.

## Acknowledgments

- Developed at UC Berkeley
- Built with Python, NumPy, and Matplotlib
- Inspired by the need for automated vascular network analysis in neuroscience research
