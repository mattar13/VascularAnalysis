import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from VesselTracer.tracer import VesselTracer
from VesselTracer.plotting import show_max_projection, plot_mean_zprofile, plot_projections

def main():
    # Initialize VesselTracer with CZI file
    czi_path = Path("test/test_files/240207_002.czi")
    tracer = VesselTracer(str(czi_path), config_path=Path("config/default_vessel_config.yaml"))

    # Print configuration
    tracer.print_config()

    # Run analysis pipeline with dead frame removal
    print("\nRunning analysis pipeline...")
    tracer.segment_roi(remove_dead_frames=True, dead_frame_threshold=1.5)
    tracer.detrend()
    tracer.smooth()
    tracer.binarize()
    
    # Create smoothed projection plot
    fig1, axes1 = plot_projections(tracer, mode='smoothed')
    plt.savefig('test_images/vessel_projections_smoothed.png', dpi=300, bbox_inches='tight')
    
    # Create binary projection plot
    fig2, axes2 = plot_projections(tracer, mode='binary')
    plt.savefig('test_images/vessel_projections_binary.png', dpi=300, bbox_inches='tight')
    
    # Create z-profile plot with regions
    fig3, ax3 = plot_mean_zprofile(tracer)
    plt.savefig('test_images/vessel_distribution.png', dpi=300, bbox_inches='tight')
    
    # Print frame range information
    print("\nValid frame range:")
    print(f"  Start frame: {tracer.valid_frame_range[0]}")
    print(f"  End frame: {tracer.valid_frame_range[1]}")
    print(f"  Total frames: {tracer.valid_frame_range[1] - tracer.valid_frame_range[0] + 1}")
    
    plt.show()

    # Get region information
    regions = tracer.determine_regions()
    print("\nRegion Information:")
    for region, (peak, sigma, bounds) in regions.items():
        print(f"\n{region}:")
        print(f"  Peak position: {peak:.1f}")
        print(f"  Width (sigma): {sigma:.1f}")
        print(f"  Bounds: {bounds[0]:.1f} - {bounds[1]:.1f}")

if __name__ == "__main__":
    main()