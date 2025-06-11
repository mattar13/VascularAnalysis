import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from VesselTracer import VesselAnalysisController
from VesselTracer.plotting import plot_projections, plot_regions, plot_paths, plot_projections_w_paths
import numpy as np
import pandas as pd

def main(input_path, output_dir=None):
    # Setup input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    # Create output directory name based on input file name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    file_stem = input_path.stem
    output_dir = Path(f'test/output/from{file_stem}_on{timestamp}')
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize controller (replaces the old VesselTracer)
        print(f"Loading file: {input_path}")
        controller = VesselAnalysisController(input_path)
        #controller.activate_gpu()
        
        # Run the complete pipeline using the controller
        print("Running analysis pipeline...")
        controller.run_analysis(
            skip_smoothing=False,
            skip_binarization=False,
            skip_regions=False,
            skip_trace=False,
        )
        
        # Generate plots using the controller's projection methods
        fig1, ax1 = plot_projections(controller, mode='volume')
        fig1.savefig(output_dir / "volume_projections.png")

        fig2, ax2 = plot_projections(controller, mode='roi')
        fig2.savefig(output_dir / "roi_projections.png")

        fig2b, ax2b = plot_projections(controller, mode='region')
        fig2b.savefig(output_dir / "region_projections.png")

        fig2a, ax2a = plot_projections(controller, mode='background')
        fig2a.savefig(output_dir / "background_projections.png")

        fig2c, ax2c = plot_projections(controller, mode='binary')
        fig2c.savefig(output_dir / "binary_projections.png")

        fig3, ax3 = plot_regions(controller)
        fig3.savefig(output_dir / "regions.png")
        
        fig4, ax4 = plot_paths(controller, region_colorcode=True)
        fig4.savefig(output_dir / "paths.png")
        
        fig5, ax5 = plot_projections_w_paths(controller, region_colorcode=True)
        fig5.savefig(output_dir / "projections_w_paths.png")
        
        plt.close('all')

        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        
        # Print some basic statistics
        if hasattr(controller, 'analysis_dfs'):
            print("\nAnalysis Summary:")
            print("----------------")
            print(f"Volume Shape: {controller.volume.shape}")
            if hasattr(controller, 'regions_df') and not controller.regions_df.empty:
                print("\nRegion Analysis:")
                print(controller.regions_df)
            if controller.path_count > 0:
                print(f"\nNumber of Vessel Paths: {controller.path_count}")
                print(f"Total Path Points: {sum(len(path_info['coordinates']) for path_info in controller.paths.values())}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting VesselTracer...")
    #input_path = Path("C:\\Users\\mtarc\\PythonScripts\\VascularAnalysis\\test\\input\\240207_002.czi")  # Replace with your input file path
    input_path = Path("F:\\240207_002 (1).czi")
    main(input_path)