import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from VesselTracer import VesselTracer
from VesselTracer.plotting import plot_projections, plot_regions, plot_paths, plot_projections_w_paths, plot_vertical_region_map
import numpy as np
import pandas as pd

def main(input_path, output_dir=None):
    # Setup input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    # Create output directory name based on input file name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = input_path.stem
    output_dir = Path(f'test/output/from{file_stem}_on{timestamp}')
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize tracer
        print(f"Loading file: {input_path}")
        tracer = VesselTracer(input_path)
        tracer.activate_gpu()
        # Run the complete pipeline
        print("Running analysis pipeline...")
        tracer.run_analysis(
            output_dir=output_dir,
            skip_smoothing=False,
            skip_binarization=False,
            skip_regions=False,
            skip_trace=False,
            skip_dataframe=False,

            #Save options
            save_volumes=True,
            save_original=True,
            save_binary=True,
            save_separate=True,
        )
        fig1, ax1 = plot_projections(tracer, mode='roi')
        fig1.savefig(output_dir / "roi_projections.png")

        fig2, ax2 = plot_projections(tracer, mode='volume')
        fig2.savefig(output_dir / "volume_projections.png")
        
        fig2b, ax2b = plot_projections(tracer, mode='region')
        fig2b.savefig(output_dir / "region_projections.png")

        fig3, ax3 = plot_regions(tracer)
        fig3.savefig(output_dir / "regions.png")
        
        fig4, ax4 = plot_paths(tracer, region_colorcode=True)
        fig4.savefig(output_dir / "paths.png")
        
        fig5, ax5 = plot_projections_w_paths(tracer, region_colorcode=True)
        fig5.savefig(output_dir / "projections_w_paths.png")
        
        fig6, ax6 = plot_vertical_region_map(tracer, projection='xz', show_vessels=True)
        fig6.savefig(output_dir / "vertical_region_map_xz.png")
        
        fig7, ax7 = plot_vertical_region_map(tracer, projection='yz', show_vessels=True)
        fig7.savefig(output_dir / "vertical_region_map_yz.png")
        
        plt.close('all')

        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        
        # Print some basic statistics
        if hasattr(tracer, 'analysis_dfs'):
            print("\nAnalysis Summary:")
            print("----------------")
            print(f"Volume Shape: {tracer.volume.shape}")
            if not tracer.regions_df.empty:
                print("\nRegion Analysis:")
                print(tracer.regions_df)
            if not tracer.paths_df.empty:
                print(f"\nNumber of Vessel Paths: {len(tracer.paths_df['Path_ID'].unique())}")
                print(f"Total Path Points: {len(tracer.paths_df)}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting VesselTracer...")
    #input_path = Path("C:\\Users\\mtarc\\PythonScripts\\VascularAnalysis\\test\\input\\240207_002.czi")  # Replace with your input file path
    input_path = Path("F:\\240207_002 (1).czi")
    main(input_path)