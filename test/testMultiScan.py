import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from VesselTracer import VesselTracer
from VesselTracer.plotting import plot_projections, plot_regions, plot_paths, plot_projections_w_paths
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
        tracer.multiscan(
            skip_smoothing=False,
            skip_binarization=False,
            skip_regions=False,
            skip_trace=False
        )
        print(f"Completed multiscan analysis of {len(tracer.roi_results)} ROIs")
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
    print("Starting MultiScan...")
    input_path = Path("C:\\Users\\mtarc\\PythonScripts\\VascularAnalysis\\test\\input\\240207_002.czi")  # Replace with your input file path
    #input_path = Path("F:\\240207_002 (1).czi")
    main(input_path) 