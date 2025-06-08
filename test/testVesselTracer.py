import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from VesselTracer import VesselTracer
from VesselTracer.plotting import plot_projections, plot_mean_zprofile, plot_path_projections
import numpy as np
import pandas as pd

def main(input_path, output_dir=None):
    # Setup input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize tracer
        print(f"Loading file: {input_path}")
        tracer = VesselTracer(input_path)
        
        # Run the complete pipeline
        print("Running analysis pipeline...")
        tracer.run_pipeline(
            output_dir=output_dir,
            save_original=True,
            save_smoothed=True,
            save_binary=True,
            save_skeleton=True,
            save_volumes=True,
            save_projections=True,
            save_regions=True,
            save_paths=True,
            skip_smoothing=False,
            skip_binarization=False,
            skip_regions=False,
            skip_trace=False,
            skip_dataframe=False
        )
        
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
    input_path = Path("C:\\Users\\Matt\\PythonDev\\VascularAnalysis\\test\\test_files\\240207_002.czi")  # Replace with your input file path
    main(input_path)