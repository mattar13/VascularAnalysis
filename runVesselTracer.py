import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from VesselTracer import VesselTracer
from VesselTracer.plotting import plot_projections, plot_mean_zprofile, plot_path_projections
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Run VesselTracer analysis on a CZI file')
    
    # Required arguments
    parser.add_argument('input_file', type=str, help='Path to the input CZI file')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory to save results (default: creates timestamped directory)')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to YAML configuration file')
    parser.add_argument('--skip-smoothing', action='store_true',
                      help='Skip the smoothing step')
    parser.add_argument('--skip-binarization', action='store_true',
                      help='Skip the binarization step')
    parser.add_argument('--skip-regions', action='store_true',
                      help='Skip the region detection step')
    
    return parser.parse_args()

def run_analysis(args):
    # Setup input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Processing {input_path.name}...")
    
    try:
        # Initialize VesselTracer
        tracer = VesselTracer(str(input_path), config_path=args.config)
        tracer.print_config()
        
        # Run analysis pipeline
        tracer.run_pipeline(
            output_dir=args.output_dir,
            skip_smoothing=args.skip_smoothing,
            skip_binarization=args.skip_binarization,
            skip_regions=args.skip_regions
        )
        
    except Exception as e:
        print(f"Error processing {input_path.name}: {str(e)}")
        raise

def main():
    args = parse_args()
    run_analysis(args)

if __name__ == "__main__":
    print("Starting VesselTracer...")
    main()