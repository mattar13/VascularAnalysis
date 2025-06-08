#!/usr/bin/env python3
"""
Run VesselTracer on input files.

This script provides a command-line interface to run the VesselTracer
analysis pipeline on one or more input files.

Usage:
    python runVesselTracer.py [options] <input_file> [<input_file> ...]

Options:
    -o, --output-dir DIR    Output directory for results
    -c, --config FILE       Path to config file
    --skip-smoothing        Skip smoothing step
    --skip-binarization     Skip binarization step
    --skip-regions          Skip region detection
    --skip-trace           Skip vessel tracing
    --skip-dataframe       Skip generating DataFrames
    --save-original        Save original volume
    --save-smoothed        Save smoothed volume
    --save-binary          Save binary volume
    --save-skeleton        Save skeleton volume
    --save-volumes         Save all volumes
    --save-projections     Save projections
    --save-regions         Save region analysis
    --save-paths           Save vessel paths
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from VesselTracer import VesselTracer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run VesselTracer analysis pipeline")
    
    # Input/output arguments
    parser.add_argument('input_files', nargs='+', help='Input file(s) to process')
    parser.add_argument('-o', '--output-dir', help='Output directory for results')
    parser.add_argument('-c', '--config', help='Path to config file')
    
    # Skip options
    parser.add_argument('--skip-smoothing', action='store_true', help='Skip smoothing step')
    parser.add_argument('--skip-binarization', action='store_true', help='Skip binarization step')
    parser.add_argument('--skip-regions', action='store_true', help='Skip region detection')
    parser.add_argument('--skip-trace', action='store_true', help='Skip vessel tracing')
    parser.add_argument('--skip-dataframe', action='store_true', help='Skip generating DataFrames')
    
    # Save options
    parser.add_argument('--save-original', action='store_true', help='Save original volume')
    parser.add_argument('--save-smoothed', action='store_true', help='Save smoothed volume')
    parser.add_argument('--save-binary', action='store_true', help='Save binary volume')
    parser.add_argument('--save-skeleton', action='store_true', help='Save skeleton volume')
    parser.add_argument('--save-volumes', action='store_true', help='Save all volumes')
    parser.add_argument('--save-projections', action='store_true', help='Save projections')
    parser.add_argument('--save-regions', action='store_true', help='Save region analysis')
    parser.add_argument('--save-paths', action='store_true', help='Save vessel paths')
    
    return parser.parse_args()

def main(input_file: Path, args):
    """Process a single input file."""
    print(f"\nProcessing {input_file.name}...")
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir) / input_file.stem
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results_{input_file.stem}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize tracer
        tracer = VesselTracer(str(input_file), config_path=args.config)
        
        # Run pipeline
        tracer.run_pipeline(
            output_dir=output_dir,
            save_original=args.save_original or args.save_volumes,
            save_smoothed=args.save_smoothed or args.save_volumes,
            save_binary=args.save_binary or args.save_volumes,
            save_skeleton=args.save_skeleton or args.save_volumes,
            save_volumes=args.save_volumes,
            save_projections=args.save_projections,
            save_regions=args.save_regions,
            save_paths=args.save_paths,
            skip_smoothing=args.skip_smoothing,
            skip_binarization=args.skip_binarization,
            skip_regions=args.skip_regions,
            skip_trace=args.skip_trace,
            skip_dataframe=args.skip_dataframe
        )
        
        print(f"Analysis complete! Results saved to: {output_dir}")
        
        # Print summary if DataFrames were generated
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
        print(f"Error processing {input_file.name}: {str(e)}")
        # Save error log
        with open(output_dir / 'error_log.txt', 'w') as f:
            f.write(f"Error processing {input_file.name}:\n{str(e)}")
        raise

if __name__ == "__main__":
    args = parse_args()
    
    print("Starting VesselTracer...")
    print(f"Processing {len(args.input_files)} file(s)")
    main(args.input_files, args)