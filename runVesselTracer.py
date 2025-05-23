import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from VesselTracer import VesselTracer
from VesselTracer.plotting import plot_projections, plot_mean_zprofile
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
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results_{input_path.stem}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {input_path.name}...")
    print(f"Results will be saved to: {output_dir}")
    
    try:
        # Initialize VesselTracer
        tracer = VesselTracer(str(input_path), config_path=args.config)
        
        # Run analysis pipeline
        print("Running analysis pipeline...")
        tracer.segment_roi(remove_dead_frames=True, dead_frame_threshold=1.5)
        
        if not args.skip_smoothing:
            print("Smoothing...")
            tracer.smooth()
        
        if not args.skip_binarization:
            print("Binarizing...")
            tracer.binarize()
        
        if not args.skip_regions:
            print("Detecting regions...")
            tracer.determine_regions()
        
        # Save visualizations
        print("Generating visualizations...")
        
        # Projections
        fig1, axes1 = plot_projections(tracer, mode='smoothed')
        fig1.savefig(output_dir / 'projections_smoothed.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        fig2, axes2 = plot_projections(tracer, mode='binary')
        fig2.savefig(output_dir / 'projections_binary.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        fig3, axes3 = plot_projections(tracer, mode='binary', depth_coded=True)
        fig3.savefig(output_dir / 'projections_depth.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # Z-profile
        fig4, axes4 = plot_mean_zprofile(tracer)
        fig4.savefig(output_dir / 'vessel_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        # Save analysis results
        print("Saving analysis results...")
        
        # Create Excel writer
        excel_path = output_dir / 'analysis_results.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Save metadata
            metadata = pd.DataFrame({
                'Parameter': ['Input File', 'Timestamp', 'Config Used', 
                            'Smoothing Applied', 'Binarization Applied', 
                            'Region Detection Applied'],
                'Value': [
                    str(input_path),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    str(args.config) if args.config else 'Default',
                    not args.skip_smoothing,
                    not args.skip_binarization,
                    not args.skip_regions
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Save region bounds if available
            if hasattr(tracer, 'region_bounds'):
                region_data = []
                for region, (peak, sigma, bounds) in tracer.region_bounds.items():
                    region_data.append({
                        'Region': region,
                        'Peak Position': peak,
                        'Sigma': sigma,
                        'Lower Bound': bounds[0],
                        'Upper Bound': bounds[1]
                    })
                regions_df = pd.DataFrame(region_data)
                regions_df.to_excel(writer, sheet_name='Region Analysis', index=False)
            
            # Save mean z-profile if available
            if hasattr(tracer, 'get_projection'):
                z_profile = tracer.get_projection([1, 2], operation='mean')
                z_profile_df = pd.DataFrame({
                    'Z Position': np.arange(len(z_profile)),
                    'Mean Intensity': z_profile
                })
                z_profile_df.to_excel(writer, sheet_name='Z Profile', index=False)
        
        print(f"Analysis complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error processing {input_path.name}: {str(e)}")
        # Save error log
        with open(output_dir / 'error_log.txt', 'w') as f:
            f.write(f"Error processing {input_path.name}:\n{str(e)}")
        raise

def main():
    args = parse_args()
    run_analysis(args)

if __name__ == "__main__":
    print("Starting VesselTracer...")
    main()