import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from VesselTracer import VesselTracer
from VesselTracer.plotting import plot_projections, plot_mean_zprofile, plot_path_projections
import numpy as np
import pandas as pd

def run_analysis(input_path, output_dir=None):
    # Setup input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = Path(f"test\\results\\{input_path.stem}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {input_path.name}...")
    print(f"Results will be saved to: {output_dir}")
    
    try:
        # Initialize VesselTracer
        tracer = VesselTracer(str(input_path))  # Using default config
        tracer.print_config()
        # Run analysis pipeline
        print("Running analysis pipeline...")
        tracer.segment_roi(remove_dead_frames=True, dead_frame_threshold=1.5)
        tracer.median_filter()
        tracer.background_smoothing()
        tracer.detrend()
        tracer.smooth()
        tracer.binarize()
        # tracer.skeletonize()
        
        # print("Detecting regions...")
        # regions = tracer.determine_regions()
        # for region, (peak, sigma, bounds) in regions.items():
        #     print(f"\n{region}:")
        #     print(f"  Peak position: {peak:.1f}")
        #     print(f"  Width (sigma): {sigma:.1f}")
        #     print(f"  Bounds: {bounds[0]:.1f} - {bounds[1]:.1f}")
        
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
        
        # Path projections
        if hasattr(tracer, 'paths'):
            fig5, axes5 = plot_path_projections(tracer)
            fig5.savefig(output_dir / 'path_projections.png', dpi=300, bbox_inches='tight')
            plt.close(fig5)
        
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
                    'Default',
                    True,
                    True,
                    True
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
            
            # Save vessel paths
            print("Extracting vessel paths...")
            if hasattr(tracer, 'paths'):
                # Get coordinates of vessel paths with their labels
                vessel_paths_df = pd.DataFrame()
                print(f"Paths: {tracer.paths}")
                # Process each path
                for path_id, path_coords in tracer.paths.items():
                    #do we even have any paths? 
                    print(f"Path {path_id} has {len(path_coords)} points")
                    # Convert path coordinates to DataFrame
                    path_df = pd.DataFrame({
                        'Path_ID': path_id,
                        'X': path_coords[:, 2],  # Note: CZI files are typically ZYX order
                        'Y': path_coords[:, 1],
                        'Z': path_coords[:, 0]
                    })
                    vessel_paths_df = pd.concat([vessel_paths_df, path_df], ignore_index=True)
                
                # Sort by Path_ID, then Z, Y, X for better organization
                vessel_paths_df = vessel_paths_df.sort_values(['Path_ID', 'Z', 'Y', 'X'])
                
                # Save to Excel
                vessel_paths_df.to_excel(writer, sheet_name='Vessel Paths', index=False)
                
                # Add summary statistics
                path_stats = pd.DataFrame({
                    'Metric': ['Total Points', 'Number of Paths', 'Unique X', 'Unique Y', 'Unique Z', 
                             'Min X', 'Max X', 'Min Y', 'Max Y', 'Min Z', 'Max Z'],
                    'Value': [
                        len(vessel_paths_df),
                        len(tracer.paths),
                        vessel_paths_df['X'].nunique(),
                        vessel_paths_df['Y'].nunique(),
                        vessel_paths_df['Z'].nunique(),
                        vessel_paths_df['X'].min(),
                        vessel_paths_df['X'].max(),
                        vessel_paths_df['Y'].min(),
                        vessel_paths_df['Y'].max(),
                        vessel_paths_df['Z'].min(),
                        vessel_paths_df['Z'].max()
                    ]
                })
                path_stats.to_excel(writer, sheet_name='Path Statistics', index=False)
                
                # Add per-path statistics
                path_details = []
                for path_id, path_coords in tracer.paths.items():
                    path_data = vessel_paths_df[vessel_paths_df['Path_ID'] == path_id]
                    path_details.append({
                        'Path_ID': path_id,
                        'Number_of_Points': len(path_data),
                        'Z_Range': path_data['Z'].max() - path_data['Z'].min(),
                        'Y_Range': path_data['Y'].max() - path_data['Y'].min(),
                        'X_Range': path_data['X'].max() - path_data['X'].min(),
                        'Min_Z': path_data['Z'].min(),
                        'Max_Z': path_data['Z'].max(),
                        'Min_Y': path_data['Y'].min(),
                        'Max_Y': path_data['Y'].max(),
                        'Min_X': path_data['X'].min(),
                        'Max_X': path_data['X'].max()
                    })
                
                path_details_df = pd.DataFrame(path_details)
                path_details_df.to_excel(writer, sheet_name='Path Details', index=False)
        
        print(f"Analysis complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error processing {input_path.name}: {str(e)}")
        # Save error log
        with open(output_dir / 'error_log.txt', 'w') as f:
            f.write(f"Error processing {input_path.name}:\n{str(e)}")
        raise

if __name__ == "__main__":
    print("Starting VesselTracer...")
    input_path = Path("C:\\Users\\Matt\\PythonDev\\VascularAnalysis\\test\\test_files\\240207_002.czi")  # Replace with your input file path
    run_analysis(input_path)