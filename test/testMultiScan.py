import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from VesselTracer import VesselAnalysisController
from VesselTracer.plotting import plot_projections, plot_regions, plot_paths, plot_projections_w_paths
import numpy as np
import pandas as pd


def main(input_path, config_path, output_dir=None):
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
        # Initialize controller
        print(f"Loading file: {input_path}")
        controller = VesselAnalysisController(input_path, config_path)
        #controller.activate_gpu()
        
        # Run the complete pipeline using the controller
        print("Running multiscan analysis pipeline...")
        controller.multiscan(
            skip_smoothing=False,
            skip_binarization=False,
            skip_regions=False,
            skip_trace=False,
        )
        
        # Generate plots using the controller's projection methods
        print("Plotting projections...")
        fig1, ax1 = plot_projections(controller, mode="volume", source='image')
        fig1.savefig(output_dir / "volume_projections.png")

        fig2, ax2 = plot_projections(controller, mode='volume', source='image')
        fig2.savefig(output_dir / "roi_projections.png")

        fig2b, ax2b = plot_projections(controller, mode='region', source='image')
        fig2b.savefig(output_dir / "region_projections.png")

        fig2a, ax2a = plot_projections(controller, mode='background', source='image')
        fig2a.savefig(output_dir / "background_projections.png")

        fig2c, ax2c = plot_projections(controller, mode='binary', source='image', depth_coded=True)
        fig2c.savefig(output_dir / "binary_projections.png")

        print("Plotting regions...")
        fig3, ax3 = plot_regions(controller)
        fig3.savefig(output_dir / "regions.png")
        
        print("Plotting paths...")
        fig4, ax4 = plot_paths(controller, region_colorcode=True)
        fig4.savefig(output_dir / "paths.png")
        
        print("Plotting projections with paths...")
        fig5, ax5 = plot_projections_w_paths(controller, region_colorcode=True)
        fig5.savefig(output_dir / "projections_w_paths.png")
        
        plt.close('all')

        # Test saving volumes
        print("\nTesting volume saving...")
        volumes_dir = output_dir / "volumes"
        volumes_dir.mkdir(exist_ok=True)
        
        # Save each type of volume
        controller.save_volume(volumes_dir, 'volume', source='image')
        controller.save_volume(volumes_dir, 'binary', source='image')
        controller.save_volume(volumes_dir, 'binary', source='image', depth_coded=True)
        controller.save_volume(volumes_dir, 'background', source='image')
        controller.save_volume(volumes_dir, 'region', source='image')

        # Save paths data
        print("\nSaving paths data...")
        paths_dir = output_dir / "paths"
        paths_dir.mkdir(exist_ok=True)
        
        # Get pixel sizes from image model
        pixel_sizes = {
            'x': controller.image_model.pixel_size_x,
            'y': controller.image_model.pixel_size_y,
            'z': controller.image_model.pixel_size_z
        }
        
        # Save detailed paths data
        paths_df = controller.create_paths_dataframe(pixel_sizes, source='image')
        paths_df.to_excel(paths_dir / "detailed_paths.xlsx", index=False)
        
        # Save path summary data
        path_summary_df = controller.create_path_summary_dataframe(pixel_sizes, source='image')
        path_summary_df.to_excel(paths_dir / "path_summary.xlsx", index=False)

        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        
        # Print some basic statistics
        print("\nAnalysis Summary:")
        print("----------------")
        print(f"Volume Shape: {controller.image_model.volume.shape}")
        if hasattr(controller, 'regions_df') and not controller.regions_df.empty:
            print("\nRegion Analysis:")
            print(controller.regions_df)
        if len(paths_df) > 0:
            print(f"\nNumber of Vessel Paths: {len(paths_df['Path_ID'].unique())}")
            print(f"Total Path Points: {len(paths_df)}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting VesselTracer MultiScan...")
    input_path = Path("C:\\Users\\mtarc\\PythonScripts\\VascularAnalysis\\test\\input\\240207_002.czi")  # Replace with your input file path
    config_path = Path("C:\\Users\\mtarc\\PythonScripts\\VascularAnalysis\\config\\default_vessel_config.yaml")
    # config_path = Path("C:\\Users\\Matt\\PythonDev\\VascularAnalysis\\config\\default_vessel_config.yaml")
    # input_path = Path("F:\\240207_002 (1).czi")
    main(input_path, config_path) 