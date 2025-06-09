import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
from mpl_toolkits.mplot3d import Axes3D

def show_max_projection(vol: np.ndarray, ax: Optional[plt.Axes] = None) -> None:
    """Show maximum intensity projection of a volume.
    
    Args:
        vol: 3D numpy array
        ax: Optional matplotlib axes to plot on
    """
    if ax is None:
        ax = plt.gca()
    ax.imshow(np.max(vol, axis=0))
    ax.axis('off')

def plot_projections(tracer, figsize=(10, 10), mode: str = 'smoothed', depth_coded: bool = False) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Create a comprehensive plot showing different projections and intensity profile.
    
    Creates a figure with:
    - Top left: Z projection (xy view)
    - Top right: Y projection (xz view)
    - Bottom left: X projection (yz view)
    - Bottom right: Mean intensity profile
    
    Args:
        tracer: VesselTracer instance with loaded data
        figsize: Figure size tuple (width, height)
        mode: Visualization mode, either 'smoothed' or 'binary'
        depth_coded: If True, creates depth-coded projections where intensity
                    represents z-position (only works with binary mode)
        
    Returns:
        Tuple of (figure, dict of axes)
    """
    # Validate mode
    if mode not in ['smoothed', 'binary']:
        raise ValueError("Mode must be either 'smoothed' or 'binary'")
    
    # Ensure required data exists
    if mode == 'smoothed' and not hasattr(tracer, 'smoothed'):
        tracer.smooth()
    elif mode == 'binary' and not hasattr(tracer, 'binary'):
        tracer.binarize()
    
    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
    
    # Create axes
    ax_z = fig.add_subplot(gs[0, 0])  # Z projection (top left)
    ax_y = fig.add_subplot(gs[0, 1])  # Y projection (top right)
    ax_x = fig.add_subplot(gs[1, 0])  # X projection (bottom left)
    ax_profile = fig.add_subplot(gs[1, 1])  # Intensity profile (bottom right)
    
    # Get data based on mode
    data = tracer.smoothed if mode == 'smoothed' else tracer.binary
    
    if depth_coded and mode == 'binary':
        # Create depth-coded volume
        Z, Y, X = data.shape
        depth_vol = np.zeros_like(data, dtype=float)
        for z in range(Z):
            depth_vol[z] = data[z] * z
            
        # Create depth-normalized projections
        z_proj = np.max(depth_vol, axis=0)
        y_proj = np.max(depth_vol, axis=1)
        x_proj = np.max(depth_vol, axis=2)
        
        # Normalize depth projections to [0,1]
        z_proj = z_proj / (Z-1) if z_proj.max() > 0 else z_proj
        y_proj = y_proj / (Z-1) if y_proj.max() > 0 else y_proj
        x_proj = x_proj / (Z-1) if x_proj.max() > 0 else x_proj
        
        # Use a colormap that shows depth well
        cmap = plt.cm.viridis
    else:
        # Regular projections
        z_proj = np.max(data, axis=0)
        y_proj = np.max(data, axis=1)
        x_proj = np.max(data, axis=2)
        cmap = 'gray'
    
    # Plot Z projection (top left)
    im_z = ax_z.imshow(z_proj, cmap=cmap)
    title = f'Z proj ({mode})'
    if depth_coded and mode == 'binary':
        title += ' [depth-coded]'
        plt.colorbar(im_z, ax=ax_z, label='Z position (normalized)')
    ax_z.set_title(title)
    ax_z.axis('on')
    
    # Add scale bar (assuming we have pixel size)
    scalebar_length_pixels = int(50 / tracer.pixel_size_x)  # 50 micron scale bar
    ax_z.plot([20, 20 + scalebar_length_pixels], [z_proj.shape[0] - 20] * 2, 
              'w-' if cmap == 'gray' else 'k-', linewidth=2)
    
    # Plot Y projection (top right)
    im_y = ax_y.imshow(y_proj.T, cmap=cmap)  # Transpose y_proj to rotate 90 degrees
    ax_y.set_title('Y proj')
    ax_y.axis('on')
    if depth_coded and mode == 'binary':
        plt.colorbar(im_y, ax=ax_y, label='Z position (normalized)')
    
    # Plot X projection (bottom left)
    im_x = ax_x.imshow(x_proj, cmap=cmap)
    ax_x.axis('on')
    if depth_coded and mode == 'binary':
        plt.colorbar(im_x, ax=ax_x, label='Z position (normalized)')
    
    # Plot mean intensity profile (bottom right)
    mean_profile = np.mean(data, axis=(1,2))
    z_positions = np.arange(len(mean_profile))
    ax_profile.plot(mean_profile, z_positions, 'b-')
    ax_profile.set_ylim(ax_profile.get_ylim()[::-1])  # Invert y-axis
    ax_profile.set_xlabel('Intensity')
    
    # Adjust spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Return figure and axes dictionary
    axes = {
        'z_proj': ax_z,
        'y_proj': ax_y,
        'x_proj': ax_x,
        'profile': ax_profile
    }
    
    return fig, axes

def plot_paths_on_axis(tracer, ax, projection='xy', region_colorcode: bool = False, is_3d: bool = False) -> None:
    """Plot vessel paths on a given axis.
    
    Args:
        tracer: VesselTracer instance with traced paths
        ax: Matplotlib axis to plot on
        projection: Projection plane ('xy', 'xz', or 'zy') for 2D plots
        region_colorcode: If True, color-code paths based on their region
        is_3d: If True, creates a 3D plot instead of a 2D projection
    """
    if not hasattr(tracer, 'paths'):
        raise ValueError("No paths found. Run trace_paths() first.")
    
    # Define colors for regions
    region_colors = {
        'superficial': 'tab:purple',
        'intermediate': 'tab:red',
        'deep': 'tab:blue',
        'diving': 'magenta'
    }
    
    # Plot each path
    for path_id in range(1, tracer.n_paths):
        path = tracer.paths[path_id]
        path_coords = path['coordinates']
        if len(path_coords) > 0:
            # Extract x, y, z coordinates
            z_coords = path_coords[:, 0]  # z is first coordinate
            y_coords = path_coords[:, 1]  # y is second coordinate
            x_coords = path_coords[:, 2]  # x is third coordinate
            
            if is_3d:
                # For 3D plots, we always plot all coordinates
                plot_x = x_coords
                plot_y = y_coords
                plot_z = z_coords
            else:
                # For 2D plots, handle different projections
                if projection == 'xy':
                    plot_x = x_coords
                    plot_y = y_coords
                elif projection == 'xz':
                    plot_x = z_coords
                    plot_y = x_coords
                elif projection == 'zy':
                    plot_x = y_coords
                    plot_y = z_coords
                else:
                    raise ValueError(f"Invalid projection '{projection}'. Must be one of: ['xy', 'xz', 'zy']")
            
            if region_colorcode and hasattr(tracer, 'region_bounds'):
                # Get the region for each point in the path
                regions = [tracer.get_region_for_z(z) for z in z_coords]
                unique_regions = np.unique(regions)

                if len(unique_regions) > 1 or unique_regions[0] == 'unknown':
                    # Plot as diving vessel if path crosses multiple regions
                    color = region_colors['diving']
                else:
                    # Plot in the color of its single region
                    region = unique_regions[0]
                    color = region_colors[region]
                
                if is_3d:
                    ax.plot(plot_x, plot_y, plot_z, color=color, linewidth=1, alpha=0.7)
                else:
                    ax.plot(plot_x, plot_y, color=color, linewidth=1, alpha=0.7)
            else:
                # Plot entire path in red
                if is_3d:
                    ax.plot(plot_x, plot_y, plot_z, 'r-', linewidth=1, alpha=0.7)
                else:
                    ax.plot(plot_x, plot_y, 'r-', linewidth=1, alpha=0.7)

def plot_paths(tracer, figsize=(15, 7), region_colorcode: bool = False, projection: str = 'xy') -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Plot vessel paths in both 2D and 3D projections.
    
    Args:
        tracer: VesselTracer instance with traced paths
        figsize: Figure size (width, height) in inches
        region_colorcode: If True, color-code paths based on their region
        projection: Projection plane for 2D view ('xy', 'xz', or 'zy')
        
    Returns:
        Tuple of (figure, dictionary of axes)
    """
    if not hasattr(tracer, 'paths'):
        raise ValueError("No paths found. Run trace_paths() first.")
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    ax_2d = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection='3d')
    
    # Plot paths on both axes using the same function
    plot_paths_on_axis(tracer, ax_2d, projection=projection, region_colorcode=region_colorcode, is_3d=False)
    plot_paths_on_axis(tracer, ax_3d, region_colorcode=region_colorcode, is_3d=True)
    
    # Set labels and titles
    ax_2d.set_xlabel('X')
    ax_2d.set_ylabel('Y')
    ax_2d.set_title(f'2D {projection.upper()} Projection')
    
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D View')
    
    # Adjust 3D view for better visualization
    ax_3d.view_init(elev=20, azim=45)
    
    # Add legend if using region colorcoding
    if region_colorcode and hasattr(tracer, 'region_bounds'):
        region_colors = {
            'superficial': 'tab:purple',
            'intermediate': 'tab:red',
            'deep': 'tab:blue',
            'diving': 'magenta'
        }
        handles = [plt.Line2D([0], [0], color=color, label=region) 
                  for region, color in region_colors.items()]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.05),
                  ncol=4, title='Regions')
    
    plt.tight_layout()
    return fig, {'2d': ax_2d, '3d': ax_3d}

def plot_projections_w_paths(tracer, figsize=(10, 10), mode: str = 'smoothed', depth_coded: bool = False, region_colorcode: bool = False) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Create a comprehensive plot showing different projections with vessel paths.
    
    Creates a figure with:
    - Top left: Z projection (xy view) with paths
    - Top right: Y projection (xz view) with paths
    - Bottom left: X projection (yz view) with paths
    - Bottom right: Mean intensity profile
    
    Args:
        tracer: VesselTracer instance with loaded data
        figsize: Figure size tuple (width, height)
        mode: Visualization mode, either 'smoothed' or 'binary'
        depth_coded: If True, creates depth-coded projections where intensity
                    represents z-position (only works with binary mode)
        region_colorcode: If True, color-code paths based on their region
        
    Returns:
        Tuple of (figure, dict of axes)
    """
    # First create the base projections plot
    fig, axes = plot_projections(tracer, figsize=figsize, mode=mode, depth_coded=depth_coded)
    
    # Add paths to each projection
    if hasattr(tracer, 'paths'):
        # Map projection names to their corresponding views
        projection_map = {
            'z_proj': 'xy',  # Z projection shows xy view
            'y_proj': 'xz',  # Y projection shows xz view
            'x_proj': 'zy'   # X projection shows zy view
        }
        
        # Plot paths on each projection
        for ax_name, projection in projection_map.items():
            plot_paths_on_axis(tracer, axes[ax_name], projection=projection, 
                             region_colorcode=region_colorcode)
    
    return fig, axes

def plot_regions(tracer, figsize=(8, 4)) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Plot the mean z-profile with detected regions alongside y-projection.
    
    Args:
        tracer: VesselTracer instance with loaded data
        figsize: Figure size tuple (width, height)
        
    Returns:
        Tuple of (figure, dict of axes)
    """
    # Get mean z-profile and y-projection
    mean_zprofile = tracer.get_projection([1, 2], operation='mean')
    y_proj = np.max(tracer.smoothed if hasattr(tracer, 'smoothed') else tracer.roi_volume, axis=2)
    
    # Determine regions if not already done
    if not hasattr(tracer, 'region_bounds'):
        tracer.determine_regions()
    
    # Create figure with two subplots
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, 
                                  gridspec_kw={'width_ratios': [3, 1]})
    
    # Plot y-projection
    ax0.imshow(y_proj, cmap='gray', aspect='auto')
    ax0.set_title('Y Projection')
    
    # Plot mean z-profile
    Z = len(mean_zprofile)
    z_positions = np.arange(Z)
    ax1.plot(mean_zprofile, z_positions, color='black')
    
    # Define colors for layers
    layer_colors = ['tab:purple', 'tab:red', 'tab:blue']
    
    # Plot regions
    for i, (region, (peak, sigma, bounds)) in enumerate(tracer.region_bounds.items()):
        # Add horizontal lines at peaks
        ax0.axhline(peak, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(peak, color='red', linestyle='--', alpha=0.5)
        
        # Add spans for regions
        ax0.axhspan(bounds[0], bounds[1], color=layer_colors[i], alpha=0.25, label=region)
        ax1.axhspan(bounds[0], bounds[1], color=layer_colors[i], alpha=0.5, label=region)
    
    # Customize ax1 (z-profile)
    ax1.invert_yaxis()
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('')  # Remove y-label since it's shared
    
    # Add legend to the first axis
    ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Return figure and axes dictionary
    axes = {
        'y_proj': ax0,
        'z_profile': ax1
    }
    
    return fig, axes
