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
        mode: Visualization mode. Options:
            - 'smoothed': Show smoothed volume
            - 'binary': Show binary volume  
            - 'background': Show background volume from median filtering
            - 'volume': Show current processed volume
        depth_coded: If True, creates depth-coded projections where intensity
                    represents z-position (only works with binary mode)
        
    Returns:
        Tuple of (figure, dict of axes)
    """
    # Validate mode
    valid_modes = ['smoothed', 'binary', 'background', 'volume']
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")
    
    # Ensure required data exists and get data based on mode
    if mode == 'smoothed':
        if not hasattr(tracer, 'smoothed'):
            tracer.smooth()
        data = tracer.smoothed
    elif mode == 'binary':
        if not hasattr(tracer, 'binary'):
            tracer.binarize()
        data = tracer.binary
    elif mode == 'background':
        if not hasattr(tracer, 'background'):
            raise ValueError("Background volume not available. Run median_filter() first.")
        data = tracer.background
    elif mode == 'volume':
        data = tracer.volume
    
    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
    
    # Create axes
    ax_z = fig.add_subplot(gs[0, 0])  # Z projection (top left)
    ax_y = fig.add_subplot(gs[0, 1])  # Y projection (top right)
    ax_x = fig.add_subplot(gs[1, 0])  # X projection (bottom left)
    ax_profile = fig.add_subplot(gs[1, 1])  # Intensity profile (bottom right)
    
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
        mode: Visualization mode. Options:
            - 'smoothed': Show smoothed volume
            - 'binary': Show binary volume  
            - 'background': Show background volume from median filtering
            - 'volume': Show current processed volume
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

def plot_vertical_region_map(tracer, figsize=(12, 8), projection: str = 'xz', 
                           show_vessels: bool = True, show_boundaries: bool = True) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Plot the vertical region map showing how regions are distributed in depth.
    
    Creates a visualization showing:
    - Left: Region map projection with color-coded regions
    - Right: Mean intensity profile with region boundaries
    
    Args:
        tracer: VesselTracer instance with region map
        figsize: Figure size tuple (width, height)
        projection: Projection plane for region map ('xz' or 'yz')
        show_vessels: If True, overlay vessel paths on the region map
        show_boundaries: If True, show region boundary lines
        
    Returns:
        Tuple of (figure, dict of axes)
    """
    # Ensure region map exists
    if not hasattr(tracer, 'region_map'):
        if not hasattr(tracer, 'region_bounds'):
            tracer.determine_regions()
        tracer.create_region_map_volume()
    
    # Create figure with two subplots
    fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=figsize, 
                                           gridspec_kw={'width_ratios': [3, 1]})
    
    # Create region map projection
    if projection == 'xz':
        # Show X-Z view (side view through Y)
        region_proj = np.max(tracer.region_map, axis=1)  # Max projection along Y axis
        vessel_data = tracer.get_projection(axis=1, operation='max', volume_type='binary') if show_vessels else None
        x_label = 'X (pixels)'
        y_label = 'Z (depth)'
    elif projection == 'yz':
        # Show Y-Z view (side view through X)
        region_proj = np.max(tracer.region_map, axis=2)  # Max projection along X axis
        vessel_data = tracer.get_projection(axis=2, operation='max', volume_type='binary') if show_vessels else None
        x_label = 'Y (pixels)'
        y_label = 'Z (depth)'
    else:
        raise ValueError("projection must be 'xz' or 'yz'")
    
    # Define region colors and labels
    region_colors = {
        0: {'color': [0.9, 0.9, 0.9], 'name': 'Unknown'},      # Light gray
        1: {'color': [0.8, 0.2, 0.8], 'name': 'Superficial'},  # Purple  
        2: {'color': [0.8, 0.2, 0.2], 'name': 'Intermediate'}, # Red
        3: {'color': [0.2, 0.2, 0.8], 'name': 'Deep'}          # Blue
    }
    
    # Create colored region map
    colored_map = np.zeros((*region_proj.shape, 3))
    for region_num, info in region_colors.items():
        mask = region_proj == region_num
        colored_map[mask] = info['color']
    
    # Plot region map
    ax_map.imshow(colored_map, aspect='auto', origin='upper', extent=[
        0, region_proj.shape[1], region_proj.shape[0], 0
    ])
    
    # Overlay vessels if requested
    if show_vessels and vessel_data is not None:
        # Create vessel overlay (white where vessels exist)
        vessel_overlay = np.zeros((*vessel_data.shape, 4))  # RGBA
        vessel_mask = vessel_data > 0
        vessel_overlay[vessel_mask] = [1, 1, 1, 0.7]  # White with transparency
        
        ax_map.imshow(vessel_overlay, aspect='auto', origin='upper', extent=[
            0, vessel_data.shape[1], vessel_data.shape[0], 0
        ])
    
    # Add region boundaries if requested
    if show_boundaries and hasattr(tracer, 'region_bounds'):
        for region_name, (peak, sigma, bounds) in tracer.region_bounds.items():
            # Draw horizontal lines at region boundaries
            ax_map.axhline(peak, color='white', linestyle='--', linewidth=2, alpha=0.8)
            ax_map.axhline(bounds[0], color='white', linestyle=':', linewidth=1, alpha=0.6)
            ax_map.axhline(bounds[1], color='white', linestyle=':', linewidth=1, alpha=0.6)
    
    ax_map.set_xlabel(x_label)
    ax_map.set_ylabel(y_label)
    ax_map.set_title(f'Vertical Region Map ({projection.upper()} view)')
    ax_map.invert_yaxis()  # Z increases downward
    
    # Create legend for regions
    import matplotlib.patches as patches
    legend_elements = []
    for region_num, info in region_colors.items():
        if region_num in np.unique(region_proj):  # Only show regions that exist
            legend_elements.append(
                patches.Patch(color=info['color'], label=info['name'])
            )
    ax_map.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Plot mean intensity profile with region boundaries
    mean_zprofile = tracer.get_projection([1, 2], operation='mean')
    z_positions = np.arange(len(mean_zprofile))
    ax_profile.plot(mean_zprofile, z_positions, color='black', linewidth=2, label='Mean Intensity')
    
    # Add region boundaries to profile plot
    if hasattr(tracer, 'region_bounds'):
        region_colors_profile = {
            'superficial': 'purple',
            'intermediate': 'red', 
            'deep': 'blue'
        }
        
        for region_name, (peak, sigma, bounds) in tracer.region_bounds.items():
            color = region_colors_profile.get(region_name, 'gray')
            
            # Add peak line
            ax_profile.axhline(peak, color=color, linestyle='--', linewidth=2, alpha=0.8)
            
            # Add region span
            ax_profile.axhspan(bounds[0], bounds[1], color=color, alpha=0.2, 
                             label=f'{region_name.title()} Region')
            
            # Add text label
            ax_profile.text(ax_profile.get_xlim()[1] * 0.1, peak, 
                          region_name.title(), 
                          verticalalignment='center',
                          color=color, fontweight='bold')
    
    ax_profile.invert_yaxis()
    ax_profile.set_xlabel('Mean Intensity')
    ax_profile.set_ylabel('Z (depth)')
    ax_profile.set_title('Depth Profile')
    ax_profile.grid(True, alpha=0.3)
    
    # Add some statistics as text
    if hasattr(tracer, 'region_map'):
        stats_text = []
        unique_regions, counts = np.unique(tracer.region_map, return_counts=True)
        total_voxels = tracer.region_map.size
        
        for region_num, count in zip(unique_regions, counts):
            region_name = region_colors[region_num]['name']
            percentage = (count / total_voxels) * 100
            stats_text.append(f'{region_name}: {percentage:.1f}%')
        
        # Add text box with statistics
        textstr = '\n'.join(stats_text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax_profile.text(0.05, 0.95, textstr, transform=ax_profile.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Return figure and axes dictionary
    axes = {
        'region_map': ax_map,
        'profile': ax_profile
    }
    
    return fig, axes
