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

def plot_projections_on_axis(ax, controller, projection: str = 'x', mode: str = 'binary', source: str = 'roi', depth_coded: bool = False, show_roi_box: bool = False) -> None:
    """Plot projections on a given axis.
    
    Args:
        ax: Matplotlib axis to plot on
        controller: VesselTracer instance with loaded data
        projection: Projection plane ('x', 'y', 'z')

        mode: Visualization mode. Options:
            - 'volume': Show original volume (global)
            - 'roi': Show ROI data (local, processed)
            - 'binary': Show binary vessel volume
            - 'region': Show region map with color-coded regions
        source: Source of data to project. Options:
            - 'roi': Use ROI model
            - 'image': Use image model
        depth_coded: If True, creates depth-coded projections where intensity
                    represents z-position (only works with binary mode)
        show_roi_box: If True, draws a red box around the ROI coordinates (only works with source='image')
    """
    # Validate mode
    valid_modes = ['volume', 'background', 'binary', 'region']
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")
    
    # Get the appropriate data object based on source
    if source == 'roi':
        if controller.roi_model is None:
            raise ValueError("ROI model not available. Run analysis pipeline first.")
        data_object = controller.roi_model
    else:  # source == 'image'
        data_object = controller.image_model
    
    # Validate that the requested mode is available in the data object
    if mode == 'volume' and data_object.volume is None:
        raise ValueError("Volume data not available.")
    elif mode == 'binary' and (not hasattr(data_object, 'binary') or data_object.binary is None):
        raise ValueError("Binary data not available. Run binarization first.")
    elif mode == 'background' and (not hasattr(data_object, 'background') or data_object.background is None):
        raise ValueError("Background data not available. Run background subtraction first.")
    elif mode == 'region' and (not hasattr(data_object, 'region') or data_object.region is None):
        raise ValueError("Region data not available. Run region detection first.")
    
    # Get projections using get_projection method
    if projection == 'z':
        _proj = data_object.get_projection(0, operation='max', volume_type=mode, depth_coded=depth_coded)  # Z projection (xy view)
    elif projection == 'y':
        _proj = data_object.get_projection(1, operation='max', volume_type=mode, depth_coded=depth_coded)  # Y projection (xz view)
    elif projection == 'x':
        _proj = data_object.get_projection(2, operation='max', volume_type=mode, depth_coded=depth_coded)  # X projection (yz view)
    
    # Choose colormap based on mode and depth coding
    if mode == 'region':
        # Use a discrete colormap for regions
        cmap = plt.cm.Set1  # Good for discrete categorical data
    elif depth_coded and mode == 'binary':
        # Use a colormap that shows depth well
        cmap = plt.cm.viridis
    else:
        cmap = 'gray'
    
    # Plot Z projection (top left)
    im_z = ax.imshow(_proj, cmap=cmap)
    title = f'Z proj ({mode})'
    if depth_coded and mode == 'binary':
        title += ' [depth-coded]'
        plt.colorbar(im_z, ax=ax, label='Z position (normalized)')
    ax.set_title(title)
    ax.axis('on')
    
    # Add scale bar (assuming we have pixel size)
    scalebar_length_pixels = int(50 / controller.image_model.pixel_size_x)  # 50 micron scale bar
    ax.plot([20, 20 + scalebar_length_pixels], [_proj.shape[0] - 20] * 2, 
            'w-' if cmap == 'gray' else 'k-', linewidth=2)
    
    # Draw ROI box if requested and source is 'image'
    if show_roi_box and source == 'image' and controller.roi_model is not None:
        # Get ROI coordinates
        min_x = controller.config.min_x
        min_y = controller.config.min_y
        micron_roi = controller.config.micron_roi
        
        # Convert micron ROI to pixels
        pixel_roi = int(micron_roi / controller.image_model.pixel_size_x)
        
        # Draw box on Z projection (xy view)
        if projection == 'z':
            rect = plt.Rectangle((min_x, min_y), pixel_roi, pixel_roi, 
                                fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
        elif projection == 'y':
            ax.hlines(min_x, min_x + pixel_roi, color='red', linewidth=2)
        elif projection == 'x':
            ax.vlines(min_y, min_y, min_y + pixel_roi, color='red', linewidth=2)

def plot_projections(controller, figsize=(10, 10), mode: str = 'binary', source: str = 'roi', depth_coded: bool = False, show_roi_box: bool = False, full_view: bool = True) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Create a comprehensive plot showing different projections and intensity profile.
    
    Creates a figure with:
    - Top left: Z projection (xy view)
    - Top right: Y projection (xz view)
    - Bottom left: X projection (yz view)
    - Bottom right: Mean intensity profile
    
    Args:
        controller: VesselTracer instance with loaded data
        figsize: Figure size tuple (width, height)
        mode: Visualization mode. Options:
            - 'volume': Show original volume (global)
            - 'roi': Show ROI data (local, processed)
            - 'binary': Show binary vessel volume
            - 'region': Show region map with color-coded regions
        source: Source of data to project. Options:
            - 'roi': Use ROI model
            - 'image': Use image model
        depth_coded: If True, creates depth-coded projections where intensity
                    represents z-position (only works with binary mode)
        show_roi_box: If True, draws a red box around the ROI coordinates (only works with source='image')
        
    Returns:
        Tuple of (figure, dict of axes)
    """
    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
    
    # Create axes
    ax_z = fig.add_subplot(gs[0, 0])  # Z projection (top left)
    ax_y = fig.add_subplot(gs[0, 1])  # Y projection (top right)
    ax_x = fig.add_subplot(gs[1, 0])  # X projection (bottom left)
    ax_profile = fig.add_subplot(gs[1, 1])  # Intensity profile (bottom right)
    
    # Plot projections on each axis
    plot_projections_on_axis(ax_z, controller, projection='z', mode=mode, source=source, depth_coded=depth_coded, show_roi_box=show_roi_box)
    plot_projections_on_axis(ax_y, controller, projection='y', mode=mode, source=source, depth_coded=depth_coded, show_roi_box=show_roi_box)
    plot_projections_on_axis(ax_x, controller, projection='x', mode=mode, source=source, depth_coded=depth_coded, show_roi_box=show_roi_box)
    
    # Get the appropriate data object based on source
    if source == 'roi':
        data_object = controller.roi_model
    else:  # source == 'image'
        data_object = controller.image_model
    
    # Plot mean intensity profile (bottom right)
    mean_profile = data_object.get_projection([1, 2], operation='mean', volume_type=mode)
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

def plot_paths_on_axis(ax, controller, 
                       projection='xy', region_colorcode: bool = False, 
                       linewidth = 5, alpha = 0.8, invert_yaxis: bool = False) -> None:
    """Plot vessel paths on a given axis.
    
    Args:
        controller: VesselTracer instance with traced paths
        ax: Matplotlib axis to plot on
        projection: Projection plane ('xy', 'xz', 'zy', or 'xyz' for 3D plot)
        region_colorcode: If True, color-code paths based on their region
        linewidth: Width of the lines
        alpha: Transparency of the lines
        invert_yaxis: If True, inverts the y-axis to match imshow's top-left origin
    """
    if controller.roi_model.paths is None:
        raise ValueError("No paths found. Run trace_paths() first.")
    
    # Get projected coordinates and colors
    if region_colorcode:
        x_paths, y_paths, z_paths, colors = controller.roi_model.get_path_coordinates(
            region_colorcode=region_colorcode,
            region_bounds=controller.roi_model.region_bounds
        )
    else:
        x_paths, y_paths, z_paths, colors = controller.roi_model.get_path_coordinates(
            region_colorcode=region_colorcode
        )
    
    for i in range(len(x_paths)):
        if projection == 'xyz':
            ax.plot(x_paths[i], y_paths[i], z_paths[i], color=colors[i], linewidth=linewidth, alpha=alpha)
        elif projection == 'xy':
            ax.plot(x_paths[i], y_paths[i], color=colors[i], linewidth=linewidth, alpha=alpha)
        elif projection == 'xz':
            ax.plot(x_paths[i], z_paths[i], color=colors[i], linewidth=linewidth, alpha=alpha)
        elif projection == 'zy':
            ax.plot(y_paths[i], z_paths[i], color=colors[i], linewidth=linewidth, alpha=alpha)
    
    if invert_yaxis:
        ax.invert_yaxis()

def plot_paths(controller, figsize=(15, 7), region_colorcode: bool = False, projection: str = 'xy') -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Plot vessel paths in both 2D and 3D projections.
    
    Args:
        controller: VesselTracer instance with traced paths
        figsize: Figure size tuple (width, height)
        region_colorcode: If True, color-code paths based on their region
        projection: Base projection for 2D plot ('xy', 'xz', or 'zy')
        
    Returns:
        Tuple of (figure, dict of axes)
    """
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Create 2D and 3D axes with shared x-axis for xy and xz
    ax_2d_xy = fig.add_subplot(gs[0, 0])
    ax_2d_xz = fig.add_subplot(gs[1, 0], sharex=ax_2d_xy)
    ax_2d_zy = fig.add_subplot(gs[0, 1], sharey=ax_2d_xy)
    ax_3d = fig.add_subplot(gs[1, 1], projection='3d')
    
    # Plot paths on 2D axis
    plot_paths_on_axis(controller, ax_2d_xy, projection='xy', region_colorcode=region_colorcode)
    plot_paths_on_axis(controller, ax_2d_xz, projection='zy', region_colorcode=region_colorcode)
    plot_paths_on_axis(controller, ax_2d_zy, projection='xz', region_colorcode=region_colorcode)

    # Plot paths on 3D axis
    plot_paths_on_axis(controller, ax_3d, projection='xyz', region_colorcode=region_colorcode)
    
    # Set titles
    ax_2d_xy.set_title(f'XY Projection')
    ax_2d_xz.set_title(f'XZ Projection')
    ax_2d_zy.set_title(f'ZY Projection')
    ax_3d.set_title('3D View')
    
    # Set equal aspect ratio for 2D projections
    ax_2d_xy.set_aspect('equal')
    # ax_2d_xz.set_aspect('equal')
    # ax_2d_zy.set_aspect('equal')
    
    # Add legend if using region colorcoding
    if region_colorcode and hasattr(controller, 'region_bounds'):
        region_colors = {
            'superficial': 'tab:purple',
            'intermediate': 'tab:red',
            'deep': 'tab:blue',
            'diving': 'magenta'
        }
        handles = [plt.Line2D([0], [0], color=color, label=region) 
                  for region, color in region_colors.items()]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=4, title='Regions')
    
    plt.tight_layout()
    return fig, {'2d_xy': ax_2d_xy, '2d_xz': ax_2d_xz, '2d_zy': ax_2d_zy, '3d': ax_3d}

def plot_projections_w_paths(controller, figsize=(10, 10), mode: str = 'binary', depth_coded: bool = False, region_colorcode: bool = False) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Create a comprehensive plot showing different projections with vessel paths.
    
    Creates a figure with:
    - Top left: Z projection (xy view) with paths
    - Top right: Y projection (xz view) with paths
    - Bottom left: X projection (yz view) with paths
    - Bottom right: Mean intensity profile
    
    Args:
        controller: VesselTracer instance with loaded data
        figsize: Figure size tuple (width, height)
        mode: Visualization mode. Options:
            - 'volume': Show original volume (global)
            - 'roi': Show ROI data (local, processed)
            - 'binary': Show binary vessel volume
            - 'region': Show region map with color-coded regions
        depth_coded: If True, creates depth-coded projections where intensity
                    represents z-position (only works with binary mode)
        region_colorcode: If True, color-code paths based on their region
        
    Returns:
        Tuple of (figure, dict of axes)
    """
    # First create the base projections plot
    fig, axes = plot_projections(controller, figsize=figsize, mode=mode, depth_coded=depth_coded)
    
    # Add paths to each projection
    if hasattr(controller, 'paths'):
        # Map projection names to their corresponding views
        projection_map = {
            'z_proj': 'xy',  # Z projection shows xy view
            'y_proj': 'xz',  # Y projection shows xz view
            'x_proj': 'zy'   # X projection shows zy view
        }
        
        # Plot paths on each projection
        for ax_name, projection in projection_map.items():
            plot_paths_on_axis(controller, axes[ax_name], projection=projection, 
                             region_colorcode=region_colorcode,
                             invert_yaxis=False)  # Don't invert y-axis for these plots
    
    return fig, axes

def plot_regions(controller, figsize=(8, 4)) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Plot the mean z-profile with detected regions alongside y-projection.
    
    Args:
        controller: VesselTracer instance with loaded data
        figsize: Figure size tuple (width, height)
        
    Returns:
        Tuple of (figure, dict of axes)
    """
    # Get the data object to use for projections
    # Use ROI model if available, otherwise use image model
    if hasattr(controller, 'roi_model') and controller.roi_model is not None:
        data_object = controller.roi_model
    else:
        data_object = controller.image_model
    
    # Get mean, min, and max z-profiles using the data object's get_projection method
    mean_zprofile = data_object.get_projection([1, 2], operation='mean')
    min_zprofile = data_object.get_projection([1, 2], operation='min')
    max_zprofile = data_object.get_projection([1, 2], operation='max')
    y_proj = data_object.get_projection(1, operation='max')
    
    # Determine regions if not already done
    if data_object.region_bounds is None:
        print("Determining regions has not been run yet")
        data_object.region_bounds = controller.tracer.determine_regions(data_object.binary)
    
    # Create figure with two subplots
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, 
                                  gridspec_kw={'width_ratios': [3, 1]})
    
    # Plot y-projection
    ax0.imshow(y_proj, cmap='gray', aspect='auto')
    ax0.set_title('Y Projection')
    
    # Plot mean z-profile
    Z = len(mean_zprofile)
    z_positions = np.arange(Z)
    ax1.plot(mean_zprofile, z_positions, color='black', label='Mean')
    #ax1.plot(min_zprofile, z_positions, color='gray', linestyle=':', label='Min')
    #ax1.plot(max_zprofile, z_positions, color='gray', linestyle=':', label='Max')
    
    # Define colors for layers
    layer_colors = ['tab:purple', 'tab:red', 'tab:blue']
    
    # Plot regions
    for i, (region, (peak, sigma, bounds)) in enumerate(controller.roi_model.region_bounds.items()):
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
    
    # Add legend for z-profiles
    #ax1.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Return figure and axes dictionary
    axes = {
        'y_proj': ax0,
        'z_profile': ax1
    }
    
    return fig, axes

def plot_region_projections(controller, figsize=(15, 5)) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Plot the binary projections for each region.
    
    Creates a figure with three subplots, one for each region's binary projection.
    
    Args:
        controller: VesselTracer instance with loaded data
        figsize: Figure size tuple (width, height)
        
    Returns:
        Tuple of (figure, dict of axes)
    """
    # Create region projections if they don't exist
    if not hasattr(controller.roi_model, 'region_projections'):
        controller.processor.create_region_projections(controller.roi_model)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Define colors for each region
    region_colors = {
        'superficial': 'tab:purple',
        'intermediate': 'tab:red',
        'deep': 'tab:blue'
    }
    
    # Plot each region projection
    for i, (region_name, ax) in enumerate(zip(['superficial', 'intermediate', 'deep'], axes)):
        projection = controller.roi_model.region_projections[region_name]
        
        # Plot projection
        im = ax.imshow(projection, cmap='gray')
        ax.set_title(f'{region_name.capitalize()} Layer')
        
        # Add scale bar
        scalebar_length_pixels = int(50 / controller.image_model.pixel_size_x)  # 50 micron scale bar
        ax.plot([20, 20 + scalebar_length_pixels], [projection.shape[0] - 20] * 2, 
                'w-', linewidth=2)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Binary Value')
    
    plt.tight_layout()
    
    # Return figure and axes dictionary
    axes_dict = {
        'superficial': axes[0],
        'intermediate': axes[1],
        'deep': axes[2]
    }
    
    return fig, axes_dict