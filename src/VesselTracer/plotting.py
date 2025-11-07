import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
from .config import DEFAULT_DIVING_COLOR, DEFAULT_REGIONS, DEFAULT_REGION_COLORS


def _extract_region_color_settings(controller) -> Tuple[Dict[str, str], str]:
    """Return configured region colors and diving color with sensible defaults."""
    config = getattr(controller, 'config', None)
    if config is not None and hasattr(config, 'get_region_color_map'):
        return config.get_region_color_map(), getattr(config, 'diving_color', DEFAULT_DIVING_COLOR)
    fallback_colors = {region: color for region, color in zip(DEFAULT_REGIONS, DEFAULT_REGION_COLORS)}
    return fallback_colors, DEFAULT_DIVING_COLOR


def _should_flip_depth(controller) -> bool:
    config = getattr(controller, 'config', None)
    return bool(config and getattr(config, 'flip_z_axis', False))


def _flip_depth_array(array: np.ndarray) -> np.ndarray:
    """Flip array along depth axis (axis 0) for visualization."""
    return np.flip(array, axis=0)


def _flip_depth_scalar(value: float, depth: Optional[int]) -> float:
    if depth is None:
        return value
    return depth - value


def _flip_depth_bounds(bounds: Tuple[float, float], depth: Optional[int]) -> Tuple[float, float]:
    if depth is None:
        return bounds
    lower, upper = bounds
    return (depth - upper, depth - lower)

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
    flip_depth = _should_flip_depth(controller)

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
    
    if flip_depth and projection in ('y', 'x'):
        _proj = _flip_depth_array(_proj)
    
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
            ax.vlines([min_x, min_x + pixel_roi], ymin = 0, ymax = _proj.shape[0], color='red', linewidth=2)
        elif projection == 'x':
            ax.hlines([min_y, min_y + pixel_roi], xmin = 0, xmax = _proj.shape[1], color='red', linewidth=2)

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
    if _should_flip_depth(controller):
        mean_profile = mean_profile[::-1]
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
    
    # Determine configured colors
    region_color_map, diving_color = _extract_region_color_settings(controller)

    get_paths_kwargs = {
        'region_colorcode': region_colorcode,
        'region_color_map': region_color_map,
        'diving_color': diving_color
    }
    if region_colorcode:
        get_paths_kwargs['region_bounds'] = controller.roi_model.region_bounds

    # Get projected coordinates and colors
    x_paths, y_paths, z_paths, colors = controller.roi_model.get_path_coordinates(**get_paths_kwargs)

    flip_depth = _should_flip_depth(controller)
    depth_extent = None
    if flip_depth and controller.roi_model is not None and controller.roi_model.volume is not None:
        depth_extent = controller.roi_model.volume.shape[0] - 1
    if flip_depth and depth_extent is not None:
        z_paths = [depth_extent - z for z in z_paths]
    
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
    if region_colorcode:
        region_color_map, diving_color = _extract_region_color_settings(controller)
        legend_regions = controller.config.regions if hasattr(controller, 'config') else list(region_color_map.keys())
        handles = [
            plt.Line2D([0], [0], color=region_color_map.get(region, 'k'), label=region)
            for region in legend_regions if region in region_color_map
        ]
        handles.append(plt.Line2D([0], [0], color=diving_color, label='diving'))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=max(len(handles), 1), title='Regions')
    
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
    
    flip_depth = _should_flip_depth(controller)
    depth_extent = None
    if flip_depth and data_object.volume is not None:
        depth_extent = data_object.volume.shape[0] - 1
    
    # Get mean, min, and max z-profiles using the data object's get_projection method
    mean_zprofile = data_object.get_projection([1, 2], operation='mean')
    min_zprofile = data_object.get_projection([1, 2], operation='min')
    max_zprofile = data_object.get_projection([1, 2], operation='max')
    y_proj = data_object.get_projection(1, operation='max')
    if flip_depth:
        mean_zprofile = mean_zprofile[::-1]
        min_zprofile = min_zprofile[::-1]
        max_zprofile = max_zprofile[::-1]
        y_proj = _flip_depth_array(y_proj)
    
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
    
    region_color_map, _ = _extract_region_color_settings(controller)

    # Plot regions
    for region, (peak, sigma, bounds) in data_object.region_bounds.items():
        # Add horizontal lines at peaks
        display_peak = _flip_depth_scalar(peak, depth_extent) if flip_depth else peak
        display_bounds = _flip_depth_bounds(bounds, depth_extent) if flip_depth else bounds
        span_color = region_color_map.get(region, 'gray')
        ax0.axhline(display_peak, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(display_peak, color='red', linestyle='--', alpha=0.5)
        
        # Add spans for regions
        ax0.axhspan(display_bounds[0], display_bounds[1], color=span_color, alpha=0.25, label=region)
        ax1.axhspan(display_bounds[0], display_bounds[1], color=span_color, alpha=0.5, label=region)
    
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
    
    region_color_map, _ = _extract_region_color_settings(controller)
    
    # Plot each region projection
    for i, (region_name, ax) in enumerate(zip(['superficial', 'intermediate', 'deep'], axes)):
        projection = controller.roi_model.region_projections[region_name]
        color = region_color_map.get(region_name, 'white')
        
        # Plot projection
        im = ax.imshow(projection, cmap='gray')
        ax.set_title(f'{region_name.capitalize()} Layer', color=color)
        
        # Add scale bar
        scalebar_length_pixels = int(50 / controller.image_model.pixel_size_x)  # 50 micron scale bar
        ax.plot([20, 20 + scalebar_length_pixels], [projection.shape[0] - 20] * 2, 
                color=color, linewidth=2)
        
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


def smooth_path_coordinates(x, y, z, method='gaussian', sigma=2.0, spline_smoothing=0.5, num_points=None):
    """Smooth vessel path coordinates to reduce jitter.
    
    Args:
        x: Array of x coordinates
        y: Array of y coordinates
        z: Array of z coordinates
        method: Smoothing method - 'gaussian' or 'spline'
        sigma: Standard deviation for Gaussian smoothing (only for 'gaussian' method)
        spline_smoothing: Smoothing parameter for spline (only for 'spline' method)
                         0 = no smoothing, larger values = more smoothing
        num_points: Number of points in the smoothed path (only for 'spline' method)
                   If None, uses the original number of points
    
    Returns:
        Tuple of (x_smooth, y_smooth, z_smooth) with smoothed coordinates
    """
    if len(x) < 4:
        # Path too short to smooth, return as-is
        return x, y, z
    
    if method == 'gaussian':
        # Apply Gaussian filter to smooth coordinates
        x_smooth = gaussian_filter1d(x, sigma=sigma, mode='nearest')
        y_smooth = gaussian_filter1d(y, sigma=sigma, mode='nearest')
        z_smooth = gaussian_filter1d(z, sigma=sigma, mode='nearest')
        
        return x_smooth, y_smooth, z_smooth
    
    elif method == 'spline':
        # Use B-spline interpolation for smooth curves
        # Create parametric variable
        points = np.array([x, y, z]).T
        
        # Fit a B-spline to the path
        try:
            # k is the degree of the spline (3 = cubic)
            k = min(3, len(x) - 1)
            tck, u = splprep([x, y, z], s=spline_smoothing, k=k)
            
            # Evaluate the spline at new points
            if num_points is None:
                num_points = len(x)
            u_new = np.linspace(0, 1, num_points)
            x_smooth, y_smooth, z_smooth = splev(u_new, tck)
            
            return x_smooth, y_smooth, z_smooth
        except:
            # If spline fitting fails, fall back to Gaussian smoothing
            return smooth_path_coordinates(x, y, z, method='gaussian', sigma=sigma)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}. Use 'gaussian' or 'spline'.")


def plot_vessel_paths_3d(controller, 
                         source: str = 'roi',
                         region_colorcode: bool = True,
                         linewidth: float = 1.5,
                         alpha: float = 0.7,
                         figsize: Tuple[int, int] = (12, 10),
                         elev: float = 20,
                         azim: float = 45,
                         show_region_planes: bool = True) -> Tuple[plt.Figure, Axes3D]:
    """Plot vessel paths in 3D with noodly lines.
    
    Note: Path smoothing is now handled in the analysis pipeline (VesselTracer.smooth_paths).
    This function plots the paths as they are stored in the model.
    
    Args:
        controller: VesselAnalysisController instance
        source: Data source ('roi' or 'image')
        region_colorcode: If True, color paths by vascular region
        linewidth: Width of the path lines
        alpha: Transparency of the lines (0-1)
        figsize: Figure size as (width, height)
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        show_region_planes: If True, show semi-transparent planes at region boundaries
        
    Returns:
        Tuple of (figure, 3D axes)
    """
    # Get the appropriate data model
    if source == 'roi':
        if controller.roi_model is None:
            raise ValueError("ROI model not available. Run analysis pipeline first.")
        model = controller.roi_model
    else:
        model = controller.image_model
    
    # Check if paths exist
    if model.paths is None or len(model.paths) == 0:
        raise ValueError("No paths found. Run trace_paths() first.")
    
    # Get path coordinates
    region_color_map, diving_color = _extract_region_color_settings(controller)
    x_paths, y_paths, z_paths, colors = model.get_path_coordinates(
        region_colorcode=region_colorcode,
        region_bounds=model.region_bounds if region_colorcode else None,
        region_color_map=region_color_map,
        diving_color=diving_color
    )
    flip_depth = _should_flip_depth(controller)
    depth_extent = None
    if flip_depth and model.volume is not None:
        depth_extent = model.volume.shape[0] - 1
    if flip_depth and depth_extent is not None:
        z_paths = [depth_extent - z for z in z_paths]
    
    # Create 3D figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each path as a line (paths are already smoothed in the pipeline)
    for i, (x, y, z, color) in enumerate(zip(x_paths, y_paths, z_paths, colors)):
        ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha)
    
    # Add region boundary planes if requested
    if show_region_planes and model.region_bounds is not None:
        # Get volume dimensions
        z_max, y_max, x_max = model.volume.shape
        x_grid, y_grid = np.meshgrid([0, x_max], [0, y_max])
        
        # Plot semi-transparent planes at region boundaries
        for region_name, (peak, sigma, bounds) in model.region_bounds.items():
            lower, upper = bounds
            if flip_depth and depth_extent is not None:
                lower, upper = _flip_depth_bounds(bounds, depth_extent)
            # Plot lower boundary
            z_plane_min = np.full_like(x_grid, lower)
            ax.plot_surface(x_grid, y_grid, z_plane_min, 
                          alpha=0.1, color='gray', linewidth=0)
            
            # Plot upper boundary
            z_plane_max = np.full_like(x_grid, upper)
            ax.plot_surface(x_grid, y_grid, z_plane_max, 
                          alpha=0.1, color='gray', linewidth=0)
    
    # Set labels and title
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_zlabel('Z (depth, pixels)', fontsize=12)
    ax.set_title(f'3D Vessel Paths ({source.upper()} volume)', fontsize=14, fontweight='bold')
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Add legend if region color-coded
    if region_colorcode:
        from matplotlib.patches import Patch
        region_color_map, diving_color = _extract_region_color_settings(controller)
        legend_regions = controller.config.regions if hasattr(controller, 'config') else list(region_color_map.keys())
        legend_elements = [
            Patch(facecolor=region_color_map.get(region, 'gray'), alpha=alpha, label=region.capitalize())
            for region in legend_regions if region in region_color_map
        ]
        legend_elements.append(Patch(facecolor=diving_color, alpha=alpha, label='Diving'))
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.8)
    
    plt.tight_layout()
    
    return fig, ax
