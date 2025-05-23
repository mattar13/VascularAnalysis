import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
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

def plot_mean_zprofile(tracer, ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Plot the mean z-profile with detected regions alongside y-projection.
    
    Args:
        tracer: VesselTracer instance with loaded data
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        
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
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4), 
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

def plot_vessel_analysis(tracer, figsize=(15,5)) -> None:
    """Create a comprehensive plot of vessel analysis results.
    
    Args:
        tracer: VesselTracer instance with completed analysis
        figsize: Figure size tuple (width, height)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original ROI
    axes[0].set_title("Maximum Intensity Projection (ROI)")
    show_max_projection(tracer.roi, axes[0])
    plt.colorbar(axes[0].images[0], ax=axes[0])
    
    # Binary segmentation
    axes[1].set_title("Binary Segmentation")
    show_max_projection(tracer.binary, axes[1])
    
    # Vessel paths
    axes[2].set_title("Vessel Paths")
    show_max_projection(tracer.roi, axes[2])  # Use ROI as background
    plot_vessel_paths(tracer.paths, axes[2])
    
    plt.tight_layout()

def plot_vessel_paths(paths: Dict[int, tuple], ax: Optional[plt.Axes] = None, 
                     color: str = 'red', linewidth: float = 0.5) -> None:
    """Plot vessel paths on given axes.
    
    Args:
        paths: Dictionary of vessel paths from skeletonization
        ax: Optional matplotlib axes to plot on
        color: Color for vessel paths
        linewidth: Line width for paths
    """
    if ax is None:
        ax = plt.gca()
        
    for path in paths.values():
        start_z, start_y, start_x, end_z, end_y, end_x = path
        ax.plot([start_x, end_x], [start_y, end_y], color=color, 
                linewidth=linewidth, alpha=0.7)

def plot_layer_analysis(tracer) -> None:
    """Plot vessel layer analysis results.
    
    Args:
        tracer: VesselTracer instance with completed analysis
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    mean_z = np.mean(tracer.binary, axis=(1,2))
    ax.plot(mean_z, label='Vessel Density')
    
    for layer, (peak, sigma, bounds) in tracer.layer_info.items():
        ax.axvline(peak, color='r', linestyle='--', alpha=0.5)
        ax.axvspan(bounds[0], bounds[1], alpha=0.2, label=layer)
    
    ax.set_xlabel('Z Position')
    ax.set_ylabel('Mean Vessel Density')
    ax.set_title('Vessel Layer Analysis')
    ax.legend()
    plt.tight_layout()

def _sort_path_points(points):
    """Sort points in a path to minimize total distance.
    
    Uses a simple nearest neighbor approach to sort points.
    
    Args:
        points: Nx3 array of points (Z,Y,X)
        
    Returns:
        Sorted array of points
    """
    if len(points) <= 2:
        return points
        
    sorted_points = [points[0]]
    remaining = points[1:].copy()
    
    while len(remaining) > 0:
        # Find closest point to last point
        last_point = sorted_points[-1]
        distances = np.linalg.norm(remaining - last_point, axis=1)
        next_idx = np.argmin(distances)
        sorted_points.append(remaining[next_idx])
        remaining = np.delete(remaining, next_idx, axis=0)
    
    return np.array(sorted_points)

def _classify_path_by_layer(path_coords, region_bounds):
    """Classify a path based on its z-coordinates and region bounds.
    
    Args:
        path_coords: Nx3 array of points (Z,Y,X)
        region_bounds: Dictionary of region bounds from tracer
        
    Returns:
        Tuple of (layer_name, is_diving)
    """
    z_coords = path_coords[:, 0]  # Z is first coordinate
    
    # Check if any points are outside all regions
    min_z = min(z_coords)
    max_z = max(z_coords)
    
    # Find which regions the path spans
    spanning_regions = []
    for region, (_, _, (lower, upper)) in region_bounds.items():
        if min_z <= upper and max_z >= lower:
            spanning_regions.append(region)
    
    # If path spans multiple regions or is outside all regions, it's a diving vessel
    if len(spanning_regions) > 1 or len(spanning_regions) == 0:
        return 'diving', True
    
    # Path is contained within a single region
    return spanning_regions[0], False

def plot_path_projections(tracer, figsize=(15, 8)):
    """Plot vessel paths in multiple projections.
    
    Creates a figure with 4 subplots, each showing both paths and depth-coded heatmap:
    1. XY projection (top left)
    2. XZ projection (bottom left)
    3. ZY projection (top right)
    4. 3D projection (right side, spanning half height)
    
    Args:
        tracer: VesselTracer instance with paths
        figsize: Tuple of (width, height) for the figure
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure with 4 subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2])  # Make 3D plot slightly wider
    
    # XY projection (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('XY Projection')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # ZY projection (top right)
    ax4 = fig.add_subplot(gs[0, 1])
    ax4.set_title('ZY Projection')
    ax4.set_xlabel('Z')
    ax4.set_ylabel('Y')
    
    # XZ projection (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('XZ Projection')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    
    # 3D projection (right side, spanning half height)
    ax2 = fig.add_subplot(gs[0:2, 2], projection='3d')
    ax2.set_title('3D View')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Define colors for each layer
    layer_colors = {
        'superficial': 'tab:purple',
        'intermediate': 'tab:red',
        'deep': 'tab:blue',
        'diving': 'magenta'
    }
    
    # First, create depth-coded heatmaps
    data = tracer.binary
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

    # Plot Z projection (top left)
    im_z = ax1.imshow(z_proj, cmap=cmap, alpha=0.25)
    title = f'Z proj (Depth-coded)'
    plt.colorbar(im_z, ax=ax1, label='Z position (normalized)')
    ax1.set_title(title)
    ax1.axis('on')
    
    # Add scale bar (assuming we have pixel size)
    scalebar_length_pixels = int(50 / tracer.pixel_size_x)  # 50 micron scale bar
    ax1.plot([20, 20 + scalebar_length_pixels], [z_proj.shape[0] - 20] * 2, 
              'w-' if cmap == 'gray' else 'k-', linewidth=2)
    
    # Plot Y projection (top right)
    im_y = ax4.imshow(x_proj.T, cmap=cmap, alpha=0.25)  # Transpose y_proj to rotate 90 degrees
    ax4.set_title('Y proj')
    ax4.axis('on')
    plt.colorbar(im_y, ax=ax4, label='Z position (normalized)')
    
    # Plot X projection (bottom left)
    im_x = ax3.imshow(y_proj, cmap=cmap, alpha=0.25)
    ax3.axis('on')
    plt.colorbar(im_x, ax=ax3, label='Z position (normalized)')
    
    # Process and plot each path
    for path_id, path_coords in tracer.paths.items():
        # Sort points to minimize total distance
        sorted_coords = _sort_path_points(path_coords)
        
        # Classify path by layer
        layer, is_diving = _classify_path_by_layer(sorted_coords, tracer.region_bounds)
        color = layer_colors[layer]
        
        # Extract coordinates
        x = sorted_coords[:, 2]  # Note: CZI files are typically ZYX order
        y = sorted_coords[:, 1]
        z = sorted_coords[:, 0]
        
        # Plot paths on top of heatmaps
        ax1.plot(x, y, '.-', color=color, alpha=0.5, markersize=2, linewidth=5.0, label=layer if is_diving else None)
        ax2.plot(x, y, z, '.-', color=color, alpha=0.5, markersize=2, linewidth=5.0)
        ax3.plot(x, z, '.-', color=color, alpha=0.5, markersize=2, linewidth=5.0)
        ax4.plot(z, y, '.-', color=color, alpha=0.5, markersize=2, linewidth=5.0)
    
    # Add legend for diving vessels
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, title='Diving Vessels')
    
    # Adjust 3D view
    ax2.view_init(elev=20, azim=45)
    
    # Make all 2D plots square
    for ax in [ax1, ax3, ax4]:
        ax.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3, ax4)
