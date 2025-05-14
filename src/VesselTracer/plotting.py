import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

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

def plot_mean_zprofile(tracer, ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the mean z-profile with detected regions.
    
    Args:
        tracer: VesselTracer instance with loaded data
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        
    Returns:
        Tuple of (figure, axes) used for plotting
    """
    # Get mean z-profile
    mean_zprofile = tracer.get_projection([1, 2], operation='mean')
    
    # Determine regions if not already done
    if not hasattr(tracer, 'region_bounds'):
        tracer.determine_regions()
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Plot profile
    z_positions = np.arange(len(mean_zprofile))
    ax.plot(z_positions, mean_zprofile, 'k-', label='Mean Intensity')
    
    # Plot regions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    for (region, (peak, sigma, bounds)), color in zip(tracer.region_bounds.items(), colors):
        # Plot peak
        ax.axvline(peak, color=color, linestyle='--', alpha=0.5)
        
        # Plot bounds
        ax.axvspan(bounds[0], bounds[1], color=color, alpha=0.2, label=region)
        
        # Add text annotation
        ax.text(peak, ax.get_ylim()[1], region, 
                rotation=90, va='bottom', ha='right', color=color)
    
    # Customize plot
    ax.set_xlabel('Z Position (slices)')
    ax.set_ylabel('Mean Vessel Density')
    ax.set_title('Vessel Distribution Across Z-Axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def plot_projections(tracer, figsize=(10, 10), mode: str = 'smoothed') -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
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
    
    # Get projections
    z_proj = np.max(data, axis=0)  # xy view
    y_proj = np.max(data, axis=1)  # xz view
    x_proj = np.max(data, axis=2)  # yz view
    mean_profile = np.mean(data, axis=(1,2))  # z profile
    
    # Plot Z projection (top left)
    ax_z.imshow(z_proj, cmap='gray')
    ax_z.set_title(f'Z proj ({mode})')
    ax_z.axis('on')
    
    # Add scale bar (assuming we have pixel size)
    scalebar_length_pixels = int(50 / tracer.pixel_size_x)  # 50 micron scale bar
    ax_z.plot([20, 20 + scalebar_length_pixels], [z_proj.shape[0] - 20] * 2, 
              'w-', linewidth=2)
    
    # Plot Y projection (top right)
    ax_y.imshow(y_proj, cmap='gray')
    ax_y.set_title('Y proj')
    ax_y.axis('on')
    
    # Plot X projection (bottom left)
    ax_x.imshow(x_proj, cmap='gray')
    ax_x.axis('on')
    
    # Plot mean intensity profile (bottom right)
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
