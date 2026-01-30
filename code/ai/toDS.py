import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
import warnings

def plot_curvature_with_histogram(coords, values, bin_width=1, 
                                  cmap='viridis', linewidth=2, alpha=1.0, 
                                  closed=True, hist_color='skyblue'):
    """
    Plot a gradient-colored polygon alongside its value histogram.
    
    Parameters:
    -----------
    coords : list of lists
        [[x1, y1], [x2, y2], ...] - polygon vertices
    values : list
        List of values corresponding to each vertex
    bin_width : int or float
        Width of histogram bins
    cmap : str
        Matplotlib colormap for polygon
    linewidth : float
        Width of polygon lines
    alpha : float
        Transparency
    closed : bool
        Whether to close the polygon
    hist_color : str
        Color for histogram bars
    """
    # Create figure with subplots
    fig, (ax_poly, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ==============================================
    # Left panel: Gradient polygon
    # ==============================================
    coords_arr = np.array(coords)
    values_arr = np.array(values)
    
    # Create line segments for polygon
    if closed:
        segments = np.array([coords_arr, np.roll(coords_arr, -1, axis=0)]).transpose(1, 0, 2)
    else:
        segments = np.array([coords_arr[:-1], coords_arr[1:]]).transpose(1, 0, 2)
    
    # Create LineCollection with gradient colors
    lc = LineCollection(segments, cmap=cmap, linewidth=linewidth, alpha=alpha)
    
    # Set colors based on values
    if len(values_arr) == len(segments):
        lc.set_array(values_arr)
    else:
        if closed:
            segment_values = (values_arr + np.roll(values_arr, -1)) / 2
        else:
            segment_values = (values_arr[:-1] + values_arr[1:]) / 2
        lc.set_array(segment_values)
    
    ax_poly.add_collection(lc)
    
    # Set axis limits
    x_min, x_max = coords_arr[:, 0].min(), coords_arr[:, 0].max()
    y_min, y_max = coords_arr[:, 1].min(), coords_arr[:, 1].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    ax_poly.set_xlim(x_min - x_padding, x_max + x_padding)
    ax_poly.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add colorbar
    plt.colorbar(lc, ax=ax_poly, label='Value')
    
    ax_poly.set_xlabel('X')
    ax_poly.set_ylabel('Y')
    ax_poly.set_aspect('equal')
    ax_poly.grid(True, alpha=0.3)
    ax_poly.set_title(f'Gradient Polygon ({cmap})')
    
    # ==============================================
    # Right panel: Histogram
    # ==============================================
    data = np.asarray(values).flatten()
    
    # Calculate bins
    data_min = np.floor(np.min(data) / bin_width) * bin_width
    data_max = np.ceil(np.max(data) / bin_width) * bin_width + 1e-10
    bins = np.arange(data_min, data_max, bin_width)
    
    if len(bins) < 2:
        bins = np.array([data_min - bin_width/2, data_min + bin_width/2])
    
    # Plot histogram
    n, bin_edges, patches = ax_hist.hist(
        data, bins=bins, density=False,
        color=hist_color, edgecolor='black', alpha=0.7,
        align='mid', rwidth=0.8
    )
    
    # Customize histogram
    ax_hist.set_xlabel('Curvature Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title(f'Value Distribution (bin width={bin_width})')
    
    if bin_width >= 1:
        ax_hist.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax_hist.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics
    stats = {
        'n': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': data_min,
        'max': data_max
    }
    
    stats_text = (
        f'n = {stats["n"]:,}\n'
        f'Mean = {stats["mean"]:.3f}\n'
        f'Median = {stats["median"]:.3f}\n'
        f'Std = {stats["std"]:.3f}\n'
        f'Range = [{stats["min"]:.1f}, {stats["max"]:.1f}]'
    )
    
    ax_hist.text(0.98, 0.98, stats_text,
                transform=ax_hist.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, fontfamily='monospace')
    
    # Add vertical line at mean
    ax_hist.axvline(x=stats['mean'], color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='Mean')
    ax_hist.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax_poly, ax_hist), stats


# ==============================================
# Example usage
# ==============================================

if __name__ == "__main__":
    # Create example data
    np.random.seed(42)
    n_points = 20
    
    # Generate star shape
    angles = np.linspace(0, 2 * np.pi, n_points)
    radius = 2 + 0.5 * np.sin(angles * 3)
    
    coords = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ]).tolist()
    
    # Simulated curvature values
    values = np.abs(np.sin(angles * 2)) * 0.5 + np.random.rand(n_points) * 0.5
    
    # Plot combined figure
    fig, axes, stats = plot_curvature_with_histogram(
        coords=coords,
        values=values,
        bin_width=0.1,  # Fine binning for curvature values
        cmap='plasma',
        linewidth=2.5,
        hist_color='lightblue'
    )
    
    print("Statistics:", stats)