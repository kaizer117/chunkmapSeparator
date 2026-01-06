'''
This submodule contains graphing utilities for visualizing contours
'''


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import programutils as putils
from matplotlib.ticker import MaxNLocator
import warnings



def plotconsHierarchy(consObj,mplcmap,suptitle):
    
    '''
    plots the 4 features of a hierarchy dataset onto the 
    '''
    
    fig, axs = plt.subplots(2, 2)
    textSizer=putils.norm01(consObj.area)
    
    
    for i in range(4):
        plt.subplot(2,2,i+1)

        cmap=mplcmap(consObj.hierarchy_norm[i])

        
        j=0
        for con in consObj.contours:
            plt.plot(con[:,0,0],con[:,0,1],color=cmap[j],marker='o',markersize=0.1)
            #plt.text(consObj.centeroid[i][0],consObj.centeroid[i][1],str(i),fontdict={'fontsize':textSizer*13+4})
            j+=1
        plt.title('hierarchy '+str(i)+' hierarchy feature')
    
    plt.suptitle(suptitle)
    
    return plt

def plotCon(img_size,con):
    # plots one contour
    plt.plot(con[:,0,0],-1*con[:,0,1],marker='o',markersize=0.1)
    plt.xlim(0,img_size[1])
    plt.ylim(-1*img_size[0],0)
    return plt

def plotCurvature(coords, values, cmap='viridis', linewidth=2, alpha=1.0, closed=True):
    """
    Plot a 2D polygon with line color gradient based on values.
    
    Parameters:
    -----------
    coords : list of lists
        [[x1, y1], [x2, y2], ...] - polygon vertices
    values : list
        List of values corresponding to each vertex (or segment)
    cmap : str
        Matplotlib colormap name ('viridis', 'plasma', 'cool', 'wistia', etc.)
    linewidth : float
        Width of polygon lines
    alpha : float
        Transparency (0 to 1)
    closed : bool
        Whether to close the polygon (connect last point to first)
    """
    # Convert to numpy arrays
    coords = np.array(coords)
    values = np.array(values)
    
    # Normalize values to [0, 1] for colormap
    vmin, vmax = values.min(), values.max()
    norm_values = (values - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(values)
    
    # Get colormap
    colormap = cm.get_cmap(cmap)
    
    # Create line segments for gradient coloring
    if closed:
        # Add first point at the end to close the polygon
        segments = np.array([coords, np.roll(coords, -1, axis=0)]).transpose(1, 0, 2)
    else:
        # For open polygon
        segments = np.array([coords[:-1], coords[1:]]).transpose(1, 0, 2)
    
    # Create LineCollection with gradient colors
    lc = LineCollection(segments, cmap=cmap, linewidth=linewidth, alpha=alpha)
    
    # Set the colors based on values (average between segment endpoints)
    if len(values) == len(segments):
        # Values already correspond to segments
        lc.set_array(values)
    else:
        # Values correspond to vertices - use average for segments
        if closed:
            segment_values = (values + np.roll(values, -1)) / 2
        else:
            segment_values = (values[:-1] + values[1:]) / 2
        lc.set_array(segment_values)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_collection(lc)
    
    # Set axis limits with padding
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add colorbar
    plt.colorbar(lc, ax=ax, label='Value')
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.title(f'Gradient Polygon ({cmap} colormap)')
    plt.tight_layout()
    
    return fig, ax

class HistogramPlotter():
    def __init__(self):
        return None

    
    def plot_fine_histogram(self, data, bin_size=1, density=False, cumulative=False,
                           color='skyblue', edgecolor='black', alpha=0.7,
                           title=None, xlabel='Value', ylabel=None,
                           figsize=(10, 6), dpi=100, show_stats=True,
                           show_grid=True, vertical_lines=None, ax=None):
        """
        Plot histogram with very fine binning (1 or 2 unit bins).
        
        Parameters:
        -----------
        data : array-like
            Input data values
        bin_size : int or float, default=1
            Width of each bin. For fine histograms, use 1 or 2.
            Can be fractional like 0.5, 0.25, etc.
        density : bool, default=False
            If True, normalize to form probability density
        cumulative : bool or -1, default=False
            If True, plot cumulative histogram
            If -1, plot reversed cumulative histogram
        color : str, default='skyblue'
            Bar color
        edgecolor : str, default='black'
            Bar edge color
        alpha : float, default=0.7
            Bar transparency
        title : str, optional
            Plot title
        xlabel : str, default='Value'
            X-axis label
        ylabel : str, optional
            Y-axis label (auto-generated if None)
        figsize : tuple, default=(10, 6)
            Figure size (width, height)
        dpi : int, default=100
            Figure resolution
        show_stats : bool, default=True
            If True, display statistics on plot
        show_grid : bool, default=True
            If True, show grid lines
        vertical_lines : list or dict, optional
            Vertical lines to add to plot. Can be:
            - List of x positions
            - Dict with {'positions': [], 'colors': [], 'styles': [], 'labels': []}
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        stats : dict
            Statistics about the data
        """
        
        # Convert to numpy array
        data = np.asarray(data).flatten()
        
        # Handle empty data
        if len(data) == 0:
            warnings.warn("Empty data provided, returning empty plot")
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                if title:
                    ax.set_title(title)
                return fig, ax, {}
            else:
                return ax.get_figure(), ax, {}
        
        # Calculate bin edges
        data_min = np.min(data)
        data_max = np.max(data)
        
        # Extend range slightly to include all data
        epsilon = 1e-10  # Small value to ensure max value is included
        data_min = np.floor(data_min / bin_size) * bin_size
        data_max = np.ceil(data_max / bin_size) * bin_size + epsilon
        
        # Create bins
        bins = np.arange(data_min, data_max, bin_size)
        
        # If only one data point or very small range, adjust bins
        if len(bins) < 2:
            bins = np.array([data_min - bin_size/2, data_min + bin_size/2])
        
        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()
        
        # Plot histogram
        n, bin_edges, patches = ax.hist(
            data, bins=bins, density=density, cumulative=cumulative,
            color=color, edgecolor=edgecolor, alpha=alpha,
            align='mid', rwidth=0.8
        )
        
        # Customize plot
        ax.set_xlabel(xlabel)
        
        if ylabel is None:
            if density:
                ylabel = 'Density'
            elif cumulative:
                ylabel = 'Cumulative Frequency'
            else:
                ylabel = 'Frequency'
        
        ax.set_ylabel(ylabel)
        
        if title:
            ax.set_title(title)
        
        # Force integer ticks on x-axis for small bin sizes
        if bin_size >= 1:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add grid
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add vertical lines if specified
        if vertical_lines is not None:
            if isinstance(vertical_lines, dict):
                positions = vertical_lines.get('positions', [])
                colors = vertical_lines.get('colors', ['red'] * len(positions))
                styles = vertical_lines.get('styles', ['--'] * len(positions))
                labels = vertical_lines.get('labels', [''] * len(positions))
                linewidths = vertical_lines.get('linewidths', [1.5] * len(positions))
                alphas = vertical_lines.get('alphas', [0.7] * len(positions))
                
                for pos, color, style, label, lw, alpha_val in zip(
                    positions, colors, styles, labels, linewidths, alphas
                ):
                    ax.axvline(x=pos, color=color, linestyle=style, 
                              linewidth=lw, alpha=alpha_val, label=label)
            
            elif isinstance(vertical_lines, (list, np.ndarray)):
                for pos in vertical_lines:
                    ax.axvline(x=pos, color='red', linestyle='--', 
                              linewidth=1.5, alpha=0.7)
        
        # Calculate statistics
        stats = {
            'n': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': data_min,
            'max': data_max,
            'range': data_max - data_min,
            'bin_size': bin_size,
            'n_bins': len(bins) - 1,
            'max_bin_count': np.max(n) if len(n) > 0 else 0,
            'max_bin_value': bin_edges[np.argmax(n)] if len(n) > 0 else 0,
            'total_count': np.sum(n) if not density else None
        }
        
        # Add statistics text to plot
        if show_stats:
            stats_text = (
                f'n = {stats["n"]:,}\n'
                f'Mean = {stats["mean"]:.3f}\n'
                f'Median = {stats["median"]:.3f}\n'
                f'Std = {stats["std"]:.3f}\n'
                f'Range = [{stats["min"]:.1f}, {stats["max"]:.1f}]\n'
                f'Bin size = {bin_size}'
            )
            
            # Place text in upper right
            ax.text(0.98, 0.98, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9, fontfamily='monospace')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig, ax, stats
    
    
    # Specialized functions for common use cases
    def plot_unit_histogram(self ,data, **kwargs):
        """
        Plot histogram with 1-unit bins.
        
        Parameters:
        -----------
        data : array-like
            Input data values
        **kwargs : additional arguments passed to plot_fine_histogram
        
        Returns:
        --------
        fig, ax, stats
        """
        return self.plot_fine_histogram(data, bin_size=1, **kwargs)
    
    
    def plot_two_unit_histogram(self, data, **kwargs):
        """
        Plot histogram with 2-unit bins.
        
        Parameters:
        -----------
        data : array-like
            Input data values
        **kwargs : additional arguments passed to plot_fine_histogram
        
        Returns:
        --------
        fig, ax, stats
        """
        return self.plot_fine_histogram(data, bin_size=2, **kwargs)
    
    
    def plot_fractional_histogram(self, data, fraction=0.5, **kwargs):
        """
        Plot histogram with fractional bins.
        
        Parameters:
        -----------
        data : array-like
            Input data values
        fraction : float, default=0.5
            Bin size as fraction of unit
        **kwargs : additional arguments passed to plot_fine_histogram
        
        Returns:
        --------
        fig, ax, stats
        """
        return self.plot_fine_histogram(data, bin_size=fraction, **kwargs)
    
    
    # Function to compare multiple datasets with fine bins
    def compare_fine_histograms(self, datasets, labels=None, bin_size=1,
                               colors=None, alpha=0.7,
                               figsize=(12, 8), title=None):
        """
        Compare multiple datasets with fine-bin histograms.
        
        Parameters:
        -----------
        datasets : list of array-like
            Multiple datasets to compare
        labels : list of str, optional
            Labels for each dataset
        bin_size : int or float, default=1
            Bin width
        colors : list of str, optional
            Colors for each dataset
        alpha : float, default=0.7
            Transparency
        figsize : tuple, default=(12, 8)
            Figure size
        title : str, optional
            Overall title
        
        Returns:
        --------
        fig, axes, all_stats
        """
        n_datasets = len(datasets)
        
        if labels is None:
            labels = [f'Dataset {i+1}' for i in range(n_datasets)]
        
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize,
                                gridspec_kw={'height_ratios': [3, 1]})
        ax_hist, ax_stats = axes
        
        all_stats = []
        
        # Plot histograms
        for i, (data, label, color) in enumerate(zip(datasets, labels, colors)):
            data = np.asarray(data).flatten()
            
            # Calculate bin edges (common for all datasets)
            if i == 0:
                all_data = np.concatenate(datasets)
                data_min = np.floor(np.min(all_data) / bin_size) * bin_size
                data_max = np.ceil(np.max(all_data) / bin_size) * bin_size + 1e-10
                bins = np.arange(data_min, data_max, bin_size)
            
            # Plot histogram
            ax_hist.hist(data, bins=bins, alpha=alpha, color=color,
                        edgecolor='black', linewidth=0.5,
                        label=label, density=True)
            
            # Calculate and store statistics
            stats = {
                'label': label,
                'n': len(data),
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
            all_stats.append(stats)
        
        # Customize histogram axis
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        if title:
            ax_hist.set_title(title)
        else:
            ax_hist.set_title(f'Comparison of {n_datasets} datasets (bin size={bin_size})')
        
        # Create statistics table
        if n_datasets <= 10:  # Only create table for reasonable number of datasets
            # Prepare data for table
            table_data = []
            headers = ['Dataset', 'n', 'Mean', 'Median', 'Std', 'Min', 'Max']
            
            for stats in all_stats:
                table_data.append([
                    stats['label'],
                    f"{stats['n']:,}",
                    f"{stats['mean']:.3f}",
                    f"{stats['median']:.3f}",
                    f"{stats['std']:.3f}",
                    f"{stats['min']:.3f}",
                    f"{stats['max']:.3f}"
                ])
            
            # Create table
            ax_stats.axis('tight')
            ax_stats.axis('off')
            table = ax_stats.table(cellText=table_data,
                                  colLabels=headers,
                                  loc='center',
                                  cellLoc='center')
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
        
        plt.tight_layout()
        return fig, axes, all_stats
    
    
    # Example usage and demonstration
    def demonstrate_fine_histograms(self):
        """Demonstrate the fine histogram plotting functions."""
        np.random.seed(42)
        
        print("Fine Histogram Plotting Demonstration")
        print("=" * 50)
        
        # Example 1: Integer data with unit bins
        print("\n1. Integer data with 1-unit bins:")
        int_data = np.random.randint(0, 10, 1000)
        fig1, ax1, stats1 = self.plot_unit_histogram(
            int_data,
            title='Integer Data with 1-unit Bins',
            xlabel='Integer Value',
            color='lightcoral'
        )
        plt.show()
        print(f"Statistics: {stats1}")
        
        # Example 2: Float data with 0.5-unit bins
        print("\n2. Float data with 0.5-unit bins:")
        float_data = np.random.normal(5, 2, 1000)
        fig2, ax2, stats2 = self.plot_fine_histogram(
            float_data,
            bin_size=0.5,
            title='Float Data with 0.5-unit Bins',
            xlabel='Value',
            color='lightblue',
            density=True
        )
        plt.show()
        print(f"Statistics: {stats2}")
        
        # Example 3: Small range data with 2-unit bins
        print("\n3. Small range data with 2-unit bins:")
        small_data = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5])
        fig3, ax3, stats3 = self.plot_two_unit_histogram(
            small_data,
            title='Small Dataset with 2-unit Bins',
            xlabel='Value',
            color='lightgreen',
            show_stats=True,
            vertical_lines=[2.5, 3.5]  # Add reference lines
        )
        plt.show()
        print(f"Statistics: {stats3}")
        
        # Example 4: Compare multiple datasets
        print("\n4. Comparing multiple datasets:")
        dataset1 = np.random.normal(0, 1, 500)
        dataset2 = np.random.normal(2, 1.5, 500)
        dataset3 = np.random.normal(-1, 0.8, 500)
        
        fig4, axes4, stats4 = self.compare_fine_histograms(
            datasets=[dataset1, dataset2, dataset3],
            labels=['Group A', 'Group B', 'Group C'],
            bin_size=0.25,
            title='Comparison of Three Groups'
        )
        plt.show()
        
        # Example 5: Custom vertical lines
        print("\n5. Histogram with custom vertical lines:")
        test_data = np.random.exponential(3, 500)
        
        # Define vertical lines with customization
        vlines = {
            'positions': [2, 5, 8],
            'colors': ['red', 'green', 'blue'],
            'styles': ['--', '-.', ':'],
            'labels': ['Threshold 1', 'Threshold 2', 'Threshold 3'],
            'linewidths': [2, 1.5, 1],
            'alphas': [0.8, 0.6, 0.4]
        }
        
        fig5, ax5, stats5 = self.plot_fine_histogram(
            test_data,
            bin_size=0.5,
            title='Histogram with Custom Vertical Lines',
            vertical_lines=vlines,
            color='gold',
            show_stats=True
        )
        ax5.legend()
        plt.show()
        
        print("\n" + "=" * 50)
        print("Demonstration complete!")
        
        return [stats1, stats2, stats3, stats4, stats5]
    
    
    # Quick plotting function for common cases
    def quick_fine_hist(self, data, bin_width=1, **kwargs):
        """
        Quick plotting with sensible defaults.
        
        Parameters:
        -----------
        data : array-like
            Data to plot
        bin_width : int or float, default=1
            Bin width
        **kwargs : additional styling arguments
        
        Returns:
        --------
        fig, ax
        """
        fig, ax, _ = self.plot_fine_histogram(
            data,
            bin_size=bin_width,
            color=kwargs.get('color', 'steelblue'),
            edgecolor=kwargs.get('edgecolor', 'white'),
            title=kwargs.get('title', f'Histogram (bin width={bin_width})'),
            show_stats=kwargs.get('show_stats', True),
            show_grid=kwargs.get('show_grid', True)
        )
        return fig, ax
    
    
    # Function to analyze and print bin counts
    def analyze_bin_counts(self, data, bin_size=1):
        """
        Analyze and print detailed bin counts for fine histograms.
        
        Parameters:
        -----------
        data : array-like
            Input data
        bin_size : int or float, default=1
            Bin width
        
        Returns:
        --------
        bin_centers : array
            Center of each bin
        counts : array
            Count in each bin
        percentage : array
            Percentage in each bin
        """
        data = np.asarray(data).flatten()
        
        # Calculate bins
        data_min = np.floor(np.min(data) / bin_size) * bin_size
        data_max = np.ceil(np.max(data) / bin_size) * bin_size + 1e-10
        bins = np.arange(data_min, data_max, bin_size)
        
        # Calculate counts
        counts, bin_edges = np.histogram(data, bins=bins)
        
        # Calculate bin centers and percentages
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        percentages = (counts / len(data)) * 100
        
        # Print detailed table
        print(f"\nDetailed Bin Analysis (bin size = {bin_size})")
        print("=" * 60)
        print(f"{'Bin Range':<20} {'Center':<10} {'Count':<10} {'%':<10} {'Cumulative %':<12}")
        print("-" * 60)
        
        cumulative = 0
        for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            cumulative += percentages[i]
            print(f"[{left:.2f}, {right:.2f})  "
                  f"{bin_centers[i]:<10.2f} "
                  f"{counts[i]:<10} "
                  f"{percentages[i]:<10.2f} "
                  f"{cumulative:<12.2f}")
        
        print("=" * 60)
        print(f"Total: {len(data):,} observations")
        print(f"Unique bins: {len(bin_centers)}")
        print(f"Most frequent bin: {bin_centers[np.argmax(counts)]:.2f} "
              f"({np.max(counts)} observations, {np.max(percentages):.1f}%)")
        
        return bin_centers, counts, percentages

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

def plotCons():
    pass

def plotconsIndexes():
    pass

def plothist():
    pass