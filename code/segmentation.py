import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.spatial.distance import pdist, squareform
import warnings

def polygon_curvature_segmentation(data, curvature_threshold=None, min_chunk=5, max_chunk=30, smooth_sigma=2):
    """
    Curvature-aware segmentation of polygon contours.
    
    Parameters:
    -----------
    data : ndarray of shape (n_points, 2)
        Polygon vertices in order (closed contour)
    curvature_threshold : float, optional
        Threshold for high curvature detection (percentile-based)
    min_chunk : int
        Minimum segment length
    max_chunk : int
        Maximum segment length
    smooth_sigma : float
        Gaussian smoothing sigma for curvature calculation
    
    Returns:
    --------
    list : Segmented polygon segments
    """
    
    # Ensure data is numpy array
    data = np.asarray(data)
    
    # 1. Calculate curvature along the contour
    curvatures = compute_curvature(data, smooth_sigma)
    
    # 2. Identify high curvature points (segment boundaries)
    if curvature_threshold is None:
        curvature_threshold = np.percentile(curvatures, 75)
    
    high_curvature_points = np.where(curvatures > curvature_threshold)[0]
    
    # 3. Adaptive segmentation based on curvature
    segments = adaptive_segmentation(data, curvatures, high_curvature_points, min_chunk, max_chunk)
    
    return segments

def compute_curvature(points, sigma=2):
    """
    Compute curvature along a closed polygon contour.
    Uses Gaussian smoothing for noise robustness.
    """
    # Make closed loop for continuous derivatives
    closed_loop = np.vstack([points, points[:3]])
    
    # Smooth the contour
    x = gaussian_filter1d(closed_loop[:, 0], sigma)
    y = gaussian_filter1d(closed_loop[:, 1], sigma)
    
    # First derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    denominator = (dx**2 + dy**2)**1.5
    denominator[denominator == 0] = 1e-6  # Avoid division by zero
    
    curvature = np.abs(dx * ddy - dy * ddx) / denominator
    
    # Return only original points (not padding)
    return curvature[:len(points)]

def adaptive_segmentation(points, curvatures, high_curvature_indices, min_chunk, max_chunk):
    """
    Adaptive segmentation based on curvature distribution.
    """
    n_points = len(points)
    
    if n_points <= max_chunk:
        return [points]
    
    # Sort high curvature points
    split_points = sorted(high_curvature_indices)
    
    # Remove duplicate or too close split points
    filtered_splits = []
    for pt in split_points:
        if not filtered_splits or pt - filtered_splits[-1] >= min_chunk:
            filtered_splits.append(pt)
    
    # Ensure we have split points
    if len(filtered_splits) < 2:
        # Fallback to equidistant segmentation
        filtered_splits = equidistant_splits(n_points, max_chunk)
    
    # Add start and end points
    if filtered_splits[0] != 0:
        filtered_splits.insert(0, 0)
    if filtered_splits[-1] != n_points - 1:
        filtered_splits.append(n_points - 1)
    
    # Merge very small segments
    filtered_splits = merge_small_segments(filtered_splits, min_chunk)
    
    # Create segments
    segments = []
    for i in range(len(filtered_splits) - 1):
        start = filtered_splits[i]
        end = filtered_splits[i + 1] + 1  # Include end point
        segments.append(points[start:end])
    
    return segments

def equidistant_splits(n_points, max_chunk):
    """Generate equidistant split points."""
    n_chunks = max(2, n_points // max_chunk + 1)
    chunk_size = n_points // n_chunks
    return [i * chunk_size for i in range(n_chunks + 1)]

def merge_small_segments(split_points, min_chunk):
    """Merge segments smaller than min_chunk."""
    if len(split_points) <= 2:
        return split_points
    
    merged = [split_points[0]]
    for i in range(1, len(split_points) - 1):
        current_gap = split_points[i] - merged[-1]
        next_gap = split_points[i + 1] - split_points[i]
        
        if current_gap >= min_chunk and next_gap >= min_chunk:
            merged.append(split_points[i])
    
    merged.append(split_points[-1])
    return merged

# Alternative approach: Multi-scale curvature segmentation
def multiscale_curvature_segmentation(data, scales=[1, 2, 4], max_segments=50):
    """
    Multi-scale curvature analysis for robust segmentation.
    """
    all_segmentations = []
    
    for scale in scales:
        curvatures = compute_curvature(data, sigma=scale)
        local_maxima = argrelextrema(curvatures, np.greater)[0]
        all_segmentations.append(local_maxima)
    
    # Consensus-based segmentation
    consensus_splits = find_consensus_splits(all_segmentations, len(data), max_segments)
    segments = create_segments_from_splits(data, consensus_splits)
    
    return segments

def find_consensus_splits(all_splits, n_points, max_segments):
    """Find split points that appear consistently across scales."""
    vote_counts = np.zeros(n_points)
    
    for splits in all_splits:
        vote_counts[splits] += 1
    
    # Keep points with highest votes
    threshold = len(all_splits) / 2  # Majority voting
    consensus = np.where(vote_counts >= threshold)[0]
    
    # Limit number of segments
    if len(consensus) > max_segments - 1:
        # Keep the strongest ones
        consensus = consensus[np.argsort(vote_counts[consensus])[-max_segments + 1:]]
        consensus = sorted(consensus)
    
    return [0] + consensus.tolist() + [n_points - 1]

def create_segments_from_splits(points, splits):
    """Create polygon segments from split indices."""
    segments = []
    for i in range(len(splits) - 1):
        start = splits[i]
        end = splits[i + 1] + 1
        segments.append(points[start:end])
    return segments