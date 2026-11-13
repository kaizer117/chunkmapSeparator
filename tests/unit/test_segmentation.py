"""Unit tests for :mod:`chunkmap.segmentation`."""

import numpy as np

from chunkmap import segmentation as seg


def test_segmentation_respects_max_chunk():
    """max_chunk used to be ignored whenever curvature splits existed."""
    x = np.linspace(0.0, 200.0, 400)
    points = np.column_stack([x, 80.0 + 40.0 * np.sin(x / 9.0)])
    segments = seg.polygon_curvature_segmentation(points, min_chunk=5,
                                                  max_chunk=15)
    lengths = [len(segment) for segment in segments]
    assert max(lengths) <= 16
    assert min(lengths) >= 2
    assert sum(len(segment) - 1 for segment in segments) >= len(points) - len(segments)


def test_equidistant_splits_stays_in_range():
    for n_points in (64, 65, 100, 101):
        splits = seg.equidistant_splits(n_points, 30)
        assert splits[0] == 0
        assert splits[-1] <= n_points - 1
        assert splits == sorted(set(splits))


def test_curvature_segmentation_covers_a_closed_circle():
    theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    points = np.column_stack([60.0 + 40.0 * np.cos(theta),
                              60.0 + 40.0 * np.sin(theta)])
    segments = seg.polygon_curvature_segmentation(points, min_chunk=5,
                                                  max_chunk=15)
    assert segments
    assert all(len(segment) >= 2 for segment in segments)
    # every point except the first of each run is represented
    assert sum(len(segment) for segment in segments) >= len(points)
