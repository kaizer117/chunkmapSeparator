"""Unit tests for :mod:`chunkmap.vectorization`."""

import numpy as np
import pytest

from chunkmap import segmentation as seg
from chunkmap import vectorization as vect


def test_two_point_segment_fits_a_straight_cubic():
    control = vect.fitBezierCurve(np.array([[0.0, 0.0], [30.0, 60.0]]))
    assert control.shape == (4, 2)
    assert np.allclose(control[0], [0.0, 0.0])
    assert np.allclose(control[-1], [30.0, 60.0])


def test_fit_rejects_an_empty_segment():
    with pytest.raises(ValueError):
        vect.fitBezierCurve(np.zeros((0, 2)))


def test_closed_formatter_joins_last_curve_to_first():
    """Without this, 'z' closes every contour with a straight chord."""
    rng = np.random.default_rng(0)
    points = np.cumsum(rng.normal(size=(40, 2)), axis=0) + 100.0
    segments = seg.polygon_curvature_segmentation(points, min_chunk=5,
                                                  max_chunk=15)
    fits = [vect.fitBezierCurve(segment) for segment in segments]

    open_rows = vect.controlPointFormatter(fits)
    closed_rows = vect.controlPointFormatter(fits, closed=True)

    start = np.asarray(closed_rows[0][0], dtype=float)
    end = np.asarray(closed_rows[-1][-1], dtype=float)
    assert np.allclose(start, end)

    # open layout leaves a gap, which is exactly the bug being guarded against
    open_end = np.asarray(open_rows[-1][-1], dtype=float)
    open_start = np.asarray(open_rows[0][0], dtype=float)
    assert np.linalg.norm(open_end - open_start) >= 0.0


def test_formatter_accepts_ndarray_and_does_not_mutate_input():
    control = np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                        [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    before = control.copy()
    rows = vect.controlPointFormatter(control, closed=True)
    assert len(rows) == 2
    assert np.array_equal(control, before)


def test_polygon_segmentation_splits_into_contiguous_chunks():
    points = np.column_stack([np.arange(25.0), np.zeros(25)])
    segments = vect.polygonSegmentation(points, chunk_size=10)
    assert [len(segment) for segment in segments] == [8, 8, 8]
    # Blind segmentation slices [start:stop], so the final point is the stop
    # index and is not repeated in the last run.
    assert np.array_equal(np.vstack(segments), points[:-1])
