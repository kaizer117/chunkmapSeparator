"""Unit tests for :mod:`chunkmap.cvutils`."""

import numpy as np
import pytest

from chunkmap import cvutils


def test_con_reshaper_flattens_one_contour():
    contour = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    points = cvutils.conReshaper(contour)
    assert points.shape == (3, 2)
    assert np.array_equal(points, np.array([[1, 2], [3, 4], [5, 6]]))


def test_contours_rehsaper_flattens_a_list_of_contours():
    contours = [np.array([[[0, 0]], [[1, 1]]]),
                np.array([[[2, 2]], [[3, 3]], [[4, 4]]])]
    points = cvutils.contoursRehsaper(contours)
    assert [len(item) for item in points] == [2, 3]
    assert all(item.shape[1] == 2 for item in points)


def test_histimg_returns_alpha_and_a_256_bin_histogram():
    image = np.zeros((4, 5, 4), np.uint8)
    image[:, :, 3] = 7
    opacity, histogram = cvutils.histimg(image)
    assert opacity.shape == (4, 5)
    assert histogram.shape == (256,)
    assert int(histogram[7]) == 20


def test_histimg_rejects_rgb():
    with pytest.raises(ValueError, match="4 channel"):
        cvutils.histimg(np.zeros((5, 5, 3), np.uint8))


def test_histimg_rejects_greyscale():
    with pytest.raises(ValueError, match="4 channel"):
        cvutils.histimg(np.zeros((5, 5), np.uint8))


def test_histimg_rejects_none():
    with pytest.raises(ValueError, match="empty"):
        cvutils.histimg(None)
