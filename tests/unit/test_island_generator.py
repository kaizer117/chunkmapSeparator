"""Unit tests for :mod:`chunkmap.island_generator`.

The ``generate`` subcommand itself is covered in
``tests/integration/test_cli_generate.py``.
"""

import numpy as np

from chunkmap import island_generator as islands


def test_generate_island_has_the_requested_size_and_coverage():
    result = islands.generate_island(islands.IslandParams(160, 128, seed=123))

    assert result.image.shape == (128, 160, 4)
    assert result.image.dtype == np.uint8
    assert 0.24 <= result.coverage <= 0.36
    assert result.image[:, :, 3].max() == 255
    assert result.image[:, :, 3].min() == 0
    assert result.island_count == 1
    assert result.region_count == 1
    assert result.stroke_px >= 2.0
    x0, y0, x1, y1 = result.bbox
    assert x0 >= result.margin_px
    assert y0 >= result.margin_px
    assert x1 < 160 - result.margin_px
    assert y1 < 128 - result.margin_px


def test_default_is_an_outline_not_a_fill():
    result = islands.generate_island(islands.IslandParams(160, 128, seed=5))
    center_y, center_x = 64, 80
    assert result.image[center_y, center_x, 3] == 0
    assert result.image[:, :, 3].max() == 255
    assert result.mask[center_y, center_x] == 255


def test_filled_mode_is_opaque_inside():
    result = islands.generate_island(
        islands.IslandParams(160, 128, seed=5, filled=True))
    center_y, center_x = 64, 80
    assert result.image[center_y, center_x, 3] == 255


def test_multiple_islands_are_placed_separately():
    result = islands.generate_island(
        islands.IslandParams(192, 160, seed=9, islands=3, coverage=0.28))
    assert result.island_count >= 2
    assert len(result.island_areas) >= 2
    assert min(result.island_areas) > 100


def test_subdivisions_are_area_aware():
    result = islands.generate_island(
        islands.IslandParams(192, 160, seed=4, subdivisions=3))
    assert result.region_count >= 2
    assert len(result.region_areas) == result.region_count
    assert min(result.region_areas) >= 0.005 * min(192, 160) ** 2
    mean_area = sum(result.region_areas) / len(result.region_areas)
    assert max(result.region_areas) <= 2.2 * mean_area


def test_same_seed_is_deterministic():
    first = islands.generate_island(islands.IslandParams(128, 128, seed=7))
    second = islands.generate_island(islands.IslandParams(128, 128, seed=7))
    assert np.array_equal(first.mask, second.mask)


def test_different_seeds_produce_different_shapes():
    first = islands.generate_island(islands.IslandParams(128, 128, seed=7))
    second = islands.generate_island(islands.IslandParams(128, 128, seed=8))
    intersection = np.count_nonzero((first.mask > 0) & (second.mask > 0))
    union = np.count_nonzero((first.mask > 0) | (second.mask > 0))
    assert intersection / max(union, 1) < 0.95
