"""Unit tests for the shared-border engine, :mod:`chunkmap.region_vectorization`.

The CLI wiring is covered separately in
``tests/integration/test_cli_regions.py``.
"""

import os
import xml.etree.ElementTree as ElementTree

import cv2 as cv
import numpy as np

from chunkmap import accuracy
from chunkmap import island_generator as islands
from chunkmap import region_vectorization as regions


def two_region_labels():
    labels = np.zeros((6, 8), dtype=np.int32)
    labels[1:5, 1:4] = 1
    labels[1:5, 4:7] = 2
    return labels


def test_two_regions_share_one_exact_border():
    layout = regions.build_layout(two_region_labels(), max_chunk=10)
    assert layout.region_count == 2
    assert layout.chain_count == 3
    assert layout.shared_chain_count == 1
    assert layout.coastline_chain_count == 2
    assert layout.path_count == 2

    shared = [chain for chain in layout.chains if chain.shared]
    assert len(shared) == 1
    border = shared[0].cubics
    assert len(border) == 1

    path_cubics = [[cubic for loop in path.loops for cubic in loop]
                   for path in layout.paths]
    for cubic in border:
        forward = [any(np.allclose(cubic, item) for item in items)
                   for items in path_cubics]
        reverse = [any(np.allclose(cubic[::-1], item) for item in items)
                   for items in path_cubics]
        assert any(forward) and any(reverse)
        if forward[0]:
            assert reverse[1]
        else:
            assert forward[1]


def test_region_paths_close_exactly():
    layout = regions.build_layout(two_region_labels(), max_chunk=10)
    for path in layout.paths:
        for loop in path.loops:
            assert np.allclose(loop[0][0], loop[-1][-1])


def test_three_and_four_region_junctions_are_decomposed():
    three = np.zeros((7, 9), dtype=np.int32)
    three[1:4, 1:4] = 1
    three[1:4, 4:8] = 2
    three[4:6, 1:8] = 3
    layout = regions.build_layout(three, max_chunk=10)
    assert layout.region_count == 3
    assert layout.path_count == 3
    assert layout.shared_chain_count == 3
    for path in layout.paths:
        assert len(path.loops) == 1
        assert np.allclose(path.loops[0][0][0], path.loops[0][-1][-1])

    four = np.zeros((8, 8), dtype=np.int32)
    four[1:4, 1:4] = 1
    four[1:4, 4:7] = 2
    four[4:7, 1:4] = 3
    four[4:7, 4:7] = 4
    cross = regions.build_layout(four, max_chunk=10)
    assert cross.region_count == 4
    assert cross.path_count == 4
    assert cross.shared_chain_count == 4
    for path in cross.paths:
        assert len(path.loops) == 1


def test_a_water_hole_is_a_second_loop():
    labels = np.zeros((9, 9), dtype=np.int32)
    labels[1:8, 1:8] = 1
    labels[3:6, 3:6] = 0
    layout = regions.build_layout(labels, max_chunk=10)
    assert layout.region_count == 1
    assert layout.path_count == 1
    assert len(layout.paths[0].loops) == 2
    assert len(layout.exact_loops) == 2
    svg = regions.build_svg(layout, colors=["#123456"])
    assert 'fill-rule="evenodd"' in svg
    assert svg.count("M ") == 2


def test_region_fills_reconstruct_the_land_mask():
    result = islands.generate_island(
        islands.IslandParams(192, 160, seed=11, islands=3, subdivisions=3))
    layout = regions.build_layout(result.labels, max_chunk=10)
    mask = (result.labels > 0).astype(np.uint8) * 255
    loops = regions.layout_loops(layout)
    reconstruction = accuracy.loops_to_mask(loops, mask.shape)
    scores = accuracy.mask_scores(mask, reconstruction, tolerance=1)
    assert scores["tolerant"]["iou"] > 0.99
    # Region fills are exact at the geometry level; the 1 px raster comparison
    # can still see the inclusive boundary convention of cv.fillPoly.
    assert abs(scores["candidate_px"] - scores["reference_px"]) < 0.05 * mask.size
    assert layout.region_count == int(result.region_count)


def test_exact_region_loops_match_the_layout_loop_order():
    result = islands.generate_island(
        islands.IslandParams(160, 128, seed=4, islands=2, subdivisions=2))
    layout = regions.build_layout(result.labels, max_chunk=10)
    exact = regions.exact_region_loops(result.labels)
    assert len(exact) == sum(len(path.loops) for path in layout.paths)
    for loop in exact:
        assert np.allclose(loop[0], loop[-1])


def test_svg_fill_has_one_path_per_region_and_no_visible_stroke():
    layout = regions.build_layout(two_region_labels(), max_chunk=10)
    svg = regions.build_svg(layout, colors=["#112233", "#445566"],
                            style="fill", seam_width=0.0)
    root = ElementTree.fromstring(svg)
    namespace = "{http://www.w3.org/2000/svg}"
    paths = root.findall(".//" + namespace + "path")
    assert len(paths) == 2
    assert all(path.get("fill-rule") == "evenodd" for path in paths)
    assert "stroke:none" in svg
    assert "#112233" in svg and "#445566" in svg


def test_boundary_style_is_one_path_of_all_chains():
    layout = regions.build_layout(two_region_labels(), max_chunk=10)
    svg = regions.build_svg(layout, style="boundary", stroke="#010203",
                            stroke_width=2.0)
    root = ElementTree.fromstring(svg)
    namespace = "{http://www.w3.org/2000/svg}"
    paths = root.findall(".//" + namespace + "path")
    assert len(paths) == 1
    assert "fill:none" in svg
    assert "#010203" in svg


def test_label_map_png_round_trip(work):
    labels = two_region_labels().astype(np.uint16)
    path = os.path.join(work, "labels.png")
    assert cv.imwrite(path, labels)
    loaded = regions.load_label_map(path)
    assert np.array_equal(loaded, labels.astype(np.int32))
