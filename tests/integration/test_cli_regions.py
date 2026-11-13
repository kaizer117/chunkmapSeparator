"""Integration tests for region-aware conversion through the CLI."""

import json
import os

import cv2 as cv
import numpy as np

from chunkmap import accuracy
from chunkmap import cli
from chunkmap import island_generator as islands


def test_cli_region_mode_end_to_end(work):
    result = islands.generate_island(
        islands.IslandParams(192, 160, seed=3, islands=2, subdivisions=2))
    image_path = os.path.join(work, "island.png")
    label_path = os.path.join(work, "island-labels.png")
    svg_path = os.path.join(work, "island-regions.svg")
    report_path = os.path.join(work, "island-regions.json")
    assert cv.imwrite(image_path, result.image)
    assert cv.imwrite(label_path, result.labels.astype(np.uint16))

    code = cli.main([image_path, "--regions", label_path, "-o", svg_path,
                     "--report-json", report_path, "--quiet"])
    assert code == 0
    assert os.path.isfile(svg_path)

    with open(svg_path, encoding="utf-8") as handle:
        document = accuracy.parse_svg(handle.read())
    assert document["paths"] == len(np.unique(result.labels)) - 1
    assert document["width"] == 192 and document["height"] == 160

    with open(report_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["regions"]["region_count"] == int(result.region_count)
    assert payload["regions"]["shared_chain_count"] > 0
    assert payload["accuracy"]["coverage"]["tolerant"]["iou"] > 0.99


def test_cli_rejects_a_label_map_of_the_wrong_size(work):
    result = islands.generate_island(islands.IslandParams(160, 128, seed=1))
    image_path = os.path.join(work, "island.png")
    label_path = os.path.join(work, "labels.png")
    assert cv.imwrite(image_path, result.image)
    assert cv.imwrite(label_path, result.labels[:-1].astype(np.uint16))
    code = cli.main([image_path, "--regions", label_path,
                     "-o", os.path.join(work, "out.svg"), "--quiet"])
    assert code == 2
