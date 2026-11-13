"""Integration tests for the orchestrated island -> SVG -> overlay -> report run."""

import json
import os

import cv2 as cv
import numpy as np

from chunkmap import accuracy
from chunkmap import cli


def write_json_config(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _small_config(work, count=2, **report):
    return {
        "name": "e2e",
        "seed": 5,
        "space": os.path.join(work, "space"),
        "generation": {
            "count": count,
            "width": 160,
            "height": 128,
            "islands": [1, 2],
            "subdivisions": [1, 2],
            "seed_step": 3,
        },
        "vectorization": {
            "alpha_threshold": 128,
            "min_chunk": 5,
            "max_chunk": 15,
        },
        "overlay": {"fill": True},
        "report": dict({"show": "summary", "csv": True}, **report),
    }


def test_orchestration_end_to_end_through_the_cli(work):
    config_path = write_json_config(os.path.join(work, "config.json"),
                                    _small_config(work))
    space = os.path.join(work, "space")

    code = cli.main(["orchestrate", "--config", config_path])
    assert code == 0

    for folder in ("islands", "regions", "svg", "overlay", "reports"):
        assert os.path.isdir(os.path.join(space, folder))

    for index in (1, 2):
        name = "island_%04d" % index
        png = os.path.join(space, "islands", name + ".png")
        region_png = os.path.join(space, "regions", name + ".png")
        svg = os.path.join(space, "svg", name + ".svg")
        overlay_path = os.path.join(space, "overlay", name + ".png")
        report_path = os.path.join(space, "reports", name + ".json")
        assert os.path.isfile(png)
        assert os.path.isfile(region_png)
        assert os.path.isfile(svg)
        assert os.path.isfile(overlay_path)
        assert os.path.isfile(report_path)

        with open(svg, encoding="utf-8") as handle:
            svg_text = handle.read()
        document = accuracy.parse_svg(svg_text)
        assert document["width"] == 160 and document["height"] == 128
        assert document["paths"] > 0 and document["loops"]
        assert "region-" in svg_text

        overlay = cv.imread(overlay_path, cv.IMREAD_UNCHANGED)
        assert overlay.shape == (128, 160, 4)
        red_pixels = ((overlay[:, :, 2] > 180) & (overlay[:, :, 1] < 160)
                      & (overlay[:, :, 0] > 40) & (overlay[:, :, 3] > 0))
        assert np.count_nonzero(red_pixels) > 0

    summary_path = os.path.join(space, "reports", "summary.json")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["aggregate"]["processed"] == 2
    assert summary["aggregate"]["failed"] == 0
    assert summary["aggregate"]["ok"] == 2
    assert len(summary["images"]) == 2
    image = summary["images"][0]
    assert image["generation"]["png"].startswith("islands/")
    assert image["generation"]["region_png"].startswith("regions/")
    assert image["vectorization"]["svg"].startswith("svg/")
    assert image["vectorization"]["region_map"].startswith("regions/")
    assert image["vectorization"]["mask_mode"] == "regions"
    assert image["vectorization"]["region_count"] >= 1
    assert image["overlay"]["png"].startswith("overlay/")
    assert 0.0 <= image["accuracy"]["iou"] <= 1.0
    assert image["accuracy"]["within_1px_pct"] > 50.0

    with open(os.path.join(space, "reports", "summary.csv"),
              encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    assert len(lines) == 3


def test_orchestration_can_still_use_the_legacy_ink_band_mode(work):
    payload = _small_config(work, count=1)
    payload["vectorization"]["region_mode"] = False
    config_path = write_json_config(os.path.join(work, "config.json"), payload)
    code = cli.main(["orchestrate", "--config", config_path, "--quiet"])
    assert code == 0

    svg_path = os.path.join(work, "space", "svg", "island_0001.svg")
    with open(svg_path, encoding="utf-8") as handle:
        text = handle.read()
    assert "outline" in text
    assert "region-" not in text
    assert not os.path.exists(os.path.join(work, "space", "regions",
                                           "island_0001.png"))


def test_threshold_failure_is_reported_and_sets_the_exit_code(work):
    payload = _small_config(work, count=1, min_iou=0.9999)
    config_path = write_json_config(os.path.join(work, "config.json"), payload)
    code = cli.main(["orchestrate", "--config", config_path, "--quiet"])
    assert code == 1

    summary_path = os.path.join(work, "space", "reports", "summary.json")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["aggregate"]["below_threshold"] == 1
    assert summary["images"][0]["status"] == "below"


def test_vectorization_failure_is_recorded_and_does_not_stop_the_batch(work):
    payload = _small_config(work, count=2)
    # Exercise the legacy ink-band path so alpha > 255 still means "no ink".
    payload["vectorization"]["region_mode"] = False
    # alpha > 255 is never true, so the converter finds no ink on every image.
    payload["vectorization"]["alpha_threshold"] = 255
    config_path = write_json_config(os.path.join(work, "config.json"), payload)

    code = cli.main(["orchestrate", "--config", config_path, "--quiet"])
    assert code == 1

    summary_path = os.path.join(work, "space", "reports", "summary.json")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["aggregate"]["processed"] == 2
    assert summary["aggregate"]["failed"] == 2
    assert all(image["status"] == "failed" for image in summary["images"])
    assert all(image["stage"] == "vectorization" for image in summary["images"])
    assert all(image["error"] for image in summary["images"])


def test_only_selects_specific_cases(work):
    config_path = write_json_config(os.path.join(work, "config.json"),
                                    _small_config(work, count=3))
    code = cli.main(["orchestrate", "--config", config_path, "--only", "2",
                     "--quiet"])
    assert code == 0

    summary_path = os.path.join(work, "space", "reports", "summary.json")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert [image["index"] for image in summary["images"]] == [2]
    assert summary["aggregate"]["processed"] == 1
