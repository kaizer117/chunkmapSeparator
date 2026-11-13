"""Unit tests for the config, case planner and overlay in
:mod:`chunkmap.orchestration`.

The batch run itself is covered in
``tests/integration/test_orchestration_run.py``.
"""

import json
import os

import numpy as np
import pytest

from chunkmap import orchestration as orch


def write_json_config(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------

def test_yaml_subset_reads_the_sample_shape():
    text = """
# leading comment
name: "demo"                  # trailing comment
seed: 42
space: outputs/demo
generation:
  count: 2
  width: 320
  height: 240
  islands: [1, 3]
  subdivisions: {min: 0, max: 2}
  filled: false
  images:
    - name: exact
      seed: 7
      islands: 2
overlay:
  color: "#ff3355"
  background: null
  fill: true
report:
  tolerance: 1
"""
    data = orch.parse_yaml_subset(text)
    assert data["name"] == "demo"
    assert data["seed"] == 42
    assert data["generation"]["islands"] == [1, 3]
    assert data["generation"]["subdivisions"] == {"min": 0, "max": 2}
    assert data["generation"]["filled"] is False
    assert data["generation"]["images"] == [
        {"name": "exact", "seed": 7, "islands": 2}
    ]
    assert data["overlay"]["color"] == "#ff3355"
    assert data["overlay"]["background"] is None
    assert data["report"]["tolerance"] == 1


def test_yaml_subset_rejects_tabs_and_bad_indentation():
    with pytest.raises(orch.ConfigError):
        orch.parse_yaml_subset("generation:\n\tcount: 2\n")
    with pytest.raises(orch.ConfigError):
        orch.parse_yaml_subset("generation:\n    count: 2\n  width: 3\n")


def test_yaml_subset_errors_name_the_line():
    with pytest.raises(orch.ConfigError) as error:
        orch.parse_yaml_subset("generation:\n  count: 2\n    width: 3\n")
    assert "line" in str(error.value).lower()


def test_yaml_subset_plain_scalar_can_contain_a_comma():
    data = orch.parse_yaml_subset("name: islands, with subdivisions\nseed: 1\n")
    assert data["name"] == "islands, with subdivisions"


def test_load_config_accepts_yaml_and_json_equivalently(work):
    yaml_path = os.path.join(work, "config.yaml")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(
            "name: equivalence\n"
            "seed: 5\n"
            "space: %s\n"
            "generation:\n"
            "  count: 2\n"
            "  width: 160\n"
            "  height: 128\n"
            "  islands: [1, 2]\n"
            "  subdivisions: [0, 1]\n"
            "vectorization:\n"
            "  alpha_threshold: 128\n"
            "overlay:\n"
            "  fill: true\n" % os.path.join(work, "space").replace("\\", "/")
        )
    yaml_config = orch.load_config(yaml_path)

    json_path = write_json_config(
        os.path.join(work, "config.json"), yaml_config.snapshot())
    json_config = orch.load_config(json_path)
    assert yaml_config.snapshot() == json_config.snapshot()

    # Windows editors and PowerShell often prepend a BOM; it must not be data.
    bom_path = os.path.join(work, "bom.json")
    with open(bom_path, "w", encoding="utf-8-sig") as handle:
        json.dump(yaml_config.snapshot(), handle)
    assert orch.load_config(bom_path).snapshot() == yaml_config.snapshot()

    assert yaml_config.vectorization.curve == "bezier"
    assert yaml_config.overlay.enabled is True
    assert yaml_config.overlay.color == "#ff3355"
    assert yaml_config.report.tolerance == 1


def test_config_rejects_unknown_keys_and_bad_ranges():
    with pytest.raises(orch.ConfigError):
        orch.normalise_config({"generation": {"islnads": 2}})
    with pytest.raises(orch.ConfigError):
        orch.normalise_config({"generation": {"islands": [4, 1]}})
    with pytest.raises(orch.ConfigError):
        orch.normalise_config({"vectorization": {"alpha_threshold": 300}})
    with pytest.raises(orch.ConfigError):
        orch.normalise_config({"overlay": {"color": "red"}})
    with pytest.raises(orch.ConfigError):
        orch.normalise_config({"generation": {"islands": [1, 99]}})


def test_explicit_images_override_generation_defaults():
    config = orch.normalise_config({
        "seed": 100,
        "generation": {
            "width": 320,
            "height": 240,
            "images": [
                {"name": "one", "seed": 7, "islands": 1},
                {"name": "two", "islands": 3, "subdivisions": 2,
                 "width": 200, "filled": True},
            ],
        },
    })
    cases = orch.build_cases(config)
    assert [case.name for case in cases] == ["one", "two"]
    assert cases[0].seed == 7
    assert cases[1].seed == 101
    assert cases[1].width == 200 and cases[1].height == 240
    assert cases[1].islands == 3 and cases[1].subdivisions == 2
    assert cases[1].filled is True


def test_build_cases_is_deterministic_and_covers_the_ranges():
    config = orch.normalise_config({
        "seed": 2026,
        "generation": {
            "count": 6,
            "width": 160,
            "height": 128,
            "islands": [1, 3],
            "subdivisions": [0, 2],
            "seed_step": 5,
        },
    })
    first = orch.build_cases(config)
    second = orch.build_cases(config)
    assert [case.to_dict() for case in first] == [case.to_dict() for case in second]
    assert [case.seed for case in first] == [2026, 2031, 2036, 2041, 2046, 2051]
    assert {case.islands for case in first} == {1, 2, 3}
    assert {case.subdivisions for case in first} == {0, 1, 2}

    selected = orch.select_cases(first, "1,3-4")
    assert [case.index for case in selected] == [1, 3, 4]
    with pytest.raises(orch.ConfigError):
        orch.select_cases(first, "9")


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------

def test_overlay_draws_the_vector_colour_outside_transparent_ink():
    base = np.zeros((40, 40, 4), dtype=np.uint8)
    loops = [np.array([[5.0, 5.0], [34.0, 5.0], [34.0, 34.0], [5.0, 34.0]])]
    config = orch.OverlayConfig(
        enabled=True, color="#ff3355", width=2.0, opacity=1.0,
        fill=False, fill_color="#ff3355", fill_opacity=0.0, background=None)
    overlay = orch.render_overlay(base, loops, config)

    assert overlay.shape == (40, 40, 4)
    assert overlay[:, :, 3].max() == 255
    assert overlay[20, 20, 3] == 0                 # centre stays transparent
    pixel = overlay[5, 15].astype(int)             # top edge of the loop
    blue, green, red, alpha = pixel
    assert alpha > 200
    assert red > 200 and blue > 60 and green < 160


def test_overlay_background_makes_the_whole_canvas_opaque():
    base = np.zeros((20, 20, 4), dtype=np.uint8)
    loops = [np.array([[4.0, 4.0], [16.0, 4.0], [16.0, 16.0], [4.0, 16.0]])]
    config = orch.OverlayConfig(
        enabled=True, color="#ff3355", width=1.0, opacity=1.0,
        fill=False, fill_color="#ff3355", fill_opacity=0.0, background="#000000")
    overlay = orch.render_overlay(base, loops, config)
    assert np.all(overlay[:, :, 3] == 255)
