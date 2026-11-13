"""Integration tests for the ``generate`` subcommand."""

import os

import cv2 as cv

from chunkmap import cli


def test_cli_generate_writes_numbered_pngs(work):
    code = cli.main(["generate", work, "2", "160", "128", "--seed", "42",
                     "--quiet"])
    assert code == 0

    files = sorted(name for name in os.listdir(work) if name.endswith(".png"))
    assert files == ["island_0001.png", "island_0002.png"]
    for name in files:
        image = cv.imread(os.path.join(work, name), cv.IMREAD_UNCHANGED)
        assert image is not None
        assert image.shape == (128, 160, 4)
        assert image[:, :, 3].max() == 255


def test_cli_generate_single_png_path(work):
    target = os.path.join(work, "one_island.png")
    code = cli.main(["generate", target, "1", "160", "128", "--seed", "1",
                     "--quiet"])
    assert code == 0
    assert os.path.isfile(target)
    image = cv.imread(target, cv.IMREAD_UNCHANGED)
    assert image is not None
    assert image.shape == (128, 160, 4)


def test_cli_generate_multiple_islands_and_subdivisions(work):
    target = os.path.join(work, "detail.png")
    code = cli.main([
        "generate", target, "1", "192", "160", "--seed", "3",
        "--islands", "3", "--subdivisions", "2", "--quiet",
    ])
    assert code == 0
    image = cv.imread(target, cv.IMREAD_UNCHANGED)
    assert image is not None
    assert image.shape == (160, 192, 4)
    assert image[:, :, 3].max() == 255
