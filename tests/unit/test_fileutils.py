"""Unit tests for :mod:`chunkmap.fileutils` (SVG writer and folder helpers)."""

import os

import pytest

from chunkmap import fileutils as futils


def test_create_folder_creates_a_missing_parent(work):
    """A fresh clone has no outputs/, so mkdir() used to fail."""
    sub = os.path.relpath(os.path.join(work, "nested", "deeper"), futils.cwd)
    created = futils.createFolder(sub)
    assert os.path.isdir(created)


def test_svg_handler_save_requires_compose():
    handler = futils.SVGHandler()
    with pytest.raises(ValueError):
        handler.save(loc="outputs")


def test_svg_handler_compose_requires_header_and_body():
    handler = futils.SVGHandler()
    handler.initialize((40, 30))
    with pytest.raises(ValueError):
        handler.compose()


def test_svg_handler_writes_header_style_and_paths():
    handler = futils.SVGHandler()
    handler.initialize((100, 50))
    handler.addstyletag({"outline": {"fill": "none", "stroke": "#111111"}})
    handler.drawline([[0, 0], [10, 0], [10, 10]], "outline")
    handler.drawcubicbezier(
        [
            [[0, 0], [1, 0], [2, 1], [3, 1]],
            [[3, 1], [4, 1], [5, 0]],
        ],
        "outline",
        closed=True,
    )
    handler.compose()

    assert '<svg width="100px" height="50px" viewBox="0 0 100 50"' in handler.file
    assert ".outline {fill:none; stroke:#111111}" in handler.file
    assert 'd="M 0,0 L 10,0 10,10"' in handler.file
    assert ('d="M 0,0 C 1,0 2,1 3,1 C 3,1 4,1 5,0 z" />' in handler.file)
    assert handler.file.rstrip().endswith("</svg>")


def test_svg_handler_clear_resets_every_section():
    handler = futils.SVGHandler()
    handler.initialize((10, 10))
    handler.addstyletag({"a": {"fill": "none"}})
    handler.drawline([[0, 0], [1, 1]], "a")
    handler.compose()
    handler.clear()
    assert handler.header == ""
    assert handler.styletag == ""
    assert handler.paths == ""
    assert handler.composed == ""
    assert handler.file == ""
