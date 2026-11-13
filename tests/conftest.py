"""Shared pytest setup.

The suite must work from a read-only user profile, so every scratch file goes
under ``outputs/_test/`` (gitignored) instead of the system temp directory, and
matplotlib's font cache is redirected there before any module imports it.
"""

import os
import re
import shutil
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEST_ROOT = os.path.join(PROJECT_ROOT, "outputs", "_test")
MPL_CACHE = os.path.join(TEST_ROOT, ".mplcache")
os.makedirs(MPL_CACHE, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE)
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture()
def work(request):
    """A clean scratch directory for one test, removed afterwards."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid).strip("_")
    path = os.path.join(TEST_ROOT, slug)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
