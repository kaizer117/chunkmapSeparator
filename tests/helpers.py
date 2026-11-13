"""Small fixtures shared by unit and integration tests."""

import os

import cv2 as cv
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES = os.path.join(PROJECT_ROOT, "resources")


def make_lineart(path, size=(240, 180), alpha=True, background=None):
    """Rectangle + circle, i.e. plenty of perfectly straight runs."""
    width, height = size
    canvas = np.zeros((height, width, 4), np.uint8)
    if background is not None:
        canvas[:, :, :3] = background
        canvas[:, :, 3] = 255
    cv.rectangle(canvas, (20, 20), (width - 20, height - 20), (0, 0, 0, 255), 3)
    cv.circle(canvas, (width // 2, height // 2), min(width, height) // 5,
              (0, 0, 0, 255), 3)
    if alpha:
        cv.imwrite(path, canvas)
    else:
        cv.imwrite(path, canvas[:, :, :3])
    return path


def make_mask(path):
    """Read an image and resolve its ink mask through the production helper."""
    from chunkmap import accuracy

    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    mask, _, _ = accuracy.prepare_mask(image)
    return mask
