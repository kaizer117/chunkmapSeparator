"""
island_generator.py - random island PNGs for the chunkmap-separator pipeline.

The generator is the implementation of the research in ``research/``:

    controlled silhouette
        -> signed distance field
        -> multi-octave Perlin fBm with domain warp
        -> coverage-solved sea level + edge safety
        -> binary mask cleanup
        -> antialiased RGBA PNG

It deliberately has no dependency beyond NumPy, OpenCV and SciPy. The noise is a
vectorised 2D Perlin implementation so the whole batch runs without a new package.

The CLI entry point lives in ``cli.py``:

    python code/cli.py generate outputs/islands 8 1024 768 --seed 42
"""

from __future__ import annotations

import argparse
import math
import os
import secrets
import sys
import time
from dataclasses import dataclass, replace
from typing import Optional

import cv2 as cv
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
from scipy.spatial import cKDTree

MAX_SEED = 0xFFFFFFFF
MAX_LONG_SIDE = 1536
MAX_ISLANDS = 12
MAX_SUBDIVISIONS = 12


class IslandGenerationError(Exception):
    """A seed/parameter combination that could not produce a valid island."""


@dataclass(frozen=True)
class IslandParams:
    """Parameters for one generated image.

    Lengths are fractions of the short side of the working grid, except
    ``coast_amp`` and ``warp`` which are in island radii. ``coverage`` is the
    fraction of the short-side square that should be land. By default the image is
    a transparent constant-width outline; set ``filled`` for a solid landmass.
    ``islands`` places several silhouettes, and ``subdivisions`` divides each one
    into area-aware regions.
    """

    width: int
    height: int
    seed: int
    coverage: float = 0.30
    margin: float = 0.08
    base_radius: float = 0.30
    aspect: float = 1.20
    lobes: int = 5
    lobe_strength: float = 0.30
    hurst: float = 0.74
    lacunarity: float = 2.0
    base_frequency: float = 3.5
    warp: float = 0.32
    warp_frequency: float = 2.0
    coast_amp: float = 0.18
    coast_band: float = 0.15
    detail_px: float = 2.0
    center_jitter: float = 0.02
    islands: int = 1
    island_spacing: float = 1.00
    subdivisions: int = 0
    min_island_area: float = 0.003
    min_region_fraction: float = 0.02
    filled: bool = False
    stroke_width: float = 0.0


@dataclass
class IslandResult:
    """The generated image plus the measurements needed by the CLI/tests."""

    image: np.ndarray
    mask: np.ndarray
    seed: int
    coverage: float
    width: int
    height: int
    bbox: tuple[int, int, int, int]
    margin_px: int
    island_count: int = 1
    region_count: int = 0
    stroke_px: float = 0.0
    region_areas: tuple[int, ...] = ()
    island_areas: tuple[int, ...] = ()
    # Region label map (0 = water, 1..n = regions) on the output grid. This is the
    # geometry behind the rendered PNG: coastlines are mask boundaries and internal
    # borders are label changes. The region-aware vectorizer consumes it so it never
    # has to guess the stroke width from the antialiased alpha band.
    labels: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------

_GRAD_X = np.array([1.0, -1.0, 0.0, 0.0, 0.7071068, -0.7071068, 0.7071068, -0.7071068],
                   dtype=np.float32)
_GRAD_Y = np.array([0.0, 0.0, 1.0, -1.0, 0.7071068, 0.7071068, -0.7071068, -0.7071068],
                   dtype=np.float32)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t


def _permutation(rng: np.random.Generator) -> np.ndarray:
    values = np.arange(256, dtype=np.int32)
    rng.shuffle(values)
    return np.concatenate([values, values]).astype(np.int32)


def _perlin2d(x: np.ndarray, y: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Vectorised 2D Perlin noise, roughly in [-0.7, 0.7]."""
    xi0 = np.floor(x).astype(np.int32)
    yi0 = np.floor(y).astype(np.int32)
    xf = x - xi0.astype(np.float32)
    yf = y - yi0.astype(np.float32)

    xi = xi0 & 255
    yi = yi0 & 255
    xi1 = (xi + 1) & 255
    yi1 = (yi + 1) & 255

    aa = perm[perm[xi] + yi] & 7
    ba = perm[perm[xi1] + yi] & 7
    ab = perm[perm[xi] + yi1] & 7
    bb = perm[perm[xi1] + yi1] & 7

    g00 = _GRAD_X[aa] * xf + _GRAD_Y[aa] * yf
    g10 = _GRAD_X[ba] * (xf - 1.0) + _GRAD_Y[ba] * yf
    g01 = _GRAD_X[ab] * xf + _GRAD_Y[ab] * (yf - 1.0)
    g11 = _GRAD_X[bb] * (xf - 1.0) + _GRAD_Y[bb] * (yf - 1.0)

    u = _fade(xf)
    v = _fade(yf)
    return _lerp(_lerp(g00, g10, u), _lerp(g01, g11, u), v)


def _fbm_split(x: np.ndarray, y: np.ndarray, params: IslandParams,
               rng: np.random.Generator, short_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (low-frequency fBm, high-frequency fBm), each roughly in [-1, 1]."""
    gain = params.lacunarity ** (-params.hurst)
    octaves = _octave_count_for_grid(params, short_px)
    low = np.zeros_like(x, dtype=np.float32)
    high = np.zeros_like(x, dtype=np.float32)
    low_norm = 0.0
    high_norm = 0.0

    for index in range(octaves):
        frequency = params.base_frequency * (params.lacunarity ** index)
        weight = gain ** index
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        offset_x = float(rng.uniform(-512.0, 512.0))
        offset_y = float(rng.uniform(-512.0, 512.0))

        u = x * frequency
        v = y * frequency
        sample_x = cos_a * u - sin_a * v + offset_x
        sample_y = sin_a * u + cos_a * v + offset_y
        value = _perlin2d(sample_x, sample_y, _permutation(rng)) * weight

        if index < 3:
            low += value
            low_norm += weight
        else:
            high += value
            high_norm += weight

    low = low / max(low_norm, 1e-6)
    high = high / max(high_norm, 1e-6)
    return _robust_normalize(low), _robust_normalize(high)


def _fbm_field(x: np.ndarray, y: np.ndarray, params: IslandParams,
               rng: np.random.Generator, octaves: int,
               base_frequency: float, hurst: float) -> np.ndarray:
    """A normalised fBm field for the warp vector components."""
    gain = params.lacunarity ** (-hurst)
    field = np.zeros_like(x, dtype=np.float32)
    norm = 0.0
    for index in range(octaves):
        frequency = base_frequency * (params.lacunarity ** index)
        weight = gain ** index
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        offset_x = float(rng.uniform(-512.0, 512.0))
        offset_y = float(rng.uniform(-512.0, 512.0))

        u = x * frequency
        v = y * frequency
        sample_x = cos_a * u - sin_a * v + offset_x
        sample_y = sin_a * u + cos_a * v + offset_y
        field += _perlin2d(sample_x, sample_y, _permutation(rng)) * weight
        norm += weight
    return _robust_normalize(field / max(norm, 1e-6))


def _robust_normalize(field: np.ndarray) -> np.ndarray:
    median = float(np.median(field))
    p01, p99 = np.percentile(field, [1.0, 99.0])
    scale = max(float(p99) - median, median - float(p01), 1e-6)
    return np.clip((field - median) / scale, -1.0, 1.0).astype(np.float32)


def _octave_count_for_grid(params: IslandParams, short_px: int) -> int:
    detail_work = max(0.25, params.detail_px * short_px / float(min(params.width, params.height)))
    ratio = max(1.0, short_px / max(1e-6, detail_work * params.base_frequency))
    octaves = int(math.floor(math.log(ratio) / math.log(params.lacunarity))) + 1
    return int(np.clip(octaves, 4, 7))


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------

def _working_size(width: int, height: int) -> tuple[int, int]:
    # Work at the requested size up to 1536 px; the antialiased alpha edge does
    # the rest. Keeping the grid moderate makes a batch of PNGs responsive.
    long_side = min(MAX_LONG_SIDE, max(256, max(width, height)))
    scale = long_side / float(max(width, height))
    return max(32, int(round(width * scale))), max(32, int(round(height * scale)))


@dataclass(frozen=True)
class _IslandShape:
    center_x: float
    center_y: float
    radius: float
    aspect: float
    rotation: float
    phases: tuple[float, ...]
    amplitudes: tuple[float, ...]


def _make_shapes(params: IslandParams, rng: np.random.Generator,
                 work_w: int, work_h: int, short_px: int) -> list[_IslandShape]:
    """Build one or more island silhouettes at distinct positions."""
    width_units = work_w / float(short_px)
    height_units = work_h / float(short_px)
    count = int(np.clip(params.islands, 1, MAX_ISLANDS))
    shapes: list[_IslandShape] = []

    if count == 1:
        center_x = width_units / 2.0 + float(rng.uniform(-params.center_jitter,
                                                          params.center_jitter))
        center_y = height_units / 2.0 + float(rng.uniform(-params.center_jitter,
                                                           params.center_jitter))
        shapes.append(_make_shape(params, rng, center_x, center_y, params.base_radius))
        return shapes

    base_each = params.base_radius / math.sqrt(count) * 0.92
    placement_margin = min(0.20, params.margin + 0.025)
    for index in range(count):
        placed = False
        for shrink in range(6):
            radius = base_each * float(rng.uniform(0.65, 1.15)) * (0.92 ** shrink)
            for attempt in range(500):
                aspect = params.aspect * float(rng.uniform(0.82, 1.18))
                reach = radius * max(math.sqrt(aspect), 1.0 / math.sqrt(aspect)) * 1.10
                min_x, max_x = placement_margin + reach, width_units - placement_margin - reach
                min_y, max_y = placement_margin + reach, height_units - placement_margin - reach
                if max_x <= min_x or max_y <= min_y:
                    break
                center_x = float(rng.uniform(min_x, max_x))
                center_y = float(rng.uniform(min_y, max_y))
                if all(
                    math.hypot(center_x - other.center_x, center_y - other.center_y)
                    >= (reach + other.radius * 1.3) * params.island_spacing
                    for other in shapes
                ):
                    shape = _make_shape(params, rng, center_x, center_y, radius, aspect)
                    shapes.append(shape)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            raise IslandGenerationError(
                "cannot place %d islands at this size and margin" % count)
    return shapes


def _make_shape(params: IslandParams, rng: np.random.Generator,
                center_x: float, center_y: float, radius: float,
                aspect: float | None = None) -> _IslandShape:
    aspect = params.aspect * float(rng.uniform(0.82, 1.18)) if aspect is None else aspect
    rotation = float(rng.uniform(0.0, 2.0 * math.pi))
    lobe_count = max(3, int(params.lobes))
    phases = tuple(float(rng.uniform(0.0, 2.0 * math.pi)) for _ in range(lobe_count - 1))
    amplitudes = tuple(
        float(rng.normal(0.0, params.lobe_strength / index))
        for index in range(2, lobe_count + 1)
    )
    return _IslandShape(center_x, center_y, radius, aspect, rotation,
                        phases, amplitudes)


def _single_base_mask(x: np.ndarray, y: np.ndarray, shape: _IslandShape) -> np.ndarray:
    rx = shape.radius * math.sqrt(shape.aspect)
    ry = shape.radius / math.sqrt(shape.aspect)
    cos_r, sin_r = math.cos(shape.rotation), math.sin(shape.rotation)
    dx = (x - shape.center_x) * cos_r + (y - shape.center_y) * sin_r
    dy = -(x - shape.center_x) * sin_r + (y - shape.center_y) * cos_r

    rho = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2).astype(np.float32)
    theta = np.arctan2(dy / ry, dx / rx).astype(np.float32)

    perturbation = np.zeros_like(theta)
    for index, (amplitude, phase) in enumerate(zip(shape.amplitudes, shape.phases), start=2):
        perturbation += amplitude * np.cos(index * theta + phase)
    radius_scale = np.clip(1.0 + perturbation, 0.55, 1.45).astype(np.float32)
    return rho <= radius_scale


def _signed_distance(base: np.ndarray, short_px: int) -> np.ndarray:
    mask = base.astype(np.uint8) * 255
    inside = cv.distanceTransform(mask, cv.DIST_L2, 5)
    outside = cv.distanceTransform(255 - mask, cv.DIST_L2, 5)
    phi = (inside - outside).astype(np.float32) / float(short_px)
    return gaussian_filter(phi, sigma=0.7).astype(np.float32)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _field_for_shape(shape: _IslandShape, params: IslandParams,
                     low: np.ndarray, high: np.ndarray,
                     warp_x: np.ndarray, warp_y: np.ndarray,
                     x: np.ndarray, y: np.ndarray,
                     xx: np.ndarray, yy: np.ndarray,
                     short_px: int, edge_penalty: np.ndarray) -> np.ndarray:
    base = _single_base_mask(x, y, shape)
    phi = _signed_distance(base, short_px)
    amplitude = params.coast_amp * shape.radius
    band = np.exp(-(phi / max(params.coast_band, 1e-4)) ** 2).astype(np.float32)

    warp_px = params.warp * shape.radius * short_px
    map_x = (xx + 0.5 + warp_px * warp_x).astype(np.float32)
    map_y = (yy + 0.5 + warp_px * warp_y).astype(np.float32)
    low_warped = cv.remap(low, map_x, map_y, cv.INTER_LINEAR,
                           borderMode=cv.BORDER_REFLECT)
    high_warped = cv.remap(high, map_x, map_y, cv.INTER_LINEAR,
                            borderMode=cv.BORDER_REFLECT)
    field = phi + 0.75 * amplitude * low_warped + 0.25 * amplitude * band * high_warped
    return field - edge_penalty


def _threshold_for_area(field: np.ndarray, target_px: float,
                        width: int, height: int) -> float:
    fraction = float(np.clip(target_px / float(width * height), 1e-6, 1.0 - 1e-6))
    return float(np.quantile(field, 1.0 - fraction))


def generate_island(params: IslandParams) -> IslandResult:
    """Generate one island image. Deterministic for a given params.seed."""
    if params.width <= 0 or params.height <= 0:
        raise ValueError("width and height must be positive")
    if not (0.02 <= params.coverage <= 0.70):
        raise ValueError("coverage must be between 0.02 and 0.70")
    if not (0.0 <= params.margin <= 0.20):
        raise ValueError("margin must be between 0.0 and 0.20")

    work_w, work_h = _working_size(params.width, params.height)
    short_px = min(work_w, work_h)
    rng = np.random.default_rng(int(params.seed) & MAX_SEED)

    yy, xx = np.indices((work_h, work_w), dtype=np.float32)
    x = (xx + 0.5) / short_px
    y = (yy + 0.5) / short_px

    # One set of noise fields is reused while the island shapes are retried.
    low, high = _fbm_split(x, y, params, rng, short_px)
    warp_x = _fbm_field(x, y, params, rng, octaves=2,
                        base_frequency=params.warp_frequency, hurst=0.80)
    warp_y = _fbm_field(x, y, params, rng, octaves=2,
                        base_frequency=params.warp_frequency, hurst=0.80)
    warp_x = _robust_normalize(warp_x)
    warp_y = _robust_normalize(warp_y)

    shapes = _make_shapes(params, rng, work_w, work_h, short_px)
    edge_penalty = _edge_penalty(x, y, work_w, work_h, short_px,
                                 params.margin + 0.025)
    total_target = params.coverage * short_px * short_px
    weights = np.asarray([shape.radius ** 2 for shape in shapes], dtype=np.float64)
    base_targets = total_target * weights / max(weights.sum(), 1e-6)
    min_area_px = params.min_island_area * short_px * short_px
    margin_px = max(1, int(round(params.margin * short_px)))
    mask = None
    coverage_scale = 1.0

    for attempt in range(5):
        masks = []
        failed = False
        targets = base_targets * coverage_scale
        for shape, target in zip(shapes, targets):
            field = _field_for_shape(shape, params, low, high, warp_x, warp_y,
                                     x, y, xx, yy, short_px, edge_penalty)
            threshold = _threshold_for_area(field, target, work_w, work_h)
            island_mask = _clean_mask(field >= threshold, keep=1,
                                      min_area_px=min_area_px)
            if island_mask is None or not _has_margin(island_mask, margin_px):
                failed = True
                break
            masks.append(island_mask)
        if failed:
            shapes = [replace(shape, radius=shape.radius * 0.92) for shape in shapes]
            base_targets = base_targets * 0.92 * 0.92
            continue

        mask = np.zeros((work_h, work_w), dtype=np.uint8)
        for island_mask in masks:
            mask |= island_mask
        if _component_count(mask, min_area_px) < len(shapes):
            separated = _separate_masks(masks, min_area_px)
            if separated is not None:
                mask = separated
        # If the islands are still fused, lower the per-island target and
        # recompute. This keeps the requested count even when packing is tight;
        # the returned coverage reports the achievable value.
        if _component_count(mask, min_area_px) >= len(shapes) or attempt >= 4:
            break
        coverage_scale *= 0.85
    else:
        raise IslandGenerationError(
            "could not keep the island inside the margin after 5 attempts")

    if mask is None:
        raise IslandGenerationError("could not build an island mask")

    out_mask = _resize_mask(mask, params.width, params.height)
    if not np.any(out_mask):
        raise IslandGenerationError("island mask is empty after scaling")

    stroke_px = _stroke_pixels(params)
    labels, island_count, region_count = _label_and_subdivide(
        out_mask > 0, params, rng, stroke_px)
    image = _render_image(out_mask > 0, labels, params, stroke_px)
    region_areas = _region_areas(labels)
    island_areas = _island_areas(out_mask > 0)

    out_coverage = _coverage(out_mask, min(params.width, params.height))
    bbox = _bbox(out_mask)
    margin_px = max(1, int(round(params.margin * min(params.width, params.height))))
    return IslandResult(image=image, mask=out_mask, seed=int(params.seed) & MAX_SEED,
                        coverage=out_coverage, width=params.width, height=params.height,
                        bbox=bbox, margin_px=margin_px,
                        island_count=island_count, region_count=region_count,
                        stroke_px=stroke_px, region_areas=region_areas,
                        island_areas=island_areas, labels=labels)


def _region_areas(labels: np.ndarray) -> tuple[int, ...]:
    values = np.unique(labels)
    return tuple(int(np.count_nonzero(labels == value))
                 for value in values if value > 0)


def _island_areas(mask: np.ndarray) -> tuple[int, ...]:
    if not np.any(mask):
        return ()
    number, _, stats, _ = cv.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    return tuple(int(stats[label, cv.CC_STAT_AREA])
                 for label in range(1, number))


def _clean_mask(mask: np.ndarray, keep: int = 1,
                min_area_px: float = 0.0) -> Optional[np.ndarray]:
    if mask is None or not np.any(mask):
        return None
    source = (mask.astype(np.uint8) * 255)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    cleaned = cv.morphologyEx(source, cv.MORPH_OPEN, kernel)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)
    filled = binary_fill_holes(cleaned > 0).astype(np.uint8)
    if not np.any(filled):
        filled = binary_fill_holes(source > 0).astype(np.uint8)
    number, labels, stats, _ = cv.connectedComponentsWithStats(filled, 8)
    if number <= 1:
        return (filled * 255).astype(np.uint8)

    candidates = []
    for label in range(1, number):
        area = int(stats[label, cv.CC_STAT_AREA])
        if area >= min_area_px:
            candidates.append((area, label))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    chosen = [label for _, label in candidates[:max(1, keep)]]
    result = np.isin(labels, chosen).astype(np.uint8) * 255
    return result


def _component_count(mask: np.ndarray, min_area_px: float = 0.0) -> int:
    if mask is None or not np.any(mask):
        return 0
    number, _, stats, _ = cv.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    return sum(1 for label in range(1, number)
               if int(stats[label, cv.CC_STAT_AREA]) >= min_area_px)


def _separate_masks(masks: list[np.ndarray],
                    min_area_px: float) -> Optional[np.ndarray]:
    """Erode nearby islands apart so a touching pair is not one component."""
    for radius in (1, 2, 3):
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        separated = np.zeros_like(masks[0], dtype=np.uint8)
        for island_mask in masks:
            separated |= cv.erode(island_mask, kernel)
        if _component_count(separated, min_area_px) == len(masks):
            return separated
    return None


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a 0/255 mask, keeping a hard edge for geometry and strokes."""
    if mask.shape[1] == width and mask.shape[0] == height:
        return mask.astype(np.uint8)
    shrinking = width * height < mask.shape[0] * mask.shape[1]
    interpolation = cv.INTER_AREA if shrinking else cv.INTER_LINEAR
    resized = cv.resize(mask, (width, height), interpolation=interpolation)
    return ((resized >= 128).astype(np.uint8) * 255)


def _stroke_pixels(params: IslandParams) -> float:
    if params.stroke_width > 0:
        return float(params.stroke_width)
    # 2 px minimum, about 2.5 px on a 1024 px canvas, 10 px at 4096.
    return max(2.0, round(min(params.width, params.height) * 0.0025, 2))


def _coverage(mask: np.ndarray, short_px: int) -> float:
    return float(np.count_nonzero(mask)) / float(short_px * short_px)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return (0, 0, 0, 0)
    return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))


def _has_margin(mask: np.ndarray, margin_px: int) -> bool:
    if margin_px <= 0:
        return True
    return not (np.any(mask[:margin_px, :]) or np.any(mask[-margin_px:, :])
                or np.any(mask[:, :margin_px]) or np.any(mask[:, -margin_px:]))


def _edge_penalty(x: np.ndarray, y: np.ndarray, width: int, height: int,
                  short_px: int, margin: float) -> np.ndarray:
    width_units = width / float(short_px)
    height_units = height / float(short_px)
    distance = np.minimum(np.minimum(x, width_units - x),
                          np.minimum(y, height_units - y)).astype(np.float32)
    scaled = np.clip((margin - distance) / max(margin, 1e-6), 0.0, 1.0)
    return (scaled * scaled).astype(np.float32)


def _farthest_seeds(points: np.ndarray, count: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Farthest-point seeds; cheap and well spread for small region counts."""
    count = min(count, len(points))
    first = int(rng.integers(0, len(points)))
    seeds = [points[first]]
    distance = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(distance))
        seeds.append(points[index])
        distance = np.minimum(
            distance, np.sum((points - points[index]) ** 2, axis=1))
    return np.asarray(seeds, dtype=np.float32)


def _assign_regions(mask: np.ndarray, seeds: np.ndarray,
                    warp=None) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    query = np.column_stack([xs, ys]).astype(np.float32)
    if warp is not None:
        query = warp(query)
    tree = cKDTree(seeds)
    _, nearest = tree.query(query)
    labels = np.zeros(mask.shape, dtype=np.int32)
    labels[ys, xs] = nearest.astype(np.int32) + 1
    return labels


def _subdivide_mask(mask: np.ndarray, requested: int,
                    rng: np.random.Generator, min_fraction: float,
                    stroke_px: float) -> np.ndarray:
    """Area-aware Voronoi-ish subdivisions clipped to one island."""
    area = int(np.count_nonzero(mask))
    if requested <= 1 or area <= 1:
        return np.zeros(mask.shape, dtype=np.int32)

    min_area = max(area * min_fraction, (max(1.0, stroke_px) * 8.0) ** 2)
    count = min(requested, max(1, int(area / min_area)))
    if count <= 1:
        return np.zeros(mask.shape, dtype=np.int32)

    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]).astype(np.float32)
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    amplitude = min(8.0, max(2.0, span * 0.025))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))

    def warp(coords: np.ndarray) -> np.ndarray:
        px, py = coords[:, 0], coords[:, 1]
        return np.column_stack([
            px + amplitude * np.sin(2.0 * math.pi * py / (span * 0.65) + phase),
            py + amplitude * np.sin(2.0 * math.pi * px / (span * 0.55) + phase * 1.7),
        ]).astype(np.float32)

    if len(points) > 6000:
        sample = rng.choice(len(points), size=6000, replace=False)
        pool = points[sample]
    else:
        pool = points
    seeds = _farthest_seeds(pool, count, rng)
    labels = _assign_regions(mask, seeds, warp)

    # Lloyd relaxation pulls the seeds toward equal-area centroids. Snapping the
    # centroid back to land keeps seeds valid on concave islands.
    for _ in range(4):
        updated = []
        for label in range(1, len(seeds) + 1):
            pts = points[labels[ys, xs] == label]
            if len(pts):
                updated.append(pts.mean(axis=0))
            else:
                updated.append(seeds[label - 1])
        updated = np.asarray(updated, dtype=np.float32)
        _, nearest = cKDTree(points).query(updated)
        seeds = points[nearest].copy()
        # Nudge duplicate seeds apart, then snap them back to land.
        for index in range(1, len(seeds)):
            if np.linalg.norm(seeds[index] - seeds[:index], axis=1).min() < 2.0:
                seeds[index] += rng.normal(0.0, max(2.0, stroke_px * 2.0), 2)
                _, snap = cKDTree(points).query(seeds[index])
                seeds[index] = points[snap]
        labels = _assign_regions(mask, seeds, warp)

    # Drop cells below the visible-area threshold by handing their pixels to the
    # nearest remaining seed. Repeat because a merge can expose another small cell.
    for _ in range(4):
        counts = np.bincount(labels.ravel(), minlength=len(seeds) + 1)
        small = [label for label in range(1, len(seeds) + 1)
                 if 0 < counts[label] < min_area]
        if not small or len(seeds) <= 1:
            break
        for label in small:
            if not np.any(labels == label):
                continue
            others = [other for other in range(1, len(seeds) + 1)
                      if other != label and counts[other] > 0]
            if not others:
                break
            tree = cKDTree(seeds[np.asarray(others) - 1])
            ys_small, xs_small = np.nonzero(labels == label)
            _, nearest = tree.query(
                np.column_stack([xs_small, ys_small]).astype(np.float32))
            labels[ys_small, xs_small] = np.asarray(others)[nearest]
            counts = np.bincount(labels.ravel(), minlength=len(seeds) + 1)
        used = np.unique(labels[labels > 0])
        if len(used) <= 1:
            return np.zeros(mask.shape, dtype=np.int32)
        remap = np.zeros(len(seeds) + 1, dtype=np.int32)
        remap[used] = np.arange(1, len(used) + 1)
        labels = remap[labels]
        seeds = seeds[used - 1]
        if len(seeds) <= 1:
            return np.zeros(mask.shape, dtype=np.int32)

    return labels


def _label_and_subdivide(mask: np.ndarray, params: IslandParams,
                         rng: np.random.Generator,
                         stroke_px: float) -> tuple[np.ndarray, int, int]:
    """Label each island and optionally subdivide it into visible regions."""
    number, components, stats, _ = cv.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    labels = np.zeros(mask.shape, dtype=np.int32)
    minimum_island = params.min_island_area * min(params.width, params.height) ** 2
    next_label = 1
    island_count = 0
    region_count = 0

    for component in range(1, number):
        area = int(stats[component, cv.CC_STAT_AREA])
        if area < minimum_island:
            continue
        island_mask = components == component
        island_count += 1
        local = _subdivide_mask(island_mask, params.subdivisions, rng,
                                params.min_region_fraction, stroke_px)
        local_labels = np.unique(local[island_mask])
        local_labels = local_labels[local_labels > 0]
        if len(local_labels) <= 1:
            labels[island_mask] = next_label
            next_label += 1
            region_count += 1
            continue
        for local_label in local_labels:
            labels[island_mask & (local == local_label)] = next_label
            next_label += 1
        region_count += len(local_labels)

    return labels, island_count, region_count


def _line_mask(mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    eroded = cv.erode(mask_u8, kernel)
    lines = (mask_u8 > 0) & (eroded == 0)

    vertical = (labels[1:, :] != labels[:-1, :]) & (labels[1:, :] > 0) & (labels[:-1, :] > 0)
    horizontal = (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] > 0) & (labels[:, :-1] > 0)
    lines[1:, :] |= vertical
    lines[:, 1:] |= horizontal
    return lines


def _stroke_band(lines: np.ndarray, stroke_px: float) -> np.ndarray:
    if not np.any(lines):
        return np.zeros(lines.shape, dtype=bool)
    distance = cv.distanceTransform(
        (255 - lines.astype(np.uint8) * 255), cv.DIST_L2, 5)
    return distance <= max(0.5, stroke_px / 2.0)


def _render_image(mask: np.ndarray, labels: np.ndarray,
                  params: IslandParams, stroke_px: float) -> np.ndarray:
    """Render the outline (default) or a filled island with region borders."""
    lines = _line_mask(mask, labels)
    band = _stroke_band(lines, stroke_px)
    height, width = mask.shape

    line_colour = np.array([36, 48, 42], dtype=np.uint8)   # BGR, dark green-grey
    fill_colour = np.array([72, 100, 72], dtype=np.uint8)  # BGR, muted green

    if params.filled:
        alpha = gaussian_filter(mask.astype(np.float32), sigma=0.6)
        colour = np.broadcast_to(fill_colour, (height, width, 3)).copy()
        colour[band] = line_colour
    else:
        alpha = gaussian_filter(band.astype(np.float32), sigma=0.6)
        peak = float(alpha.max())
        if peak > 1e-6:
            alpha = alpha / peak
        colour = np.broadcast_to(line_colour, (height, width, 3)).copy()

    image = np.empty((height, width, 4), dtype=np.uint8)
    image[:, :, :3] = colour
    image[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return image


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_STYLES = {
    "natural": {},
    "smooth": {"hurst": 0.82, "warp": 0.15, "coast_amp": 0.10,
               "coast_band": 0.12, "aspect": 1.15, "lobes": 4,
               "lobe_strength": 0.20},
    "rugged": {"hurst": 0.62, "warp": 0.35, "coast_amp": 0.24,
               "coast_band": 0.16, "aspect": 1.35, "lobes": 5,
               "lobe_strength": 0.35},
}


def style_names() -> tuple[str, ...]:
    """The --style choices, in a stable order."""
    return tuple(_STYLES)


def style_preset(name: str) -> dict[str, float]:
    """Named coast-roughness overrides for IslandParams (natural is empty)."""
    if name not in _STYLES:
        raise ValueError("unknown style %r (choose from %s)"
                         % (name, ", ".join(_STYLES)))
    return dict(_STYLES[name])


def _pairing(output: str, count: int) -> tuple[str, str]:
    """Return (output pattern, directory) for numbered PNGs."""
    if output.lower().endswith(".png"):
        if count == 1:
            directory = os.path.dirname(os.path.abspath(output))
            return output, directory
        stem = output[:-4]
        directory = os.path.dirname(os.path.abspath(stem))
        digits = max(4, len(str(count)))
        return stem + "_%0" + str(digits) + "d.png", directory
    os.makedirs(output, exist_ok=True)
    digits = max(4, len(str(count)))
    return os.path.join(output, "island_%0" + str(digits) + "d.png"), output


def build_generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py generate",
        description="Generate a batch of random island PNGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python code/cli.py generate outputs/islands 8 1024 768\n"
            "  python code/cli.py generate outputs/islands 8 1024 768 --islands 3\n"
            "  python code/cli.py generate outputs/islands 8 1024 768 --subdivisions 4\n"
            "  python code/cli.py generate outputs/islands 8 1024 768 --filled\n"
            "  python code/cli.py generate my_island.png 1 1024 1024 --style rugged\n"
        ),
    )
    parser.add_argument("output",
                        help="output directory, or a .png path (exact for count 1, "
                             "prefix otherwise)")
    parser.add_argument("count", type=int, help="number of PNG files to generate")
    parser.add_argument("width", type=int, help="output width in pixels")
    parser.add_argument("height", type=int, help="output height in pixels")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed; island i uses seed + i (default: random)")
    parser.add_argument("--coverage", type=float, default=0.30,
                        help="land fraction of the short-side square (default: %(default)s)")
    parser.add_argument("--margin", type=float, default=0.08,
                        help="water gap from the canvas edge (default: %(default)s)")
    parser.add_argument("--islands", type=int, default=1,
                        help="islands per image, in separate areas (default: %(default)s)")
    parser.add_argument("--subdivisions", type=int, default=0,
                        help="regions per island, 0 for none (default: %(default)s)")
    parser.add_argument("--stroke-width", type=float, default=0.0,
                        help="outline width in px; 0 = automatic (default: %(default)s)")
    parser.add_argument("--filled", action="store_true",
                        help="solid land fill instead of the default outline")
    parser.add_argument("--style", choices=tuple(_STYLES), default="natural",
                        help="preset roughness (default: %(default)s)")
    parser.add_argument("--quiet", action="store_true", help="print only errors")
    return parser


def _output_name(pattern: str, index: int) -> str:
    return pattern % index if "%" in pattern else pattern


def generate_main(argv: Optional[list[str]] = None) -> int:
    parser = build_generate_parser()
    args = parser.parse_args(argv)

    if args.count <= 0:
        parser.error("count must be positive")
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    if not (0.02 <= args.coverage <= 0.70):
        parser.error("coverage must be between 0.02 and 0.70")
    if not (0.0 <= args.margin <= 0.20):
        parser.error("margin must be between 0.0 and 0.20")
    if not (1 <= args.islands <= MAX_ISLANDS):
        parser.error("islands must be between 1 and %d" % MAX_ISLANDS)
    if not (0 <= args.subdivisions <= MAX_SUBDIVISIONS):
        parser.error("subdivisions must be between 0 and %d" % MAX_SUBDIVISIONS)
    if args.stroke_width < 0:
        parser.error("stroke-width must be positive (or 0 for automatic)")

    pattern, directory = _pairing(args.output, args.count)
    os.makedirs(directory or ".", exist_ok=True)
    base_seed = args.seed if args.seed is not None else secrets.randbits(32)
    base_seed &= MAX_SEED
    style = _STYLES[args.style]
    mode = "filled" if args.filled else "outline"

    started = time.time()
    failures = 0
    if not args.quiet:
        print("islands    %d x %dx%d  ->  %s" % (args.count, args.width, args.height,
                                                 directory or "."))
        print("seed       %d  (%s, %s, %d per image, %d subdivision%s)"
              % (base_seed, args.style, mode, args.islands, args.subdivisions,
                 "" if args.subdivisions == 1 else "s"))

    for index in range(1, args.count + 1):
        seed = (base_seed + index - 1) & MAX_SEED
        output = _output_name(pattern, index)
        result = None
        last_error = None
        for retry in range(2):
            trial_seed = seed if retry == 0 else (seed + 0x9E3779B9) & MAX_SEED
            params = IslandParams(
                width=args.width, height=args.height, seed=trial_seed,
                coverage=args.coverage, margin=args.margin,
                islands=args.islands, subdivisions=args.subdivisions,
                filled=args.filled, stroke_width=args.stroke_width,
                **style,
            )
            try:
                result = generate_island(params)
                break
            except (IslandGenerationError, ValueError) as error:
                last_error = error

        if result is None:
            failures += 1
            print("error: island %d (seed %d): %s" % (index, seed, last_error),
                  file=sys.stderr)
            continue

        if not cv.imwrite(output, result.image):
            failures += 1
            print("error: cannot write %s" % output, file=sys.stderr)
            continue

        if not args.quiet:
            x0, y0, x1, y1 = result.bbox
            print("[%d/%d] %s  seed %d  coverage %.1f%%  islands %d  regions %d  stroke %.1f  bbox %dx%d"
                  % (index, args.count, os.path.basename(output), result.seed,
                     result.coverage * 100.0, result.island_count,
                     result.region_count, result.stroke_px,
                     x1 - x0 + 1, y1 - y0 + 1))

    if not args.quiet:
        print("done       %d/%d in %.2fs" % (args.count - failures, args.count,
                                              time.time() - started))
    if failures:
        print("error: %d island(s) failed" % failures, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(generate_main())
