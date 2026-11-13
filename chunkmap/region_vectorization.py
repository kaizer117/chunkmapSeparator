"""
region_vectorization.py - vectorize a labelled region map with shared borders.

The normal raster -> bezier pipeline traces the **ink band**. For a map made of
line art that means every coastline and every internal border is represented by
two parallel curves, one for each side of the drawn stroke. When two regions meet,
the stroke width therefore becomes visible as a gap or a doubled line in the SVG.

This module works from the *geography* instead: a label map where 0 is water and
each positive integer is one region (island or subdivision). The boundary between
two labels is a single polyline on the pixel lattice. The module fits that
polyline once and reuses the exact same cubic segments, in opposite directions,
for the two regions that share it. The result is one filled path per region and
borders that touch exactly, with no raster stroke width in the output.

The generator already knows this label map (``IslandResult.labels``), so the
orchestration can pass it as a sidecar PNG. The CLI can also read any single
channel label map:

    python code/cli.py island.png --regions regions.png -o island-regions.svg

The algorithm, in order:

1. build every lattice edge where two labels differ (4-neighbour boundaries);
2. create directed half-edges and use the planar face-walk (next edge clockwise
   from the reverse of the incoming edge) to find each region's boundary loops;
3. split the undirected boundary graph into chains at junctions (degree != 2);
4. fit each chain once with the existing cubic fitter, clamping endpoints exactly
   to the lattice junctions;
5. rebuild each region loop from those fitted chains, reversing control points
   where the shared chain is traversed the other way;
6. write one SVG path per region.

The important invariants are:

* every chain endpoint is a lattice point, and fitted endpoints are overwritten
  with that exact point (``fitBezierCurve`` is a free fit and does not preserve
  endpoints on its own);
* a shared chain is fitted exactly once, so both regions use the same geometry;
* every region loop closes exactly (last point == first point);
* the output has one path per region, ``fill-rule="evenodd"``, no stroke, so
  adjacent fills meet without a seam caused by a stroke.
"""

from __future__ import annotations

import colorsys
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np

from . import vectorization as vect


class RegionVectorizationError(Exception):
    """A label map that cannot be turned into a valid region layout."""


# Screen coordinates: x right, y down. Clockwise order is E, S, W, N.
_DIRECTIONS = {
    (1, 0): 0,
    (0, 1): 1,
    (-1, 0): 2,
    (0, -1): 3,
}


@dataclass(frozen=True)
class BoundaryEdge:
    """One undirected unit edge of the label boundary graph."""

    p0: tuple[int, int]
    p1: tuple[int, int]
    a: int
    b: int


@dataclass(frozen=True)
class Chain:
    """A fitted boundary chain between two junctions (or a closed coastline)."""

    lattice: np.ndarray            # (n, 2) integer lattice points
    points: np.ndarray             # (m, 2) simplified pixel-centre points
    edge_ids: tuple[int, ...]
    labels: tuple[int, int]        # the two labels this border separates
    closed: bool
    cubics: tuple[np.ndarray, ...]  # each (4, 2), first/last points exact

    @property
    def shared(self) -> bool:
        """True when both sides are land regions, false for a coastline."""
        return self.labels[0] > 0 and self.labels[1] > 0


@dataclass(frozen=True)
class RegionPath:
    """One region: one or more closed loops of cubic segments."""

    region: int
    loops: tuple[tuple[np.ndarray, ...], ...]
    area_px: int
    perimeter_px: float


@dataclass(frozen=True)
class RegionLayout:
    """The complete vector geography extracted from a label map."""

    labels: np.ndarray
    width: int
    height: int
    edges: tuple[BoundaryEdge, ...]
    chains: tuple[Chain, ...]
    paths: tuple[RegionPath, ...]
    exact_loops: tuple[np.ndarray, ...]
    region_count: int
    chain_count: int
    shared_chain_count: int
    coastline_chain_count: int
    curve_count: int
    warnings: tuple[str, ...] = ()

    @property
    def path_count(self) -> int:
        return len(self.paths)


# ---------------------------------------------------------------------------
# label map input
# ---------------------------------------------------------------------------


def load_label_map(path: str) -> np.ndarray:
    """
    Read a single-channel integer label map from PNG, TIFF or NumPy.

    The orchestration writes 16-bit PNGs. 8-bit PNGs, TIFFs and .npy arrays are
    accepted as well. A multi-channel image is rejected: a colour-coded map is
    not a label map and guessing what its channels mean would be unsafe.
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npy":
        try:
            labels = np.load(path)
        except (OSError, ValueError) as error:
            raise RegionVectorizationError("cannot read label map %s (%s)"
                                           % (path, error))
    else:
        labels = cv.imread(path, cv.IMREAD_UNCHANGED)
        if labels is None:
            raise RegionVectorizationError("cannot read label map %s" % path)

    if labels.ndim == 3:
        if labels.shape[2] == 1:
            labels = labels[:, :, 0]
        else:
            raise RegionVectorizationError(
                "label map %s must have one channel, got %d; encode labels as "
                "integers in a single channel, not colours"
                % (path, labels.shape[2]))
    if labels.ndim != 2:
        raise RegionVectorizationError("label map %s must be 2D" % path)
    if not np.issubdtype(labels.dtype, np.integer):
        raise RegionVectorizationError(
            "label map %s must contain integers, got %s" % (path, labels.dtype))
    labels = labels.astype(np.int32, copy=False)
    if labels.size and labels.min() < 0:
        raise RegionVectorizationError("label map %s contains negative labels"
                                       % path)
    return labels


def check_labels(labels: np.ndarray) -> np.ndarray:
    """Validate and normalise an in-memory label map."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise RegionVectorizationError("labels must be a 2D array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise RegionVectorizationError("labels must be integer valued")
    labels = labels.astype(np.int32, copy=False)
    if labels.size and labels.min() < 0:
        raise RegionVectorizationError("labels must not be negative")
    return labels


# ---------------------------------------------------------------------------
# boundary graph
# ---------------------------------------------------------------------------


def build_boundary_edges(labels: np.ndarray) -> tuple[BoundaryEdge, ...]:
    """
    Every lattice edge where two different labels meet.

    Pixel (x, y) is the unit square [x, x+1] x [y, y+1]. The four edges below are
    therefore the exact shared border, at pixel-edge resolution, between the
    regions on either side. Canvas borders become water boundaries; the generator
    keeps a margin, but a generic label map may touch the canvas.
    """
    height, width = labels.shape
    edges: list[BoundaryEdge] = []

    # Vertical interfaces: left pixel (x-1, y) vs right pixel (x, y).
    # The shared edge is x = const, y .. y+1.
    vertical = labels[:, 1:] != labels[:, :-1]
    ys, xs = np.nonzero(vertical)
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges.append(BoundaryEdge((x + 1, y), (x + 1, y + 1),
                                  int(labels[y, x]), int(labels[y, x + 1])))

    # Horizontal interfaces: top pixel (x, y-1) vs bottom pixel (x, y).
    # The shared edge is y = const, x .. x+1.
    horizontal = labels[1:, :] != labels[:-1, :]
    ys, xs = np.nonzero(horizontal)
    for y, x in zip(ys.tolist(), xs.tolist()):
        edges.append(BoundaryEdge((x, y + 1), (x + 1, y + 1),
                                  int(labels[y, x]), int(labels[y + 1, x])))

    # Canvas borders, only where land actually touches the frame.
    top = labels[0, :]
    for x in np.nonzero(top)[0].tolist():
        edges.append(BoundaryEdge((x, 0), (x + 1, 0), int(top[x]), 0))
    bottom = labels[-1, :]
    for x in np.nonzero(bottom)[0].tolist():
        edges.append(BoundaryEdge((x, height), (x + 1, height),
                                  int(bottom[x]), 0))
    left = labels[:, 0]
    for y in np.nonzero(left)[0].tolist():
        edges.append(BoundaryEdge((0, y), (0, y + 1), int(left[y]), 0))
    right = labels[:, -1]
    for y in np.nonzero(right)[0].tolist():
        edges.append(BoundaryEdge((width, y), (width, y + 1),
                                  int(right[y]), 0))

    return tuple(edges)


def _left_label(labels: np.ndarray, u: tuple[int, int],
                v: tuple[int, int]) -> int:
    """
    Label on the left of the unit edge u -> v, sampled half a pixel to the left.

    This is used to build directed half-edges. Because edges are unit lattice
    segments, the sample always lands on a pixel centre.
    """
    dx, dy = v[0] - u[0], v[1] - u[1]
    # In screen coordinates the left normal of (dx, dy) is (dy, -dx).
    sx = (u[0] + v[0]) * 0.5 + dy * 0.5
    sy = (u[1] + v[1]) * 0.5 - dx * 0.5
    x = int(math.floor(sx))
    y = int(math.floor(sy))
    if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
        return int(labels[y, x])
    return 0


def _directed_edges(labels: np.ndarray, edges: tuple[BoundaryEdge, ...]):
    """Return (directed half-edges, outgoing index by (vertex, region))."""
    directed = []
    outgoing = defaultdict(list)
    for edge_id, edge in enumerate(edges):
        for u, v in ((edge.p0, edge.p1), (edge.p1, edge.p0)):
            region = _left_label(labels, u, v)
            directed_id = len(directed)
            directed.append((u, v, region, edge_id))
            outgoing[(u, region)].append(directed_id)
    return directed, outgoing


def _clockwise_rank(reference: tuple[int, int],
                    direction: tuple[int, int]) -> int:
    if reference not in _DIRECTIONS or direction not in _DIRECTIONS:
        raise RegionVectorizationError(
            "boundary graph produced a non-unit direction %r" % (direction,))
    return (_DIRECTIONS[direction] - _DIRECTIONS[reference]) % 4


def trace_face_loops(labels: np.ndarray, edges: tuple[BoundaryEdge, ...]):
    """
    Walk every region face and return (region, directed_edge_ids) loops.

    The walk keeps the region on the left: at the end vertex it chooses the first
    outgoing edge clockwise from the reverse of the incoming edge. This is the
    standard planar face traversal; it makes the boundary of each label exactly
    once, independent of how many regions share a junction.
    """
    directed, outgoing = _directed_edges(labels, edges)
    visited = np.zeros(len(directed), dtype=bool)
    by_region = defaultdict(list)
    for directed_id, item in enumerate(directed):
        if item[2] > 0:
            by_region[item[2]].append(directed_id)

    loops = []
    for region in sorted(by_region):
        for start in by_region[region]:
            if visited[start]:
                continue
            start_vertex = directed[start][0]
            loop = []
            current = start
            while not visited[current]:
                visited[current] = True
                loop.append(current)
                u, v, _, _ = directed[current]
                if v == start_vertex and len(loop) > 1:
                    break
                reference = (u[0] - v[0], u[1] - v[1])
                candidates = [item for item in outgoing.get((v, region), ())
                              if not visited[item]]
                if not candidates:
                    break
                current = min(
                    candidates,
                    key=lambda item: _clockwise_rank(
                        reference,
                        (directed[item][1][0] - directed[item][0][0],
                         directed[item][1][1] - directed[item][0][1])))
            end_vertex = directed[loop[-1]][1]
            if end_vertex != start_vertex:
                raise RegionVectorizationError(
                    "region %d boundary is not closed (ended at %r, expected %r)"
                    % (region, end_vertex, start_vertex))
            loops.append((region, loop))
    return directed, loops


# ---------------------------------------------------------------------------
# chains
# ---------------------------------------------------------------------------


def _simplify_lattice(points: np.ndarray) -> np.ndarray:
    """Remove interior collinear lattice points; endpoints are preserved."""
    if len(points) <= 2:
        return points
    kept = [points[0]]
    for index in range(1, len(points) - 1):
        a = kept[-1]
        b = points[index]
        c = points[index + 1]
        ab = b - a
        bc = c - b
        cross = int(ab[0]) * int(bc[1]) - int(ab[1]) * int(bc[0])
        dot = int(ab[0]) * int(bc[0]) + int(ab[1]) * int(bc[1])
        if cross != 0 or dot <= 0:
            kept.append(b)
    kept.append(points[-1])
    return np.asarray(kept, dtype=np.int32)


def extract_chains(edges: tuple[BoundaryEdge, ...]) -> list[dict]:
    """
    Split the undirected boundary graph into chains at degree != 2 vertices.

    Open chains start and end at junctions; remaining edges form closed loops
    (a coastline with no subdivisions). The returned dictionaries have
    ``lattice``, ``edge_ids``, ``labels`` and ``closed`` fields; fitting happens
    later so this function is cheap to inspect/test.
    """
    if not edges:
        return []

    incident = defaultdict(list)
    degree = defaultdict(int)
    for edge_id, edge in enumerate(edges):
        incident[edge.p0].append(edge_id)
        incident[edge.p1].append(edge_id)
        degree[edge.p0] += 1
        degree[edge.p1] += 1

    visited = set()
    chains = []
    vertices = sorted(incident)

    def walk(start_vertex, first_edge):
        points = [start_vertex]
        edge_ids = []
        current_vertex = start_vertex
        current_edge = first_edge
        while True:
            if current_edge in visited:
                break
            visited.add(current_edge)
            edge_ids.append(current_edge)
            edge = edges[current_edge]
            next_vertex = edge.p1 if edge.p0 == current_vertex else edge.p0
            points.append(next_vertex)
            current_vertex = next_vertex
            if degree[current_vertex] != 2:
                break
            candidates = [item for item in incident[current_vertex]
                          if item not in visited]
            if not candidates:
                break
            current_edge = candidates[0]
        return points, edge_ids

    # Open chains start at a junction or a dead end.
    for vertex in vertices:
        if degree[vertex] == 2:
            continue
        for edge_id in incident[vertex]:
            if edge_id in visited:
                continue
            points, edge_ids = walk(vertex, edge_id)
            edge = edges[edge_ids[0]]
            chains.append({
                "lattice": np.asarray(points, dtype=np.int32),
                "edge_ids": tuple(edge_ids),
                "labels": (int(edge.a), int(edge.b)),
                "closed": False,
            })

    # What is left is one or more closed cycles with every vertex at degree 2.
    for edge_id, edge in enumerate(edges):
        if edge_id in visited:
            continue
        points, edge_ids = walk(edge.p0, edge_id)
        if points[-1] != points[0]:
            # A dangling component can only come from a malformed graph.
            raise RegionVectorizationError(
                "boundary chain at %r did not close" % (points[0],))
        chains.append({
            "lattice": np.asarray(points, dtype=np.int32),
            "edge_ids": tuple(edge_ids),
            "labels": (int(edge.a), int(edge.b)),
            "closed": True,
        })
    return chains


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def _line_cubic(p0: np.ndarray, p3: np.ndarray) -> np.ndarray:
    return np.asarray([
        p0,
        p0 + (p3 - p0) / 3.0,
        p0 + 2.0 * (p3 - p0) / 3.0,
        p3,
    ], dtype=float)


def _open_slices(points: np.ndarray, max_chunk: int = 15,
                 corner_angle_deg: float = 55.0) -> list[tuple[int, int]]:
    """
    Split an open boundary chain at corners and at the maximum chunk length.

    A single cubic fitted through a sharp map corner will overshoot wildly (it
    has no reason to stay inside the corner). The CLI's curvature segmentation
    assumes a closed contour, so region chains use this small open-aware splitter:

    * measure the turn over a small window (3 points) so raster stair steps do
      not count as corners;
    * split at every point whose windowed turn is at least ``corner_angle_deg``;
    * additionally split any run longer than ``max_chunk`` points.

    The resulting slices share their boundary point, so the endpoint clamp in
    :func:`fit_chain` keeps every segment connected exactly.
    """
    count = len(points)
    if count <= 2:
        return [(0, count)]
    window = max(1, min(3, (count - 1) // 10))
    splits = [0]
    for index in range(1, count - 1):
        left = max(0, index - window)
        right = min(count - 1, index + window)
        before = points[index] - points[left]
        after = points[right] - points[index]
        denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
        if denominator < 1e-9:
            continue
        cosine = float(np.dot(before, after) / denominator)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if angle >= corner_angle_deg:
            splits.append(index)
    splits.append(count - 1)

    # Enforce the maximum chunk length by adding evenly spaced splits.
    expanded = [splits[0]]
    for split in splits[1:]:
        gap = split - expanded[-1]
        if max_chunk >= 2 and gap > max_chunk:
            pieces = int(math.ceil(gap / float(max_chunk)))
            for piece in range(1, pieces):
                candidate = int(round(expanded[-1] + piece * gap / float(pieces)))
                if expanded[-1] < candidate < split:
                    expanded.append(candidate)
        expanded.append(split)

    return [(expanded[index], expanded[index + 1] + 1)
            for index in range(len(expanded) - 1)]


def fit_chain(points: np.ndarray, max_chunk: int = 15,
              corner_angle_deg: float = 55.0) -> tuple[np.ndarray, ...]:
    """
    Fit a boundary polyline with the existing cubic fitter.

    Every returned cubic has its first and last point set to the source lattice
    point. That clamp is what makes adjacent chains meet exactly; the free fitter
    can otherwise move an endpoint by a fraction of a pixel.
    """
    points = np.asarray(points, dtype=float)
    # Remove duplicate consecutive points but never the first or last point.
    if len(points) < 2:
        return ()
    keep = [0]
    for index in range(1, len(points)):
        if not np.allclose(points[index], points[keep[-1]]):
            keep.append(index)
    points = points[keep]
    if len(points) < 2:
        return ()

    cubics = []
    for start, end in _open_slices(points, max_chunk=max_chunk,
                                   corner_angle_deg=corner_angle_deg):
        segment = points[start:end]
        if len(segment) < 2:
            continue
        try:
            control = np.asarray(vect.fitBezierCurve(segment), dtype=float)
        except Exception:
            control = np.zeros((0, 2))
        if control.shape != (4, 2):
            control = _line_cubic(segment[0], segment[-1])
        else:
            control = control.copy()
            control[0] = segment[0]
            control[3] = segment[-1]
        cubics.append(control)
    return tuple(cubics)


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def _loop_to_cubics(loop, directed, edge_chain, chains):
    """Turn a face loop of directed edges into an ordered list of cubics."""
    chain_ids = [edge_chain[directed[item][3]][0] for item in loop]

    # A loop can start in the middle of a chain. Rotate it to a chain boundary so
    # each shared chain appears as one contiguous run and is emitted once.
    if len(set(chain_ids)) > 1:
        for index in range(len(loop)):
            if chain_ids[index] != chain_ids[index - 1]:
                loop = loop[index:] + loop[:index]
                break

    runs = []
    for directed_id in loop:
        u, v, _, edge_id = directed[directed_id]
        chain_id, position = edge_chain[edge_id]
        chain = chains[chain_id]
        forward = (u == tuple(chain["lattice"][position])
                   and v == tuple(chain["lattice"][position + 1]))
        if runs and runs[-1][0] == chain_id:
            if runs[-1][1] != forward:
                raise RegionVectorizationError(
                    "shared chain %d changes direction inside a region loop"
                    % chain_id)
            runs[-1][2].append(directed_id)
        else:
            runs.append([chain_id, forward, [directed_id]])

    cubics = []
    for chain_id, forward, directed_ids in runs:
        chain = chains[chain_id]
        if len(directed_ids) != len(chain["edge_ids"]):
            raise RegionVectorizationError(
                "region loop uses only %d of %d edges of chain %d"
                % (len(directed_ids), len(chain["edge_ids"]), chain_id))
        part = list(chain["cubics"])
        if not forward:
            part = [cubic[::-1] for cubic in reversed(part)]
        if not part:
            continue
        if cubics and not np.allclose(cubics[-1][-1], part[0][0], atol=1e-6):
            raise RegionVectorizationError(
                "chain %d does not connect to the previous chain" % chain_id)
        cubics.extend(part)

    if not cubics:
        raise RegionVectorizationError("region loop produced no curves")
    if not np.allclose(cubics[-1][-1], cubics[0][0], atol=1e-6):
        raise RegionVectorizationError("region loop does not close exactly")

    # Exact lattice loop, kept for accuracy comparisons: this is the input
    # boundary before any curve fitting.
    exact = np.asarray([directed[item][0] for item in loop]
                       + [directed[loop[-1]][1]], dtype=np.float64) - 0.5
    return tuple(cubics), exact


def build_layout(labels: np.ndarray, max_chunk: int = 15,
                 corner_angle_deg: float = 55.0) -> RegionLayout:
    """
    Build region paths with exactly shared boundaries.

    :param labels: 2D integer map, 0 = water, positive = region id
    :param max_chunk: maximum points per fitted cubic on a boundary chain
    :param corner_angle_deg: windowed turn that forces a segmentation split
    """
    labels = check_labels(labels)
    height, width = labels.shape
    if not np.any(labels):
        raise RegionVectorizationError("the label map contains no regions")

    edges = build_boundary_edges(labels)
    if not edges:
        raise RegionVectorizationError("the label map has no boundaries")

    raw_chains = extract_chains(edges)
    chains = []
    edge_chain = {}
    for chain_id, item in enumerate(raw_chains):
        lattice = item["lattice"]
        simplified = _simplify_lattice(lattice)
        points = simplified.astype(np.float64) - 0.5
        cubics = fit_chain(points, max_chunk=max_chunk,
                           corner_angle_deg=corner_angle_deg)
        chains.append({
            "lattice": lattice,
            "points": points,
            "edge_ids": item["edge_ids"],
            "labels": item["labels"],
            "closed": item["closed"],
            "cubics": cubics,
        })
        for position, edge_id in enumerate(item["edge_ids"]):
            edge_chain[edge_id] = (chain_id, position)

    directed, face_loops = trace_face_loops(labels, edges)
    loops_by_region = defaultdict(list)
    exact_by_region = defaultdict(list)
    for region, loop in face_loops:
        cubics, exact = _loop_to_cubics(loop, directed, edge_chain, chains)
        loops_by_region[region].append(cubics)
        exact_by_region[region].append(exact)

    region_values = sorted(int(value) for value in np.unique(labels) if value > 0)
    missing = [region for region in region_values if region not in loops_by_region]
    if missing:
        raise RegionVectorizationError(
            "no boundary loop was found for region(s) %s" % missing)

    paths = []
    exact_loops = []
    for region in region_values:
        loops = tuple(loops_by_region[region])
        exact_loops.extend(exact_by_region[region])
        area = int(np.count_nonzero(labels == region))
        perimeter = 0.0
        for loop in loops:
            for cubic in loop:
                # dense polyline for a stable perimeter estimate
                t = np.linspace(0.0, 1.0, 17)[:, None]
                mt = 1.0 - t
                points = ((mt ** 3) * cubic[0] + 3.0 * (mt ** 2) * t * cubic[1]
                          + 3.0 * mt * (t ** 2) * cubic[2] + (t ** 3) * cubic[3])
                perimeter += float(np.sum(np.linalg.norm(np.diff(points, axis=0),
                                                         axis=1)))
        paths.append(RegionPath(region=region, loops=loops, area_px=area,
                                perimeter_px=perimeter))

    shared = sum(1 for chain in chains
                 if chain["labels"][0] > 0 and chain["labels"][1] > 0)
    coastline = sum(1 for chain in chains
                    if (chain["labels"][0] > 0) != (chain["labels"][1] > 0))
    warnings = []

    return RegionLayout(
        labels=labels,
        width=width,
        height=height,
        edges=edges,
        chains=tuple(Chain(
            lattice=chain["lattice"],
            points=chain["points"],
            edge_ids=chain["edge_ids"],
            labels=chain["labels"],
            closed=chain["closed"],
            cubics=chain["cubics"],
        ) for chain in chains),
        paths=tuple(paths),
        exact_loops=tuple(exact_loops),
        region_count=len(region_values),
        chain_count=len(chains),
        shared_chain_count=shared,
        coastline_chain_count=coastline,
        curve_count=sum(len(loop) for path in paths for loop in path.loops),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# SVG output
# ---------------------------------------------------------------------------


def default_palette(count: int) -> list[str]:
    """A deterministic, reasonably separated palette for region fills."""
    colours = []
    for index in range(max(1, count)):
        hue = (index * 0.6180339887498949) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.42, 0.82)
        colours.append("#%02x%02x%02x" % (int(red * 255),
                                          int(green * 255),
                                          int(blue * 255)))
    return colours


def _number(value: float, precision: int) -> str:
    text = ("%.*f" % (precision, float(value))).rstrip("0").rstrip(".")
    if text in ("", "-0"):
        return "0"
    return text


def _cubics_to_d(cubics, precision: int, close: bool) -> str:
    """One subpath: M start, then C for every cubic."""
    if not cubics:
        return ""
    first = cubics[0][0]
    parts = ["M %s,%s" % (_number(first[0], precision),
                          _number(first[1], precision))]
    current = first
    for cubic in cubics:
        cubic = np.asarray(cubic, dtype=float)
        if not np.allclose(cubic[0], current, atol=1e-6):
            parts.append("M %s,%s" % (_number(cubic[0][0], precision),
                                      _number(cubic[0][1], precision)))
        parts.append("C %s,%s %s,%s %s,%s" % (
            _number(cubic[1][0], precision), _number(cubic[1][1], precision),
            _number(cubic[2][0], precision), _number(cubic[2][1], precision),
            _number(cubic[3][0], precision), _number(cubic[3][1], precision)))
        current = cubic[3]
    if close:
        parts.append("Z")
    return " ".join(parts)


def build_svg(layout: RegionLayout, colors: Optional[list[str]] = None,
              style: str = "fill", stroke: str = "#111111",
              stroke_width: float = 1.25, precision: int = 2,
              seam_width: float = 0.5) -> str:
    """
    Compose an SVG from a region layout.

    style="fill"
        One path per region, filled, no visible outline. Adjacent fills share
        exact geometry, so no stroke width is visible between them. A hairline
        stroke in the *same colour as the fill* (``seam_width``) is used to hide
        the anti-aliasing seam that some renderers show between two adjacent
        fills.
    style="boundary"
        One path containing every boundary chain once, stroked and unfilled.
        This is the clean single-line version of the same geography.
    """
    if style not in ("fill", "boundary"):
        raise RegionVectorizationError("unknown region style %r" % (style,))
    width, height = layout.width, layout.height
    if colors is None:
        colors = default_palette(len(layout.paths))

    header = [
        '<?xml version="1.0" standalone="no"?>',
        '<svg width="%dpx" height="%dpx" viewBox="0 0 %d %d" '
        'xmlns="http://www.w3.org/2000/svg" version="1.1" '
        'shape-rendering="geometricPrecision">' % (width, height, width, height),
    ]
    style_lines = ['<style type="text/css"><![CDATA[']

    if style == "fill":
        for index, path in enumerate(layout.paths):
            colour = colors[index % len(colors)]
            if seam_width > 0:
                stroke_rule = ("stroke:%s; stroke-width:%s; "
                               "stroke-linejoin:round;" % (colour, seam_width))
            else:
                stroke_rule = "stroke:none;"
            style_lines.append(
                ".region-%d {fill:%s; fill-opacity:1; %s}"
                % (path.region, colour, stroke_rule))
        style_lines.append("]]></style>")
        body = []
        for index, path in enumerate(layout.paths):
            d = " ".join(_cubics_to_d(loop, precision, close=True)
                         for loop in path.loops)
            body.append('<path class="region-%d" fill-rule="evenodd" d="%s" />'
                        % (path.region, d))
    else:
        style_lines.append(
            ".boundary {fill:none; stroke:%s; stroke-width:%s; "
            "stroke-linecap:round; stroke-linejoin:round; "
            "vector-effect:non-scaling-stroke;}" % (stroke, stroke_width))
        style_lines.append("]]></style>")
        pieces = []
        for chain in layout.chains:
            pieces.append(_cubics_to_d(chain.cubics, precision,
                                       close=chain.closed))
        body = ['<path class="boundary" d="%s" />' % " ".join(pieces)]

    return "\n".join(header + style_lines + body + ["</svg>"]) + "\n"


# ---------------------------------------------------------------------------
# helpers for accuracy / tests
# ---------------------------------------------------------------------------


def cubics_to_polyline(cubics, samples: int = 16) -> np.ndarray:
    """Flatten a list of cubics into one polyline for raster comparison."""
    if not cubics:
        return np.zeros((0, 2), dtype=float)
    chunks = [np.asarray(cubics[0][0], dtype=float)[None, :]]
    t = np.linspace(0.0, 1.0, samples + 1)[1:, None]
    for cubic in cubics:
        cubic = np.asarray(cubic, dtype=float)
        mt = 1.0 - t
        chunks.append((mt ** 3) * cubic[0] + 3.0 * (mt ** 2) * t * cubic[1]
                      + 3.0 * mt * (t ** 2) * cubic[2] + (t ** 3) * cubic[3])
    return np.vstack(chunks)


def layout_loops(layout: RegionLayout, samples: int = 16) -> list[np.ndarray]:
    """All region loops as flattened closed polylines, for accuracy.evaluate."""
    loops = []
    for path in layout.paths:
        for loop in path.loops:
            polyline = cubics_to_polyline(loop, samples)
            if len(polyline) >= 3:
                loops.append(polyline)
    return loops


def exact_region_loops(labels: np.ndarray) -> list[np.ndarray]:
    """
    Region loops on the pixel lattice, before any curve fitting.

    This is the reference geometry for ``accuracy.evaluate`` in region mode. It
    is much cheaper than :func:`build_layout` because it only walks the boundary
    graph; it does not fit a single curve.
    """
    labels = check_labels(labels)
    edges = build_boundary_edges(labels)
    if not edges:
        return []
    directed, loops = trace_face_loops(labels, edges)
    result = []
    for _, loop in loops:
        points = np.asarray([directed[item][0] for item in loop]
                            + [directed[loop[-1]][1]], dtype=np.float64) - 0.5
        result.append(points)
    return result


