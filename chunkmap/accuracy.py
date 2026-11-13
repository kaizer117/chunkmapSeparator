"""
accuracy.py - measure how faithful a raster -> vector conversion is.

The converter takes a hand drawn raster line and replaces it with smooth cubic
bezier curves. Two independent questions have to be answered to know whether
that worked:

geometry
    how far (in pixels) does the fitted curve sit from the raster contour it
    replaces? Measured in both directions, so a curve that cuts a corner and a
    curve that bulges out are both visible.

coverage
    if the resulting SVG is rasterised again, how much of the original ink is
    covered? IoU / Dice / precision / recall against the input mask.

Nothing here is allowed to import the converter, so it can also score an SVG
produced by something else (for example the old saveCons() polygon writer):

    python code/accuracy.py input.png outputs/outline.svg
    python code/accuracy.py input.png outputs/outline.svg --json report.json
"""

import argparse
import json
import re
import sys

import cv2 as cv
import numpy as np

# the alpha cut used by main.py / the CLI: alpha > 252 counts as ink
ALPHA_THRESHOLD = 252
LUMINANCE_THRESHOLD = 128

# points sampled per cubic bezier when the path is flattened for comparison
CURVE_SAMPLES = 16


# ---------------------------------------------------------------------------
# raster side
# ---------------------------------------------------------------------------

def describe_alpha(img):
    """
    Summarise the alpha channel of an image.

    :param img: image as returned by cv.imread(..., cv.IMREAD_UNCHANGED)
    :return: dict with present / usable / min / max / mean / transparent_pct /
             opaque_pct
    """
    info = {
        "present": False,
        "usable": False,
        "min": None,
        "max": None,
        "mean": None,
        "transparent_pct": None,
        "opaque_pct": None,
    }
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return info

    alpha = img[:, :, 3]
    total = float(alpha.size)
    info["present"] = True
    info["min"] = int(alpha.min())
    info["max"] = int(alpha.max())
    info["mean"] = float(alpha.mean())
    info["transparent_pct"] = 100.0 * float(np.count_nonzero(alpha == 0)) / total
    info["opaque_pct"] = 100.0 * float(np.count_nonzero(alpha == 255)) / total
    # an alpha channel that is a single constant carries no shape information
    info["usable"] = bool(info["min"] != info["max"])
    return info


def prepare_mask(img, mode="auto", alpha_threshold=ALPHA_THRESHOLD,
                 luminance_threshold=LUMINANCE_THRESHOLD, invert=False):
    """
    Turn an image into the 8-bit ink mask the vectorizer traces.

    :param img: image as returned by cv.imread(..., cv.IMREAD_UNCHANGED)
    :param mode: 'auto', 'alpha' or 'luminance'
    :param alpha_threshold: alpha values above this count as ink
    :param luminance_threshold: luminance values below this count as ink
    :param invert: luminance mode only, treat light pixels as ink
    :return: (mask uint8 0/255, mode actually used, note string)
    """
    if img is None:
        raise ValueError("no image to build a mask from")

    note = ""
    resolved = mode
    if mode == "auto":
        alpha = describe_alpha(img)
        if alpha["present"] and alpha["usable"]:
            resolved = "alpha"
        else:
            resolved = "luminance"
            if alpha["present"]:
                note = ("alpha channel is uniform (min=%s max=%s), so it describes "
                        "nothing; using luminance" % (alpha["min"], alpha["max"]))
            elif img.ndim == 3 and img.shape[2] == 3:
                note = "image has no alpha channel; using luminance"
            else:
                note = "image is greyscale; using luminance"

    if resolved == "alpha":
        if img.ndim != 3 or img.shape[2] != 4:
            channels = 1 if img.ndim == 2 else img.shape[2]
            raise ValueError("--mask alpha needs a 4 channel RGBA image, got %d channel(s)"
                             % channels)
        mask = np.where(img[:, :, 3] > alpha_threshold, 255, 0).astype(np.uint8)
        return mask, resolved, note

    if resolved != "luminance":
        raise ValueError("unknown mask mode: %r" % (mode,))

    # flatten transparency onto white before looking at brightness
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        composite = rgb * alpha + 255.0 * (1.0 - alpha)
        grey = cv.cvtColor(composite.astype(np.uint8), cv.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        grey = img

    foreground = grey < luminance_threshold
    if invert:
        foreground = ~foreground
    mask = np.where(foreground, 255, 0).astype(np.uint8)
    return mask, resolved, note


def trace_contours(mask):
    """
    Trace the mask the same way the converter does.

    :return: (list of (n,2) int arrays, hierarchy array)
    """
    contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2) for c in contours], hierarchy


# ---------------------------------------------------------------------------
# vector side
# ---------------------------------------------------------------------------

def sample_cubic(p0, c1, c2, p3, samples=CURVE_SAMPLES):
    """
    Flatten one cubic bezier into points.

    :return: (samples, 2) array, including p3 and excluding p0
    """
    p0 = np.asarray(p0, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    t = np.linspace(0.0, 1.0, samples + 1)[1:, None]
    mt = 1.0 - t
    return (mt ** 3) * p0 + 3.0 * (mt ** 2) * t * c1 + 3.0 * mt * (t ** 2) * c2 + (t ** 3) * p3


def ctrl_to_loop(ctrlPoints, samples=CURVE_SAMPLES):
    """
    Flatten the control point layout produced by vectorization.controlPointFormatter
    ( [4 points], [3 points], [3 points], ... ) into one closed polyline.

    :return: (m, 2) float array
    """
    rows = [np.asarray(row, dtype=float) for row in ctrlPoints]
    if not rows:
        return np.zeros((0, 2))

    first = rows[0]
    chunks = [first[0][None, :], sample_cubic(first[0], first[1], first[2], first[3], samples)]
    previous = first[3]
    for row in rows[1:]:
        chunks.append(sample_cubic(previous, row[0], row[1], row[2], samples))
        previous = row[2]
    return np.vstack(chunks)


_TOKEN = re.compile(
    r"([MmZzLlHhVvCcSsQqTtAa])|([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)


def _tokenize_path(d):
    tokens = []
    for match in _TOKEN.finditer(d):
        command, number = match.group(1), match.group(2)
        tokens.append(command if command is not None else float(number))
    return tokens


def parse_path_data(d):
    """
    Parse an SVG path 'd' attribute into closed polylines.

    Supports the absolute and relative forms of M, L, H, V and C plus Z, which is
    everything the vectorizer emits. Anything else raises ValueError rather than
    being silently skipped.

    :return: list of (n, 2) float arrays
    """
    tokens = _tokenize_path(d)
    loops = []
    current = []
    position = np.zeros(2)
    start = np.zeros(2)
    command = None
    index = 0
    total = len(tokens)

    def flush():
        if len(current) > 1:
            loops.append(np.asarray(current, dtype=float))

    while index < total:
        token = tokens[index]
        if isinstance(token, str):
            command = token
            index += 1
            if command in ("Z", "z"):
                flush()
                current = []
                position = start.copy()
            continue

        if command is None:
            raise ValueError("path data starts with a number")
        upper = command.upper()
        relative = command.islower()
        offset = position if relative else np.zeros(2)

        if upper == "M":
            position = np.array([tokens[index], tokens[index + 1]], dtype=float) + offset
            index += 2
            flush()
            current = [tuple(position)]
            start = position.copy()
            command = "l" if relative else "L"
        elif upper == "L":
            position = np.array([tokens[index], tokens[index + 1]], dtype=float) + offset
            index += 2
            current.append(tuple(position))
        elif upper == "H":
            position = np.array([tokens[index] + offset[0], position[1]], dtype=float)
            index += 1
            current.append(tuple(position))
        elif upper == "V":
            position = np.array([position[0], tokens[index] + offset[1]], dtype=float)
            index += 1
            current.append(tuple(position))
        elif upper == "C":
            c1 = np.array([tokens[index], tokens[index + 1]], dtype=float) + offset
            c2 = np.array([tokens[index + 2], tokens[index + 3]], dtype=float) + offset
            end = np.array([tokens[index + 4], tokens[index + 5]], dtype=float) + offset
            index += 6
            current.extend(map(tuple, sample_cubic(position, c1, c2, end)))
            position = end
        else:
            raise ValueError("unsupported SVG path command: %r" % (command,))

    flush()
    return loops


def parse_svg(svg_text):
    """
    Pull the geometry out of an SVG document.

    :return: dict with width, height, paths, segments, loops
    """
    width = height = None
    header = re.search(r"<svg\b[^>]*>", svg_text, re.IGNORECASE | re.DOTALL)
    if header:
        tag = header.group(0)
        found = re.search(r'\bwidth="([\d.]+)', tag)
        if found:
            width = float(found.group(1))
        found = re.search(r'\bheight="([\d.]+)', tag)
        if found:
            height = float(found.group(1))
        found = re.search(r'\bviewBox="([^"]+)"', tag)
        if found and (width is None or height is None):
            parts = [float(p) for p in re.split(r"[,\s]+", found.group(1).strip()) if p]
            if len(parts) == 4:
                width, height = parts[2], parts[3]

    loops = []
    paths = 0
    segments = 0
    for d in re.findall(r'<path\b[^>]*\bd="([^"]*)"', svg_text, re.IGNORECASE | re.DOTALL):
        paths += 1
        segments += len(re.findall(r"[CcSsQqTtAa]", d))
        loops.extend(parse_path_data(d))

    return {
        "width": width,
        "height": height,
        "paths": paths,
        "segments": segments,
        "loops": loops,
    }


def loops_to_mask(loops, shape):
    """
    Rasterise closed loops with the even-odd rule.

    Every contour the vectorizer traces is either an outer boundary or a hole, so
    even-odd reproduces the original ink mask.
    """
    height, width = shape[0], shape[1]
    accumulator = np.zeros((height, width), dtype=np.uint8)
    for loop in loops:
        points = np.asarray(loop, dtype=float)
        if len(points) < 3:
            continue
        layer = np.zeros_like(accumulator)
        cv.fillPoly(layer, [np.round(points).astype(np.int32).reshape(-1, 1, 2)], 255)
        accumulator = cv.bitwise_xor(accumulator, layer)
    return accumulator


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def mask_scores(reference, candidate, tolerance=1):
    """
    IoU / Dice / precision / recall / pixel accuracy of two binary masks.

    The strict scores compare filled areas directly. That number is biased low for
    line art: cv.findContours returns the centres of the boundary pixels, so the
    polygon through them is half a pixel inside the ink on every side and a 3 px
    stroke loses about a third of its area when refilled. The 'tolerant' block
    therefore also scores pixels that land within `tolerance` px of the other
    mask, which is the honest question ("is the curve on the ink?").
    """
    reference = reference > 0
    candidate = candidate > 0
    true_positive = int(np.count_nonzero(reference & candidate))
    false_positive = int(np.count_nonzero(candidate & ~reference))
    false_negative = int(np.count_nonzero(reference & ~candidate))
    union = true_positive + false_positive + false_negative

    def ratio(numerator, denominator, empty=1.0):
        return float(numerator) / float(denominator) if denominator else empty

    scores = {
        "iou": ratio(true_positive, union),
        "dice": ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        "precision": ratio(true_positive, true_positive + false_positive),
        "recall": ratio(true_positive, true_positive + false_negative),
        "pixel_accuracy": ratio(reference.size - false_positive - false_negative, reference.size),
        "reference_px": int(np.count_nonzero(reference)),
        "candidate_px": int(np.count_nonzero(candidate)),
        "false_positive_px": false_positive,
        "false_negative_px": false_negative,
    }

    tolerant = {"tolerance_px": int(tolerance), "iou": scores["iou"],
                "dice": scores["dice"], "precision": scores["precision"],
                "recall": scores["recall"]}
    if tolerance and tolerance > 0:
        size = 2 * int(tolerance) + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
        reference_near = cv.dilate(reference.astype(np.uint8), kernel) > 0
        candidate_near = cv.dilate(candidate.astype(np.uint8), kernel) > 0
        near_tp = int(np.count_nonzero(candidate & reference_near))
        near_fp = int(np.count_nonzero(candidate & ~reference_near))
        near_fn = int(np.count_nonzero(reference & ~candidate_near))
        near_union = near_tp + near_fp + near_fn
        tolerant = {
            "tolerance_px": int(tolerance),
            "iou": ratio(near_tp, near_union),
            "dice": ratio(2 * near_tp, 2 * near_tp + near_fp + near_fn),
            "precision": ratio(near_tp, near_tp + near_fp),
            "recall": ratio(near_tp, near_tp + near_fn),
            "false_positive_px": near_fp,
            "false_negative_px": near_fn,
        }
    scores["tolerant"] = tolerant
    return scores


def _describe_distances(values):
    if len(values) == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "n": int(values.size),
    }


def boundary_scores(raster_contours, loops, shape):
    """
    Distance between the raster contour and the flattened vector curves, in pixels.

    Both directions are measured and a symmetric Hausdorff distance is reported.
    """
    height, width = shape[0], shape[1]
    raster_line = np.zeros((height, width), dtype=np.uint8)
    for contour in raster_contours:
        points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv.polylines(raster_line, [points], True, 255, 1)

    vector_line = np.zeros((height, width), dtype=np.uint8)
    for loop in loops:
        points = np.round(np.asarray(loop, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
        cv.polylines(vector_line, [points], True, 255, 1)

    if not np.any(raster_line) or not np.any(vector_line):
        return {
            "raster_to_vector": {"mean": 0.0, "p95": 0.0, "max": 0.0, "n": 0},
            "vector_to_raster": {"mean": 0.0, "p95": 0.0, "max": 0.0, "n": 0},
            "hausdorff": 0.0,
            "within_1px_pct": 0.0,
            "within_2px_pct": 0.0,
            "worst_contours": [],
        }

    distance_to_vector = cv.distanceTransform(255 - vector_line, cv.DIST_L2, 3)
    distance_to_raster = cv.distanceTransform(255 - raster_line, cv.DIST_L2, 3)

    raster_y, raster_x = np.nonzero(raster_line)
    raster_to_vector = distance_to_vector[raster_y, raster_x].astype(np.float64)
    vector_y, vector_x = np.nonzero(vector_line)
    vector_to_raster = distance_to_raster[vector_y, vector_x].astype(np.float64)

    # per contour, so a single bad fit cannot hide behind a good global average
    per_contour = []
    for i, contour in enumerate(raster_contours):
        ys = np.asarray(contour)[:, 1].astype(int)
        xs = np.asarray(contour)[:, 0].astype(int)
        inside = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        values = distance_to_vector[ys[inside], xs[inside]]
        if values.size == 0:
            continue
        per_contour.append({
            "contour": i,
            "raster_points": int(len(contour)),
            "mean_px": float(values.mean()),
            "max_px": float(values.max()),
        })
    per_contour.sort(key=lambda item: item["mean_px"], reverse=True)

    return {
        "raster_to_vector": _describe_distances(raster_to_vector),
        "vector_to_raster": _describe_distances(vector_to_raster),
        "hausdorff": float(max(raster_to_vector.max(), vector_to_raster.max())),
        "within_1px_pct": 100.0 * float(np.count_nonzero(raster_to_vector <= 1.0))
                          / float(raster_to_vector.size),
        "within_2px_pct": 100.0 * float(np.count_nonzero(raster_to_vector <= 2.0))
                          / float(raster_to_vector.size),
        "worst_contours": per_contour[:5],
    }


def evaluate(mask, loops, raster_contours=None, segments=None, control_points=None,
             tolerance=1):
    """
    Score a set of vector loops against the raster mask they came from.

    :param mask: 8-bit ink mask
    :param loops: list of (n, 2) polylines, one per contour
    :param raster_contours: the contours traced from mask (defaults to a fresh trace)
    :param segments: number of cubic segments, used for the compression ratio
    :param control_points: number of control points, used for the compression ratio
    :param tolerance: pixel tolerance for the tolerant coverage scores
    :return: metrics dict
    """
    if raster_contours is None:
        raster_contours, _ = trace_contours(mask)

    reconstruction = loops_to_mask(loops, mask.shape)
    coverage = mask_scores(mask, reconstruction, tolerance)
    boundary = boundary_scores(raster_contours, loops, mask.shape)

    # The contour points cv.findContours returns sit on the centres of the
    # boundary pixels, so even refilling the exact raster contours loses area on
    # every stroke. Score that reconstruction too: it is the ceiling any vector
    # approximation of these contours can reach under this metric.
    exact = loops_to_mask([np.asarray(c, dtype=float) for c in raster_contours], mask.shape)
    baseline = mask_scores(mask, exact, tolerance)
    relative = {
        "strict_iou": (coverage["iou"] / baseline["iou"]) if baseline["iou"] else 1.0,
        "tolerant_iou": (coverage["tolerant"]["iou"] / baseline["tolerant"]["iou"])
                        if baseline["tolerant"]["iou"] else 1.0,
    }

    raster_points = int(sum(len(np.asarray(c)) for c in raster_contours))
    segments = int(segments) if segments is not None else 0
    control_points = int(control_points) if control_points is not None else 0

    return {
        "contours": len(loops),
        "raster_points": raster_points,
        "segments": segments,
        "control_points": control_points,
        "compression": (float(raster_points) / segments) if segments else 0.0,
        "coverage": coverage,
        "baseline": baseline,
        "relative_to_exact_contours": relative,
        "boundary": boundary,
    }


def format_report(metrics, prefix="  "):
    """Human readable accuracy block as a list of lines."""
    coverage = metrics["coverage"]
    tolerant = coverage.get("tolerant", coverage)
    boundary = metrics["boundary"]
    baseline = metrics.get("baseline", coverage)
    relative = metrics.get("relative_to_exact_contours", {})
    tolerance = tolerant.get("tolerance_px", 0)
    lines = [
        "%scoverage   IoU %.4f   Dice %.4f   precision %.4f   recall %.4f   (%d px tolerance)"
        % (prefix, tolerant["iou"], tolerant["dice"], tolerant["precision"], tolerant["recall"],
           tolerance),
        "%s           strict IoU %.4f   ceiling %.4f from the exact raster contour   -> %.1f%%"
        % (prefix, coverage["iou"], baseline["iou"],
           100.0 * relative.get("strict_iou", 1.0)),
        "%sboundary   raster->vector  mean %.2f px   p95 %.2f px   max %.2f px"
        % (prefix, boundary["raster_to_vector"]["mean"], boundary["raster_to_vector"]["p95"],
           boundary["raster_to_vector"]["max"]),
        "%s           vector->raster  mean %.2f px   p95 %.2f px   max %.2f px"
        % (prefix, boundary["vector_to_raster"]["mean"], boundary["vector_to_raster"]["p95"],
           boundary["vector_to_raster"]["max"]),
        "%s           hausdorff %.2f px   within 1 px %.1f%%   within 2 px %.1f%%"
        % (prefix, boundary["hausdorff"], boundary["within_1px_pct"], boundary["within_2px_pct"]),
        "%scompress   %d raster points -> %d segments (%.1fx)   ink %d px -> %d px filled"
        % (prefix, metrics["raster_points"], metrics["segments"], metrics["compression"],
           coverage["reference_px"], coverage["candidate_px"]),
    ]
    return lines


# ---------------------------------------------------------------------------
# standalone entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="accuracy.py",
        description="Score an SVG against the raster it was traced from.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python code/accuracy.py input.png outputs/outline.svg\n"
            "  python code/accuracy.py input.png outputs/outline.svg --json report.json\n"
            "  python code/accuracy.py input.png resources/cuba_ex.svg --mask luminance\n"
        ),
    )
    parser.add_argument("raster", help="source raster image (png)")
    parser.add_argument("svg", help="SVG to score")
    parser.add_argument("--mask", choices=("auto", "alpha", "luminance"), default="auto",
                        help="how to derive ink from the raster (default: auto)")
    parser.add_argument("--alpha-threshold", type=int, default=ALPHA_THRESHOLD,
                        help="alpha above this is ink (default: %(default)s)")
    parser.add_argument("--luminance-threshold", type=int, default=LUMINANCE_THRESHOLD,
                        help="luminance below this is ink (default: %(default)s)")
    parser.add_argument("--invert", action="store_true",
                        help="luminance mode: light pixels are the ink")
    parser.add_argument("--json", metavar="PATH", help="write the full metrics as JSON")
    parser.add_argument("--quiet", action="store_true", help="only print the SVG file summary")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    image = cv.imread(args.raster, cv.IMREAD_UNCHANGED)
    if image is None:
        print("error: cannot read image %s" % args.raster, file=sys.stderr)
        return 2

    try:
        mask, mode, note = prepare_mask(
            image, args.mask, args.alpha_threshold, args.luminance_threshold, args.invert)
    except ValueError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2

    try:
        with open(args.svg, "r", encoding="utf-8") as handle:
            svg_text = handle.read()
    except OSError as error:
        print("error: cannot read %s (%s)" % (args.svg, error), file=sys.stderr)
        return 2

    document = parse_svg(svg_text)
    contours, _ = trace_contours(mask)
    metrics = evaluate(mask, document["loops"], contours,
                       document["segments"], None)
    metrics["segments"] = document["segments"]
    metrics["compression"] = (float(metrics["raster_points"]) / document["segments"]
                              if document["segments"] else 0.0)

    size_match = None
    if document["width"] and document["height"]:
        size_match = (int(document["width"]) == image.shape[1]
                      and int(document["height"]) == image.shape[0])

    if not args.quiet:
        print("raster     %s  (%dx%d)" % (args.raster, image.shape[1], image.shape[0]))
        print("mask       %s%s" % (mode, ("  [" + note + "]") if note else ""))
        print("vector     %s  %d paths, %d curves, %d loops"
              % (args.svg, document["paths"], document["segments"], len(document["loops"])))
        if size_match is not None:
            print("size       svg %sx%s -> %s" % (
                int(document["width"]), int(document["height"]),
                "matches raster" if size_match else "DIFFERS from raster"))
        for line in format_report(metrics):
            print(line)

    if args.json:
        payload = {
            "raster": args.raster,
            "svg": args.svg,
            "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "alpha": describe_alpha(image),
            "mask": {"mode": mode, "note": note},
            "vector": {
                "paths": document["paths"],
                "segments": document["segments"],
                "loops": len(document["loops"]),
                "width": document["width"],
                "height": document["height"],
                "size_matches_raster": size_match,
            },
            "metrics": metrics,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        if not args.quiet:
            print("json       %s" % args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
