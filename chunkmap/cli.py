"""
cli.py - raster line art to smooth bezier SVG.

The whole point of the project: a hand drawn line (a PNG whose ink lives in the
alpha channel) goes in, a clean vector outline comes out.

    python main.py resources/example-cuba.png
    python main.py drawing.png -o vector.svg --report
    python main.py scan.png --mask luminance --invert --stroke "#000000"
    python main.py generate outputs/islands 8 1024 768 --seed 42
    python main.py orchestrate --config configs/island-svg-sample.yaml

Only the existing pipeline functions are used: cvutils for the contour reshape,
segmentation + vectorization for the curve fitting, fileutils.SVGHandler for the
SVG itself and accuracy.py for the report. The ``generate`` subcommand delegates to
island_generator.py and writes PNGs instead of an SVG. The ``orchestrate``
subcommand delegates to orchestration.py: it generates a batch of islands,
vectorizes them, renders overlays and scores the batch.
"""

import argparse
import json
import os
import sys
import time

import cv2 as cv
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Imported lazily so the generate subcommand does not pull in matplotlib and the
# rest of the SVG pipeline. The names are populated by _load_pipeline().
cutils = futils = colutils = seg = vect = accuracy = None


def _load_pipeline():
    global cutils, futils, colutils, seg, vect, accuracy
    if accuracy is not None:
        return
    from . import accuracy as accuracy_module
    from . import colorspace as colorspace_module
    from . import cvutils as cvutils_module
    from . import fileutils as fileutils_module
    from . import segmentation as segmentation_module
    from . import vectorization as vectorization_module

    accuracy = accuracy_module
    colutils = colorspace_module
    cutils = cvutils_module
    futils = fileutils_module
    seg = segmentation_module
    vect = vectorization_module


class CliError(Exception):
    """Expected, user facing failure."""


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def load_image(path):
    if not os.path.isfile(path):
        raise CliError("input file not found: %s" % path)
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    if image is None:
        raise CliError("cannot decode image: %s" % path)
    return image


# ---------------------------------------------------------------------------
# contours
# ---------------------------------------------------------------------------

def hierarchy_depths(hierarchy):
    """
    Nesting depth of every contour (0 = outer boundary, 1 = hole, ...).

    :param hierarchy: the array cv.findContours returns
    :return: list of ints, same length as the contour list
    """
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    depths = []
    for index in range(len(hierarchy)):
        depth = 0
        parent = int(hierarchy[index][3])
        seen = 0
        while parent >= 0 and seen < len(hierarchy):
            depth += 1
            parent = int(hierarchy[parent][3])
            seen += 1
        depths.append(depth)
    return depths


def collect_contours(mask, min_points, min_area):
    """
    Trace the ink mask and drop contours that are too small to vectorise.

    :return: (kept, skipped, depths) where kept is a list of dicts
    """
    raw, hierarchy = accuracy.trace_contours(mask)
    depths = hierarchy_depths(hierarchy)

    kept = []
    skipped = []
    for index, contour in enumerate(raw):
        points = cutils.conReshaper(contour.astype(np.float64))
        if len(points) < min_points:
            skipped.append({"contour": index, "points": int(len(points)),
                            "reason": "too few points"})
            continue
        if min_area > 0 and cv.contourArea(points.astype(np.float32)) < min_area:
            skipped.append({"contour": index, "points": int(len(points)),
                            "reason": "below min area"})
            continue
        kept.append({
            "index": index,
            "points": points,
            "depth": depths[index] if index < len(depths) else 0,
        })
    return kept, skipped


# ---------------------------------------------------------------------------
# vectorisation
# ---------------------------------------------------------------------------

def segment_contour(points, args):
    """Split one raster contour into runs to fit a cubic to."""
    if args.segmentation == "curvature":
        return seg.polygon_curvature_segmentation(
            points,
            min_chunk=args.min_chunk,
            max_chunk=args.max_chunk,
            smooth_sigma=args.smooth_sigma,
        )
    return vect.polygonSegmentation(points, chunk_size=args.chunk_size)


def vectorize_contour(points, args):
    """
    Fit the contour and reshape the control points for SVG output.

    :return: (formatted control points, number of segments, number of control points)
    """
    segments = segment_contour(points, args)
    if not segments:
        return [], 0, 0

    fits = [vect.fitBezierCurve(segment) for segment in segments]
    formatted = vect.controlPointFormatter(fits, closed=True)
    control_points = sum(len(row) for row in formatted)
    return formatted, len(segments), control_points


# ---------------------------------------------------------------------------
# svg
# ---------------------------------------------------------------------------

def outline_style(stroke, width):
    return {
        "fill": "none",
        "stroke": stroke,
        "stroke-width": str(width),
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "vector-effect": "non-scaling-stroke",
    }


def palette_style(stroke, width, colour):
    style = outline_style(stroke, width)
    style["fill"] = colour
    style["fill-opacity"] = "0.75"
    return style


def build_svg(size, entries, args):
    """
    Compose the SVG document.

    :param size: (width, height) in px
    :param entries: list of dicts with 'ctrl' (bezier) or 'polygon' (straight lines)
    :return: (svg text, style dictionary)
    """
    handler = futils.SVGHandler()
    handler.initialize(size)

    styles = {}
    if args.fill == "palette":
        colours = colutils.cmapHex(args.hue, max(1, len(entries)),
                                   args.sampling, args.picking)
        for index, entry in enumerate(entries):
            class_name = "shape-%d" % index
            entry["class"] = class_name
            styles[class_name] = palette_style(args.stroke, args.stroke_width,
                                               colours[index % len(colours)])
    else:
        for entry in entries:
            entry["class"] = "outline"
        styles["outline"] = outline_style(args.stroke, args.stroke_width)

    handler.addstyletag(styles)
    for entry in entries:
        if args.curve == "polygon":
            polygon = np.vstack([entry["polygon"], entry["polygon"][:1]])
            handler.drawline(polygon, entry["class"])
        else:
            handler.drawcubicbezier(entry["ctrl"], entry["class"], closed=True)

    handler.compose()
    return handler.file


def write_text(path, text):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def progress(done, total, quiet):
    if quiet or not sys.stderr.isatty():
        return
    width = 28
    filled = int(width * done / max(1, total))
    sys.stderr.write("\r  fitting [%s%s] %d/%d" % (
        "#" * filled, "-" * (width - filled), done, total))
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def describe_alpha_line(alpha):
    if not alpha["present"]:
        return "absent (image has no alpha channel)"
    return ("present  min %d  max %d  mean %.1f  transparent %.1f%%  opaque %.1f%%  -> %s"
            % (alpha["min"], alpha["max"], alpha["mean"], alpha["transparent_pct"],
               alpha["opaque_pct"], "usable" if alpha["usable"] else "flat, not usable"))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Convert hand drawn raster line art into a smooth bezier SVG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "subcommands:\n"
            "  generate OUTPUT COUNT WIDTH HEIGHT   generate random island PNGs\n"
            "  orchestrate --config PATH           island batch -> SVG -> overlay -> accuracy\n"
            "\n"
            "examples:\n"
            "  python main.py resources/example-cuba.png\n"
            "  python main.py ink.png -o ink.svg --stroke \"#000000\" --stroke-width 1\n"
            "  python main.py scan.png --mask luminance --luminance-threshold 160\n"
            "  python main.py ink.png --segmentation blind --chunk-size 8\n"
            "  python main.py ink.png --fill palette --hue 12 --sampling random --seed 7\n"
            "  python main.py ink.png --curve polygon            # straight line baseline\n"
            "  python main.py generate outputs/islands 8 1024 768 --seed 42\n"
            "  python main.py orchestrate --config configs/island-svg-sample.yaml\n"
        ),
    )
    parser.add_argument("input", help="input raster image (png)")
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="output SVG (default: outputs/<input name>.svg)")

    vector = parser.add_argument_group("vectorisation")
    vector.add_argument("--curve", choices=("bezier", "polygon"), default="bezier",
                        help="bezier curves or straight line baseline (default: bezier)")
    vector.add_argument("--segmentation", choices=("curvature", "blind"), default="curvature",
                        help="how to split a contour before fitting (default: curvature)")
    vector.add_argument("--min-chunk", type=int, default=5,
                        help="curvature mode: shortest segment in raster points (default: 5)")
    vector.add_argument("--max-chunk", type=int, default=15,
                        help="curvature mode: longest segment in raster points (default: 15)")
    vector.add_argument("--smooth-sigma", type=float, default=2.0,
                        help="curvature mode: gaussian sigma for curvature (default: 2.0)")
    vector.add_argument("--chunk-size", type=int, default=10,
                        help="blind mode: target points per segment (default: 10)")
    vector.add_argument("--min-contour-points", type=int, default=4,
                        help="drop contours shorter than this (default: 4)")
    vector.add_argument("--min-area", type=float, default=0.0,
                        help="drop contours smaller than this area in px^2 (default: 0)")

    mask = parser.add_argument_group("ink mask")
    mask.add_argument("--mask", choices=("auto", "alpha", "luminance"), default="auto",
                      help="where the ink is (default: auto)")
    mask.add_argument("--alpha-threshold", type=int, default=252,
                      help="alpha above this is ink (default: %(default)s)")
    mask.add_argument("--luminance-threshold", type=int, default=128,
                      help="luminance below this is ink (default: %(default)s)")
    mask.add_argument("--invert", action="store_true",
                      help="luminance mode: light pixels are the ink")

    style = parser.add_argument_group("style")
    style.add_argument("--stroke", default="#111111", help="stroke colour (default: %(default)s)")
    style.add_argument("--stroke-width", type=float, default=1.25,
                       help="stroke width in px (default: %(default)s)")
    style.add_argument("--fill", choices=("none", "palette"), default="none",
                       help="none for an outline, palette for filled shapes (default: none)")
    style.add_argument("--hue", type=float, default=12.0,
                       help="palette hue in degrees (default: %(default)s)")
    style.add_argument("--sampling", choices=("linear", "random"), default="linear",
                       help="palette sampling (default: %(default)s)")
    style.add_argument("--picking", choices=("linear", "shuffle"), default="linear",
                       help="palette picking (default: %(default)s)")
    style.add_argument("--seed", type=int, default=None,
                       help="seed the palette generator for reproducible colours")
    style.add_argument("--precision", type=int, default=2,
                       help="coordinate decimals written to the SVG (default: %(default)s)")

    region = parser.add_argument_group(
        "region vectorization",
        "Read a labelled region map instead of tracing the ink band. This is the "
        "mode that gives every region one filled path and makes adjacent borders "
        "share the exact same cubic curves. --curve, --segmentation, --mask and "
        "--fill are ignored in this mode.")
    region.add_argument("--regions", metavar="PATH",
                        help="single-channel integer label map (PNG/TIFF/NPY): "
                             "0 = water, positive = region")
    region.add_argument("--region-style", choices=("fill", "boundary"), default="fill",
                        help="filled region polygons, or one boundary network "
                             "(default: fill)")
    region.add_argument("--region-fill", choices=("palette", "solid"), default="palette",
                        help="colour source for filled regions (default: palette)")
    region.add_argument("--region-palette", choices=("spread", "hue"), default="spread",
                        help="spread: distinct hues per region; hue: the legacy "
                             "--hue/--sampling palette (default: spread)")
    region.add_argument("--region-color", default="#9ec79b",
                        help="solid fill colour when --region-fill solid (default: %(default)s)")
    region.add_argument("--region-seam", type=float, default=0.5,
                        help="same-colour hairline stroke that hides renderer seams "
                             "between adjacent fills; 0 disables (default: %(default)s)")
    region.add_argument("--corner-angle", type=float, default=55.0,
                        help="windowed turn in degrees that splits a boundary chain "
                             "(default: %(default)s)")

    report = parser.add_argument_group("reporting")
    report.add_argument("--report", action=argparse.BooleanOptionalAction, default=True,
                        help="measure the result and print an accuracy report (default: on)")
    report.add_argument("--report-json", metavar="PATH",
                        help="write the full report as JSON")
    report.add_argument("--quiet", action="store_true", help="no output except errors")
    report.add_argument("--verbose", action="store_true", help="per contour detail")

    return parser


def default_output(input_path):
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(DEFAULT_OUTPUT_DIR, stem + ".svg")


def default_region_output(input_path):
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(DEFAULT_OUTPUT_DIR, stem + "-regions.svg")


def run_region_mode(args, image, started):
    """
    Vectorize a label map with shared borders.

    This is deliberately a separate path from the ink tracing above: the input
    geometry is the region map, not the antialiased stroke band, so there is no
    contour thresholding to do.
    """
    from . import region_vectorization as regions

    if not (1.0 <= args.corner_angle <= 179.0):
        raise CliError("--corner-angle must be between 1 and 179 degrees")
    if args.region_seam < 0:
        raise CliError("--region-seam must be >= 0")

    try:
        labels = regions.load_label_map(args.regions)
    except regions.RegionVectorizationError as error:
        raise CliError(str(error))

    if labels.shape[:2] != image.shape[:2]:
        raise CliError("label map is %dx%d but input image is %dx%d"
                       % (labels.shape[1], labels.shape[0],
                          image.shape[1], image.shape[0]))

    layout_started = time.time()
    try:
        layout = regions.build_layout(labels, max_chunk=args.max_chunk,
                                      corner_angle_deg=args.corner_angle)
    except regions.RegionVectorizationError as error:
        raise CliError(str(error))
    layout_elapsed = time.time() - layout_started

    if args.region_fill == "palette":
        if args.region_palette == "hue":
            if args.seed is not None:
                np.random.seed(args.seed)
            colors = colutils.cmapHex(args.hue, max(1, len(layout.paths)),
                                      args.sampling, args.picking)
        else:
            colors = regions.default_palette(max(1, len(layout.paths)))
    else:
        colors = [args.region_color] * max(1, len(layout.paths))

    svg_started = time.time()
    try:
        svg_text = regions.build_svg(
            layout, colors=colors, style=args.region_style,
            stroke=args.stroke, stroke_width=args.stroke_width,
            precision=args.precision, seam_width=args.region_seam)
    except regions.RegionVectorizationError as error:
        raise CliError(str(error))
    svg_elapsed = time.time() - svg_started

    output = args.output or default_region_output(args.input)
    write_text(output, svg_text)
    svg_bytes = len(svg_text.encode("utf-8"))
    for warning in layout.warnings:
        if not args.quiet:
            print("warning: %s" % warning, file=sys.stderr)

    metrics = None
    accuracy_elapsed = 0.0
    if args.report:
        if args.region_style == "boundary":
            if not args.quiet:
                print("warning: coverage/boundary report is skipped for "
                      "--region-style boundary (the paths are open borders)",
                      file=sys.stderr)
        else:
            accuracy_started = time.time()
            mask = (labels > 0).astype(np.uint8) * 255
            loops = regions.layout_loops(layout)
            # Compare the fitted region loops with the exact lattice loops, not
            # with the land silhouette: the latter has no internal borders and
            # would make every shared border look like a huge boundary error.
            metrics = accuracy.evaluate(mask, loops, list(layout.exact_loops),
                                        layout.curve_count, None, 1)
            metrics["segments"] = int(layout.curve_count)
            metrics["compression"] = (float(metrics["raster_points"])
                                      / layout.curve_count
                                      if layout.curve_count else 0.0)
            relative = metrics.get("relative_to_exact_contours")
            if relative:
                # The exact-loop baseline is rasterised with cv.fillPoly, whose
                # inclusive boundary can make it fractionally smaller than the
                # region fills. A ratio above 1 is a metric artefact, not a fit
                # better than the exact contour.
                relative["strict_iou"] = min(1.0, float(relative["strict_iou"]))
            accuracy_elapsed = time.time() - accuracy_started

    if not args.quiet:
        width, height = image.shape[1], image.shape[0]
        print("chunkmap-separator  regions -> %s svg" % args.region_style)
        print("input      %s  (%dx%d)" % (args.input, width, height))
        print("label map  %s  (%d region%s)" % (
            args.regions, layout.region_count,
            "" if layout.region_count == 1 else "s"))
        print("regions    %d paths  %d boundary chains (%d shared, %d coastline)"
              % (layout.path_count, layout.chain_count,
                 layout.shared_chain_count, layout.coastline_chain_count))
        print("vector     %d curves  %d raster boundary edges"
              % (layout.curve_count, len(layout.edges)))
        print("timing     layout %.2fs  svg %.2fs  report %.2fs  total %.2fs"
              % (layout_elapsed, svg_elapsed, accuracy_elapsed,
                 time.time() - started))
        print("svg        %.1f KB  ->  %s" % (svg_bytes / 1024.0, output))
        if metrics:
            for line in accuracy.format_report(metrics):
                print(line)

    if args.report_json:
        payload = {
            "input": os.path.abspath(args.input),
            "output": os.path.abspath(output),
            "label_map": os.path.abspath(args.regions),
            "image": {"width": int(image.shape[1]),
                      "height": int(image.shape[0]),
                      "channels": 1 if image.ndim == 2 else int(image.shape[2])},
            "regions": {
                "mode": "regions",
                "style": args.region_style,
                "fill": args.region_fill,
                "palette": args.region_palette,
                "region_count": layout.region_count,
                "path_count": layout.path_count,
                "chain_count": layout.chain_count,
                "shared_chain_count": layout.shared_chain_count,
                "coastline_chain_count": layout.coastline_chain_count,
                "curve_count": layout.curve_count,
                "raster_boundary_edges": len(layout.edges),
                "warnings": list(layout.warnings),
            },
            "svg": {"path": os.path.abspath(output), "bytes": svg_bytes},
            "timing": {"layout_s": layout_elapsed, "svg_s": svg_elapsed,
                       "report_s": accuracy_elapsed,
                       "total_s": time.time() - started},
            "accuracy": metrics,
            "settings": {
                "corner_angle": args.corner_angle,
                "max_chunk": args.max_chunk,
                "region_seam": args.region_seam,
                "stroke": args.stroke,
                "stroke_width": args.stroke_width,
                "hue": args.hue,
                "sampling": args.sampling,
                "picking": args.picking,
                "seed": args.seed,
                "precision": args.precision,
            },
        }
        write_text(args.report_json, json.dumps(payload, indent=2))
        if not args.quiet:
            print("json       %s" % args.report_json)

    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "generate":
        from .island_generator import generate_main
        return generate_main(argv[1:])
    if argv and argv[0] == "orchestrate":
        from .orchestration import orchestrate_main
        return orchestrate_main(argv[1:])

    args = build_parser().parse_args(argv)
    _load_pipeline()
    started = time.time()

    try:
        if not args.input.lower().endswith(".png"):
            print("warning: %s is not a .png; alpha and colour may be lossy"
                  % os.path.basename(args.input), file=sys.stderr)
        image = load_image(args.input)
        if args.regions:
            return run_region_mode(args, image, started)
        alpha = accuracy.describe_alpha(image)
        try:
            mask, mask_mode, mask_note = accuracy.prepare_mask(
                image, args.mask, args.alpha_threshold, args.luminance_threshold, args.invert)
        except ValueError as error:
            raise CliError(str(error))

        kept, skipped = collect_contours(mask, args.min_contour_points, args.min_area)
        if not kept:
            raise CliError("no contours found in the ink mask (mode: %s)" % mask_mode)

        if not args.report and not args.quiet:
            print("warning: --no-report set, output is not measured", file=sys.stderr)

        if args.seed is not None:
            np.random.seed(args.seed)

        entries = []
        failures = []
        total_points = 0
        total_segments = 0
        total_control = 0
        fit_started = time.time()

        for done, item in enumerate(kept, start=1):
            try:
                if args.curve == "polygon":
                    entry = {"polygon": np.round(item["points"], args.precision)}
                    total_segments += max(1, len(item["points"]) - 1)
                else:
                    formatted, segment_count, control_count = vectorize_contour(item["points"], args)
                    if not formatted:
                        raise CliError("segmentation produced nothing")
                    rounded = [np.round(np.asarray(row, dtype=float), args.precision)
                               for row in formatted]
                    entry = {"ctrl": rounded}
                    total_segments += segment_count
                    total_control += control_count
                entries.append(entry)
                total_points += int(len(item["points"]))
                if args.verbose:
                    print("  contour %4d  points %5d  depth %d  curves %4d"
                          % (item["index"], len(item["points"]), item["depth"],
                             total_segments), file=sys.stderr)
            except Exception as error:  # keep going, one bad contour must not kill the run
                failures.append({"contour": item["index"], "error": "%s: %s"
                                 % (type(error).__name__, error)})
            progress(done, len(kept), args.quiet)
        fit_elapsed = time.time() - fit_started

        if not entries:
            raise CliError("every contour failed to vectorise; first error: %s"
                           % (failures[0]["error"] if failures else "unknown"))

        height, width = image.shape[0], image.shape[1]
        svg_started = time.time()
        svg_text = build_svg((width, height), entries, args)
        svg_elapsed = time.time() - svg_started

        output = args.output or default_output(args.input)
        write_text(output, svg_text)
        svg_bytes = len(svg_text.encode("utf-8"))

        metrics = None
        accuracy_elapsed = 0.0
        if args.report:
            accuracy_started = time.time()
            loops = []
            for entry in entries:
                if args.curve == "polygon":
                    loops.append(np.asarray(entry["polygon"], dtype=float))
                else:
                    loops.append(accuracy.ctrl_to_loop(entry["ctrl"]))
            raster_contours = [item["points"] for item in kept]
            metrics = accuracy.evaluate(mask, loops, raster_contours,
                                        total_segments, total_control)
            accuracy_elapsed = time.time() - accuracy_started

        if not args.quiet:
            print("chunkmap-separator  raster -> %s svg" % args.curve)
            print("input      %s  (%dx%d, %d channel%s)" % (
                args.input, width, height,
                1 if image.ndim == 2 else image.shape[2],
                "" if (image.ndim == 2 or image.shape[2] == 1) else "s"))
            print("alpha      %s" % describe_alpha_line(alpha))
            print("mask       %s%s  ->  %d px (%.3f%% of image)" % (
                mask_mode, ("  [" + mask_note + "]") if mask_note else "",
                int(np.count_nonzero(mask)),
                100.0 * float(np.count_nonzero(mask)) / float(mask.size)))
            print("contours   %d kept, %d skipped, %d failed" % (
                len(kept), len(skipped), len(failures)))
            if args.curve == "polygon":
                print("vector     %d paths / %d straight segments" % (len(entries), total_segments))
            else:
                print("vector     %d paths / %d curves / %d control points" % (
                    len(entries), total_segments, total_control))
            print("timing     fit %.2fs  svg %.2fs  report %.2fs  total %.2fs" % (
                fit_elapsed, svg_elapsed, accuracy_elapsed, time.time() - started))
            print("svg        %.1f KB  ->  %s" % (svg_bytes / 1024.0, output))
            if metrics:
                for line in accuracy.format_report(metrics):
                    print(line)

        if args.report_json:
            payload = {
                "input": os.path.abspath(args.input),
                "output": os.path.abspath(output),
                "image": {"width": width, "height": height,
                          "channels": 1 if image.ndim == 2 else int(image.shape[2])},
                "alpha": alpha,
                "mask": {"mode": mask_mode, "note": mask_note,
                         "alpha_threshold": args.alpha_threshold,
                         "luminance_threshold": args.luminance_threshold,
                         "invert": bool(args.invert),
                         "ink_px": int(np.count_nonzero(mask))},
                "contours": {"kept": len(kept), "skipped": skipped, "failed": failures,
                             "raster_points": total_points},
                "vector": {"curve": args.curve, "segmentation": args.segmentation,
                           "paths": len(entries), "segments": total_segments,
                           "control_points": total_control},
                "svg": {"path": os.path.abspath(output), "bytes": svg_bytes},
                "timing": {"fit_s": fit_elapsed, "svg_s": svg_elapsed,
                           "report_s": accuracy_elapsed, "total_s": time.time() - started},
                "accuracy": metrics,
                "settings": {
                    "min_chunk": args.min_chunk, "max_chunk": args.max_chunk,
                    "smooth_sigma": args.smooth_sigma, "chunk_size": args.chunk_size,
                    "min_contour_points": args.min_contour_points, "min_area": args.min_area,
                    "stroke": args.stroke, "stroke_width": args.stroke_width,
                    "fill": args.fill, "hue": args.hue, "sampling": args.sampling,
                    "picking": args.picking, "seed": args.seed,
                    "precision": args.precision,
                },
            }
            write_text(args.report_json, json.dumps(payload, indent=2))
            if not args.quiet:
                print("json       %s" % args.report_json)

        return 0

    except CliError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

