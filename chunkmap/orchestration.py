"""
orchestration.py - integrated test for the random island generator and the
raster -> bezier vectorizer.

One config file describes a batch of random island images. The tool then, for
every image:

    1. generates the island PNG (island_generator.generate_island),
    2. vectorizes it with the real CLI (cli.py, so this is an integration test,
       not a reimplementation),
    3. keeps the SVG in its own folder,
    4. renders a PNG with the SVG paths drawn on top of the island in a distinct
       colour,
    5. scores the SVG against the PNG ink with accuracy.py and prints the
       per-image report plus a batch summary.

Folder layout inside the test space:

    <space>/islands/island_0001.png    generated raster
    <space>/svg/island_0001.svg        vectorizer output
    <space>/overlay/island_0001.png    raster + vector paths in a distinct colour
    <space>/reports/island_0001.json   full metrics for that image
    <space>/reports/summary.json       batch summary (config snapshot + aggregate)
    <space>/reports/summary.csv        one row per image, for pandas/R

Usage (from the repository root):

    python code/cli.py orchestrate --config code/configs/island-svg-sample.yaml
    python code/orchestration.py --config code/configs/island-svg-sample.json --space outputs/my-test

The config loader accepts JSON, and YAML through PyYAML when it is installed or
through a small built-in YAML subset parser otherwise (nested mappings,
sequences, inline lists/maps, quoted scalars and comments; no anchors, tags or
block scalars).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, replace
from typing import Optional

import cv2 as cv
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SPACE_ROOT = os.path.join(PROJECT_ROOT, "outputs", "orchestration")
MANAGED_DIRS = ("islands", "regions", "svg", "overlay", "reports")

from . import accuracy  # noqa: E402
from . import island_generator as islands  # noqa: E402

MAX_SEED = islands.MAX_SEED
MAX_ISLANDS = islands.MAX_ISLANDS
MAX_SUBDIVISIONS = islands.MAX_SUBDIVISIONS


class OrchestrationError(Exception):
    """A problem with the config, the test space or the run itself."""


class ConfigError(OrchestrationError):
    """The config file is missing, malformed or out of range."""


# ---------------------------------------------------------------------------
# YAML subset
# ---------------------------------------------------------------------------
#
# PyYAML is not a project dependency, so a tiny parser covers the configs this
# tool documents. It deliberately rejects the parts of YAML that would need a
# real parser instead of guessing (anchors, tags, block scalars).


_HEX_COLOUR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_INT = re.compile(r"^[-+]?\d+$")
_FLOAT = re.compile(r"^[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$")


@dataclass(frozen=True)
class _YamlLine:
    indent: int
    text: str
    number: int


def _strip_yaml_comment(line: str) -> str:
    """Drop a YAML comment, respecting quotes. A '#' needs a leading space."""
    output = []
    quote = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            output.append(char)
            if quote == '"':
                if char == "\\" and index + 1 < len(line):
                    output.append(line[index + 1])
                    index += 2
                    continue
                if char == '"':
                    quote = None
            elif quote == "'":
                if char == "'":
                    if index + 1 < len(line) and line[index + 1] == "'":
                        output.append("'")
                        index += 2
                        continue
                    quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
            output.append(char)
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            break
        else:
            output.append(char)
        index += 1
    return "".join(output).rstrip()


def _split_key_value(text: str):
    """Split 'key: value' at the first unquoted top-level colon."""
    quote = None
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if quote is not None:
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            if index + 1 == len(text) or text[index + 1].isspace():
                return text[:index].strip(), text[index + 1:].strip()
    return None


class _InlineReader:
    """Recursive parser for the inline fraction of YAML: [a, b] and {k: v}."""

    def __init__(self, text: str, number: int):
        self.text = text
        self.number = number
        self.index = 0

    def _error(self, message: str):
        raise ConfigError("YAML line %d: %s" % (self.number, message))

    def _skip(self):
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def _peek(self):
        self._skip()
        return self.text[self.index] if self.index < len(self.text) else ""

    def _consume(self, expected: str):
        if self._peek() != expected:
            self._error("expected %r" % expected)
        self.index += 1

    def parse(self):
        value = self.parse_value(flow=False)
        self._skip()
        if self.index != len(self.text):
            self._error("unexpected text %r" % self.text[self.index:].strip())
        return value

    def parse_value(self, flow=True):
        self._skip()
        if self.index >= len(self.text):
            self._error("missing value")
        char = self.text[self.index]
        if char == "[":
            return self._parse_list()
        if char == "{":
            return self._parse_map()
        if char in "'\"":
            return self._parse_quoted()
        return self._parse_bare(flow)

    def _parse_list(self):
        self._consume("[")
        result = []
        self._skip()
        if self._peek() == "]":
            self.index += 1
            return result
        while True:
            result.append(self.parse_value(flow=True))
            self._skip()
            if self._peek() == ",":
                self.index += 1
                continue
            self._consume("]")
            return result

    def _parse_map(self):
        self._consume("{")
        result = {}
        self._skip()
        if self._peek() == "}":
            self.index += 1
            return result
        while True:
            self._skip()
            if self._peek() in ("'", '"'):
                key = self._parse_quoted()
            else:
                start = self.index
                while self.index < len(self.text) and self.text[self.index] != ":":
                    if self.text[self.index] in "[{}],":
                        self._error("invalid inline mapping key")
                    self.index += 1
                key = self.text[start:self.index].strip()
                if not key:
                    self._error("empty inline mapping key")
            self._consume(":")
            value = self.parse_value(flow=True)
            result[str(key)] = value
            self._skip()
            if self._peek() == ",":
                self.index += 1
                continue
            self._consume("}")
            return result

    def _parse_quoted(self):
        quote = self.text[self.index]
        start = self.index
        self.index += 1
        while self.index < len(self.text):
            char = self.text[self.index]
            if quote == '"' and char == "\\":
                self.index += 2
                continue
            if char == quote:
                if quote == "'" and self.index + 1 < len(self.text) \
                        and self.text[self.index + 1] == "'":
                    self.index += 2
                    continue
                self.index += 1
                raw = self.text[start:self.index]
                if quote == '"':
                    try:
                        return json.loads(raw)
                    except ValueError:
                        self._error("invalid double quoted string")
                return raw[1:-1].replace("''", "'")
            self.index += 1
        self._error("unterminated quoted string")

    def _parse_bare(self, flow: bool):
        start = self.index
        stops = ",]}" if flow else ""
        while self.index < len(self.text) and self.text[self.index] not in stops:
            self.index += 1
        token = self.text[start:self.index].strip()
        if not token:
            self._error("missing value")
        return _yaml_scalar(token, self.number)


def _yaml_scalar(token: str, number: int):
    """A scalar token that has already been stripped and is not quoted."""
    lowered = token.lower()
    if lowered in ("null", "~"):
        return None
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if _INT.match(token):
        try:
            return int(token)
        except ValueError:
            pass
    if _FLOAT.match(token):
        try:
            return float(token)
        except ValueError:
            pass
    if token[0] in "&*!":
        raise ConfigError("YAML line %d: anchors, aliases and tags are not supported"
                          % number)
    return token


def _parse_yaml_scalar(text: str, number: int):
    text = text.strip()
    if not text:
        return None
    if text[0] in "&*!":
        raise ConfigError("YAML line %d: anchors, aliases and tags are not supported"
                          % number)
    if text[0] in "|>":
        raise ConfigError("YAML line %d: block scalars are not supported; quote "
                          "the value or use JSON" % number)
    return _InlineReader(text, number).parse()


def parse_yaml_subset(text: str):
    """Parse the small YAML subset used by the sample configs."""
    lines = []
    for number, raw in enumerate(text.splitlines(), start=1):
        leading = raw[:len(raw) - len(raw.lstrip(" \t"))]
        if "\t" in leading:
            raise ConfigError("YAML line %d: tabs are not allowed for indentation"
                              % number)
        stripped = _strip_yaml_comment(raw)
        if not stripped.strip():
            continue
        if stripped.strip() in ("---", "..."):
            continue
        if stripped.lstrip().startswith("%"):
            raise ConfigError("YAML line %d: directives are not supported" % number)
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append(_YamlLine(indent, stripped.strip(), number))

    if not lines:
        return {}

    class _Parser:
        def __init__(self):
            self.lines = lines

        def error(self, line, message):
            raise ConfigError("YAML line %d: %s" % (line.number, message))

        def parse(self):
            first = self.lines[0]
            if first.indent != 0:
                self.error(first, "the document must start at column 0")
            value, index = self.block(0, 0)
            if index < len(self.lines):
                self.error(self.lines[index], "unexpected content")
            return value

        def block(self, index, indent):
            line = self.lines[index]
            if line.text == "-" or line.text.startswith("- "):
                return self.sequence(index, indent)
            return self.mapping(index, indent)

        def mapping(self, index, indent):
            result = {}
            while index < len(self.lines):
                line = self.lines[index]
                if line.indent < indent:
                    break
                if line.indent > indent:
                    self.error(line, "unexpected indentation (expected %d spaces)"
                               % indent)
                if line.text == "-" or line.text.startswith("- "):
                    self.error(line, "a sequence item cannot be a mapping key")
                split = _split_key_value(line.text)
                if split is None:
                    self.error(line, "expected 'key: value'")
                key_text, rest = split
                key = _parse_yaml_scalar(key_text, line.number)
                if key is None or isinstance(key, (dict, list)):
                    self.error(line, "invalid mapping key")
                key = str(key)
                if key in result:
                    self.error(line, "duplicate key %r" % key)
                if rest == "":
                    if index + 1 < len(self.lines) \
                            and self.lines[index + 1].indent > indent:
                        value, index = self.block(index + 1,
                                                  self.lines[index + 1].indent)
                    else:
                        value = None
                        index += 1
                else:
                    value = _parse_yaml_scalar(rest, line.number)
                    index += 1
                result[key] = value
            return result, index

        def sequence(self, index, indent):
            result = []
            while index < len(self.lines):
                line = self.lines[index]
                if line.indent < indent:
                    break
                if line.indent > indent:
                    self.error(line, "unexpected indentation (expected %d spaces)"
                               % indent)
                if not (line.text == "-" or line.text.startswith("- ")):
                    break
                remainder = line.text[1:]
                spaces = len(remainder) - len(remainder.lstrip(" "))
                rest = remainder.strip()
                if rest == "":
                    if index + 1 < len(self.lines) \
                            and self.lines[index + 1].indent > indent:
                        value, index = self.block(index + 1,
                                                  self.lines[index + 1].indent)
                    else:
                        value = None
                        index += 1
                else:
                    split = _split_key_value(rest)
                    if split is not None and rest[0] not in "[{'\"":
                        # '- key: ...' is the first entry of a mapping item.
                        # Treat the rest of the line as that mapping's first line.
                        inner = indent + 1 + spaces
                        self.lines[index] = _YamlLine(inner, rest, line.number)
                        value, index = self.mapping(index, inner)
                    else:
                        value = _parse_yaml_scalar(rest, line.number)
                        index += 1
                result.append(value)
            return result, index

    return _Parser().parse()


# ---------------------------------------------------------------------------
# config loading and validation
# ---------------------------------------------------------------------------


def _read_text(path: str) -> str:
    try:
        # utf-8-sig also accepts plain UTF-8 and strips a Windows BOM, which
        # editors and PowerShell add to JSON/YAML files.
        with open(path, "r", encoding="utf-8-sig") as handle:
            return handle.read()
    except OSError as error:
        raise ConfigError("cannot read config %s (%s)" % (path, error))


def _load_json(text: str, path: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError("invalid JSON in %s at line %d column %d: %s"
                          % (path, error.lineno, error.colno, error.msg))


def _load_yaml(text: str, path: str):
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is not None:
        try:
            return yaml.safe_load(text)
        except Exception as error:  # PyYAML errors are not worth importing
            raise ConfigError("invalid YAML in %s: %s" % (path, error))
    return parse_yaml_subset(text)


def load_config_file(path: str):
    """Load a JSON or YAML config file into a raw mapping."""
    full = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(full):
        raise ConfigError("config file not found: %s" % path)
    text = _read_text(full)
    suffix = os.path.splitext(full)[1].lower()
    if suffix == ".json":
        raw = _load_json(text, full)
    elif suffix in (".yaml", ".yml"):
        raw = _load_yaml(text, full)
    else:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = _load_yaml(text, full)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping/object, got %s"
                          % type(raw).__name__)
    return raw, full


def _mapping(value, path: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("%s must be a mapping/object" % path)
    return value


def _keys(value: dict, allowed: set, path: str):
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError("%s has unknown key(s): %s (allowed: %s)"
                          % (path, ", ".join(unknown), ", ".join(sorted(allowed))))


def _int(value, path: str, minimum=None, maximum=None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("%s must be an integer (got %r)" % (path, value))
    if minimum is not None and value < minimum:
        raise ConfigError("%s must be >= %d (got %d)" % (path, minimum, value))
    if maximum is not None and value > maximum:
        raise ConfigError("%s must be <= %d (got %d)" % (path, maximum, value))
    return value


def _float(value, path: str, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("%s must be a number (got %r)" % (path, value))
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError("%s must be finite" % path)
    if minimum is not None and number < minimum:
        raise ConfigError("%s must be >= %s (got %s)" % (path, minimum, number))
    if maximum is not None and number > maximum:
        raise ConfigError("%s must be <= %s (got %s)" % (path, maximum, number))
    return number


def _bool(value, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError("%s must be true or false (got %r)" % (path, value))
    return value


def _choice(value, path: str, choices) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ConfigError("%s must be one of %s (got %r)"
                          % (path, ", ".join(choices), value))
    return value


def _string(value, path: str, allow_empty=False) -> str:
    if not isinstance(value, str):
        raise ConfigError("%s must be a string (got %r)" % (path, value))
    if not allow_empty and not value.strip():
        raise ConfigError("%s must not be empty" % path)
    return value


def _colour(value, path: str) -> str:
    text = _string(value, path).strip()
    if not _HEX_COLOUR.match(text):
        raise ConfigError("%s must be a #rgb or #rrggbb colour (got %r)"
                          % (path, text))
    return text.lower()


def _range(value, path: str, minimum: int, maximum: int) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        low = high = value
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ConfigError("%s range must be [min, max] (got %r)" % (path, value))
        low, high = value
    elif isinstance(value, dict):
        _keys(value, {"min", "max"}, path)
        if "min" not in value or "max" not in value:
            raise ConfigError("%s range needs both 'min' and 'max'" % path)
        low, high = value["min"], value["max"]
    else:
        raise ConfigError("%s must be an integer, [min, max] or {min: , max: } "
                          "(got %r)" % (path, value))
    low = _int(low, path + ".min", minimum, maximum)
    high = _int(high, path + ".max", minimum, maximum)
    if low > high:
        raise ConfigError("%s min %d is greater than max %d" % (path, low, high))
    return low, high


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip(".-_")
    if not cleaned:
        raise ConfigError("name %r has no usable filename characters" % text)
    return cleaned[:80]


@dataclass(frozen=True)
class ImageOverride:
    name: Optional[str] = None
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    islands: Optional[int] = None
    subdivisions: Optional[int] = None
    style: Optional[str] = None
    filled: Optional[bool] = None
    coverage: Optional[float] = None
    margin: Optional[float] = None
    stroke_width: Optional[float] = None

    def to_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items()
                if value is not None}


@dataclass(frozen=True)
class GenerationConfig:
    count: int
    width: int
    height: int
    style: str
    filled: bool
    coverage: float
    margin: float
    stroke_width: float
    islands_min: int
    islands_max: int
    subdivisions_min: int
    subdivisions_max: int
    seed_step: int
    images: tuple[ImageOverride, ...]

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "width": self.width,
            "height": self.height,
            "style": self.style,
            "filled": self.filled,
            "coverage": self.coverage,
            "margin": self.margin,
            "stroke_width": self.stroke_width,
            "islands": [self.islands_min, self.islands_max],
            "subdivisions": [self.subdivisions_min, self.subdivisions_max],
            "seed_step": self.seed_step,
            "images": [item.to_dict() for item in self.images],
        }


@dataclass(frozen=True)
class VectorizationConfig:
    curve: str
    segmentation: str
    min_chunk: int
    max_chunk: int
    smooth_sigma: float
    chunk_size: int
    min_contour_points: int
    min_area: float
    mask: str
    alpha_threshold: int
    luminance_threshold: int
    invert: bool
    stroke: str
    stroke_width: float
    fill: str
    hue: float
    sampling: str
    picking: str
    palette_seed: Optional[int]
    precision: int
    region_mode: bool
    region_fill: str
    region_palette: str
    region_color: str
    region_seam: float
    corner_angle: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class OverlayConfig:
    enabled: bool
    color: str
    width: float
    opacity: float
    fill: bool
    fill_color: str
    fill_opacity: float
    background: Optional[str]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ReportConfig:
    tolerance: int
    min_iou: Optional[float]
    min_within_1px: Optional[float]
    min_relative_iou: Optional[float]
    max_hausdorff_px: Optional[float]
    show: str
    json: bool
    csv: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def thresholds(self) -> dict:
        return {
            "min_iou": self.min_iou,
            "min_within_1px": self.min_within_1px,
            "min_relative_iou": self.min_relative_iou,
            "max_hausdorff_px": self.max_hausdorff_px,
        }


@dataclass(frozen=True)
class OrchestrationConfig:
    name: str
    seed: int
    space: Optional[str]
    generation: GenerationConfig
    vectorization: VectorizationConfig
    overlay: OverlayConfig
    report: ReportConfig
    source: str = ""

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "space": self.space,
            "generation": self.generation.to_dict(),
            "vectorization": self.vectorization.to_dict(),
            "overlay": self.overlay.to_dict(),
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True)
class ImageCase:
    index: int
    name: str
    seed: int
    width: int
    height: int
    islands: int
    subdivisions: int
    style: str
    filled: bool
    coverage: float
    margin: float
    stroke_width: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_TOP_LEVEL = {"name", "seed", "space", "test_space", "generation",
              "vectorization", "overlay", "report"}
_GENERATION_KEYS = {"count", "width", "height", "style", "filled", "coverage",
                    "margin", "stroke_width", "islands", "subdivisions",
                    "seed_step", "images"}
_IMAGE_KEYS = {"name", "seed", "width", "height", "islands", "subdivisions",
               "style", "filled", "coverage", "margin", "stroke_width"}
_VECTOR_KEYS = {"curve", "segmentation", "min_chunk", "max_chunk",
                "smooth_sigma", "chunk_size", "min_contour_points", "min_area",
                "mask", "alpha_threshold", "luminance_threshold", "invert",
                "stroke", "stroke_width", "fill", "hue", "sampling", "picking",
                "palette_seed", "precision", "region_mode", "region_fill",
                "region_palette", "region_color", "region_seam", "corner_angle"}
_OVERLAY_KEYS = {"enabled", "color", "width", "opacity", "fill", "fill_color",
                 "fill_opacity", "background"}
_REPORT_KEYS = {"tolerance", "min_iou", "min_within_1px", "min_relative_iou",
                "max_hausdorff_px", "show", "json", "csv"}


def _image_override(raw, path: str) -> ImageOverride:
    item = _mapping(raw, path)
    _keys(item, _IMAGE_KEYS, path)
    values = {}

    def optional(key, parser):
        if key in item and item[key] is not None:
            values[key] = parser(item[key], path + "." + key)

    optional("name", lambda value, p: _slug(_string(value, p)))
    optional("seed", lambda value, p: _int(value, p, 0, MAX_SEED))
    optional("width", lambda value, p: _int(value, p, 16, 8192))
    optional("height", lambda value, p: _int(value, p, 16, 8192))
    optional("islands", lambda value, p: _int(value, p, 1, MAX_ISLANDS))
    optional("subdivisions", lambda value, p: _int(value, p, 0, MAX_SUBDIVISIONS))
    optional("style", lambda value, p: _choice(value, p, islands.style_names()))
    optional("filled", _bool)
    optional("coverage", lambda value, p: _float(value, p, 0.02, 0.70))
    optional("margin", lambda value, p: _float(value, p, 0.0, 0.20))
    optional("stroke_width", lambda value, p: _float(value, p, 0.0, 100.0))
    return ImageOverride(**values)


def normalise_config(raw: dict, source: str = "") -> OrchestrationConfig:
    """Validate a raw config mapping and fill in every default."""
    _keys(raw, _TOP_LEVEL, "config")

    name = _string(raw.get("name", "island-svg-sample"), "config.name").strip()
    seed = _int(raw.get("seed", 2026), "config.seed", 0, MAX_SEED)
    space = raw.get("space", raw.get("test_space"))
    if "space" in raw and "test_space" in raw and raw["space"] != raw["test_space"]:
        raise ConfigError("config.space and config.test_space are both set and differ")
    if space is not None:
        space = _string(space, "config.space").strip()

    generation_raw = _mapping(raw.get("generation"), "config.generation")
    _keys(generation_raw, _GENERATION_KEYS, "config.generation")
    gen_path = "config.generation"
    style = _choice(generation_raw.get("style", "natural"), gen_path + ".style",
                    islands.style_names())
    islands_range = _range(generation_raw.get("islands", 1),
                           gen_path + ".islands", 1, MAX_ISLANDS)
    subdivisions_range = _range(generation_raw.get("subdivisions", 0),
                                gen_path + ".subdivisions", 0, MAX_SUBDIVISIONS)
    images_raw = generation_raw.get("images") or []
    if not isinstance(images_raw, list):
        raise ConfigError("%s.images must be a list" % gen_path)
    if len(images_raw) > 500:
        raise ConfigError("%s.images has %d entries; the maximum is 500"
                          % (gen_path, len(images_raw)))
    overrides = tuple(_image_override(item, "%s.images[%d]" % (gen_path, index))
                      for index, item in enumerate(images_raw))
    generation = GenerationConfig(
        count=_int(generation_raw.get("count", 8), gen_path + ".count", 1, 500),
        width=_int(generation_raw.get("width", 1024), gen_path + ".width", 16, 8192),
        height=_int(generation_raw.get("height", 768), gen_path + ".height", 16, 8192),
        style=style,
        filled=_bool(generation_raw.get("filled", False), gen_path + ".filled"),
        coverage=_float(generation_raw.get("coverage", 0.30),
                        gen_path + ".coverage", 0.02, 0.70),
        margin=_float(generation_raw.get("margin", 0.08),
                      gen_path + ".margin", 0.0, 0.20),
        stroke_width=_float(generation_raw.get("stroke_width", 0.0),
                            gen_path + ".stroke_width", 0.0, 100.0),
        islands_min=islands_range[0],
        islands_max=islands_range[1],
        subdivisions_min=subdivisions_range[0],
        subdivisions_max=subdivisions_range[1],
        seed_step=_int(generation_raw.get("seed_step", 1),
                       gen_path + ".seed_step", 0, MAX_SEED),
        images=overrides,
    )

    vector_raw = _mapping(raw.get("vectorization"), "config.vectorization")
    _keys(vector_raw, _VECTOR_KEYS, "config.vectorization")
    vec_path = "config.vectorization"
    vector = VectorizationConfig(
        curve=_choice(vector_raw.get("curve", "bezier"), vec_path + ".curve",
                      ("bezier", "polygon")),
        segmentation=_choice(vector_raw.get("segmentation", "curvature"),
                             vec_path + ".segmentation", ("curvature", "blind")),
        min_chunk=_int(vector_raw.get("min_chunk", 5), vec_path + ".min_chunk", 1, 1000),
        max_chunk=_int(vector_raw.get("max_chunk", 15), vec_path + ".max_chunk", 1, 1000),
        smooth_sigma=_float(vector_raw.get("smooth_sigma", 2.0),
                            vec_path + ".smooth_sigma", 0.0, 100.0),
        chunk_size=_int(vector_raw.get("chunk_size", 10),
                        vec_path + ".chunk_size", 1, 1000),
        min_contour_points=_int(vector_raw.get("min_contour_points", 4),
                                vec_path + ".min_contour_points", 1, 1000000),
        min_area=_float(vector_raw.get("min_area", 0.0),
                        vec_path + ".min_area", 0.0, None),
        mask=_choice(vector_raw.get("mask", "auto"), vec_path + ".mask",
                     ("auto", "alpha", "luminance")),
        alpha_threshold=_int(vector_raw.get("alpha_threshold", 128),
                             vec_path + ".alpha_threshold", 0, 255),
        luminance_threshold=_int(vector_raw.get("luminance_threshold", 128),
                                 vec_path + ".luminance_threshold", 0, 255),
        invert=_bool(vector_raw.get("invert", False), vec_path + ".invert"),
        stroke=_string(vector_raw.get("stroke", "#111111"), vec_path + ".stroke"),
        stroke_width=_float(vector_raw.get("stroke_width", 1.25),
                            vec_path + ".stroke_width", 0.01, 100.0),
        fill=_choice(vector_raw.get("fill", "none"), vec_path + ".fill",
                     ("none", "palette")),
        hue=_float(vector_raw.get("hue", 12.0), vec_path + ".hue", 0.0, 360.0),
        sampling=_choice(vector_raw.get("sampling", "linear"),
                         vec_path + ".sampling", ("linear", "random")),
        picking=_choice(vector_raw.get("picking", "linear"),
                        vec_path + ".picking", ("linear", "shuffle")),
        palette_seed=(None if vector_raw.get("palette_seed") is None else
                      _int(vector_raw["palette_seed"], vec_path + ".palette_seed",
                           0, MAX_SEED)),
        precision=_int(vector_raw.get("precision", 2), vec_path + ".precision", 0, 8),
        region_mode=_bool(vector_raw.get("region_mode", True),
                          vec_path + ".region_mode"),
        region_fill=_choice(vector_raw.get("region_fill", "palette"),
                            vec_path + ".region_fill", ("palette", "solid")),
        region_palette=_choice(vector_raw.get("region_palette", "spread"),
                               vec_path + ".region_palette", ("spread", "hue")),
        region_color=_colour(vector_raw.get("region_color", "#9ec79b"),
                             vec_path + ".region_color"),
        region_seam=_float(vector_raw.get("region_seam", 0.5),
                           vec_path + ".region_seam", 0.0, 10.0),
        corner_angle=_float(vector_raw.get("corner_angle", 55.0),
                            vec_path + ".corner_angle", 10.0, 179.0),
    )
    if vector.max_chunk < vector.min_chunk:
        raise ConfigError("%s.max_chunk (%d) must be >= min_chunk (%d)"
                          % (vec_path, vector.max_chunk, vector.min_chunk))

    overlay_raw = _mapping(raw.get("overlay"), "config.overlay")
    _keys(overlay_raw, _OVERLAY_KEYS, "config.overlay")
    overlay_path = "config.overlay"
    color = _colour(overlay_raw.get("color", "#ff3355"), overlay_path + ".color")
    background = overlay_raw.get("background")
    overlay = OverlayConfig(
        enabled=_bool(overlay_raw.get("enabled", True), overlay_path + ".enabled"),
        color=color,
        width=_float(overlay_raw.get("width", 2.0), overlay_path + ".width", 1.0, 50.0),
        opacity=_float(overlay_raw.get("opacity", 1.0), overlay_path + ".opacity", 0.0, 1.0),
        fill=_bool(overlay_raw.get("fill", False), overlay_path + ".fill"),
        fill_color=_colour(overlay_raw.get("fill_color", color),
                           overlay_path + ".fill_color"),
        fill_opacity=_float(overlay_raw.get("fill_opacity", 0.25),
                            overlay_path + ".fill_opacity", 0.0, 1.0),
        background=(None if background is None else
                    _colour(background, overlay_path + ".background")),
    )

    report_raw = _mapping(raw.get("report"), "config.report")
    _keys(report_raw, _REPORT_KEYS, "config.report")
    report_path = "config.report"

    def optional_float(key, minimum, maximum):
        value = report_raw.get(key)
        if value is None:
            return None
        return _float(value, report_path + "." + key, minimum, maximum)

    report = ReportConfig(
        tolerance=_int(report_raw.get("tolerance", 1), report_path + ".tolerance", 0, 10),
        min_iou=optional_float("min_iou", 0.0, 1.0),
        min_within_1px=optional_float("min_within_1px", 0.0, 100.0),
        min_relative_iou=optional_float("min_relative_iou", 0.0, 5.0),
        max_hausdorff_px=optional_float("max_hausdorff_px", 0.0, 10000.0),
        show=_choice(report_raw.get("show", "detailed"), report_path + ".show",
                     ("detailed", "compact", "summary")),
        json=_bool(report_raw.get("json", True), report_path + ".json"),
        csv=_bool(report_raw.get("csv", True), report_path + ".csv"),
    )

    return OrchestrationConfig(
        name=name,
        seed=seed,
        space=space,
        generation=generation,
        vectorization=vector,
        overlay=overlay,
        report=report,
        source=os.path.abspath(source) if source else "",
    )


def load_config(path: str) -> OrchestrationConfig:
    raw, full = load_config_file(path)
    return normalise_config(raw, full)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def _balanced_values(low: int, high: int, seed: int, salt: int) -> list[int]:
    """A seeded shuffle of an inclusive integer range.

    Cycling over this list makes a small sample cover every island/subdivision
    count in the configured range instead of accidentally drawing the same value
    for every image.
    """
    values = list(range(low, high + 1))
    random.Random((int(seed) << 20) ^ int(salt)).shuffle(values)
    return values


def build_cases(config: OrchestrationConfig) -> list[ImageCase]:
    generation = config.generation
    overrides = generation.images
    cases = []

    if overrides:
        count = len(overrides)
        digits = max(4, len(str(count)))
        for index, override in enumerate(overrides, start=1):
            seed = override.seed if override.seed is not None else \
                (config.seed + (index - 1) * generation.seed_step) & MAX_SEED
            name = override.name or ("island_%0*d" % (digits, index))
            cases.append(ImageCase(
                index=index,
                name=name,
                seed=seed,
                width=override.width if override.width is not None else generation.width,
                height=override.height if override.height is not None else generation.height,
                islands=override.islands if override.islands is not None else generation.islands_min,
                subdivisions=(override.subdivisions if override.subdivisions is not None
                              else generation.subdivisions_min),
                style=override.style or generation.style,
                filled=(override.filled if override.filled is not None else generation.filled),
                coverage=(override.coverage if override.coverage is not None
                          else generation.coverage),
                margin=override.margin if override.margin is not None else generation.margin,
                stroke_width=(override.stroke_width if override.stroke_width is not None
                              else generation.stroke_width),
            ))
    else:
        count = generation.count
        digits = max(4, len(str(count)))
        island_values = _balanced_values(generation.islands_min, generation.islands_max,
                                         config.seed, 0x1A2B)
        subdivision_values = _balanced_values(generation.subdivisions_min,
                                              generation.subdivisions_max,
                                              config.seed, 0x3C4D)
        for index in range(1, count + 1):
            seed = (config.seed + (index - 1) * generation.seed_step) & MAX_SEED
            cases.append(ImageCase(
                index=index,
                name="island_%0*d" % (digits, index),
                seed=seed,
                width=generation.width,
                height=generation.height,
                islands=island_values[(index - 1) % len(island_values)],
                subdivisions=subdivision_values[(index - 1) % len(subdivision_values)],
                style=generation.style,
                filled=generation.filled,
                coverage=generation.coverage,
                margin=generation.margin,
                stroke_width=generation.stroke_width,
            ))

    seen = set()
    for case in cases:
        if case.name in seen:
            raise ConfigError("duplicate image name %r; names must be unique"
                              % case.name)
        seen.add(case.name)
    return cases


def select_cases(cases: list[ImageCase], selection: Optional[str]) -> list[ImageCase]:
    if not selection:
        return cases
    wanted = set()
    for token in selection.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                raise ConfigError("--only expects indices like 1,3-5 (got %r)"
                                  % token)
            if start > end:
                raise ConfigError("--only range %r is reversed" % token)
            wanted.update(range(start, end + 1))
        else:
            try:
                wanted.add(int(token))
            except ValueError:
                raise ConfigError("--only expects indices like 1,3-5 (got %r)"
                                  % token)
    if not wanted:
        raise ConfigError("--only selected no images")
    invalid = sorted(index for index in wanted
                     if index < 1 or index > len(cases))
    if invalid:
        raise ConfigError("--only index %d is outside 1..%d"
                          % (invalid[0], len(cases)))
    return [case for case in cases if case.index in wanted]


def resolve_space(config: OrchestrationConfig, override: Optional[str]) -> str:
    raw = override if override else config.space
    if not raw:
        raw = os.path.join(DEFAULT_SPACE_ROOT, _slug(config.name))
    full = os.path.abspath(os.path.expanduser(raw))
    if os.path.exists(full) and not os.path.isdir(full):
        raise OrchestrationError("test space is not a directory: %s" % full)
    return full


# ---------------------------------------------------------------------------
# overlay rendering
# ---------------------------------------------------------------------------


def hex_to_bgr(text: str) -> np.ndarray:
    """#rrggbb -> a float BGR triplet in 0..1."""
    value = text.lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    red = int(value[0:2], 16) / 255.0
    green = int(value[2:4], 16) / 255.0
    blue = int(value[4:6], 16) / 255.0
    return np.array([blue, green, red], dtype=np.float32)


def _composite_premultiplied(rgb: np.ndarray, alpha: np.ndarray,
                             colour: np.ndarray, coverage: np.ndarray):
    """Source-over in premultiplied alpha. Both inputs are float 0..1."""
    cov = np.clip(coverage, 0.0, 1.0).astype(np.float32)[:, :, None]
    new_alpha = alpha * (1.0 - cov[:, :, 0]) + cov[:, :, 0]
    new_rgb = rgb * (1.0 - cov) + colour[None, None, :] * cov
    return new_rgb / np.maximum(new_alpha, 1e-6)[:, :, None], new_alpha


def render_overlay(base: np.ndarray, loops, config: OverlayConfig) -> np.ndarray:
    """Draw vector loops over the island PNG in a distinct colour.

    The base image's alpha is preserved where nothing is drawn, and raised where
    the vector line falls outside the raster ink, so a stray curve is visible
    instead of being hidden by the transparent background.

    :param base: BGRA image as loaded by cv.imread(..., IMREAD_UNCHANGED)
    :param loops: flattened closed loops from accuracy.parse_svg
    :return: BGRA overlay image
    """
    if base.ndim != 3 or base.shape[2] != 4:
        raise OrchestrationError("overlay needs a 4 channel BGRA base image")
    height, width = base.shape[:2]
    base_rgb = base[:, :, :3].astype(np.float32) / 255.0
    base_alpha = base[:, :, 3].astype(np.float32) / 255.0

    if config.background is not None:
        background = hex_to_bgr(config.background)
        out_rgb = base_rgb * base_alpha[:, :, None] \
            + background[None, None, :] * (1.0 - base_alpha[:, :, None])
        out_alpha = np.ones_like(base_alpha)
    else:
        out_rgb = base_rgb * base_alpha[:, :, None]
        out_alpha = base_alpha.copy()

    if config.fill and config.fill_opacity > 0.0 and loops:
        filled = accuracy.loops_to_mask(loops, (height, width)).astype(np.float32) / 255.0
        out_rgb, out_alpha = _composite_premultiplied(
            out_rgb, out_alpha, hex_to_bgr(config.fill_color),
            filled * config.fill_opacity)

    stroke = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, int(round(config.width)))
    for loop in loops:
        points = np.asarray(loop, dtype=float)
        if len(points) < 2:
            continue
        integer = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv.polylines(stroke, [integer], True, 255, thickness, lineType=cv.LINE_AA)

    coverage = stroke.astype(np.float32) / 255.0 * config.opacity
    out_rgb, out_alpha = _composite_premultiplied(
        out_rgb, out_alpha, hex_to_bgr(config.color), coverage)

    result = np.empty((height, width, 4), dtype=np.uint8)
    result[:, :, :3] = np.clip(np.rint(out_rgb * 255.0), 0, 255).astype(np.uint8)
    result[:, :, 3] = np.clip(np.rint(out_alpha * 255.0), 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------


def compact_metrics(metrics: dict) -> dict:
    coverage = metrics["coverage"]
    tolerant = coverage.get("tolerant", coverage)
    boundary = metrics["boundary"]
    baseline = metrics.get("baseline", coverage)
    relative = metrics.get("relative_to_exact_contours", {})
    return {
        "iou": float(tolerant["iou"]),
        "dice": float(tolerant["dice"]),
        "precision": float(tolerant["precision"]),
        "recall": float(tolerant["recall"]),
        "strict_iou": float(coverage["iou"]),
        "ceiling_iou": float(baseline["iou"]),
        # Clamp: the exact-contour baseline is itself rasterised with cv.fillPoly,
        # whose inclusive boundary can make the ratio exceed 1 by a fraction of a
        # percent. That is a metric artefact, not a better-than-exact fit.
        "relative_iou": min(1.0, max(0.0, float(relative.get("strict_iou", 1.0)))),
        "boundary_mean_px": float(boundary["raster_to_vector"]["mean"]),
        "boundary_p95_px": float(boundary["raster_to_vector"]["p95"]),
        "boundary_max_px": float(boundary["raster_to_vector"]["max"]),
        "vector_to_raster_mean_px": float(boundary["vector_to_raster"]["mean"]),
        "hausdorff_px": float(boundary["hausdorff"]),
        "within_1px_pct": float(boundary["within_1px_pct"]),
        "within_2px_pct": float(boundary["within_2px_pct"]),
        "compression": float(metrics["compression"]),
        "raster_points": int(metrics["raster_points"]),
        "paths": int(metrics["contours"]),
        "curves": int(metrics["segments"]),
    }


def threshold_failures(metrics: dict, report: ReportConfig) -> list[dict]:
    failures = []
    checks = (
        ("min_iou", "iou", metrics["iou"], "<"),
        ("min_within_1px", "within_1px_pct", metrics["within_1px_pct"], "<"),
        ("min_relative_iou", "relative_iou", metrics["relative_iou"], "<"),
        ("max_hausdorff_px", "hausdorff_px", metrics["hausdorff_px"], ">"),
    )
    for config_key, metric, actual, operator in checks:
        limit = getattr(report, config_key)
        if limit is None:
            continue
        failed = actual < limit if operator == "<" else actual > limit
        if failed:
            failures.append({
                "metric": metric,
                "actual": float(actual),
                "limit": float(limit),
                "operator": operator,
            })
    return failures


def _describe_failures(failures: list[dict]) -> str:
    return "; ".join("%s %.4f %s %.4f" % (item["metric"], item["actual"],
                                          item["operator"], item["limit"])
                     for item in failures)


def _stats(values: list[float]) -> Optional[dict]:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _relative(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _iso_now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _percent(value: float) -> str:
    return "%.1f%%" % (100.0 * value)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _vectorizer_argv(case: ImageCase, png: str, svg: str,
                     vector: VectorizationConfig,
                     region_png: Optional[str] = None) -> list[str]:
    if vector.region_mode and region_png:
        argv = [
            png, "-o", svg,
            "--regions", region_png,
            "--region-style", "fill",
            "--region-fill", vector.region_fill,
            "--region-palette", vector.region_palette,
            "--region-color", vector.region_color,
            "--region-seam", str(vector.region_seam),
            "--corner-angle", str(vector.corner_angle),
            "--max-chunk", str(vector.max_chunk),
            "--stroke", vector.stroke,
            "--stroke-width", str(vector.stroke_width),
            "--hue", str(vector.hue),
            "--sampling", vector.sampling,
            "--picking", vector.picking,
            "--precision", str(vector.precision),
            "--no-report",
            "--quiet",
        ]
    else:
        argv = [
            png, "-o", svg,
            "--curve", vector.curve,
            "--segmentation", vector.segmentation,
            "--min-chunk", str(vector.min_chunk),
            "--max-chunk", str(vector.max_chunk),
            "--smooth-sigma", str(vector.smooth_sigma),
            "--chunk-size", str(vector.chunk_size),
            "--min-contour-points", str(vector.min_contour_points),
            "--min-area", str(vector.min_area),
            "--mask", vector.mask,
            "--alpha-threshold", str(vector.alpha_threshold),
            "--luminance-threshold", str(vector.luminance_threshold),
            "--stroke", vector.stroke,
            "--stroke-width", str(vector.stroke_width),
            "--fill", vector.fill,
            "--hue", str(vector.hue),
            "--sampling", vector.sampling,
            "--picking", vector.picking,
            "--precision", str(vector.precision),
            "--no-report",
            "--quiet",
        ]
        if vector.invert:
            argv.append("--invert")
    if vector.palette_seed is not None:
        argv.extend(["--seed", str(vector.palette_seed)])
    return argv


def _generate_case(case: ImageCase, png: str):
    """Generate one PNG, retrying once with an alternate seed like the CLI does."""
    params = islands.IslandParams(
        width=case.width,
        height=case.height,
        seed=case.seed,
        coverage=case.coverage,
        margin=case.margin,
        islands=case.islands,
        subdivisions=case.subdivisions,
        filled=case.filled,
        stroke_width=case.stroke_width,
        **islands.style_preset(case.style),
    )
    last_error = None
    for attempt in range(2):
        trial = params if attempt == 0 else replace(
            params, seed=(case.seed + 0x9E3779B9) & MAX_SEED)
        try:
            result = islands.generate_island(trial)
        except (islands.IslandGenerationError, ValueError) as error:
            last_error = error
            continue
        if not cv.imwrite(png, result.image):
            raise OrchestrationError("cannot write %s" % png)
        return result
    raise OrchestrationError("island generation failed: %s" % last_error)


def _load_and_score(png: str, svg: str, vector: VectorizationConfig,
                    tolerance: int, region_png: Optional[str] = None):
    image = cv.imread(png, cv.IMREAD_UNCHANGED)
    if image is None:
        raise OrchestrationError("cannot read %s" % png)

    if region_png:
        from . import region_vectorization as region_vector

        try:
            labels = region_vector.load_label_map(region_png)
        except region_vector.RegionVectorizationError as error:
            raise OrchestrationError(str(error))
        if labels.shape[:2] != image.shape[:2]:
            raise OrchestrationError(
                "label map %dx%d does not match image %dx%d"
                % (labels.shape[1], labels.shape[0],
                   image.shape[1], image.shape[0]))
        mask = (labels > 0).astype(np.uint8) * 255
        mode = "regions"
        region_values = [int(value) for value in np.unique(labels) if value > 0]
        note = "%d region%s" % (len(region_values),
                                "" if len(region_values) == 1 else "s")
        contours = region_vector.exact_region_loops(labels)
        region_count = len(region_values)
    else:
        mask, mode, note = accuracy.prepare_mask(
            image, vector.mask, vector.alpha_threshold,
            vector.luminance_threshold, vector.invert)
        contours, _ = accuracy.trace_contours(mask)
        region_count = None
    if not np.any(mask):
        raise OrchestrationError("the ink mask is empty (mode %s)" % mode)

    try:
        with open(svg, "r", encoding="utf-8") as handle:
            svg_text = handle.read()
    except OSError as error:
        raise OrchestrationError("cannot read %s (%s)" % (svg, error))

    try:
        document = accuracy.parse_svg(svg_text)
    except ValueError as error:
        raise OrchestrationError("cannot parse %s (%s)" % (svg, error))
    if not document["loops"]:
        raise OrchestrationError("the SVG contains no closed path loops")

    metrics = accuracy.evaluate(mask, document["loops"], contours,
                                document["segments"], None, tolerance)
    metrics["segments"] = int(document["segments"])
    metrics["compression"] = (float(metrics["raster_points"]) / document["segments"]
                              if document["segments"] else 0.0)
    if region_png:
        relative = metrics.get("relative_to_exact_contours")
        if relative:
            relative["strict_iou"] = min(1.0, float(relative["strict_iou"]))

    size_matches = None
    if document["width"] and document["height"]:
        size_matches = (int(document["width"]) == image.shape[1]
                        and int(document["height"]) == image.shape[0])

    return {
        "image": image,
        "mask": mask,
        "mask_mode": mode,
        "mask_note": note,
        "document": document,
        "metrics": metrics,
        "compact": compact_metrics(metrics),
        "size_matches": size_matches,
        "svg_bytes": len(svg_text.encode("utf-8")),
        "region_count": region_count,
    }


def _print_image_header(position: int, total: int, case: ImageCase):
    mode = "filled" if case.filled else "outline"
    print("[%d/%d] %s  seed %d  %dx%d  %s/%s"
          % (position, total, case.name, case.seed, case.width, case.height,
             mode, case.style), flush=True)


def _print_image_detailed(record: dict, case: ImageCase):
    generation = record.get("generation") or {}
    vector = record.get("vectorization") or {}
    timings = record.get("timings") or {}

    islands_text = "%d/%d" % (generation.get("islands", 0), case.islands)
    regions = generation.get("regions", 0)
    coverage = generation.get("coverage")
    stroke = generation.get("stroke_px")
    print("      islands  %s  regions %d  coverage %s  stroke %s px"
          % (islands_text, regions,
             _percent(coverage) if coverage is not None else "?",
             ("%.1f" % stroke) if stroke is not None else "?"))
    print("      vector   %d paths / %d curves  svg %s  gen %.2fs  fit %.2fs  score %.2fs"
          % (vector.get("paths", 0), vector.get("curves", 0),
             _size_text(vector.get("svg_bytes", 0)),
             timings.get("generate", 0.0), timings.get("vectorize", 0.0),
             timings.get("accuracy", 0.0)))
    files = []
    if generation.get("png"):
        files.append(generation["png"])
    if generation.get("region_png"):
        files.append(generation["region_png"])
    if vector.get("svg"):
        files.append(vector["svg"])
    if record.get("overlay") and record["overlay"].get("png"):
        files.append(record["overlay"]["png"])
    if files:
        print("      files    %s" % "  ".join(files))

    metrics = record.get("metrics")
    if metrics:
        for line in accuracy.format_report(metrics, prefix="      "):
            print(line)

    if record["status"] == "ok":
        print("      status   OK", flush=True)
    elif record["status"] == "below":
        print("      status   BELOW  (%s)" % _describe_failures(record["below"]),
              flush=True)
    else:
        print("      status   FAILED at %s" % record.get("stage", "?"), flush=True)
        if record.get("error"):
            print("      error    %s" % record["error"], flush=True)


def _print_image_compact(record: dict):
    if record["status"] == "failed":
        print("[%d] %s  FAILED at %s: %s"
              % (record["index"], record["name"], record["stage"],
                 record["error"]), flush=True)
        return
    metrics = record["accuracy"]
    print("[%d] %s  iou %.4f  dice %.4f  within1 %.1f%%  hausdorff %.2f px  "
          "curves %d  %s"
          % (record["index"], record["name"], metrics["iou"], metrics["dice"],
             metrics["within_1px_pct"], metrics["hausdorff_px"],
             metrics["curves"], record["status"].upper()), flush=True)


def _size_text(size: int) -> str:
    if size >= 1024 * 1024:
        return "%.1f MB" % (size / (1024.0 * 1024.0))
    return "%.1f KB" % (size / 1024.0)


def _summary_record(record: dict) -> dict:
    keys = ("index", "name", "status", "stage", "error", "seed", "requested",
            "generation", "vectorization", "accuracy", "below", "overlay",
            "timings")
    return {key: record.get(key) for key in keys}


def _aggregate(records: list[dict], wall_seconds: float, processed: int,
               planned: int, report: ReportConfig) -> dict:
    successful = [record for record in records if record.get("accuracy")]
    numeric_keys = (
        "iou", "dice", "precision", "recall", "strict_iou", "ceiling_iou",
        "relative_iou", "boundary_mean_px", "boundary_p95_px", "boundary_max_px",
        "vector_to_raster_mean_px", "hausdorff_px", "within_1px_pct",
        "within_2px_pct", "compression", "paths", "curves", "raster_points",
    )
    stats = {}
    for key in numeric_keys:
        stats[key] = _stats([record["accuracy"][key] for record in successful])

    totals = {
        "paths": sum(record["accuracy"]["paths"] for record in successful),
        "curves": sum(record["accuracy"]["curves"] for record in successful),
        "raster_points": sum(record["accuracy"]["raster_points"] for record in successful),
        "svg_bytes": sum(record["vectorization"].get("svg_bytes", 0)
                         for record in successful),
        "islands": sum(record["generation"].get("islands", 0) for record in successful),
        "regions": sum(record["generation"].get("regions", 0) for record in successful),
    }
    stage_times = {
        stage: sum(record.get("timings", {}).get(stage, 0.0) for record in records)
        for stage in ("generate", "vectorize", "accuracy", "overlay")
    }

    best = worst = None
    if successful:
        ordered = sorted(successful, key=lambda item: item["accuracy"]["iou"])
        worst = {"name": ordered[0]["name"], "iou": ordered[0]["accuracy"]["iou"],
                 "within_1px_pct": ordered[0]["accuracy"]["within_1px_pct"]}
        best = {"name": ordered[-1]["name"], "iou": ordered[-1]["accuracy"]["iou"],
                "within_1px_pct": ordered[-1]["accuracy"]["within_1px_pct"]}

    return {
        "planned": planned,
        "processed": processed,
        "ok": sum(1 for record in records if record["status"] == "ok"),
        "below_threshold": sum(1 for record in records
                               if record["status"] == "below"),
        "failed": sum(1 for record in records if record["status"] == "failed"),
        "wall_seconds": wall_seconds,
        "stage_seconds": stage_times,
        "totals": totals,
        "metrics": stats,
        "best": best,
        "worst": worst,
        "thresholds": report.thresholds(),
        "tolerance": report.tolerance,
    }


def _print_aggregate(aggregate: dict, summary_json: Optional[str],
                     summary_csv: Optional[str]):
    metrics = aggregate["metrics"]
    iou = metrics["iou"]
    if not iou:
        print("summary    no scored images", flush=True)
        return

    print("summary    %d/%d images  %d ok  %d below  %d failed  wall %.2fs"
          % (aggregate["processed"], aggregate["planned"], aggregate["ok"],
             aggregate["below_threshold"], aggregate["failed"],
             aggregate["wall_seconds"]), flush=True)
    print("coverage   IoU  mean %.4f  median %.4f  min %.4f  max %.4f  (%d px tolerance)"
          % (iou["mean"], iou["median"], iou["min"], iou["max"],
             aggregate["tolerance"]))
    print("           Dice mean %.4f   precision mean %.4f   recall mean %.4f"
          % (metrics["dice"]["mean"], metrics["precision"]["mean"],
             metrics["recall"]["mean"]))
    relative = metrics["relative_iou"]
    print("           strict/ceiling mean %s  min %s  max %s"
          % (_percent(relative["mean"]), _percent(relative["min"]),
             _percent(relative["max"])))
    boundary = metrics["boundary_mean_px"]
    boundary_p95 = metrics["boundary_p95_px"]
    hausdorff = metrics["hausdorff_px"]
    within = metrics["within_1px_pct"]
    print("boundary   r->v mean %.2f px  p95 mean %.2f px  worst image mean %.2f px"
          % (boundary["mean"], boundary_p95["mean"], boundary["max"]))
    print("           hausdorff mean %.2f px  worst %.2f px  within 1 px mean %.1f%%  min %.1f%%"
          % (hausdorff["mean"], hausdorff["max"], within["mean"], within["min"]))
    print("vector     %s curves total (%d/image)  %s raster points  %.1fx compression  %s SVG"
          % (f"{aggregate['totals']['curves']:,}",
             int(round(metrics["curves"]["mean"])),
             f"{aggregate['totals']['raster_points']:,}",
             metrics["compression"]["mean"],
             _size_text(aggregate["totals"]["svg_bytes"])))

    best, worst = aggregate["best"], aggregate["worst"]
    if best and worst:
        print("images     worst %s (IoU %.4f)   best %s (IoU %.4f)"
              % (worst["name"], worst["iou"], best["name"], best["iou"]))
    stage = aggregate["stage_seconds"]
    print("timing     generate %.2fs  vector %.2fs  score %.2fs  overlay %.2fs"
          % (stage["generate"], stage["vectorize"], stage["accuracy"],
             stage["overlay"]))
    if summary_json:
        print("reports    %s" % summary_json)
    if summary_csv:
        print("           %s" % summary_csv)


def _write_csv(path: str, records: list[dict]):
    columns = [
        "index", "name", "status", "stage", "error", "below", "seed", "width",
        "height", "islands_requested", "islands", "subdivisions_requested",
        "regions", "region_count", "mode", "region_map", "coverage", "paths",
        "curves", "svg_bytes", "iou", "dice", "precision", "recall",
        "strict_iou", "ceiling_iou", "relative_iou", "boundary_mean_px",
        "boundary_p95_px", "boundary_max_px", "vector_to_raster_mean_px",
        "hausdorff_px", "within_1px_pct", "within_2px_pct", "compression",
        "generate_s", "vectorize_s", "accuracy_s", "overlay_s",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            generation = record.get("generation") or {}
            vector = record.get("vectorization") or {}
            metrics = record.get("accuracy") or {}
            timings = record.get("timings") or {}
            requested = record.get("requested") or {}
            row = {
                "index": record["index"],
                "name": record["name"],
                "status": record["status"],
                "stage": record.get("stage"),
                "error": record.get("error") or "",
                "below": _describe_failures(record.get("below") or []),
                "seed": record.get("seed"),
                "width": requested.get("width"),
                "height": requested.get("height"),
                "islands_requested": requested.get("islands"),
                "islands": generation.get("islands"),
                "subdivisions_requested": requested.get("subdivisions"),
                "regions": generation.get("regions"),
                "region_count": vector.get("region_count"),
                "mode": vector.get("mask_mode"),
                "region_map": vector.get("region_map") or "",
                "coverage": generation.get("coverage"),
                "paths": vector.get("paths") or metrics.get("paths", ""),
                "curves": vector.get("curves") or metrics.get("curves", ""),
                "svg_bytes": vector.get("svg_bytes", ""),
                "generate_s": timings.get("generate"),
                "vectorize_s": timings.get("vectorize"),
                "accuracy_s": timings.get("accuracy"),
                "overlay_s": timings.get("overlay"),
            }
            for key in ("iou", "dice", "precision", "recall", "strict_iou",
                        "ceiling_iou", "relative_iou", "boundary_mean_px",
                        "boundary_p95_px", "boundary_max_px",
                        "vector_to_raster_mean_px", "hausdorff_px",
                        "within_1px_pct", "within_2px_pct", "compression"):
                row[key] = metrics.get(key, "")
            writer.writerow(row)


def run(config: OrchestrationConfig, space: str, cases: list[ImageCase],
        args) -> int:
    started = time.time()
    started_iso = _iso_now()

    # matplotlib is imported by fileutils, and a read-only user profile makes it
    # print a caching warning on every import. Keep the cache in the project's
    # ignored outputs/ area rather than polluting the test space.
    cache = os.path.join(PROJECT_ROOT, "outputs", ".cache", "matplotlib")
    os.makedirs(cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = cache
    os.environ.setdefault("MPLBACKEND", "Agg")

    if args.clean:
        for name in MANAGED_DIRS:
            shutil.rmtree(os.path.join(space, name), ignore_errors=True)
    for name in MANAGED_DIRS:
        os.makedirs(os.path.join(space, name), exist_ok=True)

    from . import cli  # imported late: cli pulls in matplotlib through fileutils

    report_cfg = config.report
    if args.compact:
        report_cfg = replace(report_cfg, show="compact")

    if not args.quiet:
        modes = {("filled" if case.filled else "outline") for case in cases}
        mode = next(iter(modes)) if len(modes) == 1 else "mixed"
        sizes = {(case.width, case.height) for case in cases}
        size_text = ("%dx%d" % next(iter(sizes))) if len(sizes) == 1 else "mixed sizes"
        if config.generation.images:
            plan_text = "from config"
        elif config.generation.islands_min == config.generation.islands_max:
            plan_text = "islands %d" % config.generation.islands_min
        else:
            plan_text = ("islands %d-%d" % (config.generation.islands_min,
                                            config.generation.islands_max))
            if config.generation.subdivisions_min != config.generation.subdivisions_max:
                plan_text += ("  subdivisions %d-%d"
                              % (config.generation.subdivisions_min,
                                 config.generation.subdivisions_max))
        print("chunkmap-separator  orchestrate  %s" % config.name)
        print("space      %s" % space)
        print("config     %s" % config.source)
        print("plan       %d image%s  %s  seed %d  %s  %s"
              % (len(cases), "" if len(cases) == 1 else "s", size_text,
                 config.seed, mode, plan_text))
        if config.vectorization.region_mode:
            print("vector     regions  shared-border fills  %s  corner %.0f deg"
                  % (config.vectorization.region_fill,
                     config.vectorization.corner_angle))
        else:
            print("vector     %s / %s  mask %s  alpha > %d  stroke %s %.2f px"
                  % (config.vectorization.curve, config.vectorization.segmentation,
                     config.vectorization.mask, config.vectorization.alpha_threshold,
                     config.vectorization.stroke, config.vectorization.stroke_width))
        overlay_text = ("off" if not config.overlay.enabled or args.no_overlay
                        else "%s  %.1f px  opacity %.2f%s"
                        % (config.overlay.color, config.overlay.width,
                           config.overlay.opacity,
                           "  fill" if config.overlay.fill else ""))
        print("overlay    %s" % overlay_text)
        print("folders    islands/  regions/  svg/  overlay/  reports/")
        print("", flush=True)

    records = []
    summary_json = None
    summary_csv = None
    interrupted = False

    try:
        for position, case in enumerate(cases, start=1):
            if not args.quiet:
                _print_image_header(position, len(cases), case)

            record = {
                "index": case.index,
                "name": case.name,
                "status": "ok",
                "stage": "generation",
                "error": None,
                "seed": case.seed,
                "requested": {
                    "width": case.width,
                    "height": case.height,
                    "islands": case.islands,
                    "subdivisions": case.subdivisions,
                    "style": case.style,
                    "filled": case.filled,
                    "coverage": case.coverage,
                    "margin": case.margin,
                    "stroke_width": case.stroke_width,
                },
                "generation": {},
                "vectorization": {},
                "accuracy": None,
                "metrics": None,
                "overlay": None,
                "timings": {"generate": 0.0, "vectorize": 0.0,
                            "accuracy": 0.0, "overlay": 0.0},
            }
            png = os.path.join(space, "islands", case.name + ".png")
            svg = os.path.join(space, "svg", case.name + ".svg")
            region_png = os.path.join(space, "regions", case.name + ".png")
            overlay_png = os.path.join(space, "overlay", case.name + ".png")
            report_png = os.path.join(space, "reports", case.name + ".json")
            region_png_path = None

            stage = "generation"
            try:
                stage_started = time.perf_counter()
                result = _generate_case(case, png)
                record["timings"]["generate"] = time.perf_counter() - stage_started
                record["seed"] = int(result.seed)
                record["generation"] = {
                    "png": _relative(png, space),
                    "width": int(result.width),
                    "height": int(result.height),
                    "islands": int(result.island_count),
                    "regions": int(result.region_count),
                    "coverage": float(result.coverage),
                    "stroke_px": float(result.stroke_px),
                    "bbox": [int(value) for value in result.bbox],
                    "margin_px": int(result.margin_px),
                }
                if config.vectorization.region_mode and result.labels is not None:
                    if not cv.imwrite(region_png,
                                      result.labels.astype(np.uint16)):
                        raise OrchestrationError("cannot write %s" % region_png)
                    record["generation"]["region_png"] = _relative(region_png, space)
                    region_png_path = region_png

                stage = "vectorization"
                stage_started = time.perf_counter()
                code = cli.main(_vectorizer_argv(case, png, svg,
                                                 config.vectorization,
                                                 region_png_path))
                record["timings"]["vectorize"] = time.perf_counter() - stage_started
                if code != 0:
                    raise OrchestrationError("the vectorizer exited with code %d"
                                             % code)
                if not os.path.isfile(svg) or os.path.getsize(svg) == 0:
                    raise OrchestrationError("the vectorizer wrote no SVG")
                record["vectorization"] = {
                    "svg": _relative(svg, space),
                    "svg_bytes": int(os.path.getsize(svg)),
                    "mask_mode": ("regions" if region_png_path
                                  else config.vectorization.mask),
                    "alpha_threshold": config.vectorization.alpha_threshold,
                    "region_map": (None if region_png_path is None
                                   else _relative(region_png_path, space)),
                }

                stage = "accuracy"
                stage_started = time.perf_counter()
                scored = _load_and_score(png, svg, config.vectorization,
                                         report_cfg.tolerance,
                                         region_png=region_png_path)
                record["timings"]["accuracy"] = time.perf_counter() - stage_started
                record["accuracy"] = scored["compact"]
                record["metrics"] = scored["metrics"]
                record["vectorization"].update({
                    "paths": int(scored["document"]["paths"]),
                    "curves": int(scored["document"]["segments"]),
                    "loops": len(scored["document"]["loops"]),
                    "svg_bytes": int(scored["svg_bytes"]),
                    "mask_mode": scored["mask_mode"],
                    "mask_note": scored["mask_note"],
                    "alpha_threshold": config.vectorization.alpha_threshold,
                    "size_matches": scored["size_matches"],
                    "region_count": scored.get("region_count"),
                })
                if scored["size_matches"] is False:
                    print("      warning  the SVG size does not match the raster",
                          file=sys.stderr, flush=True)

                if config.overlay.enabled and not args.no_overlay:
                    stage = "overlay"
                    stage_started = time.perf_counter()
                    overlay = render_overlay(scored["image"],
                                             scored["document"]["loops"],
                                             config.overlay)
                    if not cv.imwrite(overlay_png, overlay):
                        raise OrchestrationError("cannot write %s" % overlay_png)
                    record["timings"]["overlay"] = time.perf_counter() - stage_started
                    record["overlay"] = {"png": _relative(overlay_png, space)}

                below = threshold_failures(record["accuracy"], report_cfg)
                record["below"] = below
                record["status"] = "below" if below else "ok"
                record["stage"] = "done"
            except KeyboardInterrupt:
                raise
            except Exception as error:
                record["status"] = "failed"
                record["stage"] = stage
                record["error"] = "%s: %s" % (type(error).__name__, error)

            records.append(record)
            if report_cfg.json:
                _write_json(report_png, {
                    "tool": "chunkmap-separator.orchestration",
                    "schema": 1,
                    "name": case.name,
                    "index": case.index,
                    "status": record["status"],
                    "stage": record["stage"],
                    "error": record.get("error"),
                    "seed": record["seed"],
                    "requested": record["requested"],
                    "generation": record.get("generation"),
                    "vectorization": record.get("vectorization"),
                    "accuracy": record.get("accuracy"),
                    "metrics": record.get("metrics"),
                    "below": record.get("below"),
                    "overlay": record.get("overlay"),
                    "timings": record.get("timings"),
                    "thresholds": report_cfg.thresholds(),
                })

            if record["status"] == "failed" and (args.quiet
                                                 or report_cfg.show == "summary"):
                print("error: %s  %s" % (record["name"], record.get("error")),
                      file=sys.stderr, flush=True)
            elif args.quiet and record["status"] == "below":
                print("below: %s  %s" % (record["name"],
                                         _describe_failures(record.get("below", []))),
                      file=sys.stderr, flush=True)

            if not args.quiet:
                if report_cfg.show == "compact":
                    _print_image_compact(record)
                elif report_cfg.show == "detailed":
                    _print_image_detailed(record, case)
                print("", flush=True)

            if record["status"] == "failed" and args.fail_fast:
                print("stopping after the first failure (--fail-fast)",
                      file=sys.stderr, flush=True)
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted; writing the results collected so far",
              file=sys.stderr, flush=True)

    wall = time.time() - started
    aggregate = _aggregate(records, wall, len(records), len(cases), report_cfg)

    if report_cfg.json:
        summary_json = os.path.join(space, "reports", "summary.json")
        _write_json(summary_json, {
            "tool": "chunkmap-separator.orchestration",
            "schema": 1,
            "name": config.name,
            "space": space,
            "config_source": config.source,
            "started": started_iso,
            "finished": _iso_now(),
            "config": config.snapshot(),
            "plan": [case.to_dict() for case in cases],
            "images": [_summary_record(record) for record in records],
            "aggregate": aggregate,
        })
    if report_cfg.csv:
        summary_csv = os.path.join(space, "reports", "summary.csv")
        _write_csv(summary_csv, records)

    if not args.quiet:
        print("")
        _print_aggregate(aggregate, summary_json, summary_csv)
    else:
        line = ("orchestrate  %d/%d processed  %d ok  %d below  %d failed"
                % (aggregate["processed"], aggregate["planned"],
                   aggregate["ok"], aggregate["below_threshold"],
                   aggregate["failed"]))
        if summary_json:
            line += "  -> %s" % summary_json
        print(line)

    if interrupted:
        return 130
    if aggregate["failed"] or aggregate["below_threshold"]:
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py orchestrate",
        description="Generate random islands, vectorize them, render SVG "
                    "overlays and score the result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python code/cli.py orchestrate --config code/configs/island-svg-sample.yaml\n"
            "  python code/cli.py orchestrate --config code/configs/island-svg-sample.json "
            "--space outputs/my-test\n"
            "  python code/cli.py orchestrate --config code/configs/island-svg-sample.yaml "
            "--only 1,3-5 --compact\n"
        ),
    )
    parser.add_argument("--config", required=True, metavar="PATH",
                        help="JSON or YAML config file")
    parser.add_argument("--space", metavar="DIR",
                        help="test space (overrides config.space)")
    parser.add_argument("--seed", type=int, metavar="N",
                        help="base seed (overrides config.seed)")
    parser.add_argument("--count", type=int, metavar="N",
                        help="number of images (overrides config count unless "
                             "the config lists images explicitly)")
    parser.add_argument("--only", metavar="LIST",
                        help="1-based case indices, e.g. 1,3-5")
    parser.add_argument("--no-overlay", action="store_true",
                        help="do not render the overlay PNGs")
    parser.add_argument("--clean", action="store_true",
                        help="remove islands/, svg/, overlay/ and reports/ in the "
                             "test space before running")
    parser.add_argument("--compact", action="store_true",
                        help="one line per image instead of the full report")
    parser.add_argument("--quiet", action="store_true",
                        help="only errors and the final summary")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop after the first failed image")
    return parser


def orchestrate_main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.seed is not None:
            if not 0 <= args.seed <= MAX_SEED:
                raise ConfigError("--seed must be between 0 and %d" % MAX_SEED)
            config = replace(config, seed=args.seed)
        if args.count is not None:
            if args.count < 1 or args.count > 500:
                raise ConfigError("--count must be between 1 and 500")
            if config.generation.images:
                raise ConfigError("--count cannot be combined with "
                                  "generation.images; edit the config instead")
            config = replace(config, generation=replace(config.generation,
                                                        count=args.count))
        space = resolve_space(config, args.space)
        cases = build_cases(config)
        cases = select_cases(cases, args.only)
        if not cases:
            raise ConfigError("the config produced no images")
        return run(config, space, cases, args)
    except ConfigError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    except OrchestrationError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    except OSError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def main(argv: Optional[list[str]] = None) -> int:
    return orchestrate_main(argv)


if __name__ == "__main__":
    sys.exit(main())

