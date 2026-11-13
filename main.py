"""Chunk Map Separator — repository entry point.

This is the file to run. It is intentionally tiny: the argument parser and all
commands live in :mod:`chunkmap.cli`, so they are importable by the tests and by
the orchestration batch runner.

    python main.py --help
    python main.py resources/example-cuba.png --report
    python main.py generate outputs/islands 8 1024 768 --seed 42
    python main.py orchestrate --config configs/island-svg-sample.yaml
"""

import sys

from chunkmap.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    sys.exit(main())
