"""Support ``python -m chunkmap`` as an alias for ``python main.py``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
