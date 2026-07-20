#!/usr/bin/env python3
"""Build a KISS-ICP map from recorded poses and the raw LiDAR bag."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapping import main


if __name__ == "__main__":
    main()
