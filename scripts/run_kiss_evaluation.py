#!/usr/bin/env python3
"""Run the validated KISS-ICP trajectory evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kiss_icp.pipeline import main


if __name__ == "__main__":
    main()
