#!/usr/bin/env python3
"""Run the alternative baseline_icp backend."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline_icp.pipeline import main


if __name__ == "__main__":
    main()
