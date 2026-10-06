"""Run the app from a source checkout: python app.py --backend sam2 --device cuda

Installed copies get the `filter-annotation-refiner` command instead; both call
the same entry point in refiner/cli.py.
"""

from __future__ import annotations

from refiner.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
