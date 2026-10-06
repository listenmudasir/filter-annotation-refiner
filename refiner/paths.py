"""Where things live, in a source checkout and inside a compiled build.

Nuitka's onefile mode unpacks to a temporary folder, so ``__file__`` points at
that temp directory rather than at the install location. Anything that must be
found next to the installed exe - the bundled model weights, the log folder -
has to be resolved from ``sys.argv[0]`` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running from a Nuitka/PyInstaller build rather than source."""
    return "__compiled__" in globals() or bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Install folder of a compiled build, or the repository root from source."""
    if is_frozen():
        # Not __file__: in onefile mode that is the temp extraction folder.
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundled_weights_dir() -> Path:
    """Where the installer places model checkpoints."""
    return app_dir() / "weights"


def log_dir() -> Path:
    path = app_dir() / "logs"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        # A read-only install (Program Files without elevation) must not be fatal.
        return Path.home() / ".filter-annotation-refiner" / "logs"
    return path
