"""Console entry point.

Lives inside the package so ``pip install`` can expose it as a command; app.py
stays as a thin wrapper for running straight from a checkout.
"""

from __future__ import annotations

import argparse
import sys

BACKENDS = ("sam2", "sam3", "fallback")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="filter-annotation-refiner",
        description="Convert detection datasets into polygon segmentation datasets with SAM",
    )
    p.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default="sam2",
        help="Segmentation backend. 'fallback' draws placeholder ellipses for UI testing "
             "only and never produces a usable dataset.",
    )
    p.add_argument("--device", default=None, help="SAM device, e.g. cuda or cpu")
    p.add_argument("--checkpoint", default=None, help="Optional local SAM checkpoint path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Imported here so --help works without a Qt display or a built GUI stack.
    from PySide6.QtWidgets import QApplication

    from .i18n import translator
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Filter Annotation Refiner")
    # Restore the operator's language choice before any widget is built.
    translator.load()
    window = MainWindow(args.backend, args.device, args.checkpoint)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
