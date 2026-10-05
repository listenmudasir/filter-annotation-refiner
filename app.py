from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from refiner.i18n import translator
from refiner.ui.main_window import MainWindow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO to SAM smart polygon annotation refiner")
    p.add_argument(
        "--backend",
        choices=["sam2", "sam3", "fallback"],
        default="sam2",
        help="Segmentation backend. 'fallback' draws placeholder ellipses for UI testing "
             "only and never produces a usable dataset.",
    )
    p.add_argument("--device", default=None, help="SAM device, e.g. cuda or cpu")
    p.add_argument("--checkpoint", default=None, help="Optional local SAM checkpoint path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("Filter Annotation Refiner")
    # Restore the operator's language choice before any widget is built.
    translator.load()
    win = MainWindow(args.backend, args.device, args.checkpoint)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
