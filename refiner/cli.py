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
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Build the window and load the segmentation backend, then exit. Used by "
             "build.ps1 to prove a compiled build actually works before packaging it.",
    )
    return p.parse_args(argv)


def self_test(args) -> int:
    """Prove a build is functional: Qt resolves, the model loads, a mask comes out.

    A compiled exe can start and still be useless - missing Qt plugins, sam2 not
    linked in, hydra configs absent, checkpoint not shipped. Each of those fails
    here with a specific message instead of surfacing as a blank window on a
    vendor machine.
    """
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from .models import Annotation, AnnotationKind
    from .paths import app_dir, bundled_weights_dir, is_frozen
    from .ui.main_window import MainWindow

    print(f"self-test: frozen={is_frozen()} app_dir={app_dir()}")

    app = QApplication(sys.argv[:1])
    window = MainWindow(args.backend, args.device, args.checkpoint)
    window.show()
    app.processEvents()
    if not window.isVisible():
        print("self-test FAILED: main window did not become visible")
        return 2
    print("self-test: main window constructed")

    if args.backend == "fallback":
        print("self-test: fallback backend, skipping model load")
        return 0

    print(f"self-test: weights dir {bundled_weights_dir()}")
    from .ui.workers import build_backend

    backend = build_backend(args.backend, args.device, args.checkpoint)
    backend.load()
    print(f"self-test: segmentation backend loaded on {backend.device_label}")

    # One real inference, so a broken hydra config or checkpoint cannot pass.
    from PIL import Image

    probe = Image.fromarray(np.full((256, 256, 3), 127, dtype=np.uint8))
    backend.set_image(probe)
    candidates = backend.candidates(
        Annotation(0, AnnotationKind.DETECTION, (64.0, 64.0, 192.0, 192.0)), "fast"
    )
    if not candidates:
        print("self-test FAILED: backend returned no mask for a probe box")
        return 3
    print(f"self-test: produced {len(candidates)} mask(s), score {candidates[0].sam_score:.3f}")
    print("self-test PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        try:
            return self_test(args)
        except Exception as exc:  # a build failure must be loud and specific
            import traceback

            print(f"self-test FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 1

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
