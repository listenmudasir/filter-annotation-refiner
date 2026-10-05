from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..backend.base import SegmentationBackend
from ..backend.fallback import FallbackBackend
from ..backend.sam2_backend import Sam2Backend
from ..backend.sam3_backend import Sam3Backend
from ..models import DatasetIndex, ReviewState
from ..services.refinement import ConversionCancelled, ConversionSettings, SmartRefinementEngine


def build_backend(name: str, device: str | None = None, checkpoint: str | None = None) -> SegmentationBackend:
    if name == "fallback":
        return FallbackBackend()
    if name == "sam2":
        return Sam2Backend(device, checkpoint)
    if name == "sam3":
        return Sam3Backend(device, checkpoint)
    raise ValueError(f"Unknown backend: {name}")


class ConversionWorker(QObject):
    progress = Signal(int, int, object)
    status = Signal(str)
    statistics = Signal(dict)
    error = Signal(str)
    finished = Signal(dict)

    def __init__(
        self,
        dataset: DatasetIndex,
        output_root: Path,
        backend_name: str,
        settings: ConversionSettings,
        device: str | None = None,
        checkpoint: str | None = None,
        resume: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.output_root = output_root
        self.backend_name = backend_name
        self.settings = settings
        self.device = device
        self.checkpoint = checkpoint
        self.resume = resume
        self._stop = threading.Event()
        self._pause = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def toggle_pause(self) -> bool:
        if self._pause.is_set():
            self._pause.clear()
            return False
        self._pause.set()
        return True

    def is_cancelled(self) -> bool:
        return self._stop.is_set()

    @Slot()
    def run(self) -> None:
        cancelled = False
        try:
            backend = build_backend(self.backend_name, self.device, self.checkpoint)
            self.status.emit(f"Loading {self.backend_name} backend…")
            backend.load()
            if self._stop.is_set():
                self.finished.emit({"cancelled": True, "output_root": str(self.output_root)})
                return
            self.status.emit(f"Backend ready on {backend.device_label}")
            engine = SmartRefinementEngine(self.dataset, backend, self.output_root, self.settings)
            total = len(self.dataset.records)
            stats = {"images": 0, "objects": 0, "accepted": 0, "review": 0, "failed": 0, "skipped": 0}
            for i, record in enumerate(self.dataset.records, start=1):
                if self._stop.is_set():
                    cancelled = True
                    break
                while self._pause.is_set() and not self._stop.is_set():
                    self._stop.wait(0.1)
                if self._stop.is_set():
                    cancelled = True
                    break
                self.status.emit(f"Processing {record.relative_image.as_posix()}")
                try:
                    # The cancellation check is handed to the engine so a stop takes
                    # effect between objects rather than after a whole image.
                    outcome = engine.process_record(record, resume=self.resume, should_stop=self.is_cancelled)
                    if outcome.skipped:
                        stats["skipped"] += 1
                    else:
                        stats["images"] += 1
                        stats["objects"] += len(outcome.results)
                        stats["accepted"] += sum(r.state == ReviewState.ACCEPTED for r in outcome.results)
                        stats["review"] += sum(r.state == ReviewState.REVIEW for r in outcome.results)
                    self.progress.emit(i, total, outcome)
                except ConversionCancelled:
                    cancelled = True
                    break
                except Exception as exc:
                    stats["failed"] += 1
                    tb = traceback.format_exc()
                    message = f"{type(exc).__name__}: {exc}"
                    engine.record_failure(record, str(exc), type(exc).__name__, tb)
                    self.status.emit(f"Failed: {record.relative_image.as_posix()} — {message}")
                    self.progress.emit(i, total, (record, message))
                self.statistics.emit(dict(stats))

            if cancelled:
                self.status.emit("Stopped. Images completed before the stop are saved.")
            stats["cancelled"] = cancelled
            summary = engine.write_summary()
            stats["summary"] = str(summary)
            stats["output_root"] = str(self.output_root)
            self.finished.emit(stats)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
