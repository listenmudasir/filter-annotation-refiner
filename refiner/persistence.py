from __future__ import annotations

import json
from pathlib import Path
from threading import Lock


class ProjectState:
    def __init__(self, output_root: Path):
        self.dir = output_root / ".sam_refiner"
        self.path = self.dir / "state.json"
        self._lock = Lock()
        self.data = {"completed": {}, "version": 1}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except Exception:
                pass

    @property
    def has_progress(self) -> bool:
        return bool(self.data.get("completed"))

    def run_signature(self) -> dict | None:
        """What produced the results already in this folder, if recorded."""
        signature = self.data.get("run")
        return signature if isinstance(signature, dict) else None

    def set_run_signature(self, signature: dict) -> None:
        with self._lock:
            self.data["run"] = signature
            self._save()

    def is_completed(self, relative_image: Path) -> bool:
        return relative_image.as_posix() in self.data.get("completed", {})

    def mark_completed(self, relative_image: Path, metadata: dict) -> None:
        with self._lock:
            self.data.setdefault("completed", {})[relative_image.as_posix()] = metadata
            self._save()

    def reset(self) -> None:
        with self._lock:
            self.data = {"completed": {}, "version": 1}
            self._save()

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)
