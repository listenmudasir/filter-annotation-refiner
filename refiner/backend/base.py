from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from ..models import Annotation, CandidateMask


class SegmentationBackend(ABC):
    name = "base"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def set_image(self, image: Image.Image) -> None: ...

    @abstractmethod
    def candidates(self, annotation: Annotation, preset: str) -> list[CandidateMask]: ...

    def close_image(self) -> None:
        return None

    @property
    def device_label(self) -> str:
        return "unknown"
