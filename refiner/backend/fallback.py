from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..geometry import perturb_box, polygon_to_mask
from ..models import Annotation, AnnotationKind, CandidateMask
from .base import SegmentationBackend


class FallbackBackend(SegmentationBackend):
    """Deterministic development backend. It is deliberately not presented as SAM output."""

    name = "fallback"

    def __init__(self):
        self.image: Image.Image | None = None

    def load(self) -> None:
        return None

    @property
    def device_label(self) -> str:
        return "CPU / development"

    def set_image(self, image: Image.Image) -> None:
        self.image = image.convert("RGB")

    def candidates(self, annotation: Annotation, preset: str) -> list[CandidateMask]:
        if self.image is None:
            raise RuntimeError("No image loaded")
        width, height = self.image.size
        if annotation.kind == AnnotationKind.POLYGON and annotation.polygon_xy:
            m = polygon_to_mask(annotation.polygon_xy, width, height)
            return [CandidateMask(m, 0.65, "existing polygon")]

        specs = [(0.0, 0.0, 0.0)]
        if preset in {"balanced", "maximum"}:
            specs += [(0.04, 0.0, 0.0), (-0.02, 0.0, 0.0)]
        if preset == "maximum":
            specs += [(0.07, 0.02, 0.0), (0.07, -0.02, 0.0)]
        out: list[CandidateMask] = []
        for i, (scale, dx, dy) in enumerate(specs):
            x0, y0, x1, y1 = perturb_box(annotation.bbox_xyxy, width, height, scale, dx, dy)
            mask = np.zeros((height, width), dtype=np.uint8)
            cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
            axes = (max(1, int((x1 - x0) * 0.45)), max(1, int((y1 - y0) * 0.45)))
            cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1, -1)
            out.append(CandidateMask(mask.astype(bool), 0.70 - i * 0.02, f"fallback-{i + 1}"))
        return out
