from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import MaskPolygon


class AnnotationKind(str, Enum):
    DETECTION = "detection"
    POLYGON = "polygon"


class ReviewState(str, Enum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    FAILED = "failed"
    PENDING = "pending"


@dataclass(slots=True)
class Annotation:
    class_id: int
    kind: AnnotationKind
    bbox_xyxy: tuple[float, float, float, float]
    polygon_xy: list[tuple[float, float]] = field(default_factory=list)
    source_line: str = ""
    instance_index: int = 0


@dataclass(slots=True)
class ImageRecord:
    image_path: Path
    label_path: Optional[Path]
    relative_image: Path
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DatasetIndex:
    root: Path
    records: list[ImageRecord]
    class_names: dict[int, str]
    issues: list[str] = field(default_factory=list)
    #: How labels were located, e.g. "images and labels in the same folder".
    layout: str = "unknown"

    @property
    def object_count(self) -> int:
        return sum(len(r.annotations) for r in self.records)

    @property
    def background_count(self) -> int:
        """Images with no annotations - valid YOLO background images."""
        return sum(1 for r in self.records if not r.annotations and not r.issues)

    @property
    def problem_records(self) -> list[ImageRecord]:
        return [r for r in self.records if r.issues]

    @property
    def convertible_records(self) -> list[ImageRecord]:
        return [r for r in self.records if not r.issues]

    @property
    def detection_count(self) -> int:
        return sum(a.kind == AnnotationKind.DETECTION for r in self.records for a in r.annotations)

    @property
    def polygon_count(self) -> int:
        return sum(a.kind == AnnotationKind.POLYGON for r in self.records for a in r.annotations)

    @property
    def valid_records(self) -> list[ImageRecord]:
        return [r for r in self.records if not r.issues]


@dataclass(slots=True)
class CandidateMask:
    mask: np.ndarray
    sam_score: float
    prompt_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    quality: float = 0.0


@dataclass(slots=True)
class RefinementResult:
    mask: np.ndarray
    #: Lossless geometry: every component, each with its holes.
    polygons: list[MaskPolygon]
    #: Flattened single rings as actually written to YOLO-seg.
    export_rings: list[list[tuple[float, float]]]
    #: IoU between ``mask`` and the rasterization of ``export_rings``.
    polygon_fidelity: float
    sam_score: float
    stability: float
    edge_alignment: float
    leakage: float
    fragmentation: float
    quality: float
    source_kind: AnnotationKind
    prompt_name: str
    state: ReviewState
    warnings: list[str] = field(default_factory=list)

    @property
    def part_count(self) -> int:
        return len(self.polygons)

    @property
    def hole_count(self) -> int:
        return sum(len(p.holes) for p in self.polygons)

    @property
    def point_count(self) -> int:
        return sum(len(ring) for ring in self.export_rings)
