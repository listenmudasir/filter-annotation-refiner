from __future__ import annotations

from pathlib import Path

from .models import Annotation, AnnotationKind


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _bbox_from_polygon(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def parse_yolo_line(line: str, width: int, height: int, instance_index: int = 0) -> Annotation:
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty line")
    try:
        class_id = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
    except ValueError as exc:
        raise ValueError("non-numeric YOLO annotation") from exc

    if len(values) == 4:
        xc, yc, bw, bh = values
        if bw <= 0 or bh <= 0:
            raise ValueError("YOLO box width/height must be positive")
        x0 = _clamp((xc - bw / 2) * width, 0, width - 1)
        y0 = _clamp((yc - bh / 2) * height, 0, height - 1)
        x1 = _clamp((xc + bw / 2) * width, 0, width - 1)
        y1 = _clamp((yc + bh / 2) * height, 0, height - 1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("degenerate bounding box")
        return Annotation(
            class_id=class_id,
            kind=AnnotationKind.DETECTION,
            bbox_xyxy=(x0, y0, x1, y1),
            source_line=line.strip(),
            instance_index=instance_index,
        )

    if len(values) >= 6 and len(values) % 2 == 0:
        pts: list[tuple[float, float]] = []
        for i in range(0, len(values), 2):
            x = _clamp(values[i] * width, 0, width - 1)
            y = _clamp(values[i + 1] * height, 0, height - 1)
            pts.append((x, y))
        if len(pts) < 3:
            raise ValueError("polygon requires at least 3 points")
        return Annotation(
            class_id=class_id,
            kind=AnnotationKind.POLYGON,
            bbox_xyxy=_bbox_from_polygon(pts),
            polygon_xy=pts,
            source_line=line.strip(),
            instance_index=instance_index,
        )

    raise ValueError(
        "unsupported YOLO row: expected class + 4 box values or class + polygon coordinate pairs"
    )


def load_yolo_file(path: Path | None, width: int, height: int) -> tuple[list[Annotation], list[str]]:
    if path is None or not path.exists():
        # An image with no label file is a *background* image, which is valid YOLO
        # and common in real datasets. Treating it as an error used to block the
        # whole conversion; it simply has nothing to convert.
        return [], []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [], [f"Cannot read label file: {exc}"]

    annotations: list[Annotation] = []
    issues: list[str] = []
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            annotations.append(parse_yolo_line(line, width, height, len(annotations)))
        except ValueError as exc:
            issues.append(f"Line {line_no}: {exc}")
    return annotations, issues


def polygon_to_yolo_line(class_id: int, polygon: list[tuple[float, float]], width: int, height: int) -> str:
    vals: list[str] = [str(class_id)]
    for x, y in polygon:
        vals.extend([f"{_clamp(x / width, 0.0, 1.0):.6f}", f"{_clamp(y / height, 0.0, 1.0):.6f}"])
    return " ".join(vals)


def polygons_to_yolo_lines(
    class_id: int,
    rings: list[list[tuple[float, float]]],
    width: int,
    height: int,
) -> list[str]:
    """One YOLO-seg line per mask component.

    YOLO-seg has no multi-part primitive, so a fragmented object (several hair
    strands from one box) is written as several lines sharing a class id. This is
    what Ultralytics expects and trains on; collapsing to one line would discard
    every component but the largest.
    """
    return [
        polygon_to_yolo_line(class_id, ring, width, height)
        for ring in rings
        if len(ring) >= 3
    ]
