from refiner.models import AnnotationKind
from refiner.yolo import parse_yolo_line, polygon_to_yolo_line, polygons_to_yolo_lines


def test_detection_parse():
    ann = parse_yolo_line("1 0.5 0.5 0.4 0.2", 100, 200)
    assert ann.kind == AnnotationKind.DETECTION
    assert ann.class_id == 1
    assert ann.bbox_xyxy == (30.0, 80.0, 70.0, 120.0)


def test_polygon_parse_and_write():
    ann = parse_yolo_line("2 0.1 0.1 0.8 0.1 0.8 0.8", 100, 100)
    assert ann.kind == AnnotationKind.POLYGON
    assert len(ann.polygon_xy) == 3
    out = polygon_to_yolo_line(2, ann.polygon_xy, 100, 100)
    assert out.startswith("2 ")


def test_multi_part_writer_emits_one_line_per_ring():
    rings = [
        [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0)],
        [(60.0, 60.0), (90.0, 60.0), (90.0, 90.0)],
        [(1.0, 1.0)],  # degenerate, must be skipped
    ]
    lines = polygons_to_yolo_lines(3, rings, 100, 100)
    assert len(lines) == 2
    assert all(line.startswith("3 ") for line in lines)


def test_writer_clamps_to_normalized_range():
    # Coordinates at the image edge must not serialize above 1.0.
    line = polygon_to_yolo_line(0, [(0.0, 0.0), (120.0, 0.0), (120.0, 110.0)], 100, 100)
    values = [float(v) for v in line.split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)
