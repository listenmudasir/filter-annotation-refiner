import cv2
import numpy as np
import pytest

from refiner.geometry import (
    MaskPolygon,
    bridge_holes,
    mask_iou,
    mask_to_polygon,
    mask_to_polygons,
    parts_to_export_rings,
    polygon_fidelity,
    polygons_to_mask,
    polygon_to_mask,
    simplify_to_fidelity,
)


def two_strands() -> np.ndarray:
    """Two disconnected thin bars, as SAM returns for a broken hair."""
    mask = np.zeros((200, 200), bool)
    mask[50:60, 20:180] = True
    mask[140:150, 20:180] = True
    return mask


def ring() -> np.ndarray:
    """A mask with a genuine interior hole, as for a ring-shaped stain."""
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 70, 1, -1)
    cv2.circle(mask, (100, 100), 35, 0, -1)
    return mask.astype(bool)


def island_in_hole() -> np.ndarray:
    mask = np.zeros((300, 300), np.uint8)
    cv2.circle(mask, (150, 150), 100, 1, -1)
    cv2.circle(mask, (150, 150), 60, 0, -1)
    cv2.circle(mask, (150, 150), 25, 1, -1)
    return mask.astype(bool)


def thin_hair() -> np.ndarray:
    mask = np.zeros((1024, 1024), np.uint8)
    cv2.line(mask, (50, 50), (980, 900), 1, 3)
    return mask.astype(bool)


def test_round_trip_square():
    pts = [(10, 10), (50, 10), (50, 50), (10, 50)]
    mask = polygon_to_mask(pts, 64, 64)
    parts = mask_to_polygons(mask)
    assert len(parts) == 1
    assert polygon_fidelity(mask, parts) > 0.95


def test_all_components_survive():
    """Regression: the old single-largest-contour path discarded half this mask."""
    mask = two_strands()
    parts = mask_to_polygons(mask)
    assert len(parts) == 2
    assert polygon_fidelity(mask, parts) == pytest.approx(1.0, abs=1e-6)


def test_holes_are_preserved_not_filled():
    """Regression: filling the hole inflated this mask by ~34% (IoU 0.747)."""
    mask = ring()
    parts = mask_to_polygons(mask)
    assert len(parts) == 1
    assert len(parts[0].holes) == 1
    assert polygon_fidelity(mask, parts) > 0.97

    filled = mask_to_polygons(mask, keep_holes=False)
    assert not filled[0].holes
    assert polygon_fidelity(mask, filled) < 0.80


def test_island_inside_a_hole_is_kept():
    mask = island_in_hole()
    parts = mask_to_polygons(mask)
    assert len(parts) == 2
    assert sum(len(p.holes) for p in parts) == 1
    assert polygon_fidelity(mask, parts) > 0.97


def test_bridged_ring_reproduces_the_hole():
    """YOLO-seg cannot express holes, so verify the keyhole slit rasterizes right."""
    mask = ring()
    part = mask_to_polygons(mask)[0]
    single = bridge_holes(part)
    rendered = polygons_to_mask([MaskPolygon(single)], 200, 200)
    assert mask_iou(mask, rendered) > 0.97


def test_export_rings_are_hole_free_and_measured():
    mask = island_in_hole()
    parts = mask_to_polygons(mask)
    rings, fidelity, dropped = parts_to_export_rings(mask, parts)
    assert len(rings) == 2
    assert dropped == 0
    assert fidelity > 0.97
    # What gets written must match what the fidelity number claims.
    assert mask_iou(mask, polygons_to_mask([MaskPolygon(r) for r in rings], 300, 300)) == pytest.approx(
        fidelity, abs=0.02
    )


def test_simplify_respects_the_fidelity_bound():
    mask = ring()
    parts = mask_to_polygons(mask)
    simplified, fidelity = simplify_to_fidelity(mask, parts, target_fidelity=0.98)
    assert fidelity >= 0.98
    assert sum(p.point_count for p in simplified) < sum(p.point_count for p in parts)


def test_thin_structure_is_not_oversimplified():
    """A 3px hair must keep its vertices; a blob must not."""
    hair = thin_hair()
    hair_parts, hair_fidelity = simplify_to_fidelity(hair, mask_to_polygons(hair), 0.98)
    assert hair_fidelity >= 0.98
    assert polygon_fidelity(hair, hair_parts) >= 0.98

    blob = np.zeros((1024, 1024), np.uint8)
    cv2.circle(blob, (512, 512), 200, 1, -1)
    blob = blob.astype(bool)
    blob_parts, blob_fidelity = simplify_to_fidelity(blob, mask_to_polygons(blob), 0.98)
    assert blob_fidelity >= 0.98
    # The blob tolerates far more aggressive vertex reduction than the hair.
    hair_pts = sum(p.point_count for p in hair_parts)
    blob_pts = sum(p.point_count for p in blob_parts)
    assert blob_pts < hair_pts


def test_fidelity_penalizes_polygons_leaking_outside_the_mask():
    mask = np.zeros((100, 100), bool)
    mask[40:60, 40:60] = True
    leaking = [MaskPolygon([(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)])]
    assert polygon_fidelity(mask, leaking) < 0.1


def test_empty_mask_and_empty_parts():
    empty = np.zeros((32, 32), bool)
    assert mask_to_polygons(empty) == []
    assert polygon_fidelity(empty, []) == 1.0
    assert polygon_fidelity(two_strands(), []) == 0.0


def test_min_area_filters_speckles():
    mask = two_strands()
    mask[5:7, 5:7] = True  # 4px speckle
    assert len(mask_to_polygons(mask, min_area=1.0)) == 3
    assert len(mask_to_polygons(mask, min_area=50.0)) == 2


def test_max_parts_keeps_largest():
    mask = two_strands()
    mask[100:110, 20:60] = True  # smaller third component
    parts = mask_to_polygons(mask, max_parts=2)
    assert len(parts) == 2
    assert all(p.area > 1000 for p in parts)


def test_legacy_single_polygon_helper_still_works():
    poly = mask_to_polygon(ring(), 0.5)
    assert len(poly) >= 3


def test_boxes_are_drawn_on_top_of_masks():
    """Regression: a translucent mask painted over its own box erased the box.

    With overlapping objects a later mask also covered earlier boxes, so the
    detection input became invisible in overlays and the live preview.
    """
    import numpy as np
    from PIL import Image as PILImage

    from refiner.geometry import MaskPolygon
    from refiner.models import Annotation, AnnotationKind, RefinementResult, ReviewState
    from refiner.overlay import BOX_COLOR, render_overlay

    image = PILImage.new("RGB", (400, 300), "black")
    box = (50.0, 50.0, 350.0, 250.0)
    # A mask that completely covers its own box, plus the next object's box.
    ring = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
    ann = Annotation(0, AnnotationKind.DETECTION, box)
    result = RefinementResult(
        mask=np.zeros((300, 400), bool), polygons=[MaskPolygon(ring)], export_rings=[ring],
        polygon_fidelity=0.99, sam_score=0.9, stability=0.9, edge_alignment=0.5,
        leakage=0.0, fragmentation=0.0, quality=0.85,
        source_kind=AnnotationKind.DETECTION, prompt_name="p", state=ReviewState.ACCEPTED,
    )

    rendered = np.asarray(render_overlay(image, [ann, ann], [result, result]))
    # Sample the top edge of the box; it must still carry the box colour.
    strip = rendered[48:54, 150:250].reshape(-1, 3).astype(int)
    target = np.asarray(BOX_COLOR[:3], dtype=int)
    closest = np.abs(strip - target).sum(axis=1).min()
    assert closest < 60, f"box edge was painted over by the mask (closest={closest})"
