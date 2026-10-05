from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

Point = tuple[float, float]
Ring = list[Point]


def clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    x0 = max(0.0, min(float(width - 1), x0))
    y0 = max(0.0, min(float(height - 1), y0))
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    return x0, y0, x1, y1


def perturb_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    scale: float = 0.0,
    dx_frac: float = 0.0,
    dy_frac: float = 0.0,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw2, bh2 = bw * (1 + scale), bh * (1 + scale)
    cx += dx_frac * bw
    cy += dy_frac * bh
    return clamp_box((cx - bw2 / 2, cy - bh2 / 2, cx + bw2 / 2, cy + bh2 / 2), width, height)


def polygon_to_mask(points: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    # Axis reductions rather than np.where: this runs on every candidate of every
    # object, and materializing coordinate arrays for a 4K mask dominates the cost.
    mask = mask.astype(bool, copy=False)
    rows = np.any(mask, axis=1)
    if not rows.any():
        return None
    cols = np.any(mask, axis=0)
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return x0, y0, x1, y1


def box_mask(box: tuple[float, float, float, float], width: int, height: int, expand: float = 0.0) -> np.ndarray:
    x0, y0, x1, y1 = perturb_box(box, width, height, scale=expand)
    mask = np.zeros((height, width), dtype=bool)
    mask[int(y0):max(int(y0) + 1, int(y1)), int(x0):max(int(x0) + 1, int(x1))] = True
    return mask


# --------------------------------------------------------------------------- #
# Multi-part, hole-aware mask <-> polygon conversion
# --------------------------------------------------------------------------- #


def _ring_signed_area(ring: Ring) -> float:
    pts = np.asarray(ring, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _as_i32(ring: Ring, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    pts = np.asarray(ring, dtype=np.float64)
    pts = pts - np.asarray(offset, dtype=np.float64)
    return np.round(pts).astype(np.int32).reshape((-1, 1, 2))


@dataclass(slots=True)
class MaskPolygon:
    """One connected mask component: an outer ring plus the rings of its holes.

    This is the lossless internal representation. A SAM mask of two hair strands is
    two ``MaskPolygon`` objects; a ring-shaped stain is one object with one hole.
    Flattening either of those into a single ring is a lossy *export* decision, made
    explicitly in :func:`bridge_holes` / the YOLO writer, never silently here.
    """

    exterior: Ring
    holes: list[Ring] = field(default_factory=list)

    @property
    def point_count(self) -> int:
        return len(self.exterior) + sum(len(h) for h in self.holes)

    @property
    def area(self) -> float:
        return abs(_ring_signed_area(self.exterior)) - sum(abs(_ring_signed_area(h)) for h in self.holes)


def polygons_to_mask(parts: list[MaskPolygon], width: int, height: int, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Rasterize parts back to a boolean mask, honouring holes.

    Parts are painted largest-first so that an island sitting inside another part's
    hole is drawn after that hole is punched out, instead of being erased by it.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for part in sorted(parts, key=lambda p: p.area, reverse=True):
        if len(part.exterior) >= 3:
            cv2.fillPoly(mask, [_as_i32(part.exterior, offset)], 1)
        for hole in part.holes:
            if len(hole) >= 3:
                cv2.fillPoly(mask, [_as_i32(hole, offset)], 0)
    return mask.astype(bool)


def _fidelity_window(mask: np.ndarray, parts: list[MaskPolygon], pad: int = 2) -> tuple[int, int, int, int]:
    """Smallest window covering both the mask and the polygons.

    Covering the polygons as well as the mask matters: a polygon that leaks outside
    the mask's own bounding box must count against fidelity, not be cropped away.
    """
    h, w = mask.shape
    bb = mask_bbox(mask)
    x0, y0, x1, y1 = bb if bb is not None else (0, 0, 0, 0)
    for part in parts:
        for ring in (part.exterior, *part.holes):
            if not ring:
                continue
            pts = np.asarray(ring, dtype=np.float64)
            x0 = min(x0, int(np.floor(pts[:, 0].min())))
            y0 = min(y0, int(np.floor(pts[:, 1].min())))
            x1 = max(x1, int(np.ceil(pts[:, 0].max())) + 1)
            y1 = max(y1, int(np.ceil(pts[:, 1].max())) + 1)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


class _FidelityScope:
    """Reusable crop for scoring many polygon candidates against one mask.

    ``simplify_to_fidelity`` rasterizes eight candidate simplifications; recomputing
    the window and re-slicing a full-resolution mask each time is what made this
    stage cost hundreds of milliseconds per object on 4K frames.

    Safe to reuse across candidates because ``cv2.approxPolyDP`` returns a subset of
    its input vertices, so no simplification can extend beyond the original window.
    """

    __slots__ = ("reference", "offset", "width", "height")

    def __init__(self, mask: np.ndarray, parts: list[MaskPolygon], pad: int = 2):
        x0, y0, x1, y1 = _fidelity_window(mask, parts, pad=pad)
        self.reference = np.ascontiguousarray(mask.astype(bool, copy=False)[y0:y1, x0:x1])
        self.offset = (x0, y0)
        self.width, self.height = x1 - x0, y1 - y0

    def score(self, parts: list[MaskPolygon]) -> float:
        if not parts:
            return 0.0 if self.reference.any() else 1.0
        rendered = polygons_to_mask(parts, self.width, self.height, offset=self.offset)
        return mask_iou(self.reference, rendered)


def polygon_fidelity(mask: np.ndarray, parts: list[MaskPolygon]) -> float:
    """IoU between ``mask`` and the rasterization of ``parts``.

    This is the guard rail for the whole conversion: whatever the refinement stage
    decides, the number written to the dataset is only as good as this value. It is
    evaluated on a crop around the object so it stays cheap on large images.
    """
    if not parts:
        return 0.0 if mask.any() else 1.0
    return _FidelityScope(mask, parts).score(parts)


def _simplify_parts(parts: list[MaskPolygon], epsilon: float) -> list[MaskPolygon]:
    if epsilon <= 0:
        return parts
    out: list[MaskPolygon] = []
    for part in parts:
        exterior = cv2.approxPolyDP(_as_i32(part.exterior), epsilon, True).reshape(-1, 2)
        if len(exterior) < 3:
            continue
        holes: list[Ring] = []
        for hole in part.holes:
            simplified = cv2.approxPolyDP(_as_i32(hole), epsilon, True).reshape(-1, 2)
            if len(simplified) >= 3:
                holes.append([(float(x), float(y)) for x, y in simplified])
        out.append(MaskPolygon([(float(x), float(y)) for x, y in exterior], holes))
    return out


def _max_useful_epsilon(parts: list[MaskPolygon], scope: "_FidelityScope") -> float:
    """Upper bound for the simplification search.

    Bounded by the object's *thickness* (area / perimeter), not just its extent. A
    hair crossing a 4K frame has a huge extent but is three pixels wide, so probing
    large epsilons only wastes rasterizations on candidates that erase it.
    """
    extent = max(1.0, 0.02 * float(max(scope.width, scope.height)))
    area = sum(p.area for p in parts)
    perimeter = 0.0
    for part in parts:
        for ring in (part.exterior, *part.holes):
            pts = np.asarray(ring, dtype=np.float64)
            if len(pts) >= 2:
                perimeter += float(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).sum())
    if perimeter <= 0 or area <= 0:
        return extent
    return float(min(extent, max(1.0, 3.0 * area / perimeter)))


def simplify_to_fidelity(
    mask: np.ndarray,
    parts: list[MaskPolygon],
    target_fidelity: float = 0.98,
    max_epsilon: float | None = None,
    iterations: int = 8,
) -> tuple[list[MaskPolygon], float]:
    """Drop as many vertices as possible while keeping round-trip IoU >= target.

    Replaces a hand-tuned per-class ``simplify_px``. A thin hair keeps its vertices
    because its fidelity collapses the moment it is smoothed; a blobby stain
    simplifies hard. Returns the chosen parts and their measured fidelity.
    """
    if not parts:
        return parts, polygon_fidelity(mask, parts)

    scope = _FidelityScope(mask, parts)
    if max_epsilon is None:
        max_epsilon = _max_useful_epsilon(parts, scope)

    best, best_fidelity = parts, scope.score(parts)
    lo, hi = 0.0, float(max_epsilon)
    for _ in range(max(1, iterations)):
        # Epsilon below half a pixel cannot move an integer-vertex contour, so once
        # the bracket is that tight the answer will not improve.
        if hi - lo < 0.25:
            break
        mid = (lo + hi) / 2
        candidate = _simplify_parts(parts, mid)
        fidelity = scope.score(candidate) if candidate else 0.0
        if candidate and fidelity >= target_fidelity:
            # Still faithful, so we can afford to simplify harder.
            best, best_fidelity = candidate, fidelity
            lo = mid
        else:
            hi = mid
    return best, best_fidelity


def mask_to_polygons(
    mask: np.ndarray,
    min_area: float = 1.0,
    keep_holes: bool = True,
    min_hole_area: float = 1.0,
    max_parts: int = 0,
) -> list[MaskPolygon]:
    """Convert a mask to *all* of its components, each with its holes.

    Unlike taking the single largest external contour, this preserves fragmented
    objects (broken hair strands) and ring-shaped objects (stains) intact.
    """
    # Contour extraction is proportional to the scanned area, so work on the
    # object's own bounding box: a 60px defect in a 4K frame must not pay for 8M
    # pixels. One pixel of padding keeps border-touching contours well formed.
    bb = mask_bbox(mask)
    if bb is None:
        return []
    bx0, by0, bx1, by1 = bb
    bx0, by0 = max(0, bx0 - 1), max(0, by0 - 1)
    bx1, by1 = min(mask.shape[1], bx1 + 1), min(mask.shape[0], by1 + 1)
    crop = np.ascontiguousarray(mask[by0:by1, bx0:bx1].astype(np.uint8)) * 255

    contours, hierarchy = cv2.findContours(crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []

    hierarchy = hierarchy.reshape(-1, 4)
    origin = np.asarray((bx0, by0), dtype=np.float64)
    holes_by_parent: dict[int, list[Ring]] = {}
    if keep_holes:
        for i, (_, _, _, parent) in enumerate(hierarchy):
            if parent < 0:
                continue
            if cv2.contourArea(contours[i]) < min_hole_area:
                continue
            pts = contours[i].reshape(-1, 2) + origin
            if len(pts) >= 3:
                holes_by_parent.setdefault(int(parent), []).append([(float(x), float(y)) for x, y in pts])

    parts: list[MaskPolygon] = []
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent >= 0:
            continue  # a hole; attached to its parent above
        if cv2.contourArea(contours[i]) < min_area:
            continue
        pts = contours[i].reshape(-1, 2) + origin
        if len(pts) < 3:
            continue
        parts.append(
            MaskPolygon(
                exterior=[(float(x), float(y)) for x, y in pts],
                holes=holes_by_parent.get(i, []),
            )
        )

    parts.sort(key=lambda p: p.area, reverse=True)
    if max_parts > 0:
        parts = parts[:max_parts]
    return parts


def bridge_holes(part: MaskPolygon) -> Ring:
    """Flatten one part with holes into a single ring using keyhole slits.

    Formats such as YOLO-seg have no way to express a hole, so each hole is joined
    to the outer ring by a narrow cut. The caller is expected to verify the result
    with :func:`polygon_fidelity` and fall back to the plain exterior if the cut
    did not rasterize as intended.
    """
    ring: Ring = list(part.exterior)
    if not part.holes:
        return ring

    outer_ccw = _ring_signed_area(ring) > 0
    for hole in sorted(part.holes, key=lambda h: abs(_ring_signed_area(h)), reverse=True):
        if len(hole) < 3:
            continue
        # A hole must wind opposite to the ring it is cut into, so the slit closes
        # on itself and even-odd filling leaves the hole empty.
        hole_ccw = _ring_signed_area(hole) > 0
        oriented = list(hole) if hole_ccw != outer_ccw else list(reversed(hole))

        ring_pts = np.asarray(ring, dtype=np.float64)
        hole_pts = np.asarray(oriented, dtype=np.float64)
        distances = ((ring_pts[:, None, :] - hole_pts[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(np.argmin(distances)), distances.shape)

        rotated = oriented[j:] + oriented[:j]
        # ...ring[i] -> around the hole -> back to hole[j] -> ring[i] -> rest.
        ring = ring[: i + 1] + rotated + [rotated[0]] + ring[i:]
    return ring


def _ring_window(rings: list[Ring], pad: int = 2) -> tuple[int, int, int, int]:
    pts = np.concatenate([np.asarray(r, dtype=np.float64) for r in rings if r])
    x0 = int(np.floor(pts[:, 0].min())) - pad
    y0 = int(np.floor(pts[:, 1].min())) - pad
    x1 = int(np.ceil(pts[:, 0].max())) + pad + 1
    y1 = int(np.ceil(pts[:, 1].max())) + pad + 1
    return x0, y0, x1, y1


def _part_export_ring(mask: np.ndarray, part: MaskPolygon) -> tuple[Ring, bool]:
    """One ring for a part, bridging its holes, verified against the source mask.

    Both candidates — the keyhole-bridged ring and the plain exterior — are scored
    against the mask itself rather than against each other, because punch-filling
    and even-odd filling disagree on hole boundary pixels by a percent or two. The
    better reproduction wins; falling back to the exterior reports a dropped hole
    so the loss is recorded instead of written silently.
    """
    if not part.holes:
        return part.exterior, True

    bridged = bridge_holes(part)
    if len(bridged) < 3:
        return part.exterior, False

    mh, mw = mask.shape
    x0, y0, x1, y1 = _ring_window([part.exterior, *part.holes, bridged])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(mw, max(x0 + 1, x1)), min(mh, max(y0 + 1, y1))
    reference = mask.astype(bool)[y0:y1, x0:x1]
    offset = (x0, y0)
    w, h = x1 - x0, y1 - y0

    bridged_iou = mask_iou(reference, polygons_to_mask([MaskPolygon(bridged)], w, h, offset))
    exterior_iou = mask_iou(reference, polygons_to_mask([MaskPolygon(part.exterior)], w, h, offset))
    if bridged_iou >= exterior_iou:
        return bridged, True
    return part.exterior, False


def parts_to_export_rings(mask: np.ndarray, parts: list[MaskPolygon]) -> tuple[list[Ring], float, int]:
    """Flatten parts to single rings for hole-less formats such as YOLO-seg.

    Returns the rings, the fidelity of *what will actually be written* measured
    against the source mask, and the number of holes that could not be bridged.
    """
    rings: list[Ring] = []
    dropped_holes = 0
    for part in parts:
        ring, ok = _part_export_ring(mask, part)
        if not ok:
            dropped_holes += len(part.holes)
        if len(ring) >= 3:
            rings.append(ring)
    fidelity = polygon_fidelity(mask, [MaskPolygon(r) for r in rings])
    return rings, fidelity, dropped_holes


def mask_to_polygon(mask: np.ndarray, simplify_px: float = 0.75) -> list[tuple[float, float]]:
    """Largest component's outer ring only.

    Lossy by construction; kept for diagnostics and previews. Production conversion
    uses :func:`mask_to_polygons`.
    """
    parts = mask_to_polygons(mask, keep_holes=False)
    if not parts:
        return []
    exterior = parts[0].exterior
    if simplify_px > 0:
        simplified = cv2.approxPolyDP(_as_i32(exterior), simplify_px, True).reshape(-1, 2)
        if len(simplified) >= 3:
            return [(float(x), float(y)) for x, y in simplified]
    return exterior if len(exterior) >= 3 else []
