from __future__ import annotations

import cv2
import numpy as np

from .geometry import box_mask, mask_area, mask_iou


def connected_components(mask: np.ndarray) -> tuple[int, list[int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    return max(0, n - 1), areas


def fragmentation_penalty(mask: np.ndarray, min_area: int = 4) -> float:
    count, areas = connected_components(mask)
    significant = [a for a in areas if a >= min_area]
    if len(significant) <= 1:
        return 0.0
    total = max(1, sum(significant))
    largest = max(significant)
    return float(min(1.0, 1.0 - largest / total + 0.05 * (len(significant) - 1)))


def leakage_fraction(mask: np.ndarray, bbox: tuple[float, float, float, float], width: int, height: int, allowance: float = 0.12) -> float:
    allowed = box_mask(bbox, width, height, expand=allowance)
    area = mask_area(mask)
    if area == 0:
        return 1.0
    # mask & ~allowed in one pass, counted without materialising a sum reduction.
    outside = int(np.count_nonzero(mask & ~allowed))
    return float(outside / area)


def box_coverage(mask: np.ndarray, bbox: tuple[float, float, float, float], width: int, height: int) -> float:
    src = box_mask(bbox, width, height)
    denom = int(src.sum())
    if denom == 0:
        return 0.0
    return float(np.logical_and(mask, src).sum() / denom)


def canny_edges(image_rgb: np.ndarray) -> np.ndarray:
    """Dilated Canny edge map, computed once per region and reused by every candidate.

    Recomputing this inside ``edge_alignment`` cost roughly 570 ms per object on a
    4K frame, which dominated the whole conversion.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    return cv2.dilate((edges > 0).astype(np.uint8), np.ones((5, 5), np.uint8))


def edge_alignment(image_rgb: np.ndarray, mask: np.ndarray, edges: np.ndarray | None = None) -> float:
    """Fraction of the mask boundary that lands on an image edge.

    ``edges`` must cover the same region as ``mask``; pass a cached map to avoid
    re-running Canny for every candidate.
    """
    if not mask.any():
        return 0.0
    if edges is None:
        edges = canny_edges(image_rgb)
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    boundary_count = int(cv2.countNonZero(boundary))
    if boundary_count == 0:
        return 0.0
    hit = int(cv2.countNonZero(cv2.bitwise_and(boundary, edges)))
    return float(hit / boundary_count)


def consensus_stability(mask: np.ndarray, masks: list[np.ndarray]) -> float:
    """Robust local agreement rather than averaging against every multimask alternative.

    SAM multimask deliberately returns diverse hypotheses, so averaging IoU against all
    hypotheses would make a good repeated solution look unstable. We instead measure
    agreement with the nearest cluster of masks.
    """
    if not masks:
        return 0.0
    # One mask against many: count its pixels once rather than inside every compare.
    area = mask_area(mask)
    ious = sorted((mask_iou(mask, m, area_a=area) for m in masks), reverse=True)
    # Ignore self-IoU when present and average the closest third, at least two neighbors.
    if ious and ious[0] > 0.999:
        ious = ious[1:]
    if not ious:
        return 1.0
    k = min(len(ious), max(2, len(ious) // 3))
    return float(np.mean(ious[:k]))
