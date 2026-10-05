"""Per-class mask post-processing settings.

Profiles are *derived from the dataset being converted* rather than looked up from
a table of class names. A fixed table only ever fits the dataset it was written
for; measuring the annotations instead means a hair dataset, a cell dataset and a
pedestrian dataset each get appropriate morphology without any configuration.

Explicit overrides remain available for the cases where a human knows better than
the geometry (see :func:`profile_for`).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ClassProfile:
    #: Minimum round-trip IoU the written polygon must keep. Replaces a hand-tuned
    #: pixel tolerance: vertices are dropped only while this bound holds, so thin
    #: structures keep their detail and blobs simplify hard, with no magic numbers.
    target_fidelity: float
    min_component_area: int
    close_kernel: int
    open_kernel: int
    preserve_thin: bool
    max_leakage: float
    #: Keep interior holes (ring-shaped objects) instead of filling them in.
    keep_holes: bool = True
    #: 0 keeps every component; >0 keeps only the N largest.
    max_parts: int = 0
    #: How this profile was chosen, shown in the UI and written to the run summary.
    origin: str = "default"


DEFAULT_PROFILE = ClassProfile(
    target_fidelity=0.980,
    min_component_area=8,
    close_kernel=3,
    open_kernel=0,
    preserve_thin=False,
    max_leakage=0.20,
)


@dataclass(frozen=True)
class ClassStats:
    """Geometry of one class, measured from the source annotations."""

    class_id: int
    name: str
    count: int
    median_area: float
    median_min_dim: float
    median_thinness: float  # min(w,h) / max(w,h); 1.0 is square, →0 is sliver

    @classmethod
    def from_boxes(cls, class_id: int, name: str, boxes: list[tuple[float, float, float, float]]) -> "ClassStats":
        widths = [max(1.0, b[2] - b[0]) for b in boxes]
        heights = [max(1.0, b[3] - b[1]) for b in boxes]
        areas = [w * h for w, h in zip(widths, heights)]
        min_dims = [min(w, h) for w, h in zip(widths, heights)]
        thinness = [min(w, h) / max(w, h) for w, h in zip(widths, heights)]
        return cls(
            class_id=class_id,
            name=name,
            count=len(boxes),
            median_area=float(statistics.median(areas)) if areas else 0.0,
            median_min_dim=float(statistics.median(min_dims)) if min_dims else 0.0,
            median_thinness=float(statistics.median(thinness)) if thinness else 1.0,
        )


#: A class whose boxes are this elongated is treated as a thin structure (hair,
#: scratch, crack, fibre, wire) where morphology would destroy real geometry.
THIN_ASPECT = 0.30
#: Objects this narrow cannot survive a 3x3 close/open either, whatever their shape.
THIN_MIN_DIM_PX = 14.0


def derive_profile(stats: ClassStats) -> ClassProfile:
    """Choose morphology and simplification from a class's measured geometry.

    The two questions that actually matter for mask cleanup are *how thin* the
    objects are and *how large* they are in pixels. Both are measurable from the
    boxes the user already has, so neither needs to be configured.
    """
    if stats.count == 0:
        return replace(DEFAULT_PROFILE, origin="no samples")

    thin = stats.median_thinness < THIN_ASPECT or stats.median_min_dim < THIN_MIN_DIM_PX

    # Speckle floor scales with the object: 0.1% of a typical box, never above a
    # few hundred px, and never large enough to delete a genuine thin fragment.
    area_floor = int(np.clip(round(stats.median_area * 0.001), 2, 256))
    if thin:
        area_floor = int(np.clip(round(stats.median_area * 0.0005), 2, 32))

    # Morphology kernels scale with the narrow dimension so they smooth noise
    # without closing across a genuinely thin object.
    if thin:
        close_kernel = 0
        open_kernel = 0
    else:
        close_kernel = int(np.clip(round(stats.median_min_dim * 0.04) * 2 + 1, 3, 9))
        open_kernel = 0

    # Thin shapes lose their identity under simplification, so hold a tighter bound.
    target_fidelity = 0.995 if thin else float(np.clip(0.965 + stats.median_thinness * 0.02, 0.965, 0.985))

    # Elongated objects legitimately poke outside a loose box more often.
    max_leakage = 0.28 if thin else 0.20

    descriptor = "thin/elongated" if thin else "compact"
    origin = (
        f"derived ({descriptor}, n={stats.count}, "
        f"thinness={stats.median_thinness:.2f}, min_dim={stats.median_min_dim:.0f}px)"
    )
    return ClassProfile(
        target_fidelity=target_fidelity,
        min_component_area=area_floor,
        close_kernel=close_kernel,
        open_kernel=open_kernel,
        preserve_thin=thin,
        max_leakage=max_leakage,
        keep_holes=True,
        max_parts=0,
        origin=origin,
    )


def derive_profiles(
    class_boxes: dict[int, list[tuple[float, float, float, float]]],
    class_names: dict[int, str],
) -> dict[int, ClassProfile]:
    """Derive a profile for every class present in a dataset."""
    out: dict[int, ClassProfile] = {}
    for class_id, boxes in class_boxes.items():
        name = class_names.get(class_id, f"class_{class_id}")
        out[class_id] = derive_profile(ClassStats.from_boxes(class_id, name, boxes))
    return out


def profile_for(
    class_id: int,
    derived: dict[int, ClassProfile] | None = None,
    overrides: dict[int, ClassProfile] | None = None,
) -> ClassProfile:
    """Resolve the profile for a class: explicit override, else derived, else default."""
    if overrides and class_id in overrides:
        return replace(overrides[class_id], origin="user override")
    if derived and class_id in derived:
        return derived[class_id]
    return DEFAULT_PROFILE
