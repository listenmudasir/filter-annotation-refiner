"""Single renderer for mask overlays.

The same drawing is needed by the live preview, the saved per-image overlays and
the review previews. Keeping one implementation means what the operator inspects
on screen is pixel-for-pixel what gets written to disk for manual review.
"""

from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from .models import Annotation, RefinementResult, ReviewState

BOX_COLOR = (82, 154, 255, 230)
ACCEPTED_FILL = (31, 207, 135, 90)
ACCEPTED_LINE = (31, 207, 135, 240)
REVIEW_FILL = (255, 183, 65, 105)
REVIEW_LINE = (255, 183, 65, 240)
TEXT_COLOR = (240, 246, 252, 255)
TEXT_SHADOW = (8, 14, 22, 220)


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text(draw: ImageDraw.ImageDraw, xy, text: str, font) -> None:
    x, y = xy
    # Defect imagery is low contrast; a shadow keeps labels readable on any background.
    draw.text((x + 1, y + 1), text, fill=TEXT_SHADOW, font=font)
    draw.text((x, y), text, fill=TEXT_COLOR, font=font)


def render_overlay(
    image: Image.Image,
    annotations: list[Annotation],
    results: list[RefinementResult] | None = None,
    class_names: dict[int, str] | None = None,
    show_labels: bool = True,
    coord_scale: float = 1.0,
) -> Image.Image:
    """Draw source boxes and refined polygons onto a copy of ``image``.

    ``coord_scale`` multiplies annotation coordinates, so a caller can downscale the
    image first and still draw correctly. Drawing on a small canvas rather than a
    full-resolution one is dramatically cheaper for review-only artifacts.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    results = results or []
    class_names = class_names or {}

    scale = max(1.0, canvas.width / 1200)
    width = max(2, int(round(2 * scale)))

    # Two passes. A single pass painted each box then its own translucent mask
    # over it, so every box was washed out by its own mask and, where objects
    # overlap, by later ones too. Masks go down first; boxes and captions stay crisp.
    for idx, ann in enumerate(annotations):
        if idx >= len(results):
            continue
        result = results[idx]
        accepted = result.state == ReviewState.ACCEPTED
        fill = ACCEPTED_FILL if accepted else REVIEW_FILL
        line = ACCEPTED_LINE if accepted else REVIEW_LINE
        for ring in result.export_rings:
            points = [(px * coord_scale, py * coord_scale) for px, py in ring]
            if len(points) >= 3:
                draw.polygon(points, fill=fill, outline=line)

    for idx, ann in enumerate(annotations):
        x0, y0, x1, y1 = (c * coord_scale for c in ann.bbox_xyxy)
        draw.rectangle((x0, y0, x1, y1), outline=BOX_COLOR, width=width)

        if idx >= len(results) or not show_labels:
            continue
        result = results[idx]
        name = class_names.get(ann.class_id, f"class_{ann.class_id}")
        label = f"{name}  Q{result.quality:.2f}  F{result.polygon_fidelity:.2f}"
        if result.part_count > 1:
            label += f"  x{result.part_count}"

        # Size the caption against the box, not the image: a 200px defect on a 4K
        # frame would otherwise get a 44pt label sprawling across its neighbours.
        box_width = max(1.0, x1 - x0)
        size = int(round(min(13 * scale, max(11.0, box_width / 6))))
        font = _font(max(11, size))
        text_width = draw.textlength(label, font=font)
        # Keep the caption on-canvas when the box sits near the right edge.
        tx = min(float(x0) + 3, max(0.0, canvas.width - text_width - 2))
        ty = max(2.0, y0 - size * 1.25)
        _text(draw, (tx, ty), label, font)

    return canvas


def render_record_overlay(image_path, annotations, results=None, class_names=None) -> Image.Image:
    with Image.open(image_path) as img:
        return render_overlay(img, annotations, results, class_names)


def render_comparison(
    image: Image.Image,
    annotations: list[Annotation],
    results: list[RefinementResult] | None = None,
    class_names: dict[int, str] | None = None,
    gap: int = 12,
    max_width: int = 2600,
) -> Image.Image:
    """Input detection boxes beside the segmentation masks they produced.

    This is the artifact for judging a conversion at a glance: what went in on the
    left, what came out on the right, same scale, same crop.
    """
    # Downscale the source *once*, then draw on the small canvases. Rendering two
    # full-resolution overlays and resizing both cost ~570 ms per 4K image — more
    # than the SAM image encoder itself — for an artifact that is only ever looked at.
    panel_width = max(1, (max_width - gap) // 2)
    source = image.convert("RGB")
    coord_scale = 1.0
    if source.width > panel_width:
        coord_scale = panel_width / source.width
        source = source.resize(
            (panel_width, max(1, round(source.height * coord_scale))), Image.BILINEAR
        )

    boxes_only = render_overlay(source, annotations, None, class_names, coord_scale=coord_scale)
    with_masks = render_overlay(source, annotations, results, class_names, coord_scale=coord_scale)

    width = boxes_only.width * 2 + gap
    header = max(22, int(boxes_only.height * 0.035))
    canvas = Image.new("RGB", (width, boxes_only.height + header), (12, 18, 26))
    canvas.paste(boxes_only, (0, header))
    canvas.paste(with_masks, (boxes_only.width + gap, header))

    draw = ImageDraw.Draw(canvas)
    font = _font(max(12, int(header * 0.62)))
    _text(draw, (6, max(1, (header - font.size if hasattr(font, "size") else header) // 2)),
          "INPUT  ·  detection boxes", font)
    _text(draw, (boxes_only.width + gap + 6, max(1, (header - (font.size if hasattr(font, "size") else header)) // 2)),
          "OUTPUT  ·  segmentation masks", font)
    return canvas
