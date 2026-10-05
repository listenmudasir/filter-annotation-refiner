from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refiner.backend.sam3_backend import Sam3Backend
from refiner.dataset import scan_dataset
from refiner.geometry import mask_to_polygons, parts_to_export_rings, simplify_to_fidelity


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug SAM3 on one YOLO annotation without running the full dataset")
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--image", default=None, help="Relative image path; defaults to first annotated image")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--preset", choices=["fast", "balanced", "maximum"], default="fast")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", type=Path, default=Path("sam3_debug_overlay.png"))
    args = ap.parse_args()

    ds = scan_dataset(args.dataset)
    records = [r for r in ds.records if r.annotations and not r.issues]
    if args.image:
        rec = next((r for r in records if r.relative_image.as_posix() == args.image), None)
        if rec is None:
            raise SystemExit(f"Image not found in dataset index: {args.image}")
    else:
        if not records:
            raise SystemExit("No valid annotated image found")
        rec = records[0]
    if args.instance < 0 or args.instance >= len(rec.annotations):
        raise SystemExit(f"Instance must be 0..{len(rec.annotations)-1}")
    ann = rec.annotations[args.instance]

    print(f"Image: {rec.image_path}")
    print(f"Image size: {rec.width}x{rec.height}")
    print(f"Instance: {args.instance}, class={ann.class_id}, box={ann.bbox_xyxy}")

    backend = Sam3Backend(args.device, args.checkpoint)
    try:
        backend.load()
        with Image.open(rec.image_path) as im:
            pil = im.convert("RGB")
            backend.set_image(pil)
            cands = backend.candidates(ann, args.preset)
            print(f"Candidates: {len(cands)}")
            for i, c in enumerate(cands):
                print(f"  {i:02d}: score={c.sam_score:.4f}, area={int(c.mask.sum())}, prompt={c.prompt_name}")
            best = max(cands, key=lambda c: c.sam_score)
            parts = mask_to_polygons(best.mask)
            parts, _ = simplify_to_fidelity(best.mask, parts, 0.98)
            rings, fidelity, dropped = parts_to_export_rings(best.mask, parts)
            print(
                f"Best mask area={int(best.mask.sum())}; parts={len(parts)}"
                f" holes={sum(len(p.holes) for p in parts)} points={sum(len(r) for r in rings)}"
            )
            print(f"Polygon fidelity (round-trip IoU vs mask)={fidelity:.4f}; unencodable holes={dropped}")
            canvas = pil.copy()
            draw = ImageDraw.Draw(canvas, "RGBA")
            for ring in rings:
                pts = [(int(x), int(y)) for x, y in ring]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=(40, 210, 130, 90), outline=(40, 255, 150, 255))
            x0,y0,x1,y1 = ann.bbox_xyxy
            draw.rectangle((x0,y0,x1,y1), outline=(60,130,255,255), width=2)
            canvas.save(args.out)
            print(f"Saved: {args.out.resolve()}")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 2
    finally:
        backend.close_image()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
