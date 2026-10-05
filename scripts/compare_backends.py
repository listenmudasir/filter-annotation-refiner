"""Run several backends over the same objects and compare them on your own data.

Model version numbers do not tell you which backend segments *your* defects best.
This runs each backend over an identical sample and reports the quality metrics the
converter already computes, plus agreement between backends and wall-clock cost.

    python scripts/compare_backends.py /path/to/dataset --backends sam2 fallback -n 25

Agreement (mean pairwise IoU of the chosen masks) is the practical stand-in for
"consistency": backends that disagree on an object are flagging a genuinely
ambiguous one, which is exactly what belongs in the review queue.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from refiner.dataset import class_boxes, scan_dataset  # noqa: E402
from refiner.geometry import mask_iou  # noqa: E402
from refiner.profiles import derive_profiles  # noqa: E402
from refiner.quality import canny_edges  # noqa: E402
from refiner.services.refinement import ConversionSettings, SmartRefinementEngine  # noqa: E402
from refiner.ui.workers import build_backend  # noqa: E402


def sample_objects(dataset, limit: int):
    out = []
    for record in dataset.convertible_records:
        for ann in record.annotations:
            out.append((record, ann))
            if len(out) >= limit:
                return out
    return out


def run_backend(name, dataset, samples, args):
    backend = build_backend(name, args.device, args.checkpoint if name == args.checkpoint_for else None)
    backend.load()
    engine = SmartRefinementEngine(
        dataset, backend, Path(args.scratch) / name, ConversionSettings(preset=args.preset)
    )
    results, elapsed = {}, 0.0
    current_image = None
    edges = None
    image_rgb = None
    for record, ann in samples:
        if record.image_path != current_image:
            with Image.open(record.image_path) as img:
                pil = img.convert("RGB")
                image_rgb = np.asarray(pil)
                backend.set_image(pil)
            edges = canny_edges(image_rgb)
            current_image = record.image_path
        key = (record.relative_image.as_posix(), ann.instance_index)
        start = time.perf_counter()
        try:
            results[key] = engine.refine_annotation(image_rgb, ann, "", edges)
        except Exception as exc:  # a backend failing on an object is itself a result
            results[key] = exc
        elapsed += time.perf_counter() - start
    return results, elapsed


def summarise(name, results, elapsed):
    ok = [r for r in results.values() if not isinstance(r, Exception)]
    failed = len(results) - len(ok)
    if not ok:
        return f"{name:10s} all {failed} objects failed"
    return (
        f"{name:10s} quality={statistics.mean(r.quality for r in ok):.4f}  "
        f"fidelity={statistics.mean(r.polygon_fidelity for r in ok):.4f}  "
        f"parts={statistics.mean(r.part_count for r in ok):.2f}  "
        f"pts={statistics.mean(r.point_count for r in ok):.0f}  "
        f"failed={failed}  {elapsed / max(1, len(results)):.3f}s/object"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--backends", nargs="+", default=["sam2", "fallback"])
    ap.add_argument("-n", "--objects", type=int, default=25)
    ap.add_argument("--preset", choices=["fast", "balanced", "maximum"], default="balanced")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None, help="Checkpoint for --checkpoint-for")
    ap.add_argument("--checkpoint-for", default=None, help="Backend the checkpoint belongs to")
    ap.add_argument("--scratch", default="/tmp/backend_compare")
    args = ap.parse_args()

    dataset = scan_dataset(args.dataset)
    samples = sample_objects(dataset, args.objects)
    if not samples:
        raise SystemExit("No convertible objects found in this dataset")
    print(f"dataset: {dataset.root}")
    print(f"layout : {dataset.layout}")
    print(f"classes: {dataset.class_names or '(none found)'}")
    for class_id, profile in sorted(derive_profiles(class_boxes(dataset), dataset.class_names).items()):
        print(f"  class {class_id} {dataset.class_names.get(class_id, '?'):16s} {profile.origin}")
    print(f"sample : {len(samples)} objects\n")

    all_results = {}
    for name in args.backends:
        try:
            results, elapsed = run_backend(name, dataset, samples, args)
        except Exception as exc:
            print(f"{name:10s} unavailable: {type(exc).__name__}: {exc}")
            continue
        all_results[name] = results
        print(summarise(name, results, elapsed))

    names = list(all_results)
    if len(names) > 1:
        print("\nagreement (mean IoU of selected masks, higher = more consistent):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ious = [
                    mask_iou(all_results[a][k].mask, all_results[b][k].mask)
                    for k in all_results[a]
                    if not isinstance(all_results[a][k], Exception)
                    and not isinstance(all_results[b].get(k), Exception)
                ]
                if ious:
                    low = sum(1 for v in ious if v < 0.5)
                    print(f"  {a} vs {b}: mean={statistics.mean(ious):.3f}  "
                          f"median={statistics.median(ious):.3f}  disagree(<0.5)={low}/{len(ious)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
