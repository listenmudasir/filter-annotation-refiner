from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refiner.dataset import scan_dataset


p = argparse.ArgumentParser()
p.add_argument("dataset", type=Path)
args = p.parse_args()
ds = scan_dataset(args.dataset)
problems = [(r.relative_image, r.issues) for r in ds.records if r.issues]
print(f"Images: {len(ds.records)}")
print(f"Objects: {ds.object_count}")
print(f"Boxes: {ds.detection_count}")
print(f"Polygons: {ds.polygon_count}")
print(f"Problems: {len(problems) + len(ds.issues)}")
for item in ds.issues:
    print("DATASET:", item)
for image, issues in problems:
    for issue in issues:
        print(f"{image}: {issue}")
raise SystemExit(1 if problems or ds.issues else 0)
