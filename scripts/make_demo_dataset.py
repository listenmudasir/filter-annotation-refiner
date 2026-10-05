from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
import yaml


root = Path(__file__).resolve().parents[1] / "demo_dataset"
img_dir = root / "images" / "train"
lab_dir = root / "labels" / "train"
img_dir.mkdir(parents=True, exist_ok=True)
lab_dir.mkdir(parents=True, exist_ok=True)

samples = [
    ("demo_001", [(0, (120, 80, 250, 180)), (1, (330, 60, 350, 230))]),
    ("demo_002", [(2, (180, 120, 250, 190)), (3, (360, 90, 455, 175))]),
]

for name, objects in samples:
    image = Image.new("RGB", (560, 320), (210, 207, 196))
    draw = ImageDraw.Draw(image)
    # Pleat-like background.
    for x in range(0, 560, 28):
        draw.line((x, 0, x + 12, 320), fill=(177, 174, 166), width=8)
    lines = []
    for class_id, (x0, y0, x1, y1) in objects:
        if class_id == 0:
            draw.ellipse((x0, y0, x1, y1), fill=(150, 115, 95))
        elif class_id == 1:
            draw.line((x0, y0, x1, y1), fill=(50, 45, 40), width=3)
        elif class_id == 2:
            draw.ellipse((x0, y0, x1, y1), fill=(55, 45, 35))
        else:
            draw.polygon([(x0, y0 + 15), (x1 - 10, y0), (x1, y1 - 15), (x0 + 12, y1)], fill=(175, 150, 65))
        xc = ((x0 + x1) / 2) / 560
        yc = ((y0 + y1) / 2) / 320
        bw = (x1 - x0) / 560
        bh = (y1 - y0) / 320
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    image.save(img_dir / f"{name}.jpg", quality=92)
    (lab_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

(root / "data.yaml").write_text(
    yaml.safe_dump({"path": ".", "train": "images/train", "names": ["Stain", "Hair", "Bug", "Foreign_Body"]}, sort_keys=False),
    encoding="utf-8",
)
print(root)
