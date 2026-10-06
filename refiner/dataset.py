from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image

from .models import DatasetIndex, ImageRecord
from .yolo import load_yolo_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


#: Plain-text class listings, one name per line, as written by labelImg and many
#: CVAT/Darknet exports.
NAME_FILES = ("classes.txt", "obj.names", "predefined_classes.txt")


def load_class_names(root: Path) -> dict[int, str]:
    candidates = [root / "data.yaml", root / "dataset.yaml", *sorted(root.rglob("*.yaml")), *sorted(root.rglob("*.yml"))]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            names = data.get("names", {})
            if isinstance(names, list) and names:
                return {i: str(name) for i, name in enumerate(names)}
            if isinstance(names, dict) and names:
                return {int(k): str(v) for k, v in names.items()}
        except Exception:
            pass

    # Many real YOLO exports ship no YAML at all, only a classes.txt beside the
    # labels. Without this the UI shows class_0/class_1 and the class-aware
    # post-processing profiles silently never match.
    for name in NAME_FILES:
        for path in sorted(root.rglob(name)):
            try:
                lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
            except Exception:
                continue
            lines = [line for line in lines if line]
            if lines:
                return {i: line for i, line in enumerate(lines)}
    return {}


def iter_images(root: Path) -> Iterable[Path]:
    excluded = {"masks", "review_previews", "reports", ".sam_refiner", "labels"}
    image_dirs = []
    for p in root.rglob("images"):
        if not p.is_dir():
            continue
        try:
            ancestors = set(p.relative_to(root).parts[:-1])
        except ValueError:
            ancestors = set()
        if ancestors & excluded:
            continue
        image_dirs.append(p)
    seen: set[Path] = set()
    if image_dirs:
        for image_dir in sorted(image_dirs):
            for path in sorted(image_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path not in seen:
                    seen.add(path)
                    yield path
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def candidate_label_paths(root: Path, image_path: Path) -> list[Path]:
    """Every place a YOLO label for ``image_path`` could plausibly live.

    Real datasets arrive in several shapes and a converter that only understands
    one of them is useless:

    * ``images/…`` and ``labels/…`` as sibling trees (Ultralytics convention)
    * the ``.txt`` sitting next to the image in the same folder (labelImg default)
    * a single top-level ``labels/`` mirroring the image tree

    Ordered most- to least-specific; the first that exists wins.
    """
    candidates: list[Path] = []
    parts = list(image_path.parts)
    image_indices = [i for i, p in enumerate(parts) if p == "images"]
    if image_indices:
        replaced = parts.copy()
        replaced[image_indices[-1]] = "labels"
        candidates.append(Path(*replaced).with_suffix(".txt"))

    candidates.append(image_path.with_suffix(".txt"))

    try:
        rel = image_path.relative_to(root)
        candidates.append((root / "labels" / rel).with_suffix(".txt"))
    except ValueError:
        pass

    ordered: list[Path] = []
    for path in candidates:
        if path not in ordered:
            ordered.append(path)
    return ordered


def infer_label_path(root: Path, image_path: Path) -> Path:
    candidates = candidate_label_paths(root, image_path)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def generated_output_marker(root: Path) -> Path | None:
    """Path proving ``root`` is (or sits inside) a dataset this tool generated.

    Converting an output back into a new dataset silently compounds error: run 2
    segments run 1's masks rather than the originals, and the trees nest
    (out/defect_sam_refined/comparisons/labels/...). Nothing else detects it,
    because a generated dataset is a structurally valid YOLO dataset.
    """
    root = Path(root).expanduser()
    for folder in (root, *root.parents):
        marker = folder / ".sam_refiner" / "state.json"
        if marker.exists():
            return marker
        # A tree copied without its state dir still carries these.
        summary = folder / "reports" / "summary.json"
        if summary.exists() and (folder / "overlays").is_dir():
            return summary
        if folder == folder.parent:
            break
    return None


def detect_layout(root: Path, records: list[ImageRecord]) -> str:
    """Human-readable description of how labels were found, for the UI."""
    if not records:
        return "unknown"
    same_dir = sum(
        1 for r in records
        if r.label_path is not None and r.label_path.parent == r.image_path.parent
    )
    split = sum(
        1 for r in records
        if r.label_path is not None and r.label_path.parent != r.image_path.parent
    )
    if same_dir and not split:
        return "images and labels in the same folder"
    if split and not same_dir:
        return "separate images/ and labels/ folders"
    if split or same_dir:
        return "mixed (some labels beside images, some in labels/)"
    return "no label files found"


def scan_dataset(root: str | Path) -> DatasetIndex:
    root = Path(root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(root)

    class_names = load_class_names(root)
    records: list[ImageRecord] = []
    global_issues: list[str] = []
    images = list(iter_images(root))
    if not images:
        global_issues.append("No supported images were found.")

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            global_issues.append(f"Cannot open {image_path.name}: {exc}")
            continue

        label_path = infer_label_path(root, image_path)
        annotations, issues = load_yolo_file(label_path, width, height)
        records.append(
            ImageRecord(
                image_path=image_path,
                label_path=label_path,
                relative_image=image_path.relative_to(root),
                width=width,
                height=height,
                annotations=annotations,
                issues=issues,
            )
        )

    return DatasetIndex(
        root=root,
        records=records,
        class_names=class_names,
        issues=global_issues,
        layout=detect_layout(root, records),
    )


def class_boxes(dataset: DatasetIndex) -> dict[int, list[tuple[float, float, float, float]]]:
    """Source boxes grouped by class, used to derive post-processing profiles."""
    out: dict[int, list[tuple[float, float, float, float]]] = {}
    for record in dataset.records:
        for ann in record.annotations:
            out.setdefault(ann.class_id, []).append(ann.bbox_xyxy)
    return out


def class_histogram(dataset: DatasetIndex) -> dict[int, int]:
    c = Counter(a.class_id for r in dataset.records for a in r.annotations)
    return dict(sorted(c.items()))
