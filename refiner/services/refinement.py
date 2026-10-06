from __future__ import annotations

import csv
import json
import os
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml
from PIL import Image

from ..backend.base import SegmentationBackend
from ..dataset import class_boxes
from ..geometry import (
    mask_iou,
    mask_to_polygons,
    parts_to_export_rings,
    polygon_to_mask,
    simplify_to_fidelity,
)
from ..models import Annotation, CandidateMask, DatasetIndex, ImageRecord, RefinementResult, ReviewState
from ..overlay import render_comparison, render_overlay
from ..persistence import ProjectState
from ..profiles import ClassProfile, derive_profiles, profile_for
from ..quality import (
    canny_edges,
    consensus_stability,
    edge_alignment,
    fragmentation_penalty,
    leakage_fraction,
)
from ..yolo import polygons_to_yolo_lines


@dataclass(slots=True)
class ConversionSettings:
    preset: str = "balanced"  # fast | balanced | maximum
    accept_threshold: float = 0.78
    review_threshold: float = 0.50
    save_masks: bool = True
    save_review_previews: bool = True
    #: Write an overlay JPG for *every* converted image, so the whole dataset can be
    #: flipped through in any image viewer for manual verification.
    save_overlays: bool = True
    #: Write a side-by-side "input boxes | output masks" image per converted image.
    save_comparisons: bool = True
    overlay_quality: int = 88
    #: Background threads for overlay/comparison/mask writing. These overlap the
    #: next image's GPU work; 0 writes inline.
    io_workers: int = 3
    link_images: bool = True
    #: Written polygons whose round-trip IoU against the SAM mask falls below this
    #: are sent to review regardless of their other scores.
    min_polygon_fidelity: float = 0.90


def describe_output_conflict(output_root: Path, backend_name: str, preset: str) -> str | None:
    """Why ``output_root`` cannot be resumed by this backend/preset, or None if it can.

    Exposed separately from the engine so the UI can check *before* a run starts and
    offer the operator a choice, instead of surfacing a traceback after Start.
    """
    output_root = Path(output_root).expanduser()
    if not output_root.exists():
        return None

    # A folder written before the dataset/ + qa/ split holds its labels at the top
    # level. Resume keys on the *new* label path, which does not exist there, so the
    # run would silently start over and write a second dataset alongside the first -
    # two parallel copies in one folder, neither obviously authoritative.
    legacy_labels = output_root / "labels"
    if legacy_labels.is_dir() and next(legacy_labels.rglob("*.txt"), None) is not None:
        return (
            f"{output_root} uses the older flat layout, where labels sit at the top "
            "level rather than under dataset/.\n"
            "Converting into it again would not resume; it would write a second, "
            "parallel dataset beside the first.\n"
            "Convert into a new output folder, or delete the existing one to start over."
        )

    state = ProjectState(output_root.resolve())
    if not state.has_progress:
        return None

    previous = state.run_signature()
    current = {"backend": backend_name, "preset": preset}
    if previous == current:
        return None

    if previous is None:
        detail = (
            "it was produced by an older version that did not record which backend "
            "was used, so its masks cannot be trusted to match this run"
        )
    else:
        detail = (
            f"it already holds results from backend '{previous.get('backend')}' "
            f"(preset '{previous.get('preset')}') and you are now running "
            f"'{backend_name}' (preset '{preset}')"
        )
    return (
        f"{output_root} cannot be resumed: {detail}.\n"
        "Resuming would skip the already-converted images and leave the earlier "
        "masks in the final dataset.\n"
        "Convert into a new output folder, or delete the existing one to start over."
    )


def suggest_free_output(output_root: Path) -> Path:
    """First unused sibling of ``output_root`` with a numeric suffix."""
    output_root = Path(output_root).expanduser()
    if not output_root.exists():
        return output_root
    for index in range(2, 1000):
        candidate = output_root.with_name(f"{output_root.name}_v{index}")
        if not candidate.exists():
            return candidate
    return output_root.with_name(f"{output_root.name}_new")


class ConversionCancelled(Exception):
    """Raised inside the engine when the operator stops a run.

    Distinct from a failure: a cancelled image is not written to failures.csv and
    is not marked completed, so resuming later reprocesses it from scratch.
    """


@dataclass(slots=True)
class RecordOutcome:
    record: ImageRecord
    results: list[RefinementResult]
    output_label: Path
    skipped: bool = False


class SmartRefinementEngine:
    def __init__(
        self,
        dataset: DatasetIndex,
        backend: SegmentationBackend,
        output_root: Path,
        settings: ConversionSettings,
    ):
        self.dataset = dataset
        self.backend = backend
        self.output_root = output_root.expanduser().resolve()
        self.settings = settings
        # Post-processing is tuned from this dataset's own annotation geometry, so
        # the converter behaves sensibly on data it has never seen.
        self.profiles = derive_profiles(class_boxes(dataset), dataset.class_names)
        self.profile_overrides: dict[int, ClassProfile] = {}
        # Created on first use: constructing the engine can still fail (see
        # _guard_run_provenance), and a pool made before that would leave
        # non-daemon threads behind that block interpreter shutdown.
        self._io_workers = max(0, settings.io_workers)
        self._io_pool: ThreadPoolExecutor | None = None
        # Bound in-flight images so queued masks cannot grow without limit.
        self._io_slots = threading.Semaphore(max(1, self._io_workers * 2))
        self._io_errors: list[str] = []
        self.state = ProjectState(self.output_root)
        self._guard_run_provenance()
        # The deliverable and the QA material are separated. Previously nine
        # sibling entries sat at the top level and the 7 MB of labels that the whole
        # conversion exists to produce were indistinguishable from 900 MB of
        # review imagery. Now: dataset/ is what you train on, qa/ can be deleted.
        self.dataset_dir = self.output_root / "dataset"
        self.qa_dir = self.output_root / "qa"
        self.report_dir = self.qa_dir / "reports"
        self.mask_dir = self.qa_dir / "masks"
        self.preview_dir = self.qa_dir / "review_previews"
        self.overlay_dir = self.qa_dir / "overlays"
        self.comparison_dir = self.qa_dir / "comparisons"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._report_path = self.report_dir / "objects.csv"
        self._review_path = self.report_dir / "review_queue.csv"
        self._failure_path = self.report_dir / "failures.csv"
        self._ensure_report_headers()
        self._prepare_dataset_metadata()

    REPORT_HEADER = [
        "image", "instance", "class_id", "class_name", "source_kind", "prompt",
        "sam_score", "stability", "edge_alignment", "leakage", "fragmentation",
        "fidelity", "parts", "holes", "points",
        "quality", "state", "warnings", "output_label",
    ]

    def run_signature(self) -> dict[str, str]:
        """Identity of this run's producer, used to keep an output folder coherent."""
        return {"backend": self.backend.name, "preset": self.settings.preset}

    def _guard_run_provenance(self) -> None:
        """Refuse to mix results from different backends in one output dataset.

        Resume keys only on image path, so pointing a second run with a different
        backend at a folder that already has results silently *skips* the finished
        images and leaves the first backend's masks in the final dataset. That is
        invisible afterwards: summary.json only records the last backend to run.
        """
        conflict = describe_output_conflict(
            self.output_root, self.backend.name, self.settings.preset
        )
        if conflict is None:
            self.state.set_run_signature(self.run_signature())
            return
        raise RuntimeError(conflict)

    def _ensure_report_headers(self) -> None:
        for path in (self._report_path, self._review_path):
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self.REPORT_HEADER)
                continue
            # Resuming into a report written by an older schema would append rows of a
            # different width and silently corrupt the QA data.
            with path.open("r", newline="", encoding="utf-8") as f:
                existing = next(csv.reader(f), [])
            if existing and existing != self.REPORT_HEADER:
                raise RuntimeError(
                    f"{path} was written by an incompatible version of this tool. "
                    "Convert into a fresh output folder, or delete the reports/ directory."
                )
        if not self._failure_path.exists():
            with self._failure_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["image", "error_type", "error", "traceback"])

    def _prepare_dataset_metadata(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._write_readme()
        # Preserve/normalize a source YAML when available.
        yaml_candidates = [self.dataset.root / "data.yaml", self.dataset.root / "dataset.yaml"]
        yaml_candidates += list(self.dataset.root.glob("*.yaml"))
        source_yaml = next((p for p in yaml_candidates if p.exists()), None)
        if source_yaml:
            try:
                data = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
                # The output reproduces the source's image-tree structure, so normalize
                # absolute split paths back to paths relative to the new dataset root.
                for key in ("train", "val", "test"):
                    value = data.get(key)
                    values = value if isinstance(value, list) else [value]
                    normalized = []
                    for item in values:
                        if not isinstance(item, str):
                            normalized.append(item)
                            continue
                        p = Path(item)
                        if p.is_absolute():
                            try:
                                item = p.resolve().relative_to(self.dataset.root).as_posix()
                            except Exception:
                                pass
                        normalized.append(item)
                    if isinstance(value, list):
                        data[key] = normalized
                    elif normalized:
                        data[key] = normalized[0]
                data["path"] = "."
                data["task"] = "segment"
                data["names"] = self._output_class_names(data.get("names"))
                data["nc"] = len(data["names"])
                data.setdefault("train", self._default_split())
                (self.dataset_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
                return
            except Exception:
                pass

        # No usable source YAML: write one anyway. A converted dataset without a
        # data.yaml cannot be handed to a trainer, which defeats the whole point.
        names = self._output_class_names(None)
        data = {
            "path": ".",
            "train": self._default_split(),
            "task": "segment",
            "nc": len(names),
            "names": names,
        }
        (self.dataset_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def _output_class_names(self, existing) -> dict[int, str]:
        """Names covering every class id present, filling gaps with placeholders.

        Ultralytics needs a contiguous name map; a dataset whose ids are 0,1,3,4,5
        (a gap at 2) would otherwise be rejected or silently mis-mapped.
        """
        names: dict[int, str] = {}
        if isinstance(existing, dict):
            names = {int(k): str(v) for k, v in existing.items()}
        elif isinstance(existing, list):
            names = {i: str(v) for i, v in enumerate(existing)}
        if not names:
            names = dict(self.dataset.class_names)

        observed = {ann.class_id for r in self.dataset.records for ann in r.annotations}
        highest = max([*observed, *names], default=-1)
        return {i: names.get(i, f"class_{i}") for i in range(highest + 1)}

    def _default_split(self) -> str:
        """Relative image directory to point `train:` at."""
        for record in self.dataset.records:
            parts = record.relative_image.parts
            if len(parts) > 1:
                return Path(*parts[:-1]).as_posix()
            break
        return "images"

    README_TEXT = """Filter Annotation Refiner output
================================

dataset/   The segmentation dataset. This is what you train on.
           dataset/data.yaml is ready for Ultralytics:
               yolo segment train data=dataset/data.yaml model=yolo11n-seg.pt

qa/        Quality-assurance material only. Safe to delete once you are happy
           with the conversion; nothing here is needed for training.
           overlays/         every image with its mask drawn
           comparisons/      input detection boxes beside the masks produced
           masks/            per-instance PNG masks
           review_previews/  uncertain images only
           reports/          objects.csv, review_queue.csv, failures.csv,
                             summary.json

Source labels were never modified.
"""

    def _write_readme(self) -> None:
        """Explain the two folders in the folder itself, for whoever opens it later."""
        path = self.output_root / "README.txt"
        if not path.exists():
            path.write_text(self.README_TEXT, encoding="utf-8")

    def _output_image_path(self, record: ImageRecord) -> Path:
        return self.dataset_dir / record.relative_image

    def _output_label_path(self, record: ImageRecord) -> Path:
        parts = list(record.relative_image.parts)
        image_indices = [i for i, p in enumerate(parts) if p == "images"]
        if image_indices:
            parts[image_indices[-1]] = "labels"
            return self.dataset_dir.joinpath(*parts).with_suffix(".txt")
        return self.dataset_dir / "labels" / record.relative_image.with_suffix(".txt")

    def _copy_or_link_image(self, record: ImageRecord) -> None:
        dst = self._output_image_path(record)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return
        if self.settings.link_images:
            try:
                os.link(record.image_path, dst)
                return
            except OSError:
                pass
        shutil.copy2(record.image_path, dst)

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        if min_area <= 1:
            return mask.astype(bool)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        out = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                out[labels == i] = 1
        return out.astype(bool)

    def _postprocess(self, mask: np.ndarray, profile: ClassProfile) -> np.ndarray:
        m = mask.astype(np.uint8)
        if profile.close_kernel > 1 and not profile.preserve_thin:
            k = np.ones((profile.close_kernel, profile.close_kernel), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        if profile.open_kernel > 1 and not profile.preserve_thin:
            k = np.ones((profile.open_kernel, profile.open_kernel), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        return self._remove_small_components(m.astype(bool), profile.min_component_area)

    def _candidate_metrics(
        self,
        image_rgb: np.ndarray,
        bbox: tuple[float, float, float, float],
        source_polygon: list[tuple[float, float]],
        mask: np.ndarray,
        all_masks: list[np.ndarray],
        profile: ClassProfile,
        sam_score: float,
        edges: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Score one candidate. All arguments are in ROI-window coordinates."""
        h, w = mask.shape
        stability = consensus_stability(mask, all_masks)
        edge = edge_alignment(image_rgb, mask, edges)
        leakage = leakage_fraction(mask, bbox, w, h, allowance=0.15)
        frag = fragmentation_penalty(mask, min_area=max(2, profile.min_component_area))
        source_agreement = 1.0
        if source_polygon:
            source_agreement = mask_iou(mask, polygon_to_mask(source_polygon, w, h))
        quality = (
            0.38 * np.clip(sam_score, 0.0, 1.0)
            + 0.27 * stability
            + 0.13 * edge
            + 0.10 * (1.0 - min(1.0, leakage))
            + 0.07 * (1.0 - min(1.0, frag))
            + 0.05 * source_agreement
        )
        if leakage > profile.max_leakage:
            quality -= min(0.12, (leakage - profile.max_leakage) * 0.5)
        return {
            "stability": float(stability),
            "edge_alignment": float(edge),
            "leakage": float(leakage),
            "fragmentation": float(frag),
            "source_agreement": float(source_agreement),
            "quality": float(np.clip(quality, 0.0, 1.0)),
        }

    def _build_consensus_candidate(self, candidates: list[CandidateMask]) -> CandidateMask | None:
        if len(candidates) < 2:
            return None
        best = max(candidates, key=lambda c: c.sam_score)
        cluster = [c for c in candidates if mask_iou(best.mask, c.mask) >= 0.55]
        if len(cluster) < 2:
            return None
        weights = np.asarray([max(0.05, c.sam_score) for c in cluster], dtype=np.float32)
        stack = np.stack([c.mask.astype(np.float32) for c in cluster], axis=0)
        weighted = (stack * weights[:, None, None]).sum(axis=0) / weights.sum()
        mask = weighted >= 0.5
        score = float(np.average([c.sam_score for c in cluster], weights=weights))
        return CandidateMask(mask=mask, sam_score=score, prompt_name=f"consensus({len(cluster)})")

    @staticmethod
    def _roi(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
        """Evaluation window around an object.

        Every mask operation runs inside this window. A 277px defect on a 4096px
        frame otherwise costs 15 megapixels of work per candidate per metric, which
        is what made conversion CPU-bound rather than GPU-bound. The margin is a
        full box-width on each side so a leaking mask is still visible as leaking.
        """
        x0, y0, x1, y1 = bbox
        margin = max(64.0, (x1 - x0), (y1 - y0))
        return (
            int(max(0, np.floor(x0 - margin))),
            int(max(0, np.floor(y0 - margin))),
            int(min(width, np.ceil(x1 + margin))),
            int(min(height, np.ceil(y1 + margin))),
        )

    def refine_annotation(
        self,
        image_rgb: np.ndarray,
        annotation: Annotation,
        class_name: str = "",
        edges: np.ndarray | None = None,
    ) -> RefinementResult:
        raw_candidates = self.backend.candidates(annotation, self.settings.preset)
        if not raw_candidates:
            raise RuntimeError("No segmentation candidates returned")
        profile = profile_for(annotation.class_id, self.profiles, self.profile_overrides)

        height, width = image_rgb.shape[:2]
        rx0, ry0, rx1, ry1 = self._roi(annotation.bbox_xyxy, width, height)
        roi_rgb = np.ascontiguousarray(image_rgb[ry0:ry1, rx0:rx1])
        roi_edges = edges[ry0:ry1, rx0:rx1] if edges is not None else None
        # Shift the source box into window coordinates so leakage stays meaningful.
        bx0, by0, bx1, by1 = annotation.bbox_xyxy
        roi_bbox = (bx0 - rx0, by0 - ry0, bx1 - rx0, by1 - ry0)
        roi_polygon = [(px - rx0, py - ry0) for px, py in annotation.polygon_xy]

        # Crop every candidate once; all later work is on these small arrays.
        for candidate in raw_candidates:
            candidate.mask = np.ascontiguousarray(candidate.mask[ry0:ry1, rx0:rx1])

        all_masks = [c.mask.astype(bool) for c in raw_candidates]
        consensus = self._build_consensus_candidate(raw_candidates)
        candidates = list(raw_candidates)
        if consensus is not None:
            candidates.append(consensus)
            all_masks.append(consensus.mask)

        for candidate in candidates:
            candidate.mask = self._postprocess(candidate.mask, profile)
            candidate.metrics = self._candidate_metrics(
                roi_rgb, roi_bbox, roi_polygon, candidate.mask, all_masks,
                profile, candidate.sam_score, roi_edges,
            )
            candidate.quality = candidate.metrics["quality"]

        selected = max(candidates, key=lambda c: c.quality)

        # Keep every component and every hole, then drop vertices only while the
        # round-trip IoU bound holds. The rings are what actually reach the dataset,
        # so fidelity is measured against them rather than the pre-export geometry.
        parts = mask_to_polygons(
            selected.mask,
            min_area=max(1.0, float(profile.min_component_area)),
            keep_holes=profile.keep_holes,
            max_parts=profile.max_parts,
        )
        if not parts:
            raise RuntimeError("Best SAM mask could not be converted to a valid polygon")
        parts, _ = simplify_to_fidelity(selected.mask, parts, profile.target_fidelity)
        export_rings, fidelity, dropped_holes = parts_to_export_rings(selected.mask, parts)
        if not export_rings:
            raise RuntimeError("Best SAM mask could not be converted to a valid polygon")

        # Map geometry and the final mask back to full-image coordinates.
        for part in parts:
            part.exterior = [(x + rx0, y + ry0) for x, y in part.exterior]
            part.holes = [[(x + rx0, y + ry0) for x, y in hole] for hole in part.holes]
        export_rings = [[(x + rx0, y + ry0) for x, y in ring] for ring in export_rings]
        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[ry0:ry1, rx0:rx1] = selected.mask
        selected.mask = full_mask

        m = selected.metrics
        warnings: list[str] = []
        if m["stability"] < 0.65:
            warnings.append("Candidate masks disagree")
        if m["leakage"] > profile.max_leakage:
            warnings.append("Mask extends unusually far outside the YOLO box")
        if m["fragmentation"] > 0.30:
            warnings.append("Mask is fragmented")
        if m["edge_alignment"] < 0.12:
            warnings.append("Weak image-edge support")
        if dropped_holes:
            warnings.append(f"{dropped_holes} hole(s) could not be encoded as a polygon")

        quality = selected.quality
        state = ReviewState.ACCEPTED if quality >= self.settings.accept_threshold else ReviewState.REVIEW
        if quality < self.settings.review_threshold:
            state = ReviewState.REVIEW
            warnings.append("Low overall quality; manual review recommended")
        if fidelity < self.settings.min_polygon_fidelity:
            state = ReviewState.REVIEW
            warnings.append(f"Polygon only reproduces {fidelity:.1%} of the mask")

        return RefinementResult(
            mask=selected.mask,
            polygons=parts,
            export_rings=export_rings,
            polygon_fidelity=fidelity,
            sam_score=float(selected.sam_score),
            stability=m["stability"],
            edge_alignment=m["edge_alignment"],
            leakage=m["leakage"],
            fragmentation=m["fragmentation"],
            quality=quality,
            source_kind=annotation.kind,
            prompt_name=selected.prompt_name,
            state=state,
            warnings=warnings,
        )

    def process_record(
        self,
        record: ImageRecord,
        resume: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> RecordOutcome:
        def check_cancelled() -> None:
            if should_stop is not None and should_stop():
                raise ConversionCancelled()

        if record.issues:
            raise RuntimeError("; ".join(record.issues))
        check_cancelled()
        out_label = self._output_label_path(record)
        if resume and self.state.is_completed(record.relative_image) and out_label.exists():
            return RecordOutcome(record, [], out_label, skipped=True)

        self._copy_or_link_image(record)
        out_label.parent.mkdir(parents=True, exist_ok=True)
        if not record.annotations:
            out_label.write_text("", encoding="utf-8")
            self.state.mark_completed(record.relative_image, {"objects": 0, "output": str(out_label)})
            return RecordOutcome(record, [], out_label)

        with Image.open(record.image_path) as img:
            pil = img.convert("RGB")
            image_rgb = np.asarray(pil)
            check_cancelled()
            self.backend.set_image(pil)
            # One Canny pass per image, shared by every candidate of every object.
            edges = canny_edges(image_rgb)
            results: list[RefinementResult] = []
            lines: list[str] = []
            for ann in record.annotations:
                check_cancelled()
                class_name = self.dataset.class_names.get(ann.class_id, f"class_{ann.class_id}")
                result = self.refine_annotation(image_rgb, ann, class_name, edges)
                lines.extend(
                    polygons_to_yolo_lines(ann.class_id, result.export_rings, record.width, record.height)
                )
                results.append(result)
            self.backend.close_image()

        tmp = out_label.with_suffix(".txt.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(out_label)
        if self.settings.save_masks:
            self._submit_io(self._save_masks, record, results)
        self._submit_io(self._save_visual_review, record, results)
        self._append_reports(record, results, out_label)
        self.state.mark_completed(
            record.relative_image,
            {
                "objects": len(results),
                "mean_quality": round(sum(r.quality for r in results) / len(results), 4),
                "review": sum(r.state == ReviewState.REVIEW for r in results),
                "output": str(out_label.relative_to(self.output_root)),
            },
        )
        return RecordOutcome(record, results, out_label)

    def _save_masks(self, record: ImageRecord, results: list[RefinementResult]) -> None:
        stem = record.relative_image.with_suffix("")
        target = self.mask_dir / stem
        target.mkdir(parents=True, exist_ok=True)
        for i, result in enumerate(results):
            cv2.imwrite(str(target / f"instance_{i:03d}.png"), result.mask.astype(np.uint8) * 255)

    def render_overlay_image(self, record: ImageRecord, results: list[RefinementResult]) -> Image.Image:
        with Image.open(record.image_path) as img:
            return render_overlay(img, record.annotations, results, self.dataset.class_names)

    def overlay_path(self, record: ImageRecord) -> Path:
        return self.overlay_dir / record.relative_image.with_suffix(".jpg")

    def _write_overlay(self, path: Path, canvas: Image.Image) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jpg.tmp")
        # The format must be stated: PIL infers it from the suffix, and the atomic
        # temp name ends in .tmp, which it does not recognise.
        canvas.save(tmp, format="JPEG", quality=self.settings.overlay_quality)
        tmp.replace(path)

    def _submit_io(self, fn, *args) -> None:
        """Queue background work, blocking if the writers fall behind.

        Overlay rendering and JPEG encoding are pure CPU and release the GIL, so
        running them off the inference thread lets them overlap the *next* image's
        GPU work instead of serialising behind it. The semaphore bounds how many
        images' masks can be held in memory at once.
        """
        if not self._io_workers:
            fn(*args)
            return
        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(
                max_workers=self._io_workers, thread_name_prefix="far-io"
            )
        self._io_slots.acquire()

        def run() -> None:
            try:
                fn(*args)
            except Exception as exc:  # a review artifact must not kill a run
                self._io_errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                self._io_slots.release()

        self._io_pool.submit(run)

    def wait_for_writes(self) -> list[str]:
        """Drain queued artifact writes. Must run before the summary is written."""
        if self._io_pool is None:
            return []
        self._io_pool.shutdown(wait=True)
        self._io_pool = None
        return list(self._io_errors)

    def _save_visual_review(self, record: ImageRecord, results: list[RefinementResult]) -> None:
        """Write the overlays an operator uses to eyeball the conversion.

        ``overlays/`` mirrors the dataset tree for every image; ``review_previews/``
        holds only the uncertain ones. Both come from the same renderer as the live
        preview, so what was on screen is what lands on disk.
        """
        needs_review = any(r.state == ReviewState.REVIEW for r in results)
        want_overlay = self.settings.save_overlays
        want_preview = self.settings.save_review_previews and needs_review
        want_comparison = self.settings.save_comparisons
        if not (want_overlay or want_preview or want_comparison):
            return

        if want_overlay or want_preview:
            canvas = self.render_overlay_image(record, results)
            if want_overlay:
                self._write_overlay(self.overlay_path(record), canvas)
            if want_preview:
                self._write_overlay(self.preview_dir / record.relative_image.with_suffix(".jpg"), canvas)

        if want_comparison:
            with Image.open(record.image_path) as img:
                comparison = render_comparison(
                    img, record.annotations, results, self.dataset.class_names
                )
            self._write_overlay(
                self.comparison_dir / record.relative_image.with_suffix(".jpg"), comparison
            )

    def _row(self, record: ImageRecord, ann: Annotation, result: RefinementResult, out_label: Path) -> list[str]:
        class_name = self.dataset.class_names.get(ann.class_id, f"class_{ann.class_id}")
        return [
            record.relative_image.as_posix(), str(ann.instance_index), str(ann.class_id), class_name,
            ann.kind.value, result.prompt_name, f"{result.sam_score:.4f}", f"{result.stability:.4f}",
            f"{result.edge_alignment:.4f}", f"{result.leakage:.4f}", f"{result.fragmentation:.4f}",
            f"{result.polygon_fidelity:.4f}", str(result.part_count), str(result.hole_count),
            str(result.point_count),
            f"{result.quality:.4f}", result.state.value, " | ".join(result.warnings),
            out_label.relative_to(self.output_root).as_posix(),
        ]

    def _append_reports(self, record: ImageRecord, results: list[RefinementResult], out_label: Path) -> None:
        with self._report_path.open("a", newline="", encoding="utf-8") as f_all, self._review_path.open("a", newline="", encoding="utf-8") as f_review:
            wa, wr = csv.writer(f_all), csv.writer(f_review)
            for ann, result in zip(record.annotations, results):
                row = self._row(record, ann, result, out_label)
                wa.writerow(row)
                if result.state == ReviewState.REVIEW:
                    wr.writerow(row)


    def record_failure(self, record: ImageRecord, error: str, error_type: str = "RuntimeError", traceback_text: str = "") -> None:
        with self._failure_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([record.relative_image.as_posix(), error_type, error, traceback_text])

    def write_summary(self) -> Path:
        rows: list[dict[str, str]] = []
        if self._report_path.exists():
            with self._report_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        accepted = sum(r.get("state") == ReviewState.ACCEPTED.value for r in rows)
        review = sum(r.get("state") == ReviewState.REVIEW.value for r in rows)
        qualities = [float(r["quality"]) for r in rows if r.get("quality")]
        fidelities = [float(r["fidelity"]) for r in rows if r.get("fidelity")]
        failure_count = 0
        if self._failure_path.exists():
            with self._failure_path.open("r", newline="", encoding="utf-8") as f:
                failure_count = max(0, sum(1 for _ in f) - 1)
        summary = {
            "images": len(self.dataset.records),
            "objects": self.dataset.object_count,
            "processed_object_rows": len(rows),
            "accepted": accepted,
            "review": review,
            "failed_images": failure_count,
            "mean_quality": round(float(np.mean(qualities)), 4) if qualities else None,
            "mean_polygon_fidelity": round(float(np.mean(fidelities)), 4) if fidelities else None,
            "min_polygon_fidelity": round(float(np.min(fidelities)), 4) if fidelities else None,
            "below_fidelity_threshold": sum(f < self.settings.min_polygon_fidelity for f in fidelities),
            "backend": self.backend.name,
            "preset": self.settings.preset,
            "output_root": str(self.output_root),
            "data_yaml": str(self.dataset_dir / "data.yaml"),
        }
        # Report the prompts actually present, so a folder containing results from
        # more than one backend cannot masquerade as a clean single-backend run.
        prompts = Counter(r.get("prompt", "").split(" / ")[0] for r in rows if r.get("prompt"))
        if prompts:
            summary["prompts_seen"] = dict(prompts.most_common())
            if any(p.startswith("fallback") for p in prompts):
                summary["warning"] = (
                    "This dataset contains placeholder masks from the 'fallback' backend. "
                    "Those objects are not real segmentation and must be reconverted."
                )
        path = self.report_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path
