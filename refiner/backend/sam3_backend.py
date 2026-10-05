from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..geometry import perturb_box
from ..models import Annotation, CandidateMask
from .base import SegmentationBackend


PROMPTS = {
    "fast": [("original box", 0.00, 0.00, 0.00)],
    "balanced": [
        ("original box", 0.00, 0.00, 0.00),
        ("expand 3%", 0.03, 0.00, 0.00),
        ("contract 2%", -0.02, 0.00, 0.00),
        ("expand 6%", 0.06, 0.00, 0.00),
    ],
    "maximum": [
        ("original box", 0.00, 0.00, 0.00),
        ("expand 3%", 0.03, 0.00, 0.00),
        ("contract 2%", -0.02, 0.00, 0.00),
        ("expand 6%", 0.06, 0.00, 0.00),
        ("shift left", 0.05, -0.02, 0.00),
        ("shift right", 0.05, 0.02, 0.00),
        ("shift up", 0.05, 0.00, -0.02),
        ("shift down", 0.05, 0.00, 0.02),
    ],
}


class Sam3Backend(SegmentationBackend):
    """SAM3 interactive-image adapter used for YOLO-box -> instance-mask conversion.

    The adapter deliberately uses SAM3's instance-interactive path rather than the
    open-vocabulary grounding path. YOLO already tells us which instance to segment;
    the bounding box is therefore the strongest and most deterministic prompt.
    """

    name = "sam3"

    #: Checkpoint filenames published for each release, newest first.
    KNOWN_CHECKPOINTS = ("sam3.1_multiplex.pt", "sam3.pt")

    def __init__(self, device: str | None = None, checkpoint_path: str | None = None):
        self.device = device or os.getenv("SAM3_DEVICE")
        self.checkpoint_path = checkpoint_path or os.getenv("SAM3_CHECKPOINT")
        self.model = None
        self.processor = None
        self.state = None
        self.image_size: tuple[int, int] | None = None
        self._torch = None
        self._release = "sam3"

    @property
    def device_label(self) -> str:
        return f"{self.device or 'auto'} · {self._release}"

    @staticmethod
    def _find_local_checkpoint() -> str | None:
        """Look for an already-downloaded SAM 3 / 3.1 checkpoint.

        ``build_sam3_image_model`` only ever auto-downloads the original SAM 3, so
        SAM 3.1 has to be located and passed explicitly.
        """
        roots = [
            Path(p).expanduser()
            for p in (os.getenv("SAM3_DIR", ""), "~/.cache/huggingface/hub", "~/Downloads/sam3", "~/sam3")
            if p
        ]
        for name in Sam3Backend.KNOWN_CHECKPOINTS:  # prefer 3.1
            for root in roots:
                if not root.exists():
                    continue
                for path in root.rglob(name):
                    return str(path)
        return None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as exc:
            raise RuntimeError(
                "SAM3 could not be imported. Run scripts/check_env.py and install all missing runtime dependencies."
            ) from exc

        self._torch = torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for SAM3 but torch.cuda.is_available() is False")

        checkpoint = self.checkpoint_path or self._find_local_checkpoint()
        kwargs = dict(device=self.device, enable_inst_interactivity=True, eval_mode=True)
        if checkpoint:
            kwargs.update(checkpoint_path=checkpoint, load_from_HF=False)
            self._release = "sam3.1" if "3.1" in Path(checkpoint).name else "sam3"
        try:
            self.model = build_sam3_image_model(**kwargs)
        except Exception as exc:
            if checkpoint:
                raise RuntimeError(
                    f"SAM 3 failed to load {checkpoint}. SAM 3.1 checkpoints need the matching "
                    f"model code: update your sam3 checkout (git pull) and reinstall it.\n{exc}"
                ) from exc
            raise RuntimeError(
                "SAM 3 could not obtain a checkpoint. Both facebook/sam3 and facebook/sam3.1 are "
                "gated on Hugging Face, so you must:\n"
                "  1. Request access at https://huggingface.co/facebook/sam3.1 (or .../sam3)\n"
                "  2. Authenticate: huggingface-cli login\n"
                "  3. Or download the file yourself and pass --checkpoint /path/to/sam3.1_multiplex.pt\n"
                "Note that the bundled downloader only ever fetches the original SAM 3, so SAM 3.1 "
                "must be supplied explicitly.\n"
                f"Underlying error: {exc}"
            ) from exc
        if getattr(self.model, "inst_interactive_predictor", None) is None:
            raise RuntimeError("SAM3 model was built without instance interactivity; box-to-mask conversion is unavailable")
        self.processor = Sam3Processor(self.model, device=self.device)

    def _autocast(self):
        if self._torch is not None and str(self.device).startswith("cuda"):
            # Current SAM3 ViT image paths are safest under BF16 autocast on Ampere+.
            return self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)
        return contextlib.nullcontext()

    def set_image(self, image: Image.Image) -> None:
        self.load()
        image = image.convert("RGB")
        self.image_size = image.size
        try:
            with self._torch.inference_mode(), self._autocast():
                self.state = self.processor.set_image(image)
        except Exception as exc:
            raise RuntimeError(f"SAM3 image encoding failed for {image.size[0]}x{image.size[1]} image: {exc}") from exc

    @staticmethod
    def _flatten_predictions(masks, scores) -> list[tuple[np.ndarray, float]]:
        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)
        while masks.ndim > 3 and masks.shape[0] == 1:
            masks = masks[0]
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim != 3:
            raise RuntimeError(f"Unexpected SAM3 mask tensor shape: {masks.shape}")
        out: list[tuple[np.ndarray, float]] = []
        for i in range(len(masks)):
            mask = np.asarray(masks[i])
            # Some predictor revisions return logits, others return binary masks.
            if np.issubdtype(mask.dtype, np.floating):
                mask = mask > 0.0
            else:
                mask = mask.astype(bool)
            score = float(scores[i]) if i < len(scores) and np.isfinite(scores[i]) else 0.0
            out.append((mask, score))
        return out

    @staticmethod
    def _validate_box(box, width: int, height: int) -> tuple[float, float, float, float]:
        arr = np.asarray(box, dtype=np.float32).reshape(-1)
        if arr.size != 4 or not np.isfinite(arr).all():
            raise RuntimeError(f"Invalid box prompt: {box}")
        x0, y0, x1, y1 = map(float, arr)
        x0 = float(np.clip(x0, 0, max(0, width - 1)))
        y0 = float(np.clip(y0, 0, max(0, height - 1)))
        x1 = float(np.clip(x1, 0, max(0, width - 1)))
        y1 = float(np.clip(y1, 0, max(0, height - 1)))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            raise RuntimeError(f"Degenerate box prompt after clipping: {(x0, y0, x1, y1)}")
        return x0, y0, x1, y1

    def _run_predict(self, box, point_coords=None, point_labels=None, multimask_output=True):
        # SAM's interactive predictor expects pixel-space XYXY/XY coordinates when
        # normalize_coords=True; it performs the normalization internally.
        kwargs = dict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=np.asarray(box, dtype=np.float32),
            multimask_output=multimask_output,
            normalize_coords=True,
        )
        with self._torch.inference_mode(), self._autocast():
            return self.model.predict_inst(self.state, **kwargs)

    def _predict(self, box, prompt_name: str, point_coords=None, point_labels=None) -> list[CandidateMask]:
        if self.image_size is None:
            raise RuntimeError("SAM3 image size is unavailable")
        width, height = self.image_size
        box = self._validate_box(box, width, height)

        if point_coords is not None:
            point_coords = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
            point_labels = np.asarray(point_labels, dtype=np.int32).reshape(-1)
            if len(point_coords) != len(point_labels):
                raise RuntimeError("Point prompt coordinates and labels have different lengths")

        errors: list[str] = []
        prediction = None
        # First request diverse hypotheses. If a particular public SAM3 revision has a
        # multimask-path issue, retry the deterministic single-mask path automatically.
        for multi in (True, False):
            try:
                prediction = self._run_predict(box, point_coords, point_labels, multimask_output=multi)
                masks, scores, _ = prediction
                flat = self._flatten_predictions(masks, scores)
                if any(mask.any() for mask, _ in flat):
                    break
                errors.append(f"multimask={multi}: SAM returned only empty masks")
                prediction = None
            except Exception as exc:
                errors.append(f"multimask={multi}: {type(exc).__name__}: {exc}")
                prediction = None

        if prediction is None:
            raise RuntimeError(f"SAM3 prediction failed for '{prompt_name}': " + " | ".join(errors))

        masks, scores, _ = prediction
        out: list[CandidateMask] = []
        for j, (mask, score) in enumerate(self._flatten_predictions(masks, scores)):
            if mask.shape != (height, width):
                import cv2
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not mask.any():
                continue
            out.append(CandidateMask(mask, score, f"{prompt_name} / mask {j + 1}"))
        if not out:
            raise RuntimeError(f"SAM3 produced no non-empty mask for '{prompt_name}'")
        return out

    @staticmethod
    def _interior_positive(mask: np.ndarray) -> tuple[float, float] | None:
        import cv2
        binary = mask.astype(np.uint8)
        if binary.sum() == 0:
            return None
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        if max_val <= 0:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return None
            return float(xs[len(xs) // 2]), float(ys[len(ys) // 2])
        return float(max_loc[0]), float(max_loc[1])

    @staticmethod
    def _safe_negative_points(mask: np.ndarray, box) -> list[tuple[float, float]]:
        x0, y0, x1, y1 = box
        dx, dy = (x1 - x0) * 0.12, (y1 - y0) * 0.12
        pts = [(x0 + dx, y0 + dy), (x1 - dx, y0 + dy), (x0 + dx, y1 - dy), (x1 - dx, y1 - dy)]
        h, w = mask.shape
        safe = []
        for x, y in pts:
            ix, iy = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            if not mask[iy, ix]:
                safe.append((float(ix), float(iy)))
        return safe[:2]

    def candidates(self, annotation: Annotation, preset: str) -> list[CandidateMask]:
        if self.state is None or self.image_size is None:
            raise RuntimeError("No image is loaded into SAM3")
        width, height = self.image_size
        specs = PROMPTS.get(preset, PROMPTS["balanced"])
        candidates: list[CandidateMask] = []
        prompt_errors: list[str] = []

        for prompt_name, scale, dx, dy in specs:
            box = perturb_box(annotation.bbox_xyxy, width, height, scale, dx, dy)
            try:
                candidates.extend(self._predict(box, prompt_name))
            except Exception as exc:
                # A perturbed prompt may fail while the original is valid. Preserve useful
                # candidates and report all failures only if every prompt fails.
                prompt_errors.append(str(exc))

        if preset in {"balanced", "maximum"} and candidates:
            seed = max(candidates, key=lambda c: c.sam_score)
            positive = self._interior_positive(seed.mask)
            if positive is not None:
                base_box = perturb_box(annotation.bbox_xyxy, width, height, 0.03, 0.0, 0.0)
                try:
                    candidates.extend(self._predict(base_box, "box + inferred interior point", [positive], [1]))
                except Exception as exc:
                    prompt_errors.append(str(exc))
                if preset == "maximum":
                    negatives = self._safe_negative_points(seed.mask, base_box)
                    if negatives:
                        points = [positive, *negatives]
                        labels = [1] + [0] * len(negatives)
                        try:
                            candidates.extend(self._predict(base_box, "box + interior + exterior points", points, labels))
                        except Exception as exc:
                            prompt_errors.append(str(exc))

        if not candidates:
            detail = " | ".join(prompt_errors[:4]) or "SAM returned no candidate masks"
            raise RuntimeError(f"All SAM3 prompts failed. {detail}")
        return candidates

    def close_image(self) -> None:
        self.state = None
        if self._torch is not None and str(self.device).startswith("cuda"):
            # Do not empty the CUDA allocator every image; just release Python refs.
            pass
