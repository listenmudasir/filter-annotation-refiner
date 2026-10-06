"""SAM 2.1 box-prompted backend.

SAM 2.1 is the pragmatic default for box-to-mask conversion: its checkpoints are
openly downloadable (SAM 3's are gated), its image predictor takes XYXY boxes
directly, and it accepts a *batch* of boxes against a single cached image
embedding — which is what makes dataset-scale conversion fast.

All prompts for an image are issued against one encoder pass. Measured on a 4096px
frame: ~0.5 s to encode, then ~0.1 s for a batch of box prompts, so throughput is
governed by image count rather than object count.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ..geometry import perturb_box
from ..models import Annotation, CandidateMask
from .base import SegmentationBackend

# (name, scale, dx, dy) - identical semantics to the SAM3 backend's prompt table.
PROMPTS = {
    "fast": [("original box", 0.00, 0.00, 0.00)],
    "balanced": [
        ("original box", 0.00, 0.00, 0.00),
        ("expand 4%", 0.04, 0.00, 0.00),
        ("contract 3%", -0.03, 0.00, 0.00),
    ],
    "maximum": [
        ("original box", 0.00, 0.00, 0.00),
        ("expand 4%", 0.04, 0.00, 0.00),
        ("contract 3%", -0.03, 0.00, 0.00),
        ("expand 8%", 0.08, 0.00, 0.00),
        ("shift left", 0.04, -0.02, 0.00),
        ("shift right", 0.04, 0.02, 0.00),
    ],
}

# Checkpoint filename -> hydra config name shipped inside the sam2 package.
CONFIG_BY_CHECKPOINT = {
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2_hiera_large": "configs/sam2/sam2_hiera_l.yaml",
    "sam2_hiera_base_plus": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_small": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_tiny": "configs/sam2/sam2_hiera_t.yaml",
}

#: Where people actually keep checkouts. "桌面" is Desktop on zh_CN systems.
_SEARCH_PARENTS = ("~", "~/Downloads", "~/Desktop", "~/桌面", "~/projects", "~/code", "~/src")

#: Matched against directory names under those parents. Globs rather than exact
#: names so forks and renamed clones (sam2_cell-main, sam2-main, segment-anything-2)
#: are found without the user having to set SAM2_REPO.
_REPO_GLOBS = ("sam2*", "SAM2*", "segment-anything-2*", "segment_anything_2*")


def _candidate_repos() -> list[Path]:
    """Directories that look like a SAM 2 checkout, most-specific first."""
    found: list[Path] = []

    env_repo = os.getenv("SAM2_REPO")
    if env_repo:
        found.append(Path(env_repo).expanduser())

    for parent in _SEARCH_PARENTS:
        base = Path(parent).expanduser()
        if not base.is_dir():
            continue
        for pattern in _REPO_GLOBS:
            try:
                matches = sorted(base.glob(pattern))
            except OSError:
                continue
            for path in matches:
                if path.is_dir() and path not in found:
                    found.append(path)
    return found


def _looks_like_checkout(path: Path) -> bool:
    return (path / "sam2" / "__init__.py").exists()


def _ensure_sam2_importable() -> None:
    """Import sam2, falling back to a local checkout if it is not installed.

    Keeping this tolerant matters for setup: a user who cloned the repo but never
    ran ``pip install -e`` still gets a working backend instead of an ImportError.
    """
    try:
        importlib.import_module("sam2")
        return
    except ImportError:
        pass

    searched = _candidate_repos()
    for repo in searched:
        if not _looks_like_checkout(repo):
            continue
        sys.path.insert(0, str(repo))
        try:
            importlib.import_module("sam2")
            return
        except ImportError:
            sys.path.pop(0)

    looked_in = ", ".join(str(Path(p).expanduser()) for p in _SEARCH_PARENTS)
    raise RuntimeError(
        "SAM 2 is not importable.\n"
        "  • Install it:  pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
        "  • Or point at a local checkout:  export SAM2_REPO=/path/to/sam2\n"
        f"Looked for sam2*/segment-anything-2* under: {looked_in}"
    )


def find_checkpoint(explicit: str | None = None) -> Path:
    """Locate a SAM 2 checkpoint from an explicit path, the environment, or disk."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise RuntimeError(f"SAM 2 checkpoint not found: {path}")
        return path

    env_ckpt = os.getenv("SAM2_CHECKPOINT")
    if env_ckpt:
        path = Path(env_ckpt).expanduser()
        if path.exists():
            return path

    roots = _candidate_repos()
    roots += [Path("~/.cache/sam2").expanduser(), Path("~/checkpoints").expanduser(), Path.cwd()]

    # Prefer the largest/most capable variant when several are present.
    preference = list(CONFIG_BY_CHECKPOINT)
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.pt"):
            stem = path.stem
            if stem in CONFIG_BY_CHECKPOINT and stem not in found:
                found[stem] = path
    for stem in preference:
        if stem in found:
            return found[stem]

    raise RuntimeError(
        "No SAM 2 checkpoint found. Download one (for example sam2.1_hiera_large.pt) "
        "and pass --checkpoint, or set SAM2_CHECKPOINT."
    )


def config_for_checkpoint(checkpoint: Path) -> str:
    config = CONFIG_BY_CHECKPOINT.get(checkpoint.stem)
    if config is None:
        env_config = os.getenv("SAM2_CONFIG")
        if env_config:
            return env_config
        raise RuntimeError(
            f"Unrecognised SAM 2 checkpoint '{checkpoint.name}'. Set SAM2_CONFIG to its "
            f"hydra config, e.g. configs/sam2.1/sam2.1_hiera_l.yaml"
        )
    return config


class Sam2Backend(SegmentationBackend):
    name = "sam2"

    def __init__(self, device: str | None = None, checkpoint_path: str | None = None):
        self.device = device or os.getenv("SAM2_DEVICE")
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self.image_size: tuple[int, int] | None = None
        self._torch = None
        self._checkpoint: Path | None = None

    @property
    def device_label(self) -> str:
        name = str(self.device or "auto")
        if self._checkpoint is not None:
            return f"{name} · {self._checkpoint.stem}"
        return name

    def load(self) -> None:
        if self.predictor is not None:
            return
        _ensure_sam2_importable()
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise RuntimeError(f"SAM 2 could not be imported: {exc}") from exc

        self._torch = torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for SAM 2 but torch.cuda.is_available() is False")

        self._checkpoint = find_checkpoint(self.checkpoint_path)
        config = config_for_checkpoint(self._checkpoint)
        try:
            model = build_sam2(config, str(self._checkpoint), device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build SAM 2 from {self._checkpoint.name} with config {config}: {exc}"
            ) from exc
        self.predictor = SAM2ImagePredictor(model)

    def _autocast(self):
        if self._torch is not None and str(self.device).startswith("cuda"):
            return self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)
        return contextlib.nullcontext()

    def set_image(self, image: Image.Image) -> None:
        self.load()
        rgb = image.convert("RGB")
        self.image_size = rgb.size
        try:
            with self._torch.inference_mode(), self._autocast():
                self.predictor.set_image(np.asarray(rgb))
        except Exception as exc:
            raise RuntimeError(f"SAM 2 failed to encode a {rgb.size[0]}x{rgb.size[1]} image: {exc}") from exc

    def candidates(self, annotation: Annotation, preset: str) -> list[CandidateMask]:
        if self.predictor is None or self.image_size is None:
            raise RuntimeError("No image is loaded into SAM 2")
        width, height = self.image_size
        specs = PROMPTS.get(preset, PROMPTS["balanced"])

        boxes, names = [], []
        for prompt_name, scale, dx, dy in specs:
            x0, y0, x1, y1 = perturb_box(annotation.bbox_xyxy, width, height, scale, dx, dy)
            if x1 - x0 < 1.0 or y1 - y0 < 1.0:
                continue
            boxes.append([x0, y0, x1, y1])
            names.append(prompt_name)
        if not boxes:
            raise RuntimeError(f"Degenerate box for annotation {annotation.instance_index}")

        # One decoder call for every perturbation of this object. The image embedding
        # is already cached, so extra hypotheses are nearly free.
        try:
            with self._torch.inference_mode(), self._autocast():
                masks, scores, _ = self.predictor.predict(
                    box=np.asarray(boxes, dtype=np.float32),
                    multimask_output=False,
                )
        except Exception as exc:
            raise RuntimeError(f"SAM 2 prediction failed: {exc}") from exc

        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)
        # A single box yields (1, H, W); a batch yields (B, 1, H, W).
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])

        out: list[CandidateMask] = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            # predict() returns thresholded float masks unless return_logits is set;
            # guard against a revision handing back probabilities instead of logits.
            if np.issubdtype(mask.dtype, np.floating):
                finite = mask[np.isfinite(mask)]
                threshold = 0.5 if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0 else 0.0
                mask = mask > threshold
            else:
                mask = mask.astype(bool)
            if not mask.any():
                continue
            score = float(scores[i]) if i < len(scores) and np.isfinite(scores[i]) else 0.0
            out.append(CandidateMask(np.ascontiguousarray(mask), score, names[min(i, len(names) - 1)]))

        if not out:
            raise RuntimeError("SAM 2 produced no non-empty mask for this box")
        return out

    def close_image(self) -> None:
        if self.predictor is not None:
            self.predictor.reset_predictor()
