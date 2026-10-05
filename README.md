# Filter Annotation Refiner

**Turn a detection dataset into a segmentation dataset.** Point it at your YOLO boxes,
get back trainable polygon masks — with a quality score and a side-by-side comparison
for every object, so you can tell whether to trust the result.

[![tests](https://github.com/listenmudasir/filter-annotation-refiner/actions/workflows/tests.yml/badge.svg)](https://github.com/listenmudasir/filter-annotation-refiner/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

```
detection boxes  →  SAM masks  →  smart refinement  →  YOLO-seg dataset + QA reports
```

Labelling segmentation data by hand is the expensive part of training an instance
segmentation model. If you already have bounding boxes, SAM can do most of that work —
but raw SAM output is not a dataset. This tool handles the parts in between: finding
your labels whatever layout they are in, keeping fragmented and ring-shaped masks
intact through polygon conversion, scoring every object so bad conversions surface
instead of hiding, and writing a `data.yaml` you can hand straight to a trainer.

---

## Why not just run SAM in a loop?

Converting a mask to a polygon is lossy in ways that are easy to miss and painful to
discover after training:

| Problem | Naive conversion | This tool |
|---|---|---|
| Object split into pieces (hair, scratches, cracks) | keeps the largest piece, **silently drops the rest** | one polygon per component, all kept |
| Ring-shaped object (washer, stain, cell) | hole filled in | hole preserved, keyhole-bridged on export |
| Over/under-simplification | fixed pixel tolerance, wrong for most classes | searches for the coarsest polygon that holds a **round-trip IoU bound** |
| Was the conversion any good? | no idea | per-object `fidelity`, plus overlays and comparisons to look at |

Measured on synthetic cases, the naive path loses **50%** of a two-part mask and
inflates a ring by **34%**. Both are 1.00 and 0.98 here.

## Features

- **Reads the layouts real datasets actually use** — `images/` + `labels/` sibling
  trees, or images and `.txt` files in the same folder. Class names from `data.yaml`,
  `dataset.yaml`, `classes.txt` or `obj.names`.
- **Post-processing derived from your data, not a lookup table.** Morphology,
  speckle floor and simplification bound are chosen per class from the measured
  geometry of its boxes, so thin structures (hair, cracks, wires) keep their detail
  and compact blobs simplify hard — on datasets the tool has never seen.
- **Multiple SAM backends** — SAM 2.1 (default, openly downloadable weights) and
  SAM 3 / 3.1. A placeholder backend exists for UI testing and is clearly labelled
  as such.
- **Quality scoring and a review queue.** Every object gets SAM confidence, candidate
  stability, image-edge support, leakage, fragmentation, and polygon fidelity.
  Low-confidence objects land in a filterable review queue; the rest are already
  written.
- **Visual QA for the whole dataset** — a mask overlay for every image, plus a
  side-by-side *input boxes vs output masks* comparison.
- **Safe by construction.** Source labels are never modified, output must live
  outside the source tree, labels are written atomically, and runs can be stopped
  and resumed. An output folder can never silently mix results from two different
  backends.
- **English / 中文**, switchable at runtime.

## Screenshots

| Review queue with per-object QA | Input boxes vs output masks |
|---|---|
| Zoom to any object, filter by failure reason | Written for every converted image |

## Install

Requires Python 3.12+. A GPU is strongly recommended.

```bash
git clone https://github.com/listenmudasir/filter-annotation-refiner.git
cd filter-annotation-refiner

conda create -n far python=3.12 -y && conda activate far
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Get a SAM 2.1 checkpoint

```bash
pip install "git+https://github.com/facebookresearch/sam2.git"
mkdir -p ~/sam2/checkpoints && cd ~/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

The checkpoint is found automatically from common locations, or set it explicitly:

```bash
export SAM2_CHECKPOINT=~/sam2/checkpoints/sam2.1_hiera_large.pt
export SAM2_REPO=~/sam2          # only if sam2 is not pip-installed
```

Verify the environment:

```bash
python scripts/check_env.py
```

## Run

```bash
python app.py --backend sam2 --device cuda
```

| Backend | Notes |
|---|---|
| `sam2` *(default)* | SAM 2.1. Openly downloadable weights. |
| `sam3` | SAM 3 / 3.1. Weights are **gated** — request access on Hugging Face first. |
| `fallback` | Draws placeholder ellipses. **UI testing only — never a usable dataset.** |

Try it without a model or a dataset of your own:

```bash
python scripts/make_demo_dataset.py
python app.py --backend fallback
```

## Workflow

**1 · Dataset** — drop a folder. The layout, class names, object count and any
problems are reported up front. Images without labels are treated as background
images, not errors; malformed rows are skipped rather than blocking the run.

**2 · Convert / Refine** — pick a quality preset, watch masks appear with live
throughput and ETA. Pause or stop at any point; completed images are already saved.

**3 · Review / Export** — uncertain objects only, filterable by failure reason, each
with its saved overlay zoomed to the object.

## Output

```
my_dataset_seg/
├── data.yaml              # task: segment, contiguous class ids — ready to train
├── images/                # hard-linked or copied from source
├── labels/                # YOLO-seg polygons, one line per mask component
├── masks/                 # per-instance PNG masks
├── overlays/              # every image with its masks drawn
├── comparisons/           # input boxes | output masks, side by side
├── review_previews/       # uncertain images only
└── reports/
    ├── objects.csv        # per object: scores, fidelity, parts, holes, vertices
    ├── review_queue.csv
    ├── failures.csv
    └── summary.json
```

Train directly on it:

```bash
yolo segment train data=my_dataset_seg/data.yaml model=yolo11n-seg.pt
```

## How it decides

For each box, SAM is prompted several times from perturbed variants of that box
against one cached image embedding. Candidates are scored on:

- SAM confidence
- agreement with the nearest cluster of candidates (stability)
- image-edge alignment along the mask boundary
- leakage outside an expanded source box
- fragmentation
- agreement with an existing source polygon, when there is one

The winner is post-processed with its class profile and converted to polygons. The
written geometry is then rasterized back and compared against the mask — that
**fidelity** number is what proves nothing was lost, and it is stored per object.

> The quality score is a practical annotation-QA heuristic, **not a calibrated
> probability of correctness**. Audit the review queue before treating a converted
> dataset as ground truth.

## Choosing a backend on your own data

Model version numbers do not tell you which backend segments *your* objects best:

```bash
python scripts/compare_backends.py /path/to/dataset --backends sam2 sam3 -n 25
```

Reports quality, fidelity, vertex count and seconds-per-object for each, plus
pairwise mask agreement — a direct measure of how consistent they are on your data.

## Development

```bash
pip install pytest pyflakes
QT_QPA_PLATFORM=offscreen pytest -q      # 115 tests
pyflakes refiner tests app.py
```

Diagnose a single object end to end:

```bash
python scripts/debug_one.py /path/to/dataset --preset fast --device cuda
```

Architecture notes are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Roadmap

- COCO / Pascal VOC / LabelMe import and export (currently YOLO only)
- Interactive polygon editing and point prompts in the review tab
- Class-name editor for datasets that ship without a `data.yaml`
- Iterative refinement using SAM's low-resolution logits

Contributions welcome — open an issue to discuss before large changes.

## License

MIT — see [LICENSE](LICENSE).

SAM 2 and SAM 3 are licensed separately by Meta; their checkpoints carry their own
terms. This project does not redistribute model weights.
