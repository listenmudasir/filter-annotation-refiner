import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from refiner.backend.base import SegmentationBackend
from refiner.backend.fallback import FallbackBackend
from refiner.dataset import scan_dataset
from refiner.models import CandidateMask
from refiner.services.refinement import ConversionSettings, SmartRefinementEngine


def make_dataset(tmp_path: Path, class_name: str = "Stain", size=(128, 96)) -> Path:
    src = tmp_path / "src"
    (src / "images" / "train").mkdir(parents=True)
    (src / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", size, "white").save(src / "images" / "train" / "x.jpg")
    (src / "labels" / "train" / "x.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (src / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "names": [class_name]}))
    return src


def test_fallback_end_to_end(tmp_path: Path):
    src = make_dataset(tmp_path)
    ds = scan_dataset(src)
    out = tmp_path / "out"
    engine = SmartRefinementEngine(ds, FallbackBackend(), out, ConversionSettings(preset="balanced"))
    result = engine.process_record(ds.records[0], resume=False)
    assert result.output_label.exists()
    line = result.output_label.read_text().strip().split()
    assert len(line) > 7
    assert (out / "images" / "train" / "x.jpg").exists()
    assert (out / "reports" / "objects.csv").exists()

    rows = list(csv.DictReader((out / "reports" / "objects.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert float(rows[0]["fidelity"]) > 0.95
    assert rows[0]["parts"] == "1"


class TwoStrandBackend(SegmentationBackend):
    """Returns a deliberately fragmented mask, like SAM does for broken hair."""

    name = "two-strand"

    def load(self) -> None:
        return None

    def set_image(self, image) -> None:
        self.size = image.size

    def candidates(self, annotation, preset):
        w, h = self.size
        mask = np.zeros((h, w), bool)
        mask[20:26, 10:w - 10] = True
        mask[70:76, 10:w - 10] = True
        return [CandidateMask(mask, 0.9, "two-strand")]


def test_fragmented_mask_writes_one_line_per_component(tmp_path: Path):
    """Regression: both strands must reach the label file, not just the largest."""
    src = make_dataset(tmp_path, class_name="Hair", size=(200, 100))
    ds = scan_dataset(src)
    out = tmp_path / "out"
    engine = SmartRefinementEngine(ds, TwoStrandBackend(), out, ConversionSettings(preset="fast"))
    outcome = engine.process_record(ds.records[0], resume=False)

    lines = [l for l in outcome.output_label.read_text().splitlines() if l.strip()]
    assert len(lines) == 2, "each mask component needs its own YOLO-seg line"
    assert all(l.startswith("0 ") for l in lines)

    result = outcome.results[0]
    assert result.part_count == 2
    assert result.polygon_fidelity > 0.95

    # Coordinates stay inside the normalized range.
    for line in lines:
        values = [float(v) for v in line.split()[1:]]
        assert all(0.0 <= v <= 1.0 for v in values)


def test_second_backend_cannot_resume_into_the_same_output(tmp_path: Path):
    """Regression: a fallback run then a SAM run left ellipse masks in the dataset.

    Resume keys on image path, so the second run skipped the finished images and the
    first backend's placeholder masks survived into the exported labels, invisible
    because summary.json only recorded the last backend.
    """
    src = make_dataset(tmp_path, size=(200, 100))
    ds = scan_dataset(src)
    out = tmp_path / "out"

    first = SmartRefinementEngine(ds, FallbackBackend(), out, ConversionSettings(preset="fast"))
    first.process_record(ds.records[0], resume=False)

    with pytest.raises(RuntimeError) as excinfo:
        SmartRefinementEngine(ds, TwoStrandBackend(), out, ConversionSettings(preset="fast"))
    message = str(excinfo.value)
    assert "cannot be resumed" in message
    assert "fallback" in message and "two-strand" in message


def test_changing_preset_also_blocks_resume(tmp_path: Path):
    src = make_dataset(tmp_path)
    ds = scan_dataset(src)
    out = tmp_path / "out"
    SmartRefinementEngine(ds, FallbackBackend(), out, ConversionSettings(preset="fast")).process_record(
        ds.records[0], resume=False
    )
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        SmartRefinementEngine(ds, FallbackBackend(), out, ConversionSettings(preset="maximum"))


def test_same_backend_and_preset_resumes_normally(tmp_path: Path):
    src = make_dataset(tmp_path)
    ds = scan_dataset(src)
    out = tmp_path / "out"
    settings = ConversionSettings(preset="fast")
    SmartRefinementEngine(ds, FallbackBackend(), out, settings).process_record(ds.records[0], resume=False)
    engine = SmartRefinementEngine(ds, FallbackBackend(), out, settings)
    assert engine.process_record(ds.records[0], resume=True).skipped


def test_summary_flags_placeholder_masks(tmp_path: Path):
    src = make_dataset(tmp_path)
    ds = scan_dataset(src)
    out = tmp_path / "out"
    engine = SmartRefinementEngine(ds, FallbackBackend(), out, ConversionSettings(preset="fast"))
    engine.process_record(ds.records[0], resume=False)
    summary = json.loads(engine.write_summary().read_text())
    assert "fallback-1" in summary["prompts_seen"]
    assert "not real segmentation" in summary["warning"]


def test_data_yaml_is_always_written(tmp_path: Path):
    """A converted dataset without data.yaml cannot be handed to a trainer."""
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    Image.new("RGB", (120, 90), "white").save(src / "images" / "a.jpg")
    # No data.yaml and no classes.txt, and a gap in the class ids.
    (src / "images" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n3 0.2 0.2 0.1 0.1\n")

    ds = scan_dataset(src)
    assert ds.class_names == {}
    out = tmp_path / "out"
    SmartRefinementEngine(ds, FallbackBackend(), out, ConversionSettings(preset="fast"))

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["task"] == "segment"
    assert data["path"] == "."
    # Ids must be contiguous for Ultralytics, including the unused id 2.
    assert data["names"] == {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3"}
    assert data["nc"] == 4
    assert data["train"]


def test_data_yaml_keeps_source_class_names(tmp_path: Path):
    src = make_dataset(tmp_path, class_name="Scratch")
    ds = scan_dataset(src)
    out = tmp_path / "out"
    SmartRefinementEngine(ds, FallbackBackend(), out, ConversionSettings(preset="fast"))
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"][0] == "Scratch"
    assert data["task"] == "segment"


def test_comparison_images_are_written(tmp_path: Path):
    src = make_dataset(tmp_path, size=(200, 100))
    ds = scan_dataset(src)
    out = tmp_path / "out"
    engine = SmartRefinementEngine(ds, FallbackBackend(), out, ConversionSettings(preset="fast"))
    engine.process_record(ds.records[0], resume=False)
    assert engine.wait_for_writes() == []

    comparisons = list((out / "comparisons").rglob("*.jpg"))
    assert len(comparisons) == 1
    with Image.open(comparisons[0]) as img:
        # Two panels side by side, so wider than tall relative to the source.
        assert img.width > 2 * 200 - 50
    assert not list(out.rglob("*.jpg.tmp"))


def test_comparisons_can_be_disabled(tmp_path: Path):
    src = make_dataset(tmp_path)
    ds = scan_dataset(src)
    out = tmp_path / "out"
    engine = SmartRefinementEngine(
        ds, FallbackBackend(), out, ConversionSettings(preset="fast", save_comparisons=False)
    )
    engine.process_record(ds.records[0], resume=False)
    assert engine.wait_for_writes() == []
    assert not (out / "comparisons").exists()


def test_background_writers_produce_the_same_files(tmp_path: Path):
    """Threaded artifact writing must not change what lands on disk."""
    outputs = {}
    for workers in (0, 3):
        src = make_dataset(tmp_path / f"w{workers}", size=(200, 100))
        ds = scan_dataset(src)
        out = tmp_path / f"out{workers}"
        engine = SmartRefinementEngine(
            ds, FallbackBackend(), out, ConversionSettings(preset="fast", io_workers=workers)
        )
        engine.process_record(ds.records[0], resume=False)
        assert engine.wait_for_writes() == []
        outputs[workers] = sorted(
            p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
        )
    assert outputs[0] == outputs[3]
    assert any("comparisons/" in p for p in outputs[3])
    assert not any(p.endswith(".tmp") for p in outputs[3])


def test_failed_artifact_write_is_reported_not_raised(tmp_path: Path):
    """A broken overlay must not abort a conversion, but must be surfaced."""
    src = make_dataset(tmp_path)
    ds = scan_dataset(src)
    engine = SmartRefinementEngine(
        ds, FallbackBackend(), tmp_path / "out", ConversionSettings(preset="fast", io_workers=2)
    )

    def boom(*_args):
        raise OSError("disk full")

    engine._submit_io(boom, None)
    errors = engine.wait_for_writes()
    assert errors and "disk full" in errors[0]


def test_wait_for_writes_is_safe_to_call_twice(tmp_path: Path):
    src = make_dataset(tmp_path)
    ds = scan_dataset(src)
    engine = SmartRefinementEngine(
        ds, FallbackBackend(), tmp_path / "out", ConversionSettings(preset="fast")
    )
    engine.process_record(ds.records[0], resume=False)
    assert engine.wait_for_writes() == []
    assert engine.wait_for_writes() == []


def test_report_schema_mismatch_is_refused(tmp_path: Path):
    src = make_dataset(tmp_path)
    ds = scan_dataset(src)
    out = tmp_path / "out"
    reports = out / "reports"
    reports.mkdir(parents=True)
    (reports / "objects.csv").write_text("image,instance,quality\n")

    try:
        SmartRefinementEngine(ds, FallbackBackend(), out, ConversionSettings())
    except RuntimeError as exc:
        assert "incompatible version" in str(exc)
    else:
        raise AssertionError("an older report schema must not be appended to")
