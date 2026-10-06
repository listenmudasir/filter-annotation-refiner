"""Dataset scanning must handle the layouts real YOLO exports actually use."""

from pathlib import Path

import yaml
from PIL import Image

from refiner.dataset import candidate_label_paths, class_boxes, load_class_names, scan_dataset


def image(path: Path, size=(100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def test_scan_split_folder_layout(tmp_path: Path):
    """images/ and labels/ as sibling trees - the Ultralytics convention."""
    image(tmp_path / "images" / "train" / "a.jpg")
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": ["Hair"]}))

    ds = scan_dataset(tmp_path)
    assert len(ds.records) == 1
    assert ds.object_count == 1
    assert ds.detection_count == 1
    assert ds.class_names[0] == "Hair"
    assert not ds.records[0].issues
    assert ds.layout == "separate images/ and labels/ folders"


def test_scan_same_folder_layout(tmp_path: Path):
    """Images and .txt side by side in one folder - the labelImg default."""
    flat = tmp_path / "capture"
    image(flat / "a.jpg")
    image(flat / "b.jpg")
    (flat / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (flat / "b.txt").write_text("1 0.25 0.25 0.2 0.2\n")

    ds = scan_dataset(tmp_path)
    assert len(ds.records) == 2
    assert ds.object_count == 2
    assert ds.layout == "images and labels in the same folder"
    assert all(not r.issues for r in ds.records)


def test_class_names_from_classes_txt(tmp_path: Path):
    """Most real exports ship no YAML, only a classes.txt beside the labels."""
    flat = tmp_path / "capture"
    image(flat / "a.jpg")
    (flat / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n2 0.2 0.2 0.1 0.1\n")
    (flat / "classes.txt").write_text("Bug\nForeign_Body\nStain\n")

    ds = scan_dataset(tmp_path)
    assert ds.class_names == {0: "Bug", 1: "Foreign_Body", 2: "Stain"}


def test_yaml_beats_classes_txt(tmp_path: Path):
    image(tmp_path / "images" / "a.jpg")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (tmp_path / "classes.txt").write_text("FromTxt\n")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": ["FromYaml"]}))
    assert load_class_names(tmp_path)[0] == "FromYaml"


def test_background_images_are_not_errors(tmp_path: Path):
    """An image with no label file is a valid YOLO background image."""
    flat = tmp_path / "capture"
    image(flat / "a.jpg")
    image(flat / "bg.jpg")
    (flat / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    ds = scan_dataset(tmp_path)
    assert len(ds.records) == 2
    assert ds.object_count == 1
    assert ds.background_count == 1
    assert ds.problem_records == [], "a background image must not block conversion"
    assert len(ds.convertible_records) == 2


def test_malformed_rows_are_isolated_to_their_image(tmp_path: Path):
    flat = tmp_path / "capture"
    image(flat / "good.jpg")
    image(flat / "bad.jpg")
    (flat / "good.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (flat / "bad.txt").write_text("0 not numbers here\n")

    ds = scan_dataset(tmp_path)
    assert len(ds.problem_records) == 1
    assert ds.problem_records[0].relative_image.name == "bad.jpg"
    # The healthy image is still convertible.
    assert [r.relative_image.name for r in ds.convertible_records if r.annotations] == ["good.jpg"]


def test_candidate_label_paths_cover_both_layouts(tmp_path: Path):
    img = tmp_path / "images" / "train" / "a.jpg"
    paths = candidate_label_paths(tmp_path, img)
    assert tmp_path / "labels" / "train" / "a.txt" in paths
    assert tmp_path / "images" / "train" / "a.txt" in paths


def test_sibling_label_wins_when_split_folder_is_absent(tmp_path: Path):
    """An 'images' folder with the .txt inside it must still resolve."""
    img_dir = tmp_path / "images"
    image(img_dir / "a.jpg")
    (img_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    ds = scan_dataset(tmp_path)
    assert ds.object_count == 1
    assert ds.records[0].label_path == img_dir / "a.txt"


def test_class_boxes_groups_by_class(tmp_path: Path):
    flat = tmp_path / "capture"
    image(flat / "a.jpg", size=(200, 100))
    (flat / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.1 0.1\n3 0.7 0.7 0.2 0.2\n")

    grouped = class_boxes(scan_dataset(tmp_path))
    assert sorted(grouped) == [0, 3]
    assert len(grouped[0]) == 2
    assert len(grouped[3]) == 1


def test_polygon_source_labels_are_recognised(tmp_path: Path):
    flat = tmp_path / "capture"
    image(flat / "a.jpg")
    (flat / "a.txt").write_text("0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n")

    ds = scan_dataset(tmp_path)
    assert ds.polygon_count == 1
    assert ds.detection_count == 0


def test_generated_output_is_detected(tmp_path: Path):
    """Converting an output back into a new dataset must be detectable.

    A generated dataset is a structurally valid YOLO dataset, so nothing else
    distinguishes it from a real source.
    """
    from refiner.dataset import generated_output_marker

    out = tmp_path / "ds_sam_refined"
    (out / ".sam_refiner").mkdir(parents=True)
    (out / ".sam_refiner" / "state.json").write_text("{}")
    assert generated_output_marker(out) is not None
    # Also when pointed at a subfolder of the output, which is the usual mistake.
    sub = out / "defect"
    sub.mkdir()
    assert generated_output_marker(sub) is not None


def test_output_without_state_dir_is_still_detected(tmp_path: Path):
    from refiner.dataset import generated_output_marker

    out = tmp_path / "copied_output"
    (out / "reports").mkdir(parents=True)
    (out / "reports" / "summary.json").write_text("{}")
    (out / "overlays").mkdir()
    assert generated_output_marker(out) is not None


def test_real_source_dataset_is_not_flagged(tmp_path: Path):
    from refiner.dataset import generated_output_marker

    src = tmp_path / "real"
    image(src / "images" / "a.jpg")
    (src / "labels").mkdir()
    (src / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    assert generated_output_marker(src) is None


def test_scan_reports_progress_and_can_be_cancelled(tmp_path: Path):
    """Regression: scanning a large dataset froze the GUI with no way out.

    scan_dataset opens every image header and reads every label file, so it is
    linear in dataset size - 23 s for 8k images. Run on the GUI thread without
    progress or cancellation it looked like a crash and got force-quit.
    """
    from refiner.dataset import ScanCancelled, scan_dataset

    flat = tmp_path / "many"
    for i in range(200):
        image(flat / f"img{i:03d}.jpg", size=(32, 24))
        (flat / f"img{i:03d}.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    seen: list[tuple[int, int]] = []
    ds = scan_dataset(flat, progress=lambda done, total: seen.append((done, total)))
    assert len(ds.records) == 200
    assert seen[0] == (0, 200) and seen[-1] == (200, 200)
    assert len(seen) >= 3, "progress must be reported during the scan, not only at the end"

    import pytest
    with pytest.raises(ScanCancelled):
        scan_dataset(flat, should_stop=lambda: True)


def test_cancelled_scan_stops_early(tmp_path: Path):
    from refiner.dataset import ScanCancelled, scan_dataset

    flat = tmp_path / "many"
    for i in range(300):
        image(flat / f"img{i:03d}.jpg", size=(32, 24))

    calls = {"n": 0}

    def stop_after_a_while() -> bool:
        calls["n"] += 1
        return calls["n"] > 2          # cancel on the third check

    import pytest
    with pytest.raises(ScanCancelled):
        scan_dataset(flat, should_stop=stop_after_a_while)
    assert calls["n"] <= 4, "cancellation must be noticed promptly, not at the end"


def test_class_names_found_without_a_stat_per_file(tmp_path: Path):
    """A directory whose name matches a yaml pattern must not break the walk."""
    flat = tmp_path / "ds"
    image(flat / "a.jpg")
    (flat / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (flat / "notes.yaml").mkdir(parents=True)   # a *directory* named like a YAML
    (flat / "classes.txt").write_text("Crack\nPit\n")

    assert load_class_names(flat) == {0: "Crack", 1: "Pit"}
