"""Functional tests that drive the real widgets.

These click the actual buttons and assert on the resulting state rather than only
importing the module, so a control that is wired to nothing fails the suite.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from refiner.models import Annotation, AnnotationKind, ImageRecord  # noqa: E402
from refiner.ui.main_window import MainWindow  # noqa: E402
from refiner.ui.widgets import DatasetDropZone, ImagePreview, MetricCard  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "images" / "train").mkdir(parents=True)
    (src / "labels" / "train").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(3):
        noise = rng.integers(90, 170, size=(120, 160, 3), dtype=np.uint8)
        Image.fromarray(noise).save(src / "images" / "train" / f"img{i}.jpg", quality=95)
        (src / "labels" / "train" / f"img{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n1 0.25 0.3 0.2 0.2\n")
    (src / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "names": ["Stain", "Hair"]}))
    return src


@pytest.fixture
def window(qapp, dataset) -> MainWindow:
    win = MainWindow("fallback", None, None)
    win._load_dataset(str(dataset))
    return win


def pump(qapp, seconds: float = 0.05) -> None:
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.005)


def run_conversion(qapp, window: MainWindow, timeout: float = 60.0) -> None:
    """Click Start and wait for the worker's own finished signal.

    Polling ``QThread.isRunning()`` races the thread's start-up and can report
    completion before any work has happened.
    """
    window.start_btn.click()
    assert window.worker is not None, "Start must create a worker"
    done: list[dict] = []
    window.worker.finished.connect(done.append)
    errors: list[str] = []
    window.worker.error.connect(errors.append)

    deadline = time.time() + timeout
    while not done and not errors and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    assert not errors, f"conversion reported an error: {errors}"
    assert done, "conversion did not finish within the timeout"
    pump(qapp, 0.1)


# --------------------------------------------------------------------------- #
# Dataset tab
# --------------------------------------------------------------------------- #


def test_tabs_exist(window):
    assert [window.tabs.tabText(i).strip() for i in range(window.tabs.count())] == [
        "1  ·  Dataset", "2  ·  Convert / Refine", "3  ·  Review / Export",
    ]


def test_tab_titles_contain_no_qt_mnemonics(window):
    """A bare '&' makes Qt underline the next letter: 'Convert & Refine' rendered
    as 'Convert _Refine'."""
    for i in range(window.tabs.count()):
        text = window.tabs.tabText(i)
        assert "&" not in text.replace("&&", ""), f"tab {i} has an unescaped mnemonic: {text}"


def test_dataset_scan_populates_every_metric(window):
    assert window.m_images.value.text() == "3"
    assert window.m_labels.value.text() == "3"
    assert window.m_objects.value.text() == "6"
    assert window.m_boxes.value.text() == "6"
    assert window.m_polygons.value.text() == "0"
    assert window.m_problems.value.text() == "0"
    assert window.class_table.rowCount() == 2
    assert window.class_table.item(0, 1).text() == "Stain"
    assert window.issue_table.rowCount() == 0
    assert "Ready" in window.dataset_status.text()


def test_dataset_gates_enable_after_a_clean_scan(window):
    assert window.next_btn.isEnabled()
    assert window.start_btn.isEnabled()
    assert window.scan_btn.isEnabled()


def test_next_button_switches_tab(window):
    window.tabs.setCurrentIndex(0)
    window.next_btn.click()
    assert window.tabs.currentIndex() == 1


def test_rescan_button_reloads(window, dataset):
    window.m_images.set_value("stale")
    window.scan_btn.click()
    assert window.m_images.value.text() == "3"


def test_output_path_defaults_outside_the_source(window, dataset):
    out = Path(window.output_edit.text())
    assert out.name == f"{dataset.name}_sam_refined"
    assert not out.is_relative_to(dataset)


def test_drop_zone_emits_folder(qapp, tmp_path):
    zone = DatasetDropZone()
    seen = []
    zone.folderDropped.connect(seen.append)
    zone.folderDropped.emit(str(tmp_path))
    assert seen == [str(tmp_path)]

    clicked = []
    zone.chooseClicked.connect(lambda: clicked.append(True))
    zone.findChild(type(zone.children()[-1]))  # ensure children exist
    zone.chooseClicked.emit()
    assert clicked == [True]


def test_metric_card_updates(qapp):
    card = MetricCard("Objects")
    assert card.value.text() == "—"
    card.set_value(42)
    assert card.value.text() == "42"


# --------------------------------------------------------------------------- #
# Conversion settings -> ConversionSettings
# --------------------------------------------------------------------------- #


def test_every_setting_control_reaches_conversion_settings(window):
    window.preset_combo.setCurrentIndex(2)
    window.accept_spin.setValue(0.91)
    window.review_spin.setValue(0.33)
    window.chk_masks.setChecked(False)
    window.chk_overlays.setChecked(False)
    window.chk_previews.setChecked(False)
    window.chk_links.setChecked(False)

    settings = window._settings()
    assert settings.preset == "maximum"
    assert settings.accept_threshold == pytest.approx(0.91)
    assert settings.review_threshold == pytest.approx(0.33)
    assert settings.save_masks is False
    assert settings.save_overlays is False
    assert settings.save_review_previews is False
    assert settings.link_images is False

    window.chk_overlays.setChecked(True)
    assert window._settings().save_overlays is True


def test_preset_combo_exposes_all_three_presets(window):
    values = [window.preset_combo.itemData(i) for i in range(window.preset_combo.count())]
    assert values == ["fast", "balanced", "maximum"]


def test_output_inside_source_is_refused(window, dataset, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[1]))
    window.output_edit.setText(str(dataset / "inside"))
    window.start_btn.click()
    assert warned == ["Unsafe output"]
    assert window.worker is None, "an unsafe output must not start a worker"
    assert window.start_btn.isEnabled()


# --------------------------------------------------------------------------- #
# Running a conversion through the actual buttons
# --------------------------------------------------------------------------- #


def test_start_button_converts_and_fills_the_review_tab(qapp, window, tmp_path):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    run_conversion(qapp, window)

    labels = sorted((out / "labels" / "train").glob("*.txt"))
    assert len(labels) == 3
    assert all(path.read_text().strip() for path in labels)

    # Progress and statistics widgets were driven by the worker signals.
    assert window.progress.value() == 3
    assert window.progress_text.text() == "3 / 3 images"
    assert int(window.r_objects.value.text()) == 6
    assert int(window.r_failed.value.text()) == 0
    assert int(window.r_accepted.value.text()) + int(window.r_review.value.text()) == 6

    # Finishing jumps to the review tab and re-enables the controls.
    assert window.tabs.currentIndex() == 2
    assert window.start_btn.isEnabled()
    assert not window.pause_btn.isEnabled()
    assert not window.stop_btn.isEnabled()
    assert window.open_output_btn.isEnabled()
    assert window.open_overlays_btn.isEnabled()
    assert window.refresh_review_btn.isEnabled()
    assert str(out) in window.summary_label.text()


def test_overlay_is_saved_for_every_image(qapp, window, tmp_path):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    run_conversion(qapp, window)

    overlays = sorted((out / "overlays" / "images" / "train").glob("*.jpg"))
    assert len(overlays) == 3, "an overlay must be written for every converted image"
    for path in overlays:
        with Image.open(path) as img:
            assert img.size == (160, 120)
    assert not list(out.rglob("*.jpg.tmp")), "overlay temp files must be renamed away"


def test_overlays_can_be_switched_off(qapp, window, tmp_path):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    window.chk_overlays.setChecked(False)
    run_conversion(qapp, window)
    assert not (out / "overlays").exists()
    assert sorted((out / "labels" / "train").glob("*.txt"))


def test_review_table_matches_the_queue_file(qapp, window, tmp_path):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    # Force everything into review so the table is non-trivial.
    window.accept_spin.setValue(0.99)
    run_conversion(qapp, window)

    queue = (out / "reports" / "review_queue.csv").read_text().strip().splitlines()
    assert window.review_table.rowCount() == len(queue) - 1 > 0
    assert "uncertain object" in window.review_status.text()
    # Refreshing the queue must not clobber the run summary.
    assert str(out) in window.summary_label.text()

    window.review_table.selectRow(0)
    pump(qapp)
    assert window.review_preview.view.has_image(), "selecting a row must show its overlay"
    assert window.review_table.item(0, 0).text() in window.review_detail.text()


def test_review_filters_narrow_the_queue(qapp, window, tmp_path):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    window.accept_spin.setValue(0.99)
    run_conversion(qapp, window)

    total = window.review_table.rowCount()
    assert total > 0
    assert window.review_filter_buttons["all"].text() == f"All ({total})"

    # Every visible filter must narrow to a non-empty, smaller-or-equal subset.
    for key, button in window.review_filter_buttons.items():
        if key == "all" or not button.isVisible():
            continue
        window._set_review_filter(key)
        shown = window.review_table.rowCount()
        assert 0 < shown <= total, key
        for row in range(shown):
            assert window._row_matches_filter(
                {"warnings": window.review_table.item(row, 5).text()}, key
            )

    window._set_review_filter("all")
    assert window.review_table.rowCount() == total


def test_throughput_and_eta_are_reported(qapp, window, tmp_path):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    run_conversion(qapp, window)
    text = window.throughput_text.text()
    assert "s / image" in text and "ETA" in text, text


def test_duration_formatting():
    fmt = MainWindow._format_duration
    assert fmt(5) == "5s"
    assert fmt(75) == "1m 15s"
    assert fmt(3725) == "1h 02m"
    assert fmt(-1) == "0s"


def test_refresh_button_rereads_the_queue(qapp, window, tmp_path):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    window.accept_spin.setValue(0.99)
    run_conversion(qapp, window)
    before = window.review_table.rowCount()

    window.review_table.setRowCount(0)
    window.refresh_review_btn.click()
    assert window.review_table.rowCount() == before


def test_open_folder_buttons_point_at_real_paths(qapp, window, tmp_path, monkeypatch):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    run_conversion(qapp, window)

    opened = []
    monkeypatch.setattr(
        "refiner.ui.main_window.QDesktopServices.openUrl",
        lambda url: opened.append(url.toLocalFile()),
    )
    window.open_output_btn.click()
    window.open_overlays_btn.click()
    assert opened == [str(out), str(out / "overlays")]
    assert all(Path(p).exists() for p in opened)


def seed_other_backend_output(ds_path: Path, out: Path) -> None:
    """Leave an output folder behind that a different backend produced."""
    from refiner.backend.fallback import FallbackBackend
    from refiner.dataset import scan_dataset
    from refiner.services.refinement import ConversionSettings, SmartRefinementEngine

    ds = scan_dataset(ds_path)
    engine = SmartRefinementEngine(ds, FallbackBackend(), out, ConversionSettings(preset="fast"))
    engine.process_record(ds.records[0], resume=False)
    # Drain before returning: the caller may delete this folder, and leaving
    # writers running inside a directory being removed deadlocks the test.
    assert engine.wait_for_writes() == []


def test_conflicting_output_offers_a_new_folder_before_starting(qapp, window, dataset, tmp_path, monkeypatch):
    """The clash must be caught up front, not as a traceback after Start."""
    out = tmp_path / "out"
    seed_other_backend_output(dataset, out)
    window.output_edit.setText(str(out))

    shown = {}

    def fake_exec(self):
        shown["text"] = self.text()
        # Default button is the non-destructive one.
        self.setResult(0)
        shown["default"] = self.defaultButton().text()
        self._clicked = self.defaultButton()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: self._clicked)

    resolved = window._resolve_output_conflict(out)
    assert "cannot be resumed" in shown["text"]
    assert shown["default"] == "Use a new folder"
    assert resolved == out.with_name("out_v2")
    assert out.exists(), "choosing a new folder must not touch the old one"


def test_conflict_cancel_does_not_start(qapp, window, dataset, tmp_path, monkeypatch):
    out = tmp_path / "out"
    seed_other_backend_output(dataset, out)
    window.output_edit.setText(str(out))

    def fake_exec(self):
        self._clicked = next(b for b in self.buttons() if b.text() == "Cancel")
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: self._clicked)

    assert window._resolve_output_conflict(out) is None
    window.start_btn.click()
    assert window.worker is None, "a cancelled conflict must not start a run"
    assert out.exists()


def test_delete_requires_a_second_confirmation(qapp, window, dataset, tmp_path, monkeypatch):
    """Clicking 'Delete and start over' alone must not destroy the dataset."""
    out = tmp_path / "out"
    seed_other_backend_output(dataset, out)

    def fake_exec(self):
        self._clicked = next(b for b in self.buttons() if b.text() == "Delete and start over")
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: self._clicked)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)

    assert window._resolve_output_conflict(out) is None
    assert out.exists(), "declining the confirmation must keep the folder"

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    assert window._resolve_output_conflict(out) == out
    assert not out.exists(), "confirming must remove the folder"


def test_matching_output_starts_without_a_prompt(qapp, window, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: pytest.fail("no dialog expected for a clean folder")
    )
    assert window._resolve_output_conflict(out) == out


def test_stop_releases_the_app_promptly(qapp, window, tmp_path):
    """Stop must free the UI quickly so new data can be loaded straight away."""
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    window.start_btn.click()
    done = []
    window.worker.finished.connect(done.append)

    window.stop_btn.click()
    deadline = time.time() + 10
    while not done and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    elapsed = time.time() - (deadline - 10)
    assert done, "stop did not end the run within 10s"
    assert done[0].get("cancelled") is True
    assert elapsed < 10

    thread = window.worker_thread
    assert thread is None or thread.wait(5000)
    pump(qapp, 0.1)
    assert window.start_btn.isEnabled()
    assert not window.stop_btn.isEnabled()
    assert not window.pause_btn.isEnabled()


def test_stop_while_paused_still_stops(qapp, window, tmp_path):
    """A paused worker sits in a wait loop; stopping must not deadlock."""
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    window.start_btn.click()
    done = []
    window.worker.finished.connect(done.append)
    window.pause_btn.click()
    pump(qapp, 0.1)
    window.stop_btn.click()

    deadline = time.time() + 10
    while not done and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    assert done, "a paused run must still respond to stop"
    thread = window.worker_thread
    assert thread is None or thread.wait(5000)


def test_loading_new_data_cancels_a_running_conversion(qapp, window, dataset, tmp_path):
    """Dropping a new dataset mid-run must stop the old one, not run both."""
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    window.start_btn.click()
    assert window.is_running()

    window._load_dataset(str(dataset))
    pump(qapp, 0.1)
    assert not window.is_running(), "the previous run must be stopped before rescanning"
    assert window.start_btn.isEnabled()


def test_pause_and_stop_buttons_toggle_state(qapp, window, tmp_path):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    window.start_btn.click()
    try:
        assert window.pause_btn.isEnabled()
        assert window.stop_btn.isEnabled()
        window.pause_btn.click()
        assert window.pause_btn.text() == "Resume"
        assert window.worker._pause.is_set()
        window.pause_btn.click()
        assert window.pause_btn.text() == "Pause"
        assert not window.worker._pause.is_set()
        window.stop_btn.click()
        assert window.worker._stop.is_set()
        assert not window.stop_btn.isEnabled()
    finally:
        deadline = time.time() + 30
        while window.worker_thread is not None and window.worker_thread.isRunning() and time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        pump(qapp, 0.2)


# --------------------------------------------------------------------------- #
# Zoom / pan preview
# --------------------------------------------------------------------------- #


@pytest.fixture
def preview_with_image(qapp, tmp_path):
    preview = ImagePreview()
    preview.resize(400, 300)
    path = tmp_path / "p.jpg"
    Image.new("RGB", (800, 600), "white").save(path)
    record = ImageRecord(path, None, Path("p.jpg"), 800, 600,
                         [Annotation(0, AnnotationKind.DETECTION, (10, 10, 400, 300))])
    preview.show_record(record)
    return preview


def test_controls_are_disabled_until_an_image_is_loaded(qapp):
    preview = ImagePreview()
    assert not preview.zoom_in_btn.isEnabled()
    assert not preview.fit_btn.isEnabled()
    assert preview.placeholder.isVisible() or not preview.view.has_image()


def test_zoom_buttons_change_the_scale(preview_with_image):
    preview = preview_with_image
    assert preview.zoom_in_btn.isEnabled()
    before = preview.view.scale_factor
    preview.zoom_in_btn.click()
    assert preview.view.scale_factor > before
    preview.zoom_out_btn.click()
    assert preview.view.scale_factor == pytest.approx(before, rel=1e-6)


def test_actual_size_and_fit_buttons(preview_with_image):
    preview = preview_with_image
    preview.actual_btn.click()
    assert preview.view.scale_factor == pytest.approx(1.0)
    assert preview.zoom_label.text() == "100%"
    preview.fit_btn.click()
    # The 800x600 image must shrink to fit a 400px-wide panel.
    assert preview.view.scale_factor < 1.0


def test_zoom_is_clamped_both_ways(preview_with_image):
    preview = preview_with_image
    for _ in range(80):
        preview.zoom_in_btn.click()
    assert preview.view.scale_factor <= preview.view.MAX_SCALE + 1e-6
    for _ in range(200):
        preview.zoom_out_btn.click()
    assert preview.view.scale_factor >= preview.view.MIN_SCALE - 1e-6


def test_zoom_label_tracks_the_scale(preview_with_image):
    preview = preview_with_image
    preview.actual_btn.click()
    assert preview.zoom_label.text() == "100%"
    preview.zoom_in_btn.click()
    assert preview.zoom_label.text() == "125%"


def test_manual_zoom_survives_an_image_change(preview_with_image, tmp_path):
    preview = preview_with_image
    preview.actual_btn.click()
    scale = preview.view.scale_factor

    other = tmp_path / "q.jpg"
    Image.new("RGB", (800, 600), "black").save(other)
    record = ImageRecord(other, None, Path("q.jpg"), 800, 600, [])
    preview.show_record(record)
    assert preview.view.scale_factor == pytest.approx(scale), "zoom must persist across images"

    preview.fit_btn.click()
    preview.show_record(record)
    assert preview.view.scale_factor < 1.0, "after Fit, new images fit again"


def test_message_clears_any_previous_image(preview_with_image):
    """Regression: an error caption used to appear over the previous object's mask."""
    preview = preview_with_image
    assert preview.view.has_image()
    preview.set_message("nothing to show")
    assert not preview.view.has_image(), "a message must replace the image, not overlay it"
    # isHidden() rather than isVisible(): the panel itself is never shown in tests,
    # and Qt reports children of an unshown widget as not visible.
    assert not preview.placeholder.isHidden()
    assert preview.placeholder.text() == "nothing to show"
    assert not preview.zoom_in_btn.isEnabled()


def test_focus_zooms_to_the_object_not_the_frame(qapp, tmp_path):
    """A small defect on a large frame must not be fitted to invisibility."""
    preview = ImagePreview()
    preview.resize(400, 300)
    path = tmp_path / "big.jpg"
    Image.new("RGB", (4000, 3000), "white").save(path)
    record = ImageRecord(path, None, Path("big.jpg"), 4000, 3000,
                         [Annotation(0, AnnotationKind.DETECTION, (2000, 1500, 2100, 1600))])

    preview.focus_btn.setChecked(True)
    preview.show_record(record)
    focused = preview.view.scale_factor

    preview.focus_btn.setChecked(False)
    whole_frame = preview.view.scale_factor
    assert focused > whole_frame * 5, "focusing a 100px object on a 4000px frame must zoom in hard"


def test_focus_falls_back_to_whole_frame_without_boxes(qapp, tmp_path):
    preview = ImagePreview()
    preview.resize(400, 300)
    path = tmp_path / "empty.jpg"
    Image.new("RGB", (800, 600), "white").save(path)
    preview.show_record(ImageRecord(path, None, Path("empty.jpg"), 800, 600, []))
    assert preview.view.has_image()
    assert preview.view.scale_factor < 1.0


def test_unreadable_image_reports_instead_of_crashing(qapp, tmp_path):
    preview = ImagePreview()
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")
    record = ImageRecord(broken, None, Path("broken.jpg"), 10, 10, [])
    preview.show_record(record)
    assert "Cannot preview" in preview.placeholder.text()


# --------------------------------------------------------------------------- #
# Input annotation preview (Dataset tab)
# --------------------------------------------------------------------------- #


def test_input_preview_shows_parsed_boxes_before_converting(window):
    """The operator must be able to verify label parsing without a conversion."""
    assert window.input_preview.view.has_image()
    assert window.input_position.text() == "1 / 3"
    caption = window.input_caption.text()
    assert "img0.jpg" in caption
    assert "160x160" not in caption and "160" in caption  # width x height reported
    assert "Stain" in caption and "Hair" in caption, "class breakdown must be shown"


def test_input_preview_navigates(window):
    first = window.input_caption.text()
    window.input_next_btn.click()
    assert window.input_position.text() == "2 / 3"
    assert window.input_caption.text() != first
    window.input_prev_btn.click()
    assert window.input_position.text() == "1 / 3"
    assert window.input_caption.text() == first


def test_input_preview_wraps_at_both_ends(window):
    window.input_prev_btn.click()
    assert window.input_position.text() == "3 / 3"
    window.input_next_btn.click()
    assert window.input_position.text() == "1 / 3"


def test_random_lands_on_a_different_image(window):
    start = window.input_position.text()
    window.input_random_btn.click()
    assert window.input_position.text() != start, "random must move off the current image"


def test_input_preview_handles_a_dataset_with_no_annotations(qapp, tmp_path):
    from refiner.ui.main_window import MainWindow

    src = tmp_path / "empty"
    (src / "images").mkdir(parents=True)
    Image.new("RGB", (80, 60), "white").save(src / "images" / "bg.jpg")
    win = MainWindow("fallback")
    win._load_dataset(str(src))
    assert not win.input_preview.view.has_image()
    assert window_nav_disabled(win)


def window_nav_disabled(win) -> bool:
    return not (
        win.input_prev_btn.isEnabled()
        or win.input_next_btn.isEnabled()
        or win.input_random_btn.isEnabled()
    )


# --------------------------------------------------------------------------- #
# Review view toggle
# --------------------------------------------------------------------------- #


def test_review_view_toggle_picks_the_matching_artifact(qapp, window, tmp_path):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    window.accept_spin.setValue(0.99)
    run_conversion(qapp, window)

    window.review_table.selectRow(0)
    pump(qapp)
    rel = window.review_table.item(0, 0).text()
    assert window._review_view == "overlay"
    assert window._saved_overlay_for(rel).parent.parent.parent.name == "overlays"

    window.view_comparison_btn.click()
    assert window._review_view == "comparison"
    assert window._saved_overlay_for(rel).parent.parent.parent.name == "comparisons"
    assert window.review_preview.view.has_image()

    window.view_overlay_btn.click()
    assert window._saved_overlay_for(rel).parent.parent.parent.name == "overlays"


def test_comparison_view_falls_back_when_comparisons_were_not_saved(qapp, window, tmp_path):
    out = tmp_path / "out"
    window.output_edit.setText(str(out))
    window.chk_comparisons.setChecked(False)
    window.accept_spin.setValue(0.99)
    run_conversion(qapp, window)

    window.review_table.selectRow(0)
    pump(qapp)
    window.view_comparison_btn.click()
    rel = window.review_table.item(0, 0).text()
    # No comparisons/ folder exists, so it must degrade to the overlay rather than
    # showing an error.
    assert window._saved_overlay_for(rel).parent.parent.parent.name == "overlays"
    assert window.review_preview.view.has_image()
