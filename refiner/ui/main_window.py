from __future__ import annotations

import csv
import random
import shutil
import time
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QThread, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
    QProgressBar, QPushButton, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..dataset import class_histogram, generated_output_marker, scan_dataset
from ..models import DatasetIndex
from ..services.refinement import (
    ConversionSettings, RecordOutcome, describe_output_conflict, suggest_free_output,
)
from ..i18n import LANGUAGES, tr, translator
from .theme import APP_QSS

#: (key, label) for the review-queue quick filters. The substrings match the
#: warnings the refinement engine writes, so a new warning only needs adding here.
REVIEW_FILTERS = [
    ("all", "filter.all"),
    ("disagree", "filter.disagree"),
    ("edge", "filter.edge"),
    ("leak", "filter.leak"),
    ("fragmented", "filter.fragmented"),
    ("fidelity", "filter.fidelity"),
    ("quality", "filter.quality"),
]

REVIEW_FILTER_MATCHES = {
    "disagree": ("candidate masks disagree",),
    "edge": ("weak image-edge support",),
    "leak": ("extends unusually far",),
    "fragmented": ("is fragmented",),
    "fidelity": ("only reproduces",),
    "quality": ("low overall quality",),
}
from .widgets import DatasetDropZone, ImagePreview, MetricCard
from .workers import ConversionWorker


class MainWindow(QMainWindow):
    def __init__(self, backend_name: str = "sam2", device: str | None = None, checkpoint: str | None = None):
        super().__init__()
        self.backend_name = backend_name
        self.device = device
        self.checkpoint = checkpoint
        self.dataset: DatasetIndex | None = None
        self.output_root: Path | None = None
        self.worker: ConversionWorker | None = None
        self.worker_thread: QThread | None = None
        self._run_started_at: float | None = None
        self._run_started_index: int = 0
        self._review_rows: list[dict[str, str]] = []
        self._last_stats: dict | None = None
        self._input_index = 0
        self._review_view = "overlay"
        self.resize(1500, 930)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(APP_QSS)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("Header")
        h = QHBoxLayout(header)
        h.setContentsMargins(20, 13, 20, 13)
        title_col = QVBoxLayout()
        self.title_label = QLabel()
        self.title_label.setObjectName("Title")
        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("Subtitle")
        title_col.addWidget(self.title_label)
        title_col.addWidget(self.subtitle_label)
        h.addLayout(title_col)
        h.addStretch(1)

        self.language_caption = QLabel()
        self.language_caption.setObjectName("Muted")
        h.addWidget(self.language_caption)
        self.language_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self.language_combo.addItem(name, code)
        self.language_combo.setCurrentIndex(
            max(0, self.language_combo.findData(translator.language))
        )
        self.language_combo.setMinimumWidth(110)
        self.language_combo.currentIndexChanged.connect(self._on_language_selected)
        h.addWidget(self.language_combo)
        outer.addWidget(header)

        self.tabs = QTabWidget()
        # "&" marks a keyboard mnemonic in Qt, which is why these rendered as
        # "Convert _Refine" with an underlined R. Use a separator instead.
        self.tabs.addTab(self._build_dataset_page(), "")
        self.tabs.addTab(self._build_conversion_page(), "")
        self.tabs.addTab(self._build_review_page(), "")
        outer.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

        translator.languageChanged.connect(self._retranslate)
        self._retranslate()

    def _on_language_selected(self, index: int) -> None:
        code = self.language_combo.itemData(index)
        if code:
            translator.set_language(code)

    def _retranslate(self) -> None:
        """Re-apply every user-facing string for the active language.

        Called on construction and whenever the language changes, so switching does
        not require a restart. Status lines that depend on current state are
        regenerated from that state rather than being translated in place.
        """
        self.setWindowTitle(tr("app.title"))
        self.title_label.setText(tr("app.title"))
        self.subtitle_label.setText(tr("app.subtitle"))
        self.language_caption.setText(tr("app.language"))
        for index, key in enumerate(("tab.dataset", "tab.convert", "tab.review")):
            self.tabs.setTabText(index, tr(key))

        self._retranslate_dataset_page()
        self._retranslate_conversion_page()
        self._retranslate_review_page()

    # ---------------- Dataset page ----------------
    def _build_dataset_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(14)

        self.dataset_intro = QLabel()
        self.dataset_intro.setObjectName("SectionTitle")
        layout.addWidget(self.dataset_intro)
        self.dataset_hint = QLabel()
        self.dataset_hint.setWordWrap(True)
        self.dataset_hint.setObjectName("Muted")
        layout.addWidget(self.dataset_hint)

        self.drop_zone = DatasetDropZone()
        self.drop_zone.chooseClicked.connect(self._choose_dataset)
        self.drop_zone.folderDropped.connect(self._load_dataset)
        layout.addWidget(self.drop_zone)

        path_row = QHBoxLayout()
        self.dataset_path_caption = QLabel()
        path_row.addWidget(self.dataset_path_caption)
        self.dataset_path = QLineEdit()
        self.dataset_path.setReadOnly(True)
        path_row.addWidget(self.dataset_path, 1)
        self.choose_btn = QPushButton()
        self.choose_btn.clicked.connect(self._choose_dataset)
        self.choose_btn.setVisible(False)
        path_row.addWidget(self.choose_btn)
        self.scan_btn = QPushButton()
        self.scan_btn.clicked.connect(self._rescan)
        self.scan_btn.setEnabled(False)
        path_row.addWidget(self.scan_btn)
        layout.addLayout(path_row)

        metrics = QHBoxLayout()
        self.m_images = MetricCard("")
        self.m_labels = MetricCard("")
        self.m_objects = MetricCard("")
        self.m_boxes = MetricCard("")
        self.m_polygons = MetricCard("")
        self.m_problems = MetricCard("")
        for card in [self.m_images, self.m_labels, self.m_objects, self.m_boxes, self.m_polygons, self.m_problems]:
            metrics.addWidget(card)
        layout.addLayout(metrics)

        split = QSplitter(Qt.Horizontal)
        classes_card = QFrame(); classes_card.setObjectName("Card")
        cl = QVBoxLayout(classes_card); cl.setContentsMargins(12, 12, 12, 12)
        self.classes_title = QLabel(); self.classes_title.setObjectName("SectionTitle"); cl.addWidget(self.classes_title)
        self.class_table = QTableWidget(0, 3)
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.class_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        cl.addWidget(self.class_table)
        split.addWidget(classes_card)

        issues_card = QFrame(); issues_card.setObjectName("Card")
        il = QVBoxLayout(issues_card); il.setContentsMargins(12, 12, 12, 12)
        self.issues_title = QLabel(); self.issues_title.setObjectName("SectionTitle"); il.addWidget(self.issues_title)
        self.issue_table = QTableWidget(0, 2)
        self.issue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.issue_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        il.addWidget(self.issue_table)
        split.addWidget(issues_card)

        # Looking at one parsed annotation catches a misread label format in seconds.
        # Without this the first visual confirmation arrives after a full conversion.
        input_card = QFrame(); input_card.setObjectName("Card")
        pv = QVBoxLayout(input_card); pv.setContentsMargins(12, 12, 12, 12); pv.setSpacing(8)
        self.input_title = QLabel(); self.input_title.setObjectName("SectionTitle")
        pv.addWidget(self.input_title)
        self.input_preview = ImagePreview()
        pv.addWidget(self.input_preview, 1)

        nav = QHBoxLayout(); nav.setSpacing(6)
        self.input_prev_btn = QPushButton("‹"); self.input_prev_btn.setFixedWidth(38)
        self.input_prev_btn.clicked.connect(lambda: self._step_input_preview(-1))
        self.input_next_btn = QPushButton("›"); self.input_next_btn.setFixedWidth(38)
        self.input_next_btn.clicked.connect(lambda: self._step_input_preview(1))
        self.input_random_btn = QPushButton()
        self.input_random_btn.clicked.connect(self._random_input_preview)
        self.input_position = QLabel("—"); self.input_position.setObjectName("Muted")
        nav.addWidget(self.input_prev_btn); nav.addWidget(self.input_next_btn)
        nav.addWidget(self.input_random_btn); nav.addWidget(self.input_position)
        nav.addStretch(1)
        pv.addLayout(nav)
        self.input_caption = QLabel(); self.input_caption.setObjectName("Muted")
        self.input_caption.setWordWrap(True)
        pv.addWidget(self.input_caption)
        split.addWidget(input_card)

        split.setSizes([360, 420, 720])
        layout.addWidget(split, 1)

        bottom = QHBoxLayout()
        self.dataset_status = QLabel()
        self.dataset_status.setObjectName("Muted")
        bottom.addWidget(self.dataset_status, 1)
        self.next_btn = QPushButton()
        self.next_btn.setObjectName("Primary")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        bottom.addWidget(self.next_btn)
        layout.addLayout(bottom)
        return page

    # ---------------- Input annotation preview ----------------
    def _annotated_records(self) -> list:
        """Records worth previewing: the ones that actually carry annotations."""
        if not self.dataset:
            return []
        return [r for r in self.dataset.records if r.annotations]

    def _step_input_preview(self, delta: int) -> None:
        records = self._annotated_records()
        if not records:
            return
        self._input_index = (self._input_index + delta) % len(records)
        self._show_input_preview()

    def _random_input_preview(self) -> None:
        records = self._annotated_records()
        if len(records) > 1:
            # Spot-checking a 2000-image dataset means sampling, not paging from 1.
            choices = [i for i in range(len(records)) if i != self._input_index]
            self._input_index = random.choice(choices)
        self._show_input_preview()

    def _show_input_preview(self) -> None:
        records = self._annotated_records()
        enabled = len(records) > 1
        for button in (self.input_prev_btn, self.input_next_btn, self.input_random_btn):
            button.setEnabled(enabled)

        if not records:
            self.input_preview.set_message(tr("input.empty"))
            self.input_position.setText("—")
            self.input_caption.setText("")
            return

        self._input_index = max(0, min(self._input_index, len(records) - 1))
        record = records[self._input_index]
        # No results passed, so only the source boxes are drawn - this is the input.
        self.input_preview.show_record(record, None, self.dataset.class_names)
        self.input_position.setText(
            tr("input.position", current=self._input_index + 1, total=len(records))
        )
        counts: dict[str, int] = {}
        for ann in record.annotations:
            name = self.dataset.class_names.get(ann.class_id, f"class_{ann.class_id}")
            counts[name] = counts.get(name, 0) + 1
        breakdown = ", ".join(f"{name} ×{n}" for name, n in sorted(counts.items()))
        self.input_caption.setText(
            f"{record.relative_image.as_posix()}  ·  {record.width}×{record.height}  ·  {breakdown}"
        )

    def _retranslate_dataset_page(self) -> None:
        self.dataset_intro.setText(tr("dataset.intro"))
        self.dataset_hint.setText(tr("dataset.hint"))
        self.dataset_path_caption.setText(tr("dataset.label"))
        self.scan_btn.setText(tr("dataset.rescan"))
        self.choose_btn.setText(tr("dataset.change"))
        self.next_btn.setText(tr("dataset.next"))
        self.drop_zone.retranslate()
        for card, key in (
            (self.m_images, "dataset.metric.images"),
            (self.m_labels, "dataset.metric.labels"),
            (self.m_objects, "dataset.metric.objects"),
            (self.m_boxes, "dataset.metric.boxes"),
            (self.m_polygons, "dataset.metric.polygons"),
            (self.m_problems, "dataset.metric.problems"),
        ):
            card.set_label(tr(key))
        self.input_title.setText(tr("input.title"))
        self.input_random_btn.setText(tr("input.random"))
        self.input_prev_btn.setToolTip(tr("input.prev"))
        self.input_next_btn.setToolTip(tr("input.next"))
        self._show_input_preview()
        self.classes_title.setText(tr("dataset.classes"))
        self.class_table.setHorizontalHeaderLabels(
            [tr("dataset.classes.id"), tr("dataset.classes.name"), tr("dataset.classes.count")]
        )
        self.issues_title.setText(tr("dataset.validation"))
        self.issue_table.setHorizontalHeaderLabels(
            [tr("dataset.validation.image"), tr("dataset.validation.issue")]
        )
        # Regenerate rather than translate in place: the text depends on the scan.
        if self.dataset is None:
            self.dataset_status.setText(tr("dataset.choose"))
        else:
            self._update_dataset_status()

    # ---------------- Conversion page ----------------
    def _build_conversion_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)

        top = QHBoxLayout()
        self.convert_title = QLabel()
        self.convert_title.setObjectName("SectionTitle")
        top.addWidget(self.convert_title)
        top.addStretch(1)
        self.convert_dataset_label = QLabel()
        self.convert_dataset_label.setObjectName("Muted")
        top.addWidget(self.convert_dataset_label)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)

        settings_card = QFrame(); settings_card.setObjectName("Card")
        sl = QVBoxLayout(settings_card); sl.setContentsMargins(14, 14, 14, 14); sl.setSpacing(10)
        self.settings_title = QLabel(); self.settings_title.setObjectName("SectionTitle"); sl.addWidget(self.settings_title)
        self.quality_caption = QLabel(); sl.addWidget(self.quality_caption)
        self.preset_combo = QComboBox()
        for _key, _value in (("convert.preset.fast", "fast"), ("convert.preset.balanced", "balanced"), ("convert.preset.maximum", "maximum")):
            self.preset_combo.addItem(tr(_key), _value)
        self.preset_combo.setCurrentIndex(1)
        sl.addWidget(self.preset_combo)

        self.chk_consensus = QCheckBox()
        self.chk_consensus.setChecked(True); self.chk_consensus.setEnabled(False)
        self.chk_boundary = QCheckBox()
        self.chk_boundary.setChecked(True); self.chk_boundary.setEnabled(False)
        self.chk_class = QCheckBox()
        self.chk_class.setChecked(True); self.chk_class.setEnabled(False)
        self.chk_uncertainty = QCheckBox()
        self.chk_uncertainty.setChecked(True); self.chk_uncertainty.setEnabled(False)
        for w in [self.chk_consensus, self.chk_boundary, self.chk_class, self.chk_uncertainty]: sl.addWidget(w)

        row = QHBoxLayout(); self.accept_caption = QLabel(); row.addWidget(self.accept_caption)
        self.accept_spin = QDoubleSpinBox(); self.accept_spin.setRange(0.5, 0.99); self.accept_spin.setSingleStep(0.01); self.accept_spin.setValue(0.78)
        row.addWidget(self.accept_spin); sl.addLayout(row)
        row2 = QHBoxLayout(); self.review_caption = QLabel(); row2.addWidget(self.review_caption)
        self.review_spin = QDoubleSpinBox(); self.review_spin.setRange(0.1, 0.9); self.review_spin.setSingleStep(0.01); self.review_spin.setValue(0.50)
        row2.addWidget(self.review_spin); sl.addLayout(row2)

        self.chk_masks = QCheckBox(); self.chk_masks.setChecked(True); sl.addWidget(self.chk_masks)
        self.chk_overlays = QCheckBox()
        self.chk_overlays.setChecked(True)
        self.chk_overlays.setToolTip(
            "Writes overlays/<image>.jpg with the mask drawn on every converted image, "
            "so the whole dataset can be checked by eye in any image viewer."
        )
        sl.addWidget(self.chk_overlays)
        self.chk_comparisons = QCheckBox()
        self.chk_comparisons.setChecked(True)
        self.chk_comparisons.setToolTip(
            "Writes comparisons/<image>.jpg showing the input detection boxes next to "
            "the segmentation masks they produced, for judging the conversion at a glance."
        )
        sl.addWidget(self.chk_comparisons)
        self.chk_previews = QCheckBox(); self.chk_previews.setChecked(True); sl.addWidget(self.chk_previews)
        self.chk_links = QCheckBox(); self.chk_links.setChecked(True); sl.addWidget(self.chk_links)
        self.chk_resume = QCheckBox(); self.chk_resume.setChecked(True); sl.addWidget(self.chk_resume)

        sl.addSpacing(4); self.output_caption = QLabel(); sl.addWidget(self.output_caption)
        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.browse_btn = out_btn = QPushButton()
        out_btn.clicked.connect(self._choose_output)
        out_row.addWidget(self.output_edit, 1); out_row.addWidget(out_btn)
        sl.addLayout(out_row)
        self.output_hint = QLabel()
        self.output_hint.setWordWrap(True); self.output_hint.setObjectName("Muted"); sl.addWidget(self.output_hint)
        sl.addStretch(1)
        self.start_btn = QPushButton()
        self.start_btn.setObjectName("Primary"); self.start_btn.setEnabled(False); self.start_btn.clicked.connect(self._start_conversion)
        sl.addWidget(self.start_btn)
        splitter.addWidget(settings_card)

        preview_card = QFrame(); preview_card.setObjectName("Card")
        pl = QVBoxLayout(preview_card); pl.setContentsMargins(12, 12, 12, 12)
        ptop = QHBoxLayout(); self.preview_title = QLabel(); self.preview_title.setObjectName("SectionTitle"); ptop.addWidget(self.preview_title); ptop.addStretch(1)
        self.legend_label = QLabel(); self.legend_label.setObjectName("Muted"); ptop.addWidget(self.legend_label); pl.addLayout(ptop)
        self.preview = ImagePreview(); pl.addWidget(self.preview, 1)
        self.current_status = QLabel(); self.current_status.setObjectName("Muted"); pl.addWidget(self.current_status)
        splitter.addWidget(preview_card)

        stats_card = QFrame(); stats_card.setObjectName("Card")
        rl = QVBoxLayout(stats_card); rl.setContentsMargins(14, 14, 14, 14); rl.setSpacing(10)
        self.status_title = QLabel(); self.status_title.setObjectName("SectionTitle"); rl.addWidget(self.status_title)
        self.progress = QProgressBar(); self.progress.setValue(0); rl.addWidget(self.progress)
        self.progress_text = QLabel(); self.progress_text.setObjectName("Muted"); rl.addWidget(self.progress_text)
        self.throughput_text = QLabel("—"); self.throughput_text.setObjectName("Muted"); rl.addWidget(self.throughput_text)
        self.r_objects = MetricCard("", "0")
        self.r_accepted = MetricCard("", "0")
        self.r_review = MetricCard("", "0")
        self.r_failed = MetricCard("", "0")
        self.r_skipped = MetricCard("", "0")
        for card in [self.r_objects, self.r_accepted, self.r_review, self.r_failed, self.r_skipped]: rl.addWidget(card)
        self.backend_info = QLabel()
        self.backend_info.setObjectName("Muted"); rl.addWidget(self.backend_info)
        rl.addStretch(1)
        controls = QHBoxLayout()
        self.pause_btn = QPushButton(); self.pause_btn.setEnabled(False); self.pause_btn.clicked.connect(self._pause_conversion)
        self.stop_btn = QPushButton(); self.stop_btn.setObjectName("Danger"); self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self._stop_conversion)
        controls.addWidget(self.pause_btn); controls.addWidget(self.stop_btn)
        rl.addLayout(controls)
        splitter.addWidget(stats_card)
        splitter.setSizes([330, 820, 300])
        layout.addWidget(splitter, 1)
        return page

    def _retranslate_conversion_page(self) -> None:
        self.convert_title.setText(tr("convert.title"))
        if self.dataset is None:
            self.convert_dataset_label.setText(tr("convert.nodataset"))
        self.settings_title.setText(tr("convert.quality"))
        self.quality_caption.setText(tr("convert.quality"))
        self.settings_title.setText(tr("convert.settings"))
        for index, key in enumerate(
            ("convert.preset.fast", "convert.preset.balanced", "convert.preset.maximum")
        ):
            self.preset_combo.setItemText(index, tr(key))
        self.chk_consensus.setText(tr("convert.opt.consensus"))
        self.chk_boundary.setText(tr("convert.opt.boundary"))
        self.chk_class.setText(tr("convert.opt.classaware"))
        self.chk_uncertainty.setText(tr("convert.opt.uncertainty"))
        self.accept_caption.setText(tr("convert.accept"))
        self.review_caption.setText(tr("convert.review"))
        self.chk_masks.setText(tr("convert.save.masks"))
        self.chk_overlays.setText(tr("convert.save.overlays"))
        self.chk_comparisons.setText(tr("convert.save.comparisons"))
        self.chk_previews.setText(tr("convert.save.previews"))
        self.chk_links.setText(tr("convert.save.links"))
        self.chk_resume.setText(tr("convert.resume.skip"))
        self.chk_resume.setToolTip(tr("convert.resume.tip"))
        self.output_caption.setText(tr("convert.output"))
        self.browse_btn.setText(tr("convert.browse"))
        self.output_hint.setText(tr("convert.output.hint"))
        self.start_btn.setText(tr("convert.start"))
        self.preview_title.setText(tr("convert.preview"))
        self.legend_label.setText(tr("convert.legend"))
        self.status_title.setText(tr("convert.status"))
        self.backend_info.setText(tr("convert.device", device=self.device or "auto"))
        for card, key in (
            (self.r_objects, "convert.metric.objects"),
            (self.r_accepted, "convert.metric.accepted"),
            (self.r_review, "convert.metric.review"),
            (self.r_failed, "convert.metric.failed"),
            (self.r_skipped, "convert.metric.skipped"),
        ):
            card.set_label(tr(key))
        self.pause_btn.setText(
            tr("convert.resume") if (self.worker and self.worker._pause.is_set()) else tr("convert.pause")
        )
        self.stop_btn.setText(tr("convert.stop"))
        if not self.current_status.text() or not self.is_running():
            self.current_status.setText(self.current_status.text() or tr("convert.waiting"))
        self.progress_text.setText(
            tr("convert.images", current=self.progress.value(), total=self.progress.maximum())
        )

    def _render_summary(self) -> None:
        """Run summary, rebuilt from the stored stats so it survives a language change."""
        stats = self._last_stats
        if not stats:
            self.summary_label.setText(tr("review.none"))
            return
        self.summary_label.setText(tr(
            "review.summary",
            prefix=tr("review.summary.stopped") if stats.get("cancelled") else "",
            output=stats.get("output_root", ""),
            objects=stats.get("objects", 0),
            accepted=stats.get("accepted", 0),
            review=stats.get("review", 0),
            failed=stats.get("failed", 0),
        ))

    def _retranslate_review_page(self) -> None:
        self._render_summary()
        self.review_title.setText(tr("review.title"))
        self.review_note.setText(tr("review.note"))
        self.view_overlay_btn.setText(tr("review.view.overlay"))
        self.view_comparison_btn.setText(tr("review.view.comparison"))
        self.view_comparison_btn.setToolTip(tr("review.view.comparison.tip"))
        self.refresh_review_btn.setText(tr("review.refresh"))
        self.open_dataset_btn.setText(tr("review.open.dataset"))
        self.diagnostics_btn.setText(tr("review.diagnostics"))
        for action in self.diagnostics_menu.actions():
            data = action.data()
            if data:
                action.setText(tr(data[0]))
        self._diag_root_action.setText(tr("diag.root"))
        self.review_table.setHorizontalHeaderLabels([
            tr("review.col.image"), tr("review.col.class"), tr("review.col.instance"),
            tr("review.col.quality"), tr("review.col.sam"), tr("review.col.reason"),
        ])
        if not self.review_table.selectionModel().selectedRows():
            self.review_detail.setText(tr("review.select"))
        if not self._review_rows:
            self.review_status.setText(tr("review.none"))
        else:
            self._populate_review_table()

    # ---------------- Review page ----------------
    def _build_review_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(18, 16, 18, 18); layout.setSpacing(12)
        top = QHBoxLayout(); self.review_title = QLabel(); self.review_title.setObjectName("SectionTitle"); top.addWidget(self.review_title)
        top.addStretch(1)
        # The dataset is the deliverable, so it gets the primary button; everything
        # else is diagnostic and collapses into one menu rather than a row of buttons
        # that grows with each new artifact type.
        self.open_dataset_btn = QPushButton(); self.open_dataset_btn.setObjectName("Primary")
        self.open_dataset_btn.setEnabled(False)
        self.open_dataset_btn.clicked.connect(self._open_dataset_folder)
        self.diagnostics_btn = QPushButton(); self.diagnostics_btn.setEnabled(False)
        self.diagnostics_menu = QMenu(self)
        for key, name in (
            ("diag.overlays", "overlays"),
            ("diag.comparisons", "comparisons"),
            ("diag.masks", "masks"),
            ("diag.previews", "review_previews"),
            ("diag.reports", "reports"),
        ):
            action = self.diagnostics_menu.addAction("")
            action.setData((key, name))
            action.triggered.connect(lambda _c=False, n=name: self._open_path(self._artifact_dir(n)))
        self.diagnostics_menu.addSeparator()
        self._diag_root_action = self.diagnostics_menu.addAction("")
        self._diag_root_action.triggered.connect(lambda: self._open_path(self.output_root))
        self.diagnostics_btn.setMenu(self.diagnostics_menu)
        # Kept for compatibility with existing call sites/tests.
        self.open_output_btn = self.open_dataset_btn
        self.open_overlays_btn = self.diagnostics_btn
        # Which saved artifact the right-hand panel shows.
        self.view_overlay_btn = QPushButton(); self.view_overlay_btn.setCheckable(True); self.view_overlay_btn.setChecked(True)
        self.view_overlay_btn.clicked.connect(lambda: self._set_review_view("overlay"))
        self.view_comparison_btn = QPushButton(); self.view_comparison_btn.setCheckable(True)
        self.view_comparison_btn.clicked.connect(lambda: self._set_review_view("comparison"))
        top.addWidget(self.view_overlay_btn); top.addWidget(self.view_comparison_btn)
        self.refresh_review_btn = QPushButton(); self.refresh_review_btn.clicked.connect(self._refresh_review); self.refresh_review_btn.setEnabled(False)
        top.addWidget(self.refresh_review_btn); top.addWidget(self.open_dataset_btn); top.addWidget(self.diagnostics_btn); layout.addLayout(top)
        self.review_note = QLabel()
        self.review_note.setWordWrap(True); self.review_note.setObjectName("Muted"); layout.addWidget(self.review_note)
        # Kept separate from summary_label so refreshing the queue cannot overwrite
        # the run summary (output path and counts) after a clean conversion.
        self.review_status = QLabel()
        self.review_status.setObjectName("Muted"); self.review_status.setWordWrap(True)
        layout.addWidget(self.review_status)

        filters = QHBoxLayout()
        filters.setSpacing(6)
        self.review_filter_buttons: dict[str, QPushButton] = {}
        for key, label_key in REVIEW_FILTERS:
            button = QPushButton(tr(label_key))
            button.setCheckable(True)
            button.setChecked(key == "all")
            button.clicked.connect(lambda _checked, k=key: self._set_review_filter(k))
            self.review_filter_buttons[key] = button
            filters.addWidget(button)
        filters.addStretch(1)
        layout.addLayout(filters)
        self._review_filter = "all"

        split = QSplitter(Qt.Horizontal)
        self.review_table = QTableWidget(0, 6)
        
        self.review_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.review_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.review_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.review_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.review_table.itemSelectionChanged.connect(self._review_selection_changed)
        split.addWidget(self.review_table)
        right = QFrame(); right.setObjectName("Card"); rr = QVBoxLayout(right); rr.setContentsMargins(12, 12, 12, 12)
        self.review_preview = ImagePreview(); rr.addWidget(self.review_preview, 1)
        self.review_detail = QLabel()
        self.review_detail.setWordWrap(True); self.review_detail.setObjectName("Muted"); rr.addWidget(self.review_detail)
        split.addWidget(right); split.setSizes([720, 650]); layout.addWidget(split, 1)

        self.summary_label = QLabel(); self.summary_label.setObjectName("Muted"); layout.addWidget(self.summary_label)
        return page

    # ---------------- Dataset actions ----------------
    def _choose_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select YOLO dataset folder")
        if path: self._load_dataset(path)

    def _load_dataset(self, path: str) -> None:
        if self.is_running():
            # Loading new data while a run is in flight must not leave the old run
            # writing into the old output behind the operator's back.
            if not self._cancel_running_conversion():
                QMessageBox.warning(self, tr("convert.stillstopping.title"), tr("convert.stillstopping.body"))
                return
        try:
            self.dataset_status.setText(tr("dataset.scanning"))
            self.dataset = scan_dataset(path)
        except Exception as exc:
            QMessageBox.critical(self, tr("dataset.scanfailed"), str(exc)); return
        self._input_index = 0
        # The drop zone has done its job; the space is worth more to the preview.
        self.drop_zone.setVisible(False)
        self.choose_btn.setVisible(True)
        self.dataset_path.setText(str(self.dataset.root)); self.scan_btn.setEnabled(True)
        self.convert_dataset_label.setText(self.dataset.root.name)
        self.output_root = self.dataset.root.parent / f"{self.dataset.root.name}_sam_refined"
        self.output_edit.setText(str(self.output_root))
        self._populate_dataset_summary()

    def _rescan(self) -> None:
        if self.dataset: self._load_dataset(str(self.dataset.root))

    def _populate_dataset_summary(self) -> None:
        assert self.dataset is not None
        ds = self.dataset
        problem_records = [r for r in ds.records if r.issues]
        matched = sum(r.label_path is not None and r.label_path.exists() for r in ds.records)
        self.m_images.set_value(len(ds.records)); self.m_labels.set_value(matched); self.m_objects.set_value(ds.object_count)
        self.m_boxes.set_value(ds.detection_count); self.m_polygons.set_value(ds.polygon_count)
        problem_count = len(problem_records) + len(ds.issues); self.m_problems.set_value(problem_count)

        hist = class_histogram(ds); self.class_table.setRowCount(0)
        for cid, count in hist.items():
            row = self.class_table.rowCount(); self.class_table.insertRow(row)
            self.class_table.setItem(row, 0, QTableWidgetItem(str(cid)))
            self.class_table.setItem(row, 1, QTableWidgetItem(ds.class_names.get(cid, f"class_{cid}")))
            self.class_table.setItem(row, 2, QTableWidgetItem(str(count)))
        self.issue_table.setRowCount(0)
        for issue in ds.issues:
            self._append_issue("Dataset", issue)
        for r in problem_records:
            for issue in r.issues: self._append_issue(r.relative_image.as_posix(), issue)

        # Problems no longer block the run: a handful of malformed rows in a large
        # dataset must not stop the other thousands of images from converting. Bad
        # records are skipped and reported instead.
        ready = bool([r for r in ds.convertible_records if r.annotations])
        self.next_btn.setEnabled(ready); self.start_btn.setEnabled(ready)
        self._update_dataset_status()
        self._show_input_preview()

    def _update_dataset_status(self) -> None:
        ds = self.dataset
        if ds is None:
            self.dataset_status.setText(tr("dataset.choose"))
            return
        problem_count = len(ds.problem_records) + len(ds.issues)
        convertible = [r for r in ds.convertible_records if r.annotations]

        bits = [tr("dataset.layout", layout=ds.layout)]
        marker = generated_output_marker(ds.root)
        if marker is not None:
            bits.append(tr("dataset.isoutput", marker=marker.parent.parent))
        if convertible:
            bits.append(tr("dataset.ready", images=len(convertible), objects=ds.object_count))
            if ds.background_count:
                bits.append(tr("dataset.background", count=ds.background_count))
            if problem_count:
                bits.append(tr("dataset.skipped", count=problem_count))
            if not ds.class_names:
                bits.append(tr("dataset.nonames"))
        elif ds.object_count == 0:
            bits.append(tr("dataset.noobjects"))
        else:
            bits.append(tr("dataset.allbad", count=problem_count))
        self.dataset_status.setText("  ".join(bits))

    def _append_issue(self, image: str, issue: str) -> None:
        row = self.issue_table.rowCount(); self.issue_table.insertRow(row)
        self.issue_table.setItem(row, 0, QTableWidgetItem(image)); self.issue_table.setItem(row, 1, QTableWidgetItem(issue))

    # ---------------- Conversion actions ----------------
    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output parent/folder")
        if path: self.output_edit.setText(path)

    def _settings(self) -> ConversionSettings:
        return ConversionSettings(
            preset=str(self.preset_combo.currentData()),
            accept_threshold=float(self.accept_spin.value()),
            review_threshold=float(self.review_spin.value()),
            save_masks=self.chk_masks.isChecked(),
            save_review_previews=self.chk_previews.isChecked(),
            save_overlays=self.chk_overlays.isChecked(),
            save_comparisons=self.chk_comparisons.isChecked(),
            link_images=self.chk_links.isChecked(),
        )

    def _start_conversion(self) -> None:
        if not self.dataset: return
        output = Path(self.output_edit.text()).expanduser()
        source_root = self.dataset.root.resolve()
        resolved_output = output.resolve()
        if resolved_output == source_root or resolved_output.is_relative_to(source_root):
            QMessageBox.warning(
                self, tr("convert.unsafe.title"), tr("convert.unsafe.body")
            )
            return

        marker = generated_output_marker(self.dataset.root)
        if marker is not None:
            answer = QMessageBox.warning(
                self,
                tr("dataset.isoutput.title"),
                tr("dataset.isoutput.body", source=self.dataset.root, marker=marker),
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return

        output = self._resolve_output_conflict(output)
        if output is None:
            return
        self.output_edit.setText(str(output))
        self.output_root = output
        self.start_btn.setEnabled(False); self.pause_btn.setEnabled(True); self.stop_btn.setEnabled(True)
        self.progress.setValue(0); self.current_status.setText(tr("convert.starting"))
        self._run_started_at = None; self._run_started_index = 0
        self.throughput_text.setText(tr("convert.estimating"))
        self.worker_thread = QThread(self)
        self.worker = ConversionWorker(self.dataset, output, self.backend_name, self._settings(), self.device, self.checkpoint, resume=self.chk_resume.isChecked())
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self.current_status.setText)
        self.worker.statistics.connect(self._on_statistics)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished.connect(self._on_finished)
        # Direct connection: quit() is thread-safe and must run in the worker thread.
        # Queued, it would be delivered on the GUI thread, which deadlocks whenever
        # the GUI thread is itself waiting for this thread to finish.
        self.worker.finished.connect(self.worker_thread.quit, Qt.DirectConnection)
        # Null our references before the C++ object is destroyed, so a later
        # is_running() check cannot touch a dangling QThread.
        self.worker_thread.finished.connect(self._on_thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _resolve_output_conflict(self, output: Path) -> Path | None:
        """Offer a way out when the output folder holds another backend's results.

        Checked before the worker starts so the operator gets a choice rather than
        a traceback, and so the non-destructive option is the default.
        """
        conflict = describe_output_conflict(output, self.backend_name, self._settings().preset)
        if conflict is None:
            return output

        suggestion = suggest_free_output(output)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(tr("conflict.title"))
        box.setText(tr("conflict.body", conflict=conflict))
        box.setInformativeText(tr("conflict.suggestion", folder=suggestion.name))
        use_new = box.addButton(tr("conflict.usenew"), QMessageBox.AcceptRole)
        delete = box.addButton(tr("conflict.delete"), QMessageBox.DestructiveRole)
        box.addButton(tr("conflict.cancel"), QMessageBox.RejectRole)
        box.setDefaultButton(use_new)
        box.exec()

        clicked = box.clickedButton()
        if clicked is use_new:
            return suggestion
        if clicked is not delete:
            return None

        # Deleting a whole converted dataset is irreversible, so confirm the path
        # explicitly rather than treating the first click as consent.
        confirm = QMessageBox.question(
            self,
            tr("conflict.delete.title"),
            tr("conflict.delete.body", folder=output),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return None
        try:
            shutil.rmtree(output)
        except OSError as exc:
            QMessageBox.critical(self, tr("conflict.delete.failed"), str(exc))
            return None
        return output

    def _pause_conversion(self) -> None:
        if self.worker:
            paused = self.worker.toggle_pause(); self.pause_btn.setText(tr("convert.resume") if paused else tr("convert.pause"))
            self.current_status.setText("Paused — completed images are safely saved." if paused else "Resuming…")

    def is_running(self) -> bool:
        if self.worker_thread is None:
            return False
        try:
            return self.worker_thread.isRunning()
        except RuntimeError:
            # The QThread was deleteLater'd; our Python reference outlived it.
            self.worker_thread = None
            return False

    def _on_thread_finished(self) -> None:
        """Drop references once the worker thread's event loop has exited."""
        self.worker_thread = None
        self.worker = None

    def _stop_conversion(self) -> None:
        if not self.worker:
            return
        # Unpause first: a paused worker is sitting in a wait loop and would not
        # observe the stop until it was resumed.
        self.worker._pause.clear()
        self.worker.stop()
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText(tr("convert.pause"))
        self.stop_btn.setEnabled(False)
        self.current_status.setText(tr("convert.stopping"))

    def _cancel_running_conversion(self, wait_ms: int = 15000) -> bool:
        """Stop a run and wait for the thread, so new data can be loaded at once."""
        if not self.is_running():
            return True
        self._stop_conversion()
        thread = self.worker_thread
        if thread is None:
            return True
        deadline = time.monotonic() + wait_ms / 1000
        while thread.isRunning() and time.monotonic() < deadline:
            # Keep the event loop turning: the worker's queued signals (progress,
            # statistics, finished) still need delivering while we wait.
            QApplication.processEvents()
            thread.wait(20)
        return not thread.isRunning()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        if seconds >= 3600:
            return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
        if seconds >= 60:
            return f"{seconds // 60}m {seconds % 60:02d}s"
        return f"{seconds}s"

    def _update_throughput(self, current: int, total: int) -> None:
        """Rate and ETA, measured from the first completed image.

        The first image pays model load and CUDA warm-up, so it is excluded from
        the rate rather than being allowed to inflate the estimate for the run.
        """
        now = time.monotonic()
        if self._run_started_at is None or current <= 1:
            self._run_started_at = now
            self._run_started_index = current
            self.throughput_text.setText("estimating…")
            return
        done = current - self._run_started_index
        elapsed = now - self._run_started_at
        if done <= 0 or elapsed <= 0:
            return
        per_image = elapsed / done
        remaining = max(0, total - current) * per_image
        self.throughput_text.setText(tr(
            "convert.rate",
            rate=f"{per_image:.2f}",
            elapsed=self._format_duration(elapsed),
            eta=self._format_duration(remaining),
        ))

    def _on_progress(self, current: int, total: int, payload) -> None:
        self.progress.setMaximum(max(1, total)); self.progress.setValue(current); self.progress_text.setText(tr("convert.images", current=current, total=total))
        self._update_throughput(current, total)
        names = self.dataset.class_names if self.dataset else None
        if isinstance(payload, RecordOutcome) and payload.skipped:
            # Resume skips work already on disk, so there are no results to draw.
            # Showing the record alone renders a bare box, which reads as a failed
            # conversion; show the overlay the previous run wrote instead.
            self._show_skipped_preview(payload.record)
        elif isinstance(payload, RecordOutcome):
            self.preview.show_record(payload.record, payload.results, names)
        elif isinstance(payload, tuple) and payload:
            self.preview.show_record(payload[0], [], names)

    def _show_skipped_preview(self, record) -> None:
        """Show what a previous run produced for an image resume just skipped."""
        names = self.dataset.class_names if self.dataset else None
        saved = self._saved_overlay_for(record.relative_image.as_posix())
        if saved is not None:
            try:
                with Image.open(saved) as img:
                    self.preview.show_image(img.convert("RGB"), key=str(saved))
                return
            except Exception:
                pass
        self.preview.show_record(record, [], names)

    def _on_statistics(self, stats: dict) -> None:
        self.r_objects.set_value(stats.get("objects", 0)); self.r_accepted.set_value(stats.get("accepted", 0))
        self.r_review.set_value(stats.get("review", 0)); self.r_failed.set_value(stats.get("failed", 0))
        self.r_skipped.set_value(stats.get("skipped", 0))

    def _on_worker_error(self, message: str) -> None:
        self.start_btn.setEnabled(True); self.pause_btn.setEnabled(False); self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, tr("convert.failed"), message)

    def _on_finished(self, stats: dict) -> None:
        self.start_btn.setEnabled(True); self.pause_btn.setEnabled(False); self.stop_btn.setEnabled(False); self.pause_btn.setText("Pause")
        self.open_output_btn.setEnabled(True); self.refresh_review_btn.setEnabled(True)
        self.open_overlays_btn.setEnabled(True)
        self.worker = None
        cancelled = bool(stats.get("cancelled"))
        self.current_status.setText(tr("convert.stopped") if cancelled else tr("convert.finished"))
        self._last_stats = dict(stats)
        self._render_summary()
        self._refresh_review()
        if not cancelled:
            self.tabs.setCurrentIndex(2)

    # ---------------- Review actions ----------------
    def _refresh_review(self) -> None:
        self.review_table.setRowCount(0)
        if not self.output_root: return
        reports = self._artifact_dir("reports")
        path = (reports / "review_queue.csv") if reports else self.output_root / "nonexistent"
        if not path.exists():
            self.review_status.setText(tr("review.noqueue")); return
        with path.open("r", newline="", encoding="utf-8") as f:
            self._review_rows = list(csv.DictReader(f))
        self._populate_review_table()

    @staticmethod
    def _row_matches_filter(data: dict[str, str], key: str) -> bool:
        if key == "all":
            return True
        warnings = data.get("warnings", "").lower()
        return any(needle in warnings for needle in REVIEW_FILTER_MATCHES.get(key, ()))

    def _set_review_filter(self, key: str) -> None:
        self._review_filter = key
        for other, button in self.review_filter_buttons.items():
            button.setChecked(other == key)
        self._populate_review_table()

    def _populate_review_table(self) -> None:
        self.review_table.setRowCount(0)
        rows = [r for r in self._review_rows if self._row_matches_filter(r, self._review_filter)]
        for data in rows:
            row = self.review_table.rowCount(); self.review_table.insertRow(row)
            values = [data.get("image", ""), data.get("class_name", ""), data.get("instance", ""), data.get("quality", ""), data.get("sam_score", ""), data.get("warnings", "")]
            for col, value in enumerate(values): self.review_table.setItem(row, col, QTableWidgetItem(value))

        # Show how many objects each reason accounts for, so the dominant failure
        # mode is visible without opening the CSV.
        for key, label_key in REVIEW_FILTERS:
            count = len(self._review_rows) if key == "all" else sum(
                self._row_matches_filter(r, key) for r in self._review_rows
            )
            self.review_filter_buttons[key].setText(f"{tr(label_key)} ({count})")
            self.review_filter_buttons[key].setVisible(key == "all" or count > 0)

        if not self._review_rows:
            self.review_status.setText(tr("review.empty"))
        elif not rows:
            self.review_status.setText(tr("review.nomatch", total=len(self._review_rows)))
        elif self._review_filter != "all":
            self.review_status.setText(tr("review.filtered", shown=len(rows), total=len(self._review_rows)))
        else:
            self.review_status.setText(tr("review.count", count=len(rows)))

    # ---------------- Output layout ----------------
    #: Artifact folders, newest layout first. Runs made before the dataset/ + qa/
    #: split keep working because the legacy location is still searched.
    ARTIFACT_LOCATIONS = {
        "overlays": ("qa/overlays", "overlays"),
        "comparisons": ("qa/comparisons", "comparisons"),
        "masks": ("qa/masks", "masks"),
        "review_previews": ("qa/review_previews", "review_previews"),
        "reports": ("qa/reports", "reports"),
        "dataset": ("dataset", "."),
    }

    def _artifact_dir(self, name: str) -> Path | None:
        """Where ``name`` lives in this output folder, whichever layout it uses."""
        if not self.output_root:
            return None
        for relative in self.ARTIFACT_LOCATIONS.get(name, (name,)):
            candidate = (self.output_root / relative).resolve()
            if candidate.is_dir():
                return candidate
        return None

    def _open_path(self, path: Path | None) -> None:
        if path is not None and path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_dataset_folder(self) -> None:
        self._open_path(self._artifact_dir("dataset") or self.output_root)

    def _saved_overlay_for(self, image_rel: str) -> Path | None:
        """Artifact to display for this object, honouring the chosen view mode.

        Both views are files written during the conversion, so what is reviewed on
        screen is exactly what is on disk - no second rendering path to drift.
        """
        if not self.output_root:
            return None
        relative = Path(image_rel).with_suffix(".jpg")
        if self._review_view == "comparison":
            names = ("comparisons", "overlays", "review_previews")
        else:
            names = ("overlays", "review_previews", "comparisons")
        for name in names:
            folder = self._artifact_dir(name)
            if folder is None:
                continue
            candidate = folder / relative
            if candidate.exists():
                return candidate
        return None

    def _set_review_view(self, mode: str) -> None:
        self._review_view = mode
        self.view_overlay_btn.setChecked(mode == "overlay")
        self.view_comparison_btn.setChecked(mode == "comparison")
        self._review_selection_changed()

    def _review_selection_changed(self) -> None:
        rows = self.review_table.selectionModel().selectedRows()
        if not rows or not self.output_root: return
        row = rows[0].row(); image_rel = self.review_table.item(row, 0).text()
        instance = self.review_table.item(row, 2).text()
        record = next(
            (r for r in (self.dataset.records if self.dataset else []) if r.relative_image.as_posix() == image_rel),
            None,
        )
        # Zoom to the object under review, not to the whole frame: the saved overlay
        # is a baked image so the viewer cannot work the boxes out for itself.
        focus = None
        if record is not None:
            try:
                ann = record.annotations[int(instance)]
                focus = self.review_preview._boxes_rect([ann.bbox_xyxy])
            except (ValueError, IndexError):
                focus = self.review_preview._boxes_rect([a.bbox_xyxy for a in record.annotations])

        overlay = self._saved_overlay_for(image_rel)
        if overlay is not None:
            try:
                with Image.open(overlay) as img:
                    self.review_preview.show_image(img.convert("RGB"), key=str(overlay), focus_rect=focus)
            except Exception as exc:
                self.review_preview.set_message(f"Cannot open {overlay.name}: {exc}")
        elif record is not None:
            self.review_preview.show_record(record, [], self.dataset.class_names)
        else:
            self.review_preview.set_message(
                tr("review.nooverlay", image=image_rel, root=self.output_root)
            )
        self.review_detail.setText(
            f"{image_rel} • class {self.review_table.item(row,1).text()} • instance {self.review_table.item(row,2).text()}\nQuality: {self.review_table.item(row,3).text()} • SAM: {self.review_table.item(row,4).text()}\n{self.review_table.item(row,5).text()}"
        )

    def _open_output(self) -> None:
        if self.output_root and self.output_root.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_root)))

    def _open_overlays(self) -> None:
        if not self.output_root:
            return
        overlays = self.output_root / "overlays"
        target = overlays if overlays.exists() else self.output_root
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
