from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from ..i18n import tr, translator
from ..models import ImageRecord, RefinementResult
from ..overlay import render_overlay


class MetricCard(QFrame):
    def __init__(self, label: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        self.value = QLabel(value)
        self.value.setObjectName("MetricValue")
        self.label = QLabel(label)
        self.label.setObjectName("MetricLabel")
        layout.addWidget(self.value)
        layout.addWidget(self.label)

    def set_value(self, value: str | int) -> None:
        self.value.setText(str(value))

    def set_label(self, text: str) -> None:
        self.label.setText(text)


class DatasetDropZone(QFrame):
    folderDropped = Signal(str)
    chooseClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 24, 22, 24)
        layout.setSpacing(8)
        icon = QLabel("▣")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("font-size: 34px; color: #64a9ff;")
        self.title = QLabel()
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.hint = QLabel()
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)
        self.hint.setObjectName("Muted")
        self.button = QPushButton()
        self.button.setObjectName("Primary")
        self.button.clicked.connect(self.chooseClicked.emit)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.button)
        row.addStretch(1)
        layout.addWidget(icon)
        layout.addWidget(self.title)
        layout.addWidget(self.hint)
        layout.addSpacing(6)
        layout.addLayout(row)
        self.retranslate()

    def retranslate(self) -> None:
        self.title.setText(tr("dataset.drop.title"))
        self.hint.setText(tr("dataset.drop.hint"))
        self.button.setText(tr("dataset.drop.button"))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and any(Path(u.toLocalFile()).is_dir() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                self.folderDropped.emit(str(path))
                event.acceptProposedAction()
                return


class _ZoomView(QGraphicsView):
    """Scrollable canvas with wheel zoom anchored under the cursor."""

    zoomChanged = Signal(float)

    MIN_SCALE = 0.02
    MAX_SCALE = 40.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self.item)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("background:#08111a; border:1px solid #24384c; border-radius:8px;")
        self.setMinimumSize(420, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    @property
    def scale_factor(self) -> float:
        return float(self.transform().m11())

    def has_image(self) -> bool:
        return not self.item.pixmap().isNull()

    def set_pixmap(self, pixmap: QPixmap, keep_view: bool) -> None:
        previous = self.transform()
        centre = self.mapToScene(self.viewport().rect().center())
        self.item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        if keep_view and not pixmap.isNull():
            self.setTransform(previous)
            self.centerOn(centre)
        else:
            self.fit()

    def clear(self) -> None:
        self.item.setPixmap(QPixmap())
        self._scene.setSceneRect(QRectF())

    def fit(self) -> None:
        if not self.has_image():
            return
        self.fitInView(self.item, Qt.KeepAspectRatio)
        self.zoomChanged.emit(self.scale_factor)

    def fit_rect(self, rect: QRectF) -> None:
        """Fit a sub-region, clamped to the image."""
        if not self.has_image():
            return
        bounds = QRectF(self.item.pixmap().rect())
        rect = rect.intersected(bounds)
        if rect.width() < 1 or rect.height() < 1:
            self.fit()
            return
        self.fitInView(rect, Qt.KeepAspectRatio)
        self.zoomChanged.emit(self.scale_factor)

    def reset_to_actual_size(self) -> None:
        if not self.has_image():
            return
        self.setTransform(self.transform().fromScale(1.0, 1.0))
        self.zoomChanged.emit(self.scale_factor)

    def zoom_by(self, factor: float) -> None:
        if not self.has_image():
            return
        target = self.scale_factor * factor
        # Clamp so the image can never be scaled into invisibility or overflow.
        if target < self.MIN_SCALE:
            factor = self.MIN_SCALE / self.scale_factor
        elif target > self.MAX_SCALE:
            factor = self.MAX_SCALE / self.scale_factor
        if abs(factor - 1.0) < 1e-9:
            return
        self.scale(factor, factor)
        self.zoomChanged.emit(self.scale_factor)

    def wheelEvent(self, event) -> None:
        if not self.has_image():
            return
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(1.0015 ** delta)
            event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()
        event.accept()


class ImagePreview(QWidget):
    """Image panel with zoom, pan and fit controls.

    Wheel zooms under the cursor, drag pans, double-click fits. Once the operator
    has zoomed, the view is held across image changes so a position of interest is
    not lost every time the live preview advances to the next image.
    """

    def __init__(self, parent=None, placeholder: str | None = None):
        super().__init__(parent)
        self._placeholder_key = "viewer.placeholder" if placeholder is None else None
        self._user_zoomed = False
        self._current_key: str | None = None
        self._focus_rect: QRectF | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.view = _ZoomView(self)
        self.view.zoomChanged.connect(self._on_zoom_changed)
        layout.addWidget(self.view, 1)

        self.placeholder = QLabel(placeholder or tr("viewer.placeholder"))
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setObjectName("Muted")
        self.placeholder.setWordWrap(True)
        layout.addWidget(self.placeholder)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        self.zoom_out_btn = self._tool("−", lambda: self._manual_zoom(1 / 1.25))
        self.zoom_in_btn = self._tool("+", lambda: self._manual_zoom(1.25))
        self.fit_btn = self._tool(tr("viewer.fit"), self.fit)
        self.actual_btn = self._tool(tr("viewer.actual"), self._actual_size)
        self.focus_btn = QPushButton(tr("viewer.focus"))
        self.focus_btn.setCheckable(True)
        self.focus_btn.setChecked(True)
        self.focus_btn.toggled.connect(self._on_focus_toggled)
        self.zoom_label = QLabel("—")
        self.zoom_label.setObjectName("Muted")
        self.zoom_label.setMinimumWidth(54)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        for widget in (
            self.zoom_out_btn, self.zoom_in_btn, self.zoom_label,
            self.fit_btn, self.actual_btn, self.focus_btn,
        ):
            controls.addWidget(widget)
        controls.addStretch(1)
        self.hint = QLabel()
        self.hint.setObjectName("Muted")
        controls.addWidget(self.hint)
        layout.addLayout(controls)
        self._set_controls_enabled(False)
        self.retranslate()
        translator.languageChanged.connect(self.retranslate)

    def _tool(self, text: str, slot) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(slot)
        return button

    def retranslate(self) -> None:
        self.fit_btn.setText(tr("viewer.fit"))
        self.actual_btn.setText(tr("viewer.actual"))
        self.focus_btn.setText(tr("viewer.focus"))
        self.hint.setText(tr("viewer.hint"))
        self.zoom_out_btn.setToolTip(tr("viewer.zoomout"))
        self.zoom_in_btn.setToolTip(tr("viewer.zoomin"))
        self.fit_btn.setToolTip(tr("viewer.fit.tip"))
        self.actual_btn.setToolTip(tr("viewer.actual.tip"))
        self.focus_btn.setToolTip(tr("viewer.focus.tip"))
        if self._placeholder_key and not self.view.has_image():
            self.placeholder.setText(tr(self._placeholder_key))
        # Widths depend on the translated text; Chinese labels are wider per glyph.
        for button in (self.zoom_out_btn, self.zoom_in_btn, self.fit_btn, self.actual_btn, self.focus_btn):
            button.setMinimumWidth(button.fontMetrics().horizontalAdvance(button.text()) + 24)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for button in (self.zoom_out_btn, self.zoom_in_btn, self.fit_btn, self.actual_btn):
            button.setEnabled(enabled)

    # ---------------- zoom ----------------
    def _manual_zoom(self, factor: float) -> None:
        self._user_zoomed = True
        self.view.zoom_by(factor)

    def _actual_size(self) -> None:
        self._user_zoomed = True
        self.view.reset_to_actual_size()

    def fit(self) -> None:
        self._user_zoomed = False
        self._apply_default_view()

    def _on_focus_toggled(self, _checked: bool) -> None:
        self._user_zoomed = False
        self._apply_default_view()

    def _apply_default_view(self) -> None:
        """Fit to the objects when Focus is on, otherwise to the whole frame."""
        if self.focus_btn.isChecked() and self._focus_rect is not None:
            self.view.fit_rect(self._focus_rect)
        else:
            self.view.fit()

    def _on_zoom_changed(self, scale: float) -> None:
        self.zoom_label.setText(f"{scale * 100:.0f}%")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._user_zoomed:
            self._apply_default_view()

    # ---------------- content ----------------
    @staticmethod
    def _pil_to_pixmap(image: Image.Image) -> QPixmap:
        arr = np.ascontiguousarray(np.asarray(image.convert("RGB")))
        h, w, _ = arr.shape
        return QPixmap.fromImage(QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy())

    def set_message(self, message: str) -> None:
        """Show a message *instead of* any image.

        The previous pixmap must be cleared: leaving it visible alongside an error
        shows the last-selected object's mask next to a caption saying there is no
        mask, which is worse than showing nothing.
        """
        self.placeholder.setText(message)
        self.placeholder.setVisible(True)
        self.view.clear()
        self._current_key = None
        self._set_controls_enabled(False)
        self.zoom_label.setText("—")

    @staticmethod
    def _boxes_rect(boxes: list[tuple[float, float, float, float]], margin: float = 0.6) -> QRectF | None:
        """Padded bounding rect of the annotated objects."""
        if not boxes:
            return None
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        pad = max(24.0, margin * max(x1 - x0, y1 - y0))
        return QRectF(x0 - pad, y0 - pad, (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad)

    def show_image(
        self,
        image: Image.Image,
        key: str | None = None,
        focus_rect: QRectF | None = None,
    ) -> None:
        pixmap = self._pil_to_pixmap(image)
        same_image = key is not None and key == self._current_key
        self._current_key = key
        self._focus_rect = focus_rect
        self.view.set_pixmap(pixmap, keep_view=self._user_zoomed or same_image)
        self.placeholder.setVisible(False)
        self._set_controls_enabled(True)
        if not (self._user_zoomed or same_image):
            self._apply_default_view()
        self._on_zoom_changed(self.view.scale_factor)

    def show_record(
        self,
        record: ImageRecord,
        results: list[RefinementResult] | None = None,
        class_names: dict[int, str] | None = None,
    ) -> None:
        try:
            with Image.open(record.image_path) as img:
                canvas = render_overlay(img, record.annotations, results, class_names)
        except Exception as exc:
            self.set_message(tr("viewer.cannotpreview", name=record.image_path.name, error=exc))
            return
        self.show_image(
            canvas,
            key=str(record.image_path),
            focus_rect=self._boxes_rect([a.bbox_xyxy for a in record.annotations]),
        )
