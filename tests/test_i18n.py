"""Translation catalogue and live language switching."""

from __future__ import annotations

import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from refiner.i18n import CATALOGUES, EN, LANGUAGES, ZH, Translator  # noqa: E402
from refiner.ui.main_window import REVIEW_FILTERS  # noqa: E402
from refiner.ui.widgets import ImagePreview  # noqa: E402

PLACEHOLDER = re.compile(r"\{(\w+)\}")


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_every_language_has_a_catalogue():
    assert set(LANGUAGES) == set(CATALOGUES)


def test_chinese_covers_every_english_key():
    missing = sorted(set(EN) - set(ZH))
    assert not missing, f"untranslated keys: {missing}"


def test_no_orphaned_translations():
    orphans = sorted(set(ZH) - set(EN))
    assert not orphans, f"keys translated but no longer used: {orphans}"


def test_placeholders_match_across_languages():
    """A mismatched {placeholder} would raise or silently drop data at runtime."""
    for key, english in EN.items():
        assert PLACEHOLDER.findall(english).sort() == PLACEHOLDER.findall(ZH[key]).sort(), key


def test_review_filter_labels_are_translation_keys():
    for _key, label_key in REVIEW_FILTERS:
        assert label_key in EN, label_key


def test_translator_falls_back_to_english_then_key():
    t = Translator()
    t.set_language("zh")
    assert t.tr("app.language") == ZH["app.language"]
    # A key present only in English still resolves.
    assert t.tr("viewer.actual") == ZH["viewer.actual"]
    assert t.tr("does.not.exist") == "does.not.exist"


def test_translator_formats_placeholders():
    t = Translator()
    assert "7" in t.tr("convert.images", current=7, total=9)


def test_translator_survives_missing_placeholder_argument():
    """A formatting error must not crash the UI mid-conversion."""
    t = Translator()
    assert t.tr("convert.images") == EN["convert.images"]


def test_language_change_emits_once():
    t = Translator()
    seen = []
    t.languageChanged.connect(seen.append)
    t.set_language("zh")
    t.set_language("zh")  # already active
    t.set_language("nope")  # unknown
    assert seen == ["zh"]


def test_window_retranslates_live(qapp, monkeypatch, tmp_path):
    from refiner import i18n
    from refiner.ui.main_window import MainWindow

    monkeypatch.setattr(i18n, "CONFIG_PATH", tmp_path / "ui.json")
    i18n.translator.set_language("en")
    win = MainWindow("fallback")
    assert win.tabs.tabText(0) == EN["tab.dataset"]
    assert win.start_btn.text() == EN["convert.start"]
    assert win.review_title.text() == EN["review.title"]

    i18n.translator.set_language("zh")
    assert win.tabs.tabText(0) == ZH["tab.dataset"]
    assert win.start_btn.text() == ZH["convert.start"]
    assert win.review_title.text() == ZH["review.title"]
    assert win.m_images.label.text() == ZH["dataset.metric.images"]
    assert win.drop_zone.button.text() == ZH["dataset.drop.button"]

    i18n.translator.set_language("en")
    assert win.start_btn.text() == EN["convert.start"]


def test_language_selector_lists_both_languages(qapp, monkeypatch, tmp_path):
    from refiner import i18n
    from refiner.ui.main_window import MainWindow

    monkeypatch.setattr(i18n, "CONFIG_PATH", tmp_path / "ui.json")
    i18n.translator.set_language("en")
    win = MainWindow("fallback")
    codes = [win.language_combo.itemData(i) for i in range(win.language_combo.count())]
    assert codes == list(LANGUAGES)
    # The name of each language is shown in that language, not translated.
    assert win.language_combo.itemText(codes.index("zh")) == "中文"


def test_language_choice_is_persisted(monkeypatch, tmp_path):
    from refiner import i18n

    config = tmp_path / "ui.json"
    monkeypatch.setattr(i18n, "CONFIG_PATH", config)
    t = Translator()
    t.set_language("zh")
    assert config.exists()

    restored = Translator()
    restored.load()
    assert restored.language == "zh"


def test_viewer_buttons_retranslate(qapp, monkeypatch, tmp_path):
    from refiner import i18n

    monkeypatch.setattr(i18n, "CONFIG_PATH", tmp_path / "ui.json")
    i18n.translator.set_language("en")
    preview = ImagePreview()
    assert preview.fit_btn.text() == EN["viewer.fit"]
    i18n.translator.set_language("zh")
    assert preview.fit_btn.text() == ZH["viewer.fit"]
    # Chinese glyphs are wider; the button must grow rather than clip.
    assert preview.focus_btn.minimumWidth() >= preview.focus_btn.fontMetrics().horizontalAdvance(
        preview.focus_btn.text()
    )
    i18n.translator.set_language("en")
