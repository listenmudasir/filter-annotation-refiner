"""Minimal runtime translation layer.

Qt's .ts/lupdate toolchain needs a build step and compiled .qm files, which would
make the app harder to install for no benefit at this size. A plain catalogue keyed
by stable identifiers gives live language switching, falls back to English for any
missing entry, and can be checked by a test for missing or orphaned keys.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, Signal

#: Language code -> name shown in the selector, in that language.
LANGUAGES = {"en": "English", "zh": "繁體中文"}

DEFAULT_LANGUAGE = "en"

EN: dict[str, str] = {
    # Header / chrome
    "app.title": "Filter Annotation Refiner",
    "app.subtitle": "Detection boxes → SAM masks → smart refinement → trainable polygon segmentation",
    "app.language": "Language",
    "tab.dataset": "1  ·  Dataset",
    "tab.convert": "2  ·  Convert / Refine",
    "tab.review": "3  ·  Review / Export",

    # Dataset page
    "dataset.intro": "Start with your existing detection dataset",
    "dataset.hint": (
        "The app reads each YOLO .txt annotation, uses the box as SAM guidance, generates "
        "multiple mask candidates, scores/refines them, and saves polygon segmentation "
        "labels in a new dataset."
    ),
    "dataset.drop.title": "Drop your dataset folder here",
    "dataset.drop.hint": "Images + YOLO .txt labels are detected automatically. Originals are never overwritten.",
    "dataset.drop.button": "Choose Dataset Folder",
    "dataset.label": "Dataset:",
    "dataset.rescan": "Rescan",
    "dataset.change": "Change dataset…",
    "dataset.metric.images": "Images",
    "dataset.metric.labels": "Matched labels",
    "dataset.metric.objects": "Objects",
    "dataset.metric.boxes": "Detection boxes",
    "dataset.metric.polygons": "Existing polygons",
    "dataset.metric.problems": "Problems",
    "dataset.classes": "Class distribution",
    "dataset.classes.id": "ID",
    "dataset.classes.name": "Class",
    "dataset.classes.count": "Objects",
    "dataset.validation": "Dataset validation",
    "dataset.validation.image": "Image",
    "dataset.validation.issue": "Issue",
    "dataset.choose": "Choose a dataset to begin.",
    "dataset.scanning": "Scanning dataset…",
    "dataset.next": "Configure Conversion  →",
    "dataset.layout": "Layout: {layout}.",
    "dataset.ready": "✓ Ready: {images} images and {objects} objects can be converted.",
    "dataset.background": "{count} background image(s) without objects will be copied with empty labels.",
    "dataset.skipped": "{count} image(s) with problems will be skipped and listed in reports/failures.csv.",
    "dataset.nonames": "No class names found (no data.yaml or classes.txt) — class names will be generated.",
    "dataset.noobjects": (
        "No objects were found. Check that labels sit beside the images or in a matching labels/ folder."
    ),
    "dataset.allbad": "All {count} image(s) have validation problems; nothing can be converted.",
    "dataset.scanfailed": "Dataset scan failed",
    "dataset.isoutput": "⚠ This looks like a dataset this tool generated ({marker}). Converting it again segments the previous masks, not the original objects.",
    "dataset.isoutput.title": "This is already a converted dataset",
    "dataset.isoutput.body": "{source}\n\nappears to be output from a previous conversion ({marker}).\n\nConverting it again runs SAM on masks that were already generated, instead of on your original images, and nests the output trees. You almost certainly want the original detection dataset instead.\n\nContinue anyway?",

    # Conversion page
    "convert.title": "Automatic SAM conversion + smart refinement",
    "convert.nodataset": "No dataset loaded",
    "convert.settings": "Conversion settings",
    "convert.quality": "Refinement quality",
    "convert.preset.fast": "Fast — 1 prompt",
    "convert.preset.balanced": "Balanced — multi-prompt",
    "convert.preset.maximum": "Maximum — exhaustive candidates",
    "convert.opt.consensus": "Multi-prompt consensus",
    "convert.opt.boundary": "Image-boundary validation",
    "convert.opt.classaware": "Class-aware post-processing",
    "convert.opt.uncertainty": "Uncertainty / stability scoring",
    "convert.accept": "Auto-accept ≥",
    "convert.review": "Review warning <",
    "convert.save.masks": "Save per-instance PNG masks",
    "convert.save.overlays": "Save mask overlay image for every image",
    "convert.save.comparisons": "Save side-by-side box vs mask comparison",
    "convert.save.previews": "Save previews for uncertain images",
    "convert.save.links": "Hard-link images when possible",
    "convert.output": "Output dataset",
    "convert.browse": "Browse",
    "convert.output.hint": "A separate segmentation dataset is created. Source labels are never modified.",
    "convert.start": "▶  Start Conversion",
    "convert.preview": "Live mask preview",
    "convert.legend": "Blue: source box   Green: accepted   Amber: review",
    "convert.waiting": "Waiting to start",
    "convert.status": "Run status",
    "convert.images": "{current} / {total} images",
    "convert.estimating": "estimating…",
    "convert.rate": "{rate}s / image   •   elapsed {elapsed}   •   ETA {eta}",
    "convert.metric.objects": "Objects converted",
    "convert.metric.accepted": "High quality",
    "convert.metric.review": "Review queue",
    "convert.metric.failed": "Failed images",
    "convert.metric.skipped": "Skipped (already done)",
    "convert.resume.skip": "Resume: skip already-converted images",
    "convert.resume.tip": "Leave on to continue an interrupted run. Turn off to reconvert every image from scratch.",
    "convert.device": "Device: {device}",
    "convert.pause": "Pause",
    "convert.resume": "Resume",
    "convert.stop": "Stop Safely",
    "convert.starting": "Starting…",
    "convert.stopping": "Stopping…",
    "convert.stopped": "Stopped. Completed images are saved; re-running resumes from here.",
    "convert.finished": "Conversion finished. Outputs and quality reports are saved.",
    "convert.failed": "Conversion failed",
    "convert.unsafe.title": "Unsafe output",
    "convert.unsafe.body": (
        "Output must be outside the source dataset so generated images/labels can never be "
        "re-scanned as source data."
    ),
    "convert.stillstopping.title": "Still stopping",
    "convert.stillstopping.body": "The previous conversion has not stopped yet. Try again in a moment.",

    # Review page
    "review.title": "Review queue & export",
    "review.note": (
        "Only uncertain masks are placed here. High-quality masks are already saved as polygon "
        "labels. Review is a safety net, not the primary workflow. Every converted image also "
        "gets a mask overlay in the output's _debug_and_logs/overlays/ folder for manual checking."
    ),
    "review.refresh": "Refresh",
    "review.open.overlays": "Open Overlays",
    "review.open.output": "Open Output Folder",
    "review.open.dataset": "📁  Open Dataset",
    "review.diagnostics": "⚙  Diagnostics",
    "diag.overlays": "Mask overlays",
    "diag.comparisons": "Box vs mask comparisons",
    "diag.masks": "Per-instance PNG masks",
    "diag.previews": "Uncertain-image previews",
    "diag.reports": "Reports (CSV / summary.json)",
    "diag.root": "Output root folder",
    "review.col.image": "Image",
    "review.col.class": "Class",
    "review.col.instance": "Instance",
    "review.col.quality": "Quality",
    "review.col.sam": "SAM",
    "review.col.reason": "Reason",
    "review.select": "Select an uncertain object to inspect its saved preview.",
    "review.none": "No conversion has completed yet.",
    "review.empty": "✓ No uncertain objects are currently in the review queue.",
    "review.noqueue": "No review queue exists yet.",
    "review.count": "{count} uncertain object(s). Select a row to see its saved overlay.",
    "review.filtered": "{shown} of {total} uncertain object(s). Select a row to see its saved overlay.",
    "review.nomatch": "No objects match this filter ({total} in the queue).",
    "review.summary": (
        "{prefix}Output: {output}   •   Converted objects: {objects}   •   High quality: {accepted}"
        "   •   Review: {review}   •   Failed images: {failed}"
    ),
    "review.summary.stopped": "Stopped early — ",
    "review.nooverlay": (
        "No saved overlay found for {image}.\nLooked under {root}/_debug_and_logs/overlays and /review_previews.\n"
        "This usually means the review queue is from a different run than the currently loaded "
        "dataset and output folder."
    ),

    "input.title": "Input annotations",
    "input.empty": "Load a dataset to preview its detection annotations.",
    "input.position": "{current} / {total}",
    "input.random": "Random",
    "input.prev": "Previous annotated image",
    "input.next": "Next annotated image",
    "review.view.overlay": "SAM mask",
    "review.view.comparison": "Box vs mask",
    "review.view.comparison.tip": "Show the saved side-by-side comparison: input detection box next to the mask it produced.",

    # Review filters
    "filter.all": "All",
    "filter.disagree": "Candidates disagree",
    "filter.edge": "Weak edge support",
    "filter.leak": "Leaking outside box",
    "filter.fragmented": "Fragmented",
    "filter.fidelity": "Low polygon fidelity",
    "filter.quality": "Low quality",

    "conflict.title": "Output folder already has results",
    "conflict.body": "{conflict}",
    "conflict.suggestion": "Converting into '{folder}' instead keeps both datasets intact.",
    "conflict.usenew": "Use a new folder",
    "conflict.delete": "Delete and start over",
    "conflict.cancel": "Cancel",
    "conflict.delete.title": "Delete the existing dataset?",
    "conflict.delete.body": "This permanently deletes:\n\n{folder}\n\nIncluding its labels, masks, overlays and reports. This cannot be undone.",
    "conflict.delete.failed": "Could not delete the folder",

    # Image viewer
    "viewer.placeholder": "Live preview will appear here during conversion",
    "viewer.zoomin": "Zoom in",
    "viewer.zoomout": "Zoom out",
    "viewer.fit": "Fit",
    "viewer.fit.tip": "Fit the whole image to the panel (double-click)",
    "viewer.actual": "1:1",
    "viewer.actual.tip": "Show at 100%",
    "viewer.focus": "Focus",
    "viewer.focus.tip": (
        "Zoom to the annotated objects instead of the whole frame.\n"
        "A small defect on a 4K image is invisible when the frame is fitted."
    ),
    "viewer.hint": "Scroll to zoom · drag to pan",
    "viewer.cannotpreview": "Cannot preview {name}: {error}",
}

ZH: dict[str, str] = {
    # Traditional Chinese, Taiwan technical conventions:
    # 資料集 (not 数据集), 影像 (not 图像), 檔案 (not 文件), 品質 (not 质量),
    # 遮罩 (not 掩膜), 偵測 (not 检测), 物件 (not 目标), 設定 (not 设置).
    "app.title": "標註精修工具",
    "app.subtitle": "偵測框 → SAM 遮罩 → 智慧精修 → 可訓練的多邊形分割",
    "app.language": "語言",
    "tab.dataset": "1  ·  資料集",
    "tab.convert": "2  ·  轉換 / 精修",
    "tab.review": "3  ·  審閱 / 匯出",

    "dataset.intro": "從您現有的偵測資料集開始",
    "dataset.hint": (
        "本軟體讀取每個 YOLO .txt 標註，以偵測框作為 SAM 的提示，產生多個候選遮罩，"
        "進行評分與精修，並將多邊形分割標註儲存為新的資料集。"
    ),
    "dataset.drop.title": "將資料集資料夾拖放至此",
    "dataset.drop.hint": "自動辨識影像與 YOLO .txt 標註。原始檔案絕不會被覆寫。",
    "dataset.drop.button": "選擇資料集資料夾",
    "dataset.label": "資料集：",
    "dataset.rescan": "重新掃描",
    "dataset.change": "更換資料集…",
    "dataset.metric.images": "影像",
    "dataset.metric.labels": "已配對標註",
    "dataset.metric.objects": "物件",
    "dataset.metric.boxes": "偵測框",
    "dataset.metric.polygons": "既有多邊形",
    "dataset.metric.problems": "問題",
    "dataset.classes": "類別分布",
    "dataset.classes.id": "編號",
    "dataset.classes.name": "類別",
    "dataset.classes.count": "物件數",
    "dataset.validation": "資料集檢查",
    "dataset.validation.image": "影像",
    "dataset.validation.issue": "問題",
    "dataset.choose": "請先選擇一個資料集。",
    "dataset.scanning": "正在掃描資料集…",
    "dataset.next": "設定轉換  →",
    "dataset.layout": "目錄結構：{layout}。",
    "dataset.ready": "✓ 就緒：可轉換 {images} 張影像、{objects} 個物件。",
    "dataset.background": "{count} 張無物件的背景影像將被複製並產生空白標註。",
    "dataset.skipped": "{count} 張有問題的影像將被略過，並記錄於 _debug_and_logs/reports/failures.csv。",
    "dataset.nonames": "找不到類別名稱（缺少 data.yaml 或 classes.txt）—— 將自動產生類別名稱。",
    "dataset.noobjects": "找不到任何物件。請確認標註檔與影像位於同一目錄，或位於對應的 labels/ 資料夾中。",
    "dataset.allbad": "全部 {count} 張影像皆有檢查問題，無法進行轉換。",
    "dataset.scanfailed": "資料集掃描失敗",
    "dataset.isoutput": "⚠ 此目錄疑似本工具產生的輸出（{marker}）。再次轉換將對先前的遮罩進行分割，而非原始物件。",
    "dataset.isoutput.title": "該目錄已是轉換後的資料集",
    "dataset.isoutput.body": (
        "{source}\n\n疑似上一次轉換的輸出（{marker}）。\n\n"
        "再次轉換會讓 SAM 處理已產生的遮罩，而非原始影像，並且會產生層層巢狀的輸出目錄。"
        "您需要的很可能是原始的偵測資料集。\n\n仍要繼續嗎？"
    ),

    "convert.title": "自動 SAM 轉換 + 智慧精修",
    "convert.nodataset": "尚未載入資料集",
    "convert.settings": "轉換設定",
    "convert.quality": "精修品質",
    "convert.preset.fast": "快速 —— 單一提示",
    "convert.preset.balanced": "均衡 —— 多重提示",
    "convert.preset.maximum": "最高 —— 窮舉候選",
    "convert.opt.consensus": "多重提示一致性",
    "convert.opt.boundary": "影像邊緣驗證",
    "convert.opt.classaware": "依類別後處理",
    "convert.opt.uncertainty": "不確定性 / 穩定性評分",
    "convert.accept": "自動接受 ≥",
    "convert.review": "需審閱閾值 <",
    "convert.save.masks": "儲存每個實例的 PNG 遮罩",
    "convert.save.overlays": "為每張影像儲存遮罩疊圖",
    "convert.save.comparisons": "儲存偵測框與遮罩的並排對照圖",
    "convert.save.previews": "儲存不確定影像的預覽圖",
    "convert.save.links": "盡可能使用硬連結",
    "convert.resume.skip": "續傳：略過已轉換的影像",
    "convert.resume.tip": "保持勾選可接續上次中斷的執行；取消勾選則全部重新轉換。",
    "convert.output": "輸出資料集",
    "convert.browse": "瀏覽",
    "convert.output.hint": "將建立獨立的分割資料集。來源標註絕不會被修改。",
    "convert.start": "▶  開始轉換",
    "convert.preview": "即時遮罩預覽",
    "convert.legend": "藍色：來源偵測框   綠色：已接受   琥珀色：待審閱",
    "convert.waiting": "等待開始",
    "convert.status": "執行狀態",
    "convert.images": "{current} / {total} 張影像",
    "convert.estimating": "估算中…",
    "convert.rate": "{rate} 秒 / 張   •   已用 {elapsed}   •   預計剩餘 {eta}",
    "convert.metric.objects": "已轉換物件",
    "convert.metric.accepted": "高品質",
    "convert.metric.review": "待審閱",
    "convert.metric.failed": "失敗影像",
    "convert.metric.skipped": "已略過（先前完成）",
    "convert.device": "裝置：{device}",
    "convert.pause": "暫停",
    "convert.resume": "繼續",
    "convert.stop": "安全停止",
    "convert.starting": "正在啟動…",
    "convert.stopping": "正在停止…",
    "convert.stopped": "已停止。已完成的影像均已儲存；重新執行將從此處接續。",
    "convert.finished": "轉換完成。輸出結果與品質報告已儲存。",
    "convert.failed": "轉換失敗",
    "convert.unsafe.title": "輸出路徑不安全",
    "convert.unsafe.body": "輸出目錄必須位於來源資料集之外，以免產生的影像／標註被當作來源資料重新掃描。",
    "convert.stillstopping.title": "仍在停止中",
    "convert.stillstopping.body": "上一次轉換尚未停止，請稍候再試。",

    "review.title": "審閱佇列與匯出",
    "review.note": (
        "此處僅列出不確定的遮罩。高品質遮罩已儲存為多邊形標註。審閱是安全網，而非主要流程。"
        "每張已轉換的影像在輸出目錄的 _debug_and_logs/overlays/ 資料夾中都有遮罩疊圖，可供人工檢查。"
    ),
    "review.refresh": "重新整理",
    "review.open.overlays": "開啟疊圖",
    "review.open.output": "開啟輸出資料夾",
    "review.open.dataset": "📁  開啟資料集",
    "review.diagnostics": "⚙  診斷",
    "diag.overlays": "遮罩疊圖",
    "diag.comparisons": "偵測框與遮罩對照圖",
    "diag.masks": "各實例 PNG 遮罩",
    "diag.previews": "不確定影像預覽",
    "diag.reports": "報告（CSV / summary.json）",
    "diag.root": "輸出根目錄",
    "review.view.overlay": "SAM 遮罩",
    "review.view.comparison": "偵測框 vs 遮罩",
    "review.view.comparison.tip": "顯示已儲存的並排對照圖：輸入偵測框與其產生的遮罩。",
    "review.col.image": "影像",
    "review.col.class": "類別",
    "review.col.instance": "實例",
    "review.col.quality": "品質",
    "review.col.sam": "SAM",
    "review.col.reason": "原因",
    "review.select": "選擇一個不確定的物件以檢視其已儲存的預覽圖。",
    "review.none": "尚未完成任何轉換。",
    "review.empty": "✓ 目前審閱佇列中沒有不確定的物件。",
    "review.noqueue": "尚未產生審閱佇列。",
    "review.count": "{count} 個不確定物件。選擇一列以檢視其疊圖。",
    "review.filtered": "共 {total} 個不確定物件，目前顯示 {shown} 個。選擇一列以檢視其疊圖。",
    "review.nomatch": "沒有符合此篩選條件的物件（佇列中共 {total} 個）。",
    "review.summary": (
        "{prefix}輸出：{output}   •   已轉換物件：{objects}   •   高品質：{accepted}"
        "   •   待審閱：{review}   •   失敗影像：{failed}"
    ),
    "review.summary.stopped": "提前停止 —— ",
    "review.nooverlay": (
        "找不到 {image} 的疊圖。\n已於 {root} 下的 _debug_and_logs/overlays 與 _debug_and_logs/review_previews 中尋找。\n"
        "這通常表示審閱佇列來自與目前載入的資料集和輸出目錄不同的另一次執行。"
    ),

    "input.title": "輸入標註",
    "input.empty": "載入資料集後可在此預覽其偵測標註。",
    "input.position": "第 {current} / {total} 張",
    "input.random": "隨機",
    "input.prev": "上一張有標註的影像",
    "input.next": "下一張有標註的影像",

    "filter.all": "全部",
    "filter.disagree": "候選不一致",
    "filter.edge": "邊緣支持薄弱",
    "filter.leak": "超出偵測框",
    "filter.fragmented": "遮罩破碎",
    "filter.fidelity": "多邊形保真度低",
    "filter.quality": "品質偏低",

    "conflict.title": "輸出目錄中已有轉換結果",
    "conflict.body": "{conflict}",
    "conflict.suggestion": "改為轉換至「{folder}」可同時保留兩個資料集。",
    "conflict.usenew": "使用新目錄",
    "conflict.delete": "刪除並重新開始",
    "conflict.cancel": "取消",
    "conflict.delete.title": "確定刪除既有資料集？",
    "conflict.delete.body": "將永久刪除：\n\n{folder}\n\n其中的標註、遮罩、疊圖與報告都會一併刪除，且無法復原。",
    "conflict.delete.failed": "無法刪除該目錄",

    "viewer.placeholder": "轉換過程中將在此顯示即時預覽",
    "viewer.zoomin": "放大",
    "viewer.zoomout": "縮小",
    "viewer.fit": "符合視窗",
    "viewer.fit.tip": "將整張影像縮放至面板大小（亦可雙擊）",
    "viewer.actual": "1:1",
    "viewer.actual.tip": "以 100% 顯示",
    "viewer.focus": "聚焦",
    "viewer.focus.tip": "縮放至已標註的物件，而非整個畫面。\n在 4K 影像上，符合整個畫面時小瑕疵幾乎不可見。",
    "viewer.hint": "滾輪縮放 · 拖曳平移",
    "viewer.cannotpreview": "無法預覽 {name}：{error}",
}

CATALOGUES: dict[str, dict[str, str]] = {"en": EN, "zh": ZH}

CONFIG_PATH = Path.home() / ".config" / "filter-annotation-refiner" / "ui.json"


class Translator(QObject):
    """Holds the active language and notifies widgets to retranslate."""

    languageChanged = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._language = DEFAULT_LANGUAGE

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, code: str) -> None:
        if code not in CATALOGUES or code == self._language:
            return
        self._language = code
        self.save()
        self.languageChanged.emit(code)

    def tr(self, key: str, **kwargs) -> str:
        """Translate ``key``, falling back to English and then to the key itself."""
        text = CATALOGUES.get(self._language, EN).get(key) or EN.get(key) or key
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError):
                return text
        return text

    # ---------------- persistence ----------------
    def load(self) -> None:
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            code = data.get("language")
            if code in CATALOGUES:
                self._language = code
        except Exception:
            pass

    def save(self) -> None:
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(json.dumps({"language": self._language}), encoding="utf-8")
        except Exception:
            pass


translator = Translator()


def tr(key: str, **kwargs) -> str:
    return translator.tr(key, **kwargs)
