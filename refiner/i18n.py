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
LANGUAGES = {"en": "English", "zh": "中文"}

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
        "gets a mask overlay in the output's overlays/ folder for manual checking."
    ),
    "review.refresh": "Refresh",
    "review.open.overlays": "Open Overlays",
    "review.open.output": "Open Output Folder",
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
        "No saved overlay found for {image}.\nLooked under {root}/overlays and /review_previews.\n"
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
    "app.title": "标注精修工具",
    "app.subtitle": "检测框 → SAM 掩膜 → 智能优化 → 可训练的多边形分割",
    "app.language": "语言",
    "tab.dataset": "1  ·  数据集",
    "tab.convert": "2  ·  转换 / 优化",
    "tab.review": "3  ·  审查 / 导出",

    "dataset.intro": "从现有的检测数据集开始",
    "dataset.hint": (
        "本软件读取每个 YOLO .txt 标注，以检测框作为 SAM 的提示，生成多个候选掩膜，"
        "进行评分与优化，并将多边形分割标注保存为新的数据集。"
    ),
    "dataset.drop.title": "将数据集文件夹拖放到此处",
    "dataset.drop.hint": "自动识别图像与 YOLO .txt 标注。原始文件绝不会被覆盖。",
    "dataset.drop.button": "选择数据集文件夹",
    "dataset.label": "数据集：",
    "dataset.rescan": "重新扫描",
    "dataset.change": "更换数据集…",
    "dataset.metric.images": "图像",
    "dataset.metric.labels": "匹配的标注",
    "dataset.metric.objects": "目标",
    "dataset.metric.boxes": "检测框",
    "dataset.metric.polygons": "已有多边形",
    "dataset.metric.problems": "问题",
    "dataset.classes": "类别分布",
    "dataset.classes.id": "编号",
    "dataset.classes.name": "类别",
    "dataset.classes.count": "目标数",
    "dataset.validation": "数据集校验",
    "dataset.validation.image": "图像",
    "dataset.validation.issue": "问题",
    "dataset.choose": "请先选择一个数据集。",
    "dataset.scanning": "正在扫描数据集…",
    "dataset.next": "配置转换  →",
    "dataset.layout": "目录结构：{layout}。",
    "dataset.ready": "✓ 就绪：可转换 {images} 张图像、{objects} 个目标。",
    "dataset.background": "{count} 张无目标的背景图像将被复制并生成空标注。",
    "dataset.skipped": "{count} 张有问题的图像将被跳过，并记录在 reports/failures.csv 中。",
    "dataset.nonames": "未找到类别名称（缺少 data.yaml 或 classes.txt）—— 将自动生成类别名称。",
    "dataset.noobjects": "未找到任何目标。请确认标注文件与图像同目录，或位于对应的 labels/ 文件夹中。",
    "dataset.allbad": "全部 {count} 张图像均存在校验问题，无法进行转换。",
    "dataset.scanfailed": "数据集扫描失败",
    "dataset.isoutput": "⚠ 此目录疑似本工具生成的输出（{marker}）。再次转换将对先前的掩膜进行分割，而非原始目标。",
    "dataset.isoutput.title": "该目录已是转换后的数据集",
    "dataset.isoutput.body": "{source}\n\n疑似上一次转换的输出（{marker}）。\n\n再次转换会让 SAM 处理已生成的掩膜，而不是原始图像，并且会产生层层嵌套的输出目录。您需要的很可能是原始的检测数据集。\n\n仍要继续吗？",

    "convert.title": "自动 SAM 转换 + 智能优化",
    "convert.nodataset": "尚未载入数据集",
    "convert.settings": "转换设置",
    "convert.quality": "优化质量",
    "convert.preset.fast": "快速 —— 单一提示",
    "convert.preset.balanced": "均衡 —— 多提示",
    "convert.preset.maximum": "最高 —— 穷举候选",
    "convert.opt.consensus": "多提示一致性",
    "convert.opt.boundary": "图像边缘校验",
    "convert.opt.classaware": "按类别后处理",
    "convert.opt.uncertainty": "不确定性 / 稳定性评分",
    "convert.accept": "自动接受 ≥",
    "convert.review": "需审查阈值 <",
    "convert.save.masks": "保存每个实例的 PNG 掩膜",
    "convert.save.overlays": "为每张图像保存掩膜叠加图",
    "convert.save.comparisons": "保存检测框与掩膜的并排对比图",
    "convert.save.previews": "保存不确定图像的预览图",
    "convert.save.links": "尽可能使用硬链接",
    "convert.output": "输出数据集",
    "convert.browse": "浏览",
    "convert.output.hint": "将创建独立的分割数据集。源标注绝不会被修改。",
    "convert.start": "▶  开始转换",
    "convert.preview": "实时掩膜预览",
    "convert.legend": "蓝色：源检测框   绿色：已接受   琥珀色：待审查",
    "convert.waiting": "等待开始",
    "convert.status": "运行状态",
    "convert.images": "{current} / {total} 张图像",
    "convert.estimating": "正在估算…",
    "convert.rate": "{rate} 秒 / 张   •   已用 {elapsed}   •   预计剩余 {eta}",
    "convert.metric.objects": "已转换目标",
    "convert.metric.accepted": "高质量",
    "convert.metric.review": "待审查",
    "convert.metric.failed": "失败图像",
    "convert.metric.skipped": "已跳过（先前已完成）",
    "convert.resume.skip": "断点续传：跳过已转换的图像",
    "convert.resume.tip": "保持勾选可继续上次中断的运行；取消勾选则全部重新转换。",
    "convert.device": "设备：{device}",
    "convert.pause": "暂停",
    "convert.resume": "继续",
    "convert.stop": "安全停止",
    "convert.starting": "正在启动…",
    "convert.stopping": "正在停止…",
    "convert.stopped": "已停止。已完成的图像均已保存；重新运行将从此处继续。",
    "convert.finished": "转换完成。输出结果与质量报告已保存。",
    "convert.failed": "转换失败",
    "convert.unsafe.title": "输出路径不安全",
    "convert.unsafe.body": "输出目录必须位于源数据集之外，以免生成的图像/标注被当作源数据重新扫描。",
    "convert.stillstopping.title": "仍在停止中",
    "convert.stillstopping.body": "上一次转换尚未停止，请稍候再试。",

    "review.title": "审查队列与导出",
    "review.note": (
        "此处仅列出不确定的掩膜。高质量掩膜已保存为多边形标注。审查是安全网，而非主要流程。"
        "每张已转换的图像在输出目录的 overlays/ 文件夹中都有掩膜叠加图，可供人工检查。"
    ),
    "review.refresh": "刷新",
    "review.open.overlays": "打开叠加图",
    "review.open.output": "打开输出文件夹",
    "review.col.image": "图像",
    "review.col.class": "类别",
    "review.col.instance": "实例",
    "review.col.quality": "质量",
    "review.col.sam": "SAM",
    "review.col.reason": "原因",
    "review.select": "选择一个不确定的目标以查看已保存的预览图。",
    "review.none": "尚未完成任何转换。",
    "review.empty": "✓ 当前审查队列中没有不确定的目标。",
    "review.noqueue": "尚未生成审查队列。",
    "review.count": "{count} 个不确定目标。选择一行以查看其叠加图。",
    "review.filtered": "共 {total} 个不确定目标，当前显示 {shown} 个。选择一行以查看其叠加图。",
    "review.nomatch": "没有符合此筛选条件的目标（队列中共 {total} 个）。",
    "review.summary": (
        "{prefix}输出：{output}   •   已转换目标：{objects}   •   高质量：{accepted}"
        "   •   待审查：{review}   •   失败图像：{failed}"
    ),
    "review.summary.stopped": "提前停止 —— ",
    "review.nooverlay": (
        "未找到 {image} 的叠加图。\n已在 {root}/overlays 与 /review_previews 中查找。\n"
        "这通常表示审查队列来自与当前载入的数据集和输出目录不同的另一次运行。"
    ),

    "input.title": "输入标注",
    "input.empty": "载入数据集后可在此预览其检测标注。",
    "input.position": "第 {current} / {total} 张",
    "input.random": "随机",
    "input.prev": "上一张有标注的图像",
    "input.next": "下一张有标注的图像",
    "review.view.overlay": "SAM 掩膜",
    "review.view.comparison": "检测框 vs 掩膜",
    "review.view.comparison.tip": "显示已保存的并排对比图：输入检测框与其生成的掩膜。",

    "filter.all": "全部",
    "filter.disagree": "候选不一致",
    "filter.edge": "边缘支持弱",
    "filter.leak": "超出检测框",
    "filter.fragmented": "掩膜碎片化",
    "filter.fidelity": "多边形保真度低",
    "filter.quality": "质量偏低",

    "conflict.title": "输出目录中已有转换结果",
    "conflict.body": "{conflict}",
    "conflict.suggestion": "改为转换到“{folder}”可同时保留两个数据集。",
    "conflict.usenew": "使用新目录",
    "conflict.delete": "删除并重新开始",
    "conflict.cancel": "取消",
    "conflict.delete.title": "确定删除已有数据集？",
    "conflict.delete.body": "将永久删除：\n\n{folder}\n\n其中的标注、掩膜、叠加图与报告都会一并删除，且无法恢复。",
    "conflict.delete.failed": "无法删除该目录",

    "viewer.placeholder": "转换过程中将在此显示实时预览",
    "viewer.zoomin": "放大",
    "viewer.zoomout": "缩小",
    "viewer.fit": "适应",
    "viewer.fit.tip": "将整张图像缩放至面板大小（双击亦可）",
    "viewer.actual": "1:1",
    "viewer.actual.tip": "以 100% 显示",
    "viewer.focus": "聚焦",
    "viewer.focus.tip": "缩放到已标注的目标，而非整幅画面。\n在 4K 图像上，适应整幅画面时小缺陷几乎不可见。",
    "viewer.hint": "滚轮缩放 · 拖动平移",
    "viewer.cannotpreview": "无法预览 {name}：{error}",
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
