APP_QSS = r"""
QWidget {
    background: #0d1520;
    color: #e7edf5;
    font-family: Inter, "Segoe UI", Arial;
    font-size: 13px;
}
QMainWindow { background: #0b121b; }
QFrame#Header { background: #111c29; border-bottom: 1px solid #26384c; }
QLabel#Title { font-size: 21px; font-weight: 700; color: #f3f7fb; }
QLabel#Subtitle { color: #95a8bc; }
QLabel#SectionTitle { font-size: 15px; font-weight: 700; color: #f1f6fb; }
QLabel#Muted { color: #8fa3b6; }
QLabel#MetricValue { font-size: 25px; font-weight: 700; color: #ffffff; }
QLabel#MetricLabel { color: #8fa3b6; font-size: 11px; }
QFrame#Card {
    background: #111c28;
    border: 1px solid #24374b;
    border-radius: 10px;
}
QFrame#DropZone {
    background: #101b27;
    border: 2px dashed #315578;
    border-radius: 12px;
}
QPushButton {
    background: #172638;
    border: 1px solid #2b425a;
    border-radius: 7px;
    padding: 8px 13px;
    font-weight: 600;
}
QPushButton:hover { background: #1d3046; border-color: #3e6284; }
QPushButton:disabled { color: #647587; background: #101923; }
QPushButton#Primary {
    background: #1677ff;
    border-color: #2b87ff;
    color: white;
    padding: 10px 18px;
}
QPushButton#Primary:hover { background: #2784ff; }
QPushButton#Success { background: #137a52; border-color: #23956a; }
QPushButton#Danger { background: #6f2a35; border-color: #9b3e4d; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    background: #0c151f;
    border: 1px solid #2b4156;
    border-radius: 6px;
    padding: 7px;
    min-height: 20px;
}
QComboBox QAbstractItemView { background: #101b27; selection-background-color: #1d5fa7; }
QProgressBar {
    border: 1px solid #293e54; border-radius: 6px; background: #0b141e; text-align: center;
}
QProgressBar::chunk { background: #1d7ff2; border-radius: 5px; }
QTableWidget, QListWidget, QTreeWidget {
    background: #0c151f;
    alternate-background-color: #101b27;
    border: 1px solid #25394d;
    border-radius: 7px;
    gridline-color: #1d2d3d;
}
QHeaderView::section {
    background: #172636; color: #cbd7e3; border: none; padding: 7px; font-weight: 600;
}
QTabWidget::pane { border: 1px solid #25394d; top: -1px; }
QTabBar::tab { background: #111c28; color: #93a7ba; padding: 10px 18px; border: 1px solid #24374b; }
QTabBar::tab:selected { background: #17283b; color: white; border-bottom: 2px solid #2f8cff; }
QCheckBox { spacing: 7px; }
QScrollBar:vertical { background: #0c151f; width: 10px; }
QScrollBar::handle:vertical { background: #2a4056; border-radius: 5px; min-height: 30px; }
"""
