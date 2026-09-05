"""Dark visual theme and semantic colors for the planning graph."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette


CANVAS_BACKGROUND = "#080B0F"
PANEL_BACKGROUND = "#11161C"
CONTROL_BACKGROUND = "#1A2129"
CONTROL_HOVER = "#222C36"
CONTROL_PRESSED = "#2A3744"
BORDER = "#34414E"
TEXT = "#F2F5F8"
MUTED_TEXT = "#A6B0BA"
NODE_HEADER = "#40556A"
NODE_BODY = "#252C34"
NODE_BORDER = "#526273"
SELECTION = "#4A9EFF"
GRID_DOT = (255, 255, 255, 128)

NODE_ACCENTS = {
    "Idea": "#70A7FF",
    "Goal": "#63D297",
    "Question": "#B58BFF",
    "Task": "#F3C969",
    "Evidence": "#58C8D4",
    "Decision": "#FF956D",
    "Constraint": "#FF7890",
    "Result": "#93D36E",
    "Note": "#AAB4BE",
}

EDGE_COLORS = {
    "Related": "#94A3B2",
    "Supports": "#55C77A",
    "Contradicts": "#F06B6B",
    "Depends On": "#9B8CFF",
    "Leads To": "#5BA8E6",
    "Part Of": "#C99561",
    "Refines": "#E5A24C",
    "Blocks": "#D95B5B",
}

GROUP_COLOR_PRESETS = (
    "#3B5B78",
    "#3E6B58",
    "#65517C",
    "#7A5A3C",
    "#6D4651",
    "#3E6D73",
    "#555F70",
    "#4B6541",
)


DARK_STYLESHEET = f"""
QWidget {{
    background: {PANEL_BACKGROUND};
    color: {TEXT};
    selection-background-color: {SELECTION};
    selection-color: #FFFFFF;
}}
QMainWindow, QMenuBar, QStatusBar, QToolBar {{
    background: {PANEL_BACKGROUND};
    color: {TEXT};
}}
QMenuBar::item:selected, QMenu::item:selected {{
    background: {CONTROL_HOVER};
}}
QMenu {{
    background: {PANEL_BACKGROUND};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px 6px 10px;
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 7px;
}}
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {{
    background: {CONTROL_BACKGROUND};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px;
}}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QListWidget:focus {{
    border: 1px solid {SELECTION};
}}
QComboBox QAbstractItemView {{
    background: {CONTROL_BACKGROUND};
    color: {TEXT};
    selection-background-color: {CONTROL_HOVER};
}}
QPushButton {{
    background: {CONTROL_BACKGROUND};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 10px;
}}
QPushButton:hover {{ background: {CONTROL_HOVER}; }}
QPushButton:pressed {{ background: {CONTROL_PRESSED}; }}
QSplitter::handle {{ background: {BORDER}; }}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {PANEL_BACKGROUND};
    border: none;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 24px;
    min-width: 24px;
}}
QToolTip {{
    background: {CONTROL_BACKGROUND};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
"""


def apply_dark_theme(app) -> None:
    """Apply a consistent Fusion palette and stylesheet to a QApplication."""

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(PANEL_BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(CONTROL_BACKGROUND))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL_BACKGROUND))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(CONTROL_BACKGROUND))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(CONTROL_BACKGROUND))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(SELECTION))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    app.setStyleSheet(DARK_STYLESHEET)
