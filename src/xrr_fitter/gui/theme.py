"""Palette-aware application theme built on a 4px spacing grid.

The stylesheet styles controls only; window backgrounds stay native so the
application follows the platform light and dark appearance.  Semantic state
colors are exposed through dynamic properties (``statusKind``, ``primary``,
``sectionCard``, ``sectionHeader``, ``commandBar``) so widgets opt in without
hand-written inline stylesheets.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QWidget


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    accent: str
    accent_hover: str
    accent_text: str
    surface: str
    surface_border: str
    muted_text: str
    ok: str
    warn: str
    error: str
    selection_bg: str


LIGHT_TOKENS = ThemeTokens(
    accent="#2F6BD8",
    accent_hover="#255CC0",
    accent_text="#FFFFFF",
    surface="rgba(0, 0, 0, 12)",
    surface_border="rgba(0, 0, 0, 34)",
    muted_text="rgba(0, 0, 0, 140)",
    ok="#2E7D32",
    warn="#9A6700",
    error="#B3261E",
    selection_bg="rgba(47, 107, 216, 46)",
)

DARK_TOKENS = ThemeTokens(
    accent="#5A8DEE",
    accent_hover="#6E9CF2",
    accent_text="#101418",
    surface="rgba(255, 255, 255, 14)",
    surface_border="rgba(255, 255, 255, 42)",
    muted_text="rgba(255, 255, 255, 150)",
    ok="#7BC67E",
    warn="#E3B341",
    error="#F28B82",
    selection_bg="rgba(90, 141, 238, 70)",
)

# The 4px grid the module docstring refers to.  Layouts import these instead of
# writing literals so one dock's density stays comparable to the next; the
# right-hand docks scroll inside a fixed viewport, so raising a step here costs
# vertical budget in every stacked row at once.
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16

# Point sizes for text drawn outside the stylesheet (Matplotlib artists) and for
# the few labels that need to depart from the widget default.
FONT_PT_SM = 9
FONT_PT_MD = 11
FONT_PT_LG = 14

# One interactive-control height for buttons, tool buttons and single-line
# inputs alike.  These used to carry three different minimums (22, 20, and
# unset), which is what made neighbouring controls in one row look mismatched.
CONTROL_MIN_H = 22


@dataclass(frozen=True, slots=True)
class PlotPalette:
    """Structural colours for a Matplotlib figure, as RGBA 0..1 tuples.

    A ``Figure`` is not styled by the Qt stylesheet, so the diagnostic plots
    need the palette handed to them explicitly. Only structural colours live
    here; the data series keep their fixed Okabe-Ito hues, which stay legible
    against either background.
    """

    background: tuple[float, float, float, float]
    foreground: tuple[float, float, float, float]
    muted: tuple[float, float, float, float]
    grid: tuple[float, float, float, float]


def _rgba(value: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    channels = tuple(int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    return (*channels, alpha)


LIGHT_PLOT_PALETTE = PlotPalette(
    background=_rgba("#FFFFFF"),
    foreground=_rgba("#202020"),
    muted=_rgba("#6B6B70"),
    grid=_rgba("#202020", 0.16),
)

DARK_PLOT_PALETTE = PlotPalette(
    background=_rgba("#1E1F22"),
    foreground=_rgba("#E8E8EA"),
    muted=_rgba("#A0A0A6"),
    grid=_rgba("#E8E8EA", 0.20),
)


def plot_palette(tokens: ThemeTokens) -> PlotPalette:
    """Return the figure palette matching a resolved token set."""
    return DARK_PLOT_PALETTE if tokens is DARK_TOKENS else LIGHT_PLOT_PALETTE


def palette_tokens(palette: QPalette) -> ThemeTokens:
    window = palette.color(QPalette.ColorRole.Window)
    return DARK_TOKENS if window.lightness() < 128 else LIGHT_TOKENS


def _button_styles(tokens: ThemeTokens) -> str:
    return f"""
QPushButton {{
    border: 1px solid {tokens.surface_border};
    border-radius: 6px;
    padding: {SPACE_XS}px {SPACE_MD}px;
    min-height: {CONTROL_MIN_H}px;
    background: {tokens.surface};
}}
QPushButton:hover:enabled {{ border-color: {tokens.accent}; }}
QPushButton:pressed {{ background: {tokens.selection_bg}; }}
QPushButton:disabled {{ color: {tokens.muted_text}; }}
QPushButton[primary="true"] {{
    background: {tokens.accent};
    border-color: {tokens.accent};
    color: {tokens.accent_text};
    font-weight: 600;
}}
QPushButton[primary="true"]:hover:enabled {{ background: {tokens.accent_hover}; }}
QPushButton[primary="true"]:disabled {{
    background: {tokens.surface};
    border-color: {tokens.surface_border};
    color: {tokens.muted_text};
    font-weight: 400;
}}
QPushButton[commandBar="true"] {{
    padding: {SPACE_XS}px {SPACE_SM}px;
    min-height: {CONTROL_MIN_H}px;
}}
QToolButton {{
    border: 1px solid transparent;
    border-radius: 5px;
    padding: {SPACE_XS}px {SPACE_SM}px;
    min-height: {CONTROL_MIN_H}px;
}}
QToolButton:hover:enabled {{ border-color: {tokens.surface_border}; }}
QToolButton:checked {{
    background: {tokens.selection_bg};
    border-color: {tokens.accent};
}}
"""


def _container_styles(tokens: ThemeTokens) -> str:
    return f"""
QFrame[sectionCard="true"] {{
    border: 1px solid {tokens.surface_border};
    border-radius: 8px;
    background: transparent;
}}
QLabel[sectionHeader="true"] {{
    font-weight: 700;
    padding: 2px 0px;
}}
QLabel[mutedText="true"] {{ color: {tokens.muted_text}; }}
QLabel[emptyTitle="true"] {{ font-size: {FONT_PT_LG}pt; font-weight: 700; }}
QLabel[statusKind="ok"] {{ color: {tokens.ok}; }}
QLabel[statusKind="warn"] {{ color: {tokens.warn}; }}
QLabel[statusKind="error"] {{ color: {tokens.error}; }}
QTabBar::tab {{ padding: {SPACE_XS}px {SPACE_MD}px; }}
QGroupBox {{
    border: 1px solid {tokens.surface_border};
    border-radius: 8px;
    margin-top: {SPACE_MD}px;
    padding-top: {SPACE_XS}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {SPACE_SM}px;
    padding: 0px {SPACE_XS}px;
    font-weight: 700;
}}
"""


def _chrome_styles(tokens: ThemeTokens) -> str:
    return f"""
QToolBar#mainToolbar {{
    padding: {SPACE_XS}px {SPACE_SM}px;
    spacing: {SPACE_SM}px;
    border-bottom: 1px solid {tokens.surface_border};
}}
QStatusBar#mainStatusBar {{ border-top: 1px solid {tokens.surface_border}; }}
QStatusBar#mainStatusBar QLabel {{ padding: 0px {SPACE_SM}px; }}
QProgressBar {{
    border: 1px solid {tokens.surface_border};
    border-radius: 5px;
    text-align: center;
    min-height: 14px;
}}
QProgressBar::chunk {{ background: {tokens.accent}; border-radius: 4px; }}
QTreeWidget, QTableWidget, QListWidget {{
    border: 1px solid {tokens.surface_border};
    border-radius: 6px;
    selection-background-color: {tokens.selection_bg};
}}
QHeaderView::section {{
    padding: 3px {SPACE_SM}px;
    border: 0px;
    border-bottom: 1px solid {tokens.surface_border};
    font-weight: 600;
}}
QScrollArea#analysisScroll {{ border: 0px; background: transparent; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ min-height: {CONTROL_MIN_H}px; }}
"""


def build_stylesheet(palette: QPalette) -> str:
    tokens = palette_tokens(palette)
    return "".join(
        (
            _button_styles(tokens),
            _container_styles(tokens),
            _chrome_styles(tokens),
        )
    )


def apply_theme(application: QApplication) -> str:
    """Install the palette-matched stylesheet and return the applied sheet."""
    sheet = build_stylesheet(application.palette())
    if application.styleSheet() != sheet:
        application.setStyleSheet(sheet)
    return sheet


def set_status_kind(widget: QWidget, kind: str) -> None:
    """Set the semantic status property and repolish so QSS reevaluates."""
    if kind not in ("", "ok", "warn", "error"):
        raise ValueError(f"unsupported status kind: {kind}")
    if widget.property("statusKind") == kind:
        return
    widget.setProperty("statusKind", kind)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
