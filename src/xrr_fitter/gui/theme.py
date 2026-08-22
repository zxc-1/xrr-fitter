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
    info: str
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
    info="#1565C0",
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
    info="#8AB4F8",
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

# The side of one cell in the plot's floating mode bar.  That bar is the one
# place a button holds a glyph and nothing else, so it opts out of CONTROL_MIN_H
# and the general button padding: a square cell a little larger than the glyph
# keeps the bar a narrow strip over the data instead of a slab across it.
GLYPH_CELL_PX = 24

# Unlike the tokens above, these do not follow the appearance.  A series that
# changed hue when the desktop switched to dark would invalidate the reference a
# user has already built ("the blue curve is my data"), so each hue is instead
# chosen to survive both canvas backgrounds.  Okabe-Ito throughout, which keeps
# the set separable under the common colour vision deficiencies.
DATA_OBSERVED = "#0072B2"
DATA_CANDIDATE = "#D55E00"
DATA_PREVIEW = "#009E73"
DATA_RANGE = "#E69F00"

# Cycled per component in the structure diagram so neighbouring layers separate
# at a glance.  The first three are the series hues above: a reader who has
# learnt "blue is the data" loses nothing by meeting blue again as a layer,
# whereas a second unrelated six-colour set would be six more things to learn.
DATA_SEQUENCE = (DATA_OBSERVED, DATA_CANDIDATE, DATA_PREVIEW, "#CC79A7", DATA_RANGE, "#56B4E9")

# Air and the substrate are semi-infinite, so they are not layers and must not
# borrow a sequence hue.
DATA_NEUTRAL = "#9AA0A6"

# Captions drawn on top of a fill.  Two are needed because the fills span most
# of the luminance range: white reaches only 2.25:1 on the orange and black only
# 4.05:1 on the deep blue, so neither one works for the whole sequence.
BAND_LABEL_ON_DARK = "#FFFFFF"
BAND_LABEL_ON_LIGHT = "#101010"


def _luminance(value: str) -> float:
    """WCAG relative luminance of a ``#RRGGBB`` string."""
    channels = []
    for index in (1, 3, 5):
        channel = int(value[index : index + 2], 16) / 255.0
        channels.append(channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def band_label_colour(fill: str) -> str:
    """The more readable of the two caption colours against `fill`.

    Computed rather than tabulated so a fill added to `DATA_SEQUENCE` later
    gets a correct label without a matching edit somewhere else.
    """
    if _contrast_ratio(BAND_LABEL_ON_DARK, fill) >= _contrast_ratio(BAND_LABEL_ON_LIGHT, fill):
        return BAND_LABEL_ON_DARK
    return BAND_LABEL_ON_LIGHT


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


def current_plot_palette() -> PlotPalette:
    """Resolve the plot palette from the running application's palette.

    Plots are drawn outside the stylesheet, so each draw reads the palette
    afresh; that is what lets a system appearance change reach both the
    matplotlib figures and the pyqtgraph panes on the next repaint without any
    switching logic.  This lives in ``theme`` rather than ``plots.diagnostics``
    so ``plots.live`` can share it without importing ``diagnostics`` (a cycle).
    """
    application = QApplication.instance()
    if application is None:
        return LIGHT_PLOT_PALETTE
    return plot_palette(palette_tokens(application.palette()))


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
QWidget#plotInteractionToolbar {{
    background: {tokens.surface};
    border: 1px solid {tokens.surface_border};
    border-radius: 7px;
}}
/* The floating mode bar sits on top of the data, so its buttons must not carry
   the padding and min-height the general rule gives a button holding text: that
   turned an 18px glyph into a 39x43 slab and made the bar wide enough to crowd
   the plot title.  Square, glyph-sized cells keep the bar the narrow strip a
   charting mode bar is supposed to be. */
QWidget#plotInteractionToolbar QToolButton {{
    padding: 0px;
    min-width: {GLYPH_CELL_PX}px;
    max-width: {GLYPH_CELL_PX}px;
    min-height: {GLYPH_CELL_PX}px;
    max-height: {GLYPH_CELL_PX}px;
    border-radius: 4px;
}}
QLabel[sectionHeader="true"] {{
    font-weight: 700;
    padding: 2px 0px;
}}
QLabel[mutedText="true"] {{ color: {tokens.muted_text}; }}
QLabel[emptyTitle="true"] {{ font-size: {FONT_PT_LG}pt; font-weight: 700; }}
QLabel[statusKind="ok"] {{ color: {tokens.ok}; }}
QLabel[statusKind="info"] {{ color: {tokens.info}; }}
QLabel[statusKind="warn"] {{ color: {tokens.warn}; }}
QLabel[statusKind="error"] {{ color: {tokens.error}; }}
QLabel#confidenceBadge {{ font-weight: 600; }}
QTabBar::tab {{ padding: {SPACE_XS}px {SPACE_MD}px; }}
/* Nine diagnostic labels want 495px of text; the tab stack gets 568px once both
   docks are open.  At the general 12px side padding the chrome alone eats 216px
   and every label elides down to two characters, which is where 加权残差 /
   残差热图 / 参数热图 become indistinguishable.  Trimming just this bar's side
   padding to 4px fits all nine names in full with no ellipsis, and leaves the
   roomier padding to the tab bars that hold two or three labels. */
QTabWidget#diagnosticTabs QTabBar::tab {{ padding: {SPACE_XS}px {SPACE_XS}px; }}
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
    if kind not in ("", "ok", "info", "warn", "error"):
        raise ValueError(f"unsupported status kind: {kind}")
    if widget.property("statusKind") == kind:
        return
    widget.setProperty("statusKind", kind)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
