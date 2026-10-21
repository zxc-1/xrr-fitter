"""Qt text is sized in pixels so it reads the same size on every desktop.

A point is 1/72 inch, so Qt turns it into pixels through the logical DPI: 96 on
Windows, Linux and the offscreen test platform, 72 on macOS. Every pt size in the
widgets therefore drew 25% smaller on a Mac than the layout it was tuned in, while
the system body text stayed 13px: the verdict that should outrank the body (11pt,
15px here) came out at 11px there, smaller than the sentence beneath it.

Matplotlib text is the exception and stays in points: a figure renders at its own
dpi, independent of the desktop's logical DPI.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUI = ROOT / "src" / "xrr_fitter" / "gui"

# 带样式表字号的那几档：判定框的结论与理由、空状态标题、提示框、代码块、步骤圆点、层行小字。
PROBE = r"""
import json
from PySide6.QtGui import QFontInfo
from PySide6.QtWidgets import QApplication, QLabel
from xrr_fitter.gui import theme

app = QApplication([])
app.setPalette(theme.light_palette())
theme.apply_theme(app)
roles = {
    "confidenceBadge": ("confidenceBadge", {}),
    "confidenceReason": ("confidenceReason", {}),
    "emptyTitle": ("", {"emptyTitle": True}),
    "hintBox": ("", {"hintBox": True}),
    "codeBlock": ("", {"codeBlock": True}),
    "stepDot": ("", {"stepDot": True}),
    "structureLayerDetail": ("structureLayerDetail", {}),
}
sizes = {}
for role, (name, properties) in roles.items():
    label = QLabel("判定 Ag")
    label.setObjectName(name)
    for key, value in properties.items():
        label.setProperty(key, value)
    label.ensurePolished()
    sizes[role] = QFontInfo(label.font()).pixelSize()
print(json.dumps(sizes))
"""


def _role_pixels(dpi: int) -> dict[str, int]:
    environment = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "QT_FONT_DPI": str(dpi),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
        timeout=120,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_stylesheet_text_keeps_its_pixel_size_at_the_macos_logical_dpi() -> None:
    windows = _role_pixels(96)
    macos = _role_pixels(72)

    assert macos == windows


def test_the_shipped_stylesheet_has_no_point_sizes(qapp) -> None:
    from xrr_fitter.gui import theme

    sheet = theme.build_stylesheet(theme.light_palette())

    assert re.findall(r"font-size:\s*[\d.]+pt", sheet) == []


def _point_size_calls() -> list[str]:
    found: list[str] = []
    for path in sorted(GUI.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found.extend(
            f"{path.relative_to(ROOT)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"setPointSize", "setPointSizeF"}
        )
    return found


def test_no_widget_font_is_sized_in_points() -> None:
    assert _point_size_calls() == []
