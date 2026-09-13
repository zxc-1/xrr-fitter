"""Fixtures shared by the two halves of the data-import suite.

Import splits into two subjects that answer to different code: what
``ImportDialog`` asks before a manual import (``test_import_dialog.py``), and what
``DataPanel`` does with a batch once the answers exist (``test_data_import.py``).
Both need the same synthetic curve and the same saved preset, so the fixtures live
here rather than being duplicated -- following ``plot_support.py`` next door.
"""

from __future__ import annotations

from pathlib import Path

import xrr_fitter.api as api


def _saved_preset() -> api.MeasurementPreset:
    return api.MeasurementPreset(
        "gui-lab",
        api.BeamSpec("monochromatic", wavelength_a=1.5406),
        api.InstrumentSpec(instrument_id="gui-lab"),
    )


def _write_curve(path: Path, *, scale: float = 1000.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {scale / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _write_headed_curve(path: Path, header: str) -> Path:
    """一条 32 点曲线，前面加一行自述轴名的表头。"""
    _write_curve(path)
    path.write_text(header + "\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _panel(qtbot, document=None):
    from xrr_fitter.gui.data.panel import DataPanel
    from xrr_fitter.gui.document import ProjectDocument

    panel = DataPanel(ProjectDocument() if document is None else document)
    qtbot.addWidget(panel)
    return panel


def _instrument() -> api.InstrumentSpec:
    return api.InstrumentSpec(instrument_id="gui-import")
