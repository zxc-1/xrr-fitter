#!/usr/bin/env python3
"""Opt-in native Qt rendering and event checks; no claim of desktop drag input."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def require_native_platform(actual: str, expected: str) -> None:
    if expected not in {"windows", "cocoa"} or actual != expected:
        raise ValueError(f"expected native Qt platform {expected}, found {actual}")


def _source_identity() -> dict:
    root = Path(__file__).resolve().parents[1]

    def git(*arguments):
        return subprocess.check_output(("git", *arguments), cwd=root).decode("utf-8").strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(git("status", "--porcelain=v1", "--untracked-files=all")),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _project(root: Path):
    import xrr_fitter.api as api

    curve = root / "native-gui.xy"
    curve.write_text(
        "\n".join(f"{0.05 + i * 0.02:.6f} {1000 / (i + 1):.12g}" for i in range(32)) + "\n",
        encoding="utf-8",
    )
    project = api.add_dataset(api.new_project(), curve, api.InstrumentSpec())
    layers = tuple(
        api.LayerSpec(f"film{i}", api.MaterialSpec("SiO2", "SiO2", 2.20), 40.0 + 20 * i, roughness_a=3.0)
        for i in range(3)
    )
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j), layers, api.MaterialSpec("Si", "Si", 2.329)
    )
    return api.set_structure(project, "native-gui", structure)


def _capture(widget, target: Path) -> dict:
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.processEvents()
    pixmap = widget.grab()
    if pixmap.isNull() or not pixmap.save(str(target)):
        raise RuntimeError("native widget screenshot could not be saved")
    return {
        "file": target.name,
        "width": pixmap.width(),
        "height": pixmap.height(),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def _expose(widget) -> None:
    from PySide6.QtTest import QTest

    widget.show()
    widget.activateWindow()
    if not QTest.qWaitForWindowExposed(widget, 5000):
        raise RuntimeError("native GUI window was not exposed")


def _selection(panel, index: int, component) -> None:
    from PySide6.QtCore import Qt

    item = panel.editor.tree.currentItem()
    if item is None or item.data(0, Qt.ItemDataRole.UserRole) != index:
        raise RuntimeError("native tree selection moved away from the edited component")
    if panel.structure.components[index] is not component or panel.editor.stack.selected_index() != index:
        raise RuntimeError("native tree and structure selection disagree")
    if panel.editor.tree.topLevelItemCount() != 5:
        raise RuntimeError("native tree lost a row after reordering")


def _exercise_structure(panel) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest

    tree = panel.editor.tree
    component = panel.structure.components[1]
    point = tree.visualItemRect(tree.topLevelItem(2)).center()
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=point)
    _selection(panel, 1, component)
    QTest.keySequence(tree, QKeySequence("Ctrl+Up"))
    _selection(panel, 0, component)
    QTest.keySequence(tree, QKeySequence("Ctrl+Down"))
    _selection(panel, 1, component)


def exercise_gui(report: dict, root: Path, expected: str) -> None:
    import PySide6
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    from shiboken6 import isValid

    import xrr_fitter.api as api
    from xrr_fitter.gui.application import create_application
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow
    from xrr_fitter.gui.structure.panel import StructurePanel

    app = create_application([])
    require_native_platform(app.platformName(), expected)
    report.update(qt_platform=app.platformName(), pyside6=PySide6.__version__)
    owned = []
    try:
        document = ProjectDocument(_project(root))
        panel = StructurePanel(document)
        owned.append(panel)
        panel.resize(760, 680)
        _expose(panel)
        _exercise_structure(panel)
        report["structure"] = {"rows": 5, "selection": 1, "order": [x.name for x in panel.structure.components]}
        report["captures"] = [_capture(panel, root / "structure.png")]
        panel.hide()
        window = MainWindow(document)
        owned.append(window)
        window.resize(1280, 800)
        _expose(window)
        report["captures"].append(_capture(window, root / "workspace.png"))
        target = root / "native-gui.xrrproj.json"
        window.save_project(target)
        reopened = api.load_project(target)
        if not api.inspect_sources(reopened).valid:
            raise RuntimeError("saved native GUI project sources are not valid")
        window.open_project(target)
        report["project_roundtrip"] = "PASS"
        if not window.close():
            raise RuntimeError("native GUI rejected a clean close")
    finally:
        for widget in reversed(owned):
            if isValid(widget):
                widget.hide()
                widget.deleteLater()
        for _index in range(2):
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        report["remaining_widgets"] = len(QApplication.allWidgets())
    if report["remaining_widgets"]:
        raise RuntimeError("native GUI left widgets alive after teardown")


def run_acceptance(root: Path, expected: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "acceptance.json"
    # Reserve the report before running; an earlier run must never be overwritten.
    with target.open("x", encoding="utf-8") as stream:
        stream.write('{"state": "RUNNING"}\n')
    report = {
        "schema": "xrr-native-gui-acceptance-v1",
        "state": "FAIL",
        "host": platform.system(),
        "python": platform.python_version(),
        "input": "QtTest events on native platform",
        "native_desktop_drag": "NOT_RUN",
        "source": _source_identity(),
    }
    try:
        exercise_gui(report, root, expected)
        report["state"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-platform", choices=("windows", "cocoa"), required=True)
    args = parser.parse_args()
    run_acceptance(args.report_dir, args.expected_platform)


if __name__ == "__main__":
    main()
