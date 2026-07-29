from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QMessageBox, QSplitter

import xrr_fitter.api as api


class _UserCloseEvent:
    def __init__(self) -> None:
        self.accepted = False
        self.ignored = False

    def spontaneous(self) -> bool:
        return True

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


class _OperationController(QObject):
    running_changed = Signal(bool)

    def __init__(self, *, running: bool) -> None:
        super().__init__()
        self.is_running = running
        self.cancel_calls = 0
        self.force_stop_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1

    def force_stop(self) -> None:
        self.force_stop_calls += 1

    def set_running(self, running: bool) -> None:
        self.is_running = running
        self.running_changed.emit(running)


class _ProjectApiStub:
    def __init__(self, initial, *loaded) -> None:
        self.initial = initial
        self.loaded = list(loaded)
        self.calls: list[tuple[object, ...]] = []

    def new_project(self):
        return self.initial

    def load_project(self, path):
        self.calls.append(("load", Path(path)))
        return self.loaded.pop(0)

    def save_project(self, project, path) -> None:
        self.calls.append(("save", project, Path(path)))


def test_project_document_owns_api_created_state_and_persistence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument

    initial = api.new_project()
    loaded = api.set_expert_mode(api.new_project(), True)
    saved = api.set_expert_mode(api.new_project(), True)
    stub = _ProjectApiStub(initial, loaded, saved)
    monkeypatch.setattr(api, "new_project", stub.new_project)
    monkeypatch.setattr(api, "load_project", stub.load_project)
    monkeypatch.setattr(api, "save_project", stub.save_project)
    document = ProjectDocument()
    snapshots = [
        (document.project, document.path, document.is_dirty),
    ]

    source = tmp_path / "project.xrrproj.json"
    document.open(source)
    snapshots.append((document.project, document.path, document.is_dirty))

    updated = api.set_expert_mode(loaded, False)
    document.replace_project(updated)
    snapshots.append((document.project, document.path, document.is_dirty))

    target = tmp_path / "saved.xrrproj.json"
    document.save(target)
    snapshots.append((document.project, document.path, document.is_dirty))

    assert (snapshots, stub.calls) == (
        [
            (initial, None, False),
            (loaded, source, False),
            (updated, source, True),
            (saved, target, False),
        ],
        [
            ("load", source),
            ("save", updated, target),
            ("load", target),
        ],
    )


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}"
            for index in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _saved_project_with_source(tmp_path: Path) -> tuple[Path, Path, str]:
    source = _write_curve(tmp_path / "sample.xy")
    project = api.add_dataset(
        api.new_project(),
        source,
        api.InstrumentSpec(instrument_id="gui-document"),
    )
    target = tmp_path / "workspace.xrrproj.json"
    api.save_project(project, target)
    return source, target, project.datasets[0].dataset_id


def test_open_project_rejects_persisted_mask_shape_mismatch(tmp_path: Path) -> None:
    from xrr_fitter.gui.document import ProjectDocument

    _source, target, _dataset_id = _saved_project_with_source(tmp_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["datasets"][0]["fit_mask"] = [True]
    target.write_text(json.dumps(payload), encoding="utf-8")
    document = ProjectDocument()
    before = document.project

    try:
        document.open(target)
    except ValueError as error:
        assert "fit mask must match derived data length" in str(error)
    else:
        raise AssertionError("persisted mask shape mismatch was accepted")

    assert document.project is before
    assert document.path is None
    assert document.is_dirty is False


def test_open_project_revalidation_restores_persisted_mask_after_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument

    source, target, dataset_id = _saved_project_with_source(tmp_path)
    project = api.load_project(target)
    mask = np.asarray(project.datasets[0].fit_mask, dtype=bool)
    mask[8] = False
    api.save_project(api.set_fit_mask(project, dataset_id, mask), target)
    read_bytes = Path.read_bytes
    source_reads = 0

    def fail_first_parse(path: Path) -> bytes:
        nonlocal source_reads
        if path == source:
            source_reads += 1
            if source_reads == 2:
                raise OSError("transient source read failure")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_first_parse)
    document = ProjectDocument()

    document.open(target)

    assert source_reads == 5
    assert document.project.datasets[0].fit_mask[8] is False


def test_main_window_has_three_accessible_columns(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    splitter = window.findChild(QSplitter, "workspaceSplitter")
    assert splitter is not None
    assert splitter.orientation() == Qt.Orientation.Horizontal
    assert splitter.count() == 3
    assert tuple(
        splitter.widget(index).objectName() for index in range(splitter.count())
    ) == ("projectColumn", "plotColumn", "analysisColumn")
    assert window.minimumWidth() == 1280
    assert window.minimumHeight() == 760
    assert window.windowTitle() == "XRR 全自动拟合"


def test_close_dirty_project_asks_before_discarding(qtbot, monkeypatch) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.document.mark_dirty()
    questions: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **_kwargs: (
            questions.append(args),
            QMessageBox.StandardButton.No,
        )[1],
    )
    event = _UserCloseEvent()

    window.closeEvent(event)

    assert len(questions) == 1
    assert event.accepted is False
    assert event.ignored is True


def test_close_hidden_dirty_project_does_not_open_discard_dialog(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.document.mark_dirty()
    questions: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: questions.append((args, kwargs)),
    )
    event = _UserCloseEvent()

    window.closeEvent(event)

    assert questions == []
    assert event.accepted is True
    assert event.ignored is False


def test_programmatic_close_of_visible_dirty_window_does_not_open_dialog(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.document.mark_dirty()
    questions: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: questions.append((args, kwargs)),
    )

    assert window.close() is True
    assert questions == []


def test_main_window_close_while_fitting_is_async_and_idempotent(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    controller = _OperationController(running=True)
    window = MainWindow(operation_controller=controller)
    qtbot.addWidget(window)
    first = _UserCloseEvent()
    second = _UserCloseEvent()

    window.closeEvent(first)
    remaining = window.close_cancel_timer.remainingTime()
    qtbot.wait(25)
    window.closeEvent(second)

    assert controller.cancel_calls == 1
    assert first.ignored is True and first.accepted is False
    assert second.ignored is True and second.accepted is False
    assert window.close_pending is True
    assert window.close_cancel_timer.isActive()
    assert window.close_cancel_timer.interval() == 5000
    assert 0 < window.close_cancel_timer.remainingTime() < remaining


def test_pending_close_natural_stop_reenters_idle_dirty_gate(qtbot, monkeypatch) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    controller = _OperationController(running=True)
    window = MainWindow(operation_controller=controller)
    qtbot.addWidget(window)
    window.show()
    window.document.mark_dirty()
    questions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda _parent, title, message, *_args: (
            questions.append((title, message)),
            QMessageBox.StandardButton.No,
        )[1],
    )
    event = _UserCloseEvent()

    window.closeEvent(event)
    controller.set_running(False)

    qtbot.waitUntil(lambda: bool(questions), timeout=1000)
    assert event.ignored is True
    assert event.accepted is False
    assert window.close_pending is False
    assert window.close_cancel_timer.isActive() is False
    assert questions[0][0] == "未保存的项目更改"
    assert window.isVisible()


def test_main_window_close_timeout_decline_keeps_window_open(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    controller = _OperationController(running=True)
    window = MainWindow(operation_controller=controller)
    qtbot.addWidget(window)
    window.show()
    event = _UserCloseEvent()

    window.closeEvent(event)
    window.close_cancel_deadline_reached()
    first_prompt = window.force_close_prompt
    window.close_cancel_deadline_reached()

    prompt_snapshot = (
        isinstance(first_prompt, QMessageBox),
        window.force_close_prompt is first_prompt,
        "检查点" in first_prompt.text(),
        "损坏" in first_prompt.text(),
    )
    first_prompt.button(QMessageBox.StandardButton.No).click()
    qtbot.waitUntil(lambda: window.force_close_prompt is None)

    assert (
        prompt_snapshot,
        controller.cancel_calls,
        controller.force_stop_calls,
        event.ignored,
        event.accepted,
        window.close_pending,
        window.close_cancel_timer.isActive(),
        window.isVisible(),
    ) == ((True, True, True, True), 1, 0, True, False, False, False, True)
