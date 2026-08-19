from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox, QPushButton

import xrr_fitter.api as api


def _write_curve(path: Path, *, scale: float = 1000.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {scale / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _project_with_source(source: Path) -> api.XrrProject:
    return api.add_dataset(
        api.new_project(),
        source,
        api.InstrumentSpec(instrument_id="gui-source"),
    )


def _window_with_source(qtbot, source: Path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project_with_source(source)))
    qtbot.addWidget(window)
    return window


def _source_dialog_case(qtbot, tmp_path):
    source = _write_curve(tmp_path / "sample.xy")
    replacement = _write_curve(tmp_path / "replacement.xy", scale=1700.0)
    return _window_with_source(qtbot, source), source, replacement


def test_main_window_open_project_surfaces_hash_mismatch_and_blocks_fit(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    source = _write_curve(tmp_path / "sample.xy")
    project = _project_with_source(source)
    dataset = project.datasets[0]
    target = tmp_path / "stale.xrrproj.json"
    api.save_project(project, target)
    source.write_text(source.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    window = MainWindow()
    qtbot.addWidget(window)

    window.open_project(target)

    warning = window.source_warning_text(dataset.dataset_id)
    assert window.source_hash_status(dataset.dataset_id) == "hash_mismatch"
    assert "源文件已变化" in warning
    assert dataset.source_sha256 in warning
    assert api.inspect_sources(window.document.project).datasets[0].actual_sha256 in warning
    assert window.findChild(QPushButton, "reloadSourceButton").isEnabled()
    assert window.findChild(QPushButton, "relinkSourceButton").isEnabled()


def test_reload_dialog_confirms_old_path_expected_and_actual_hash_before_commit(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    source = _write_curve(tmp_path / "sample.xy")
    window = _window_with_source(qtbot, source)
    dataset_id = window.document.project.datasets[0].dataset_id
    expected = window.document.project.datasets[0].source_sha256
    source.write_text(source.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    observed = api.preview_source_update(window.document.project, dataset_id).observed_sha256
    questions: list[tuple[str, str]] = []
    answers = iter((QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda _parent, title, message, *_args: (
            questions.append((title, message)),
            next(answers),
        )[1],
    )

    window.reload_source_dialog()
    assert window.document.project.datasets[0].source_sha256 == expected
    assert window.document.is_dirty is False

    window.reload_source_dialog()

    assert questions[0][0] == "确认重新加载数据源"
    assert str(source) in questions[0][1]
    assert expected in questions[0][1]
    assert observed in questions[0][1]
    assert window.document.project.datasets[0].source_sha256 == observed
    assert window.source_hash_status(dataset_id) == "ok"
    assert window.document.is_dirty is True


def test_relink_dialog_confirms_old_and_new_source_identity_and_can_cancel(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    old_source = _write_curve(tmp_path / "old.xy")
    replacement = _write_curve(tmp_path / "replacement.xy", scale=1800.0)
    window = _window_with_source(qtbot, old_source)
    before = window.document.project
    preview = api.preview_source_update(
        before,
        before.datasets[0].dataset_id,
        replacement,
    )
    questions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(replacement), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda _parent, title, message, *_args: (
            questions.append((title, message)),
            QMessageBox.StandardButton.No,
        )[1],
    )

    window.relink_source_dialog()

    assert window.document.project is before
    assert questions[0][0] == "确认重新链接数据源"
    assert str(old_source) in questions[0][1]
    assert preview.expected_sha256 in questions[0][1]
    assert str(replacement) in questions[0][1]
    assert preview.observed_sha256 in questions[0][1]


@pytest.mark.parametrize("operation", ("reload", "relink"))
def test_source_dialog_rejects_bytes_changed_after_confirmation_preview(
    qtbot,
    tmp_path,
    monkeypatch,
    operation: str,
) -> None:
    window, source, replacement = _source_dialog_case(qtbot, tmp_path)
    before = window.document.project
    target = source if operation == "reload" else replacement
    critical: list[tuple[str, str]] = []
    if operation == "relink":
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(replacement), ""),
        )

    def mutate_after_preview(*_args, **_kwargs):
        target.write_text(
            target.read_text(encoding="utf-8") + "# changed after preview\n",
            encoding="utf-8",
        )
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", mutate_after_preview)
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: critical.append((title, message)),
    )

    getattr(window, f"{operation}_source_dialog")()

    assert window.document.project is before
    assert len(critical) == 1
    assert "proposed source changed after preview" in critical[0][1]
    assert "检查数据源路径、读取权限和文件格式后重试" in critical[0][1]


@pytest.mark.parametrize("operation", ("reload", "relink"))
@pytest.mark.parametrize("stage", ("preview", "commit"))
def test_source_dialog_catches_preview_and_commit_errors_once(
    qtbot,
    tmp_path,
    monkeypatch,
    operation: str,
    stage: str,
) -> None:
    window, _source, replacement = _source_dialog_case(qtbot, tmp_path)
    messages: list[tuple[str, str]] = []
    title = "重新加载数据源失败" if operation == "reload" else "重新链接数据源失败"
    if operation == "relink":
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(replacement), ""),
        )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    if stage == "preview":
        monkeypatch.setattr(
            api,
            "preview_source_update",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("preview failed")),
        )
        expected = "KeyError: 'preview failed'"
    else:
        monkeypatch.setattr(
            api,
            "accept_source_update",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("commit failed")),
        )
        expected = "RuntimeError: commit failed"
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, actual_title, message: messages.append((actual_title, message)),
    )

    getattr(window, f"{operation}_source_dialog")()

    assert len(messages) == 1
    assert messages[0][0] == title
    assert expected in messages[0][1]
    assert "检查数据源路径、读取权限和文件格式后重试" in messages[0][1]


def test_source_dialogs_report_no_active_dataset_before_opening_a_file_chooser(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    messages: list[tuple[str, str]] = []
    choosers: list[bool] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (choosers.append(True), ("", ""))[1],
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: messages.append((title, message)),
    )

    window.reload_source_dialog()
    window.relink_source_dialog()

    assert choosers == []
    assert messages == [
        ("重新加载数据源失败", "当前没有活动数据集"),
        ("重新链接数据源失败", "当前没有活动数据集"),
    ]


def test_reload_dialog_reports_parameter_settings_removed_by_reconciliation(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    source = _write_curve(tmp_path / "sample.xy")
    window = _window_with_source(qtbot, source)
    dataset = window.document.project.datasets[0]
    preview = api.SourceUpdatePreview(
        dataset.dataset_id,
        dataset.source_path,
        dataset.source_path,
        dataset.source_sha256,
        dataset.source_sha256,
        False,
    )
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window.document,
        "preview_source_update",
        lambda *_args, **_kwargs: preview,
    )
    monkeypatch.setattr(
        window.document,
        "accept_source_update",
        lambda _preview: ("component.99.thickness_a", "component.0.roughness_a"),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, message: messages.append((title, message)),
    )

    window.reload_source_dialog()

    assert len(messages) == 1
    assert messages[0][0] == "数据源已更新"
    assert dataset.dataset_id in messages[0][1]
    assert "component.99.thickness_a" in messages[0][1]
    assert "component.0.roughness_a" in messages[0][1]
    assert "已移除" in messages[0][1]


def test_save_as_rebases_relative_source_without_making_it_missing(
    tmp_path,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument

    source = _write_curve(tmp_path / "sources" / "sample.xy")
    first = source.parent / "workspace.xrrproj.json"
    api.save_project(_project_with_source(source), first)
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["datasets"][0]["source_path"] = "./sample.xy"
    first.write_text(json.dumps(payload), encoding="utf-8")
    document = ProjectDocument()
    document.open(first)
    second = tmp_path / "moved" / "workspace.xrrproj.json"
    second.parent.mkdir()

    document.save(second)

    assert document.path == second
    assert document.project.base_directory == str(second.parent.resolve())
    assert Path(document.project.datasets[0].source_path).is_absolute() is False
    assert api.inspect_sources(document.project).valid is True
    assert document.is_dirty is False
