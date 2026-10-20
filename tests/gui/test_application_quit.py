"""⌘Q, the Dock's 退出 and logging out reach the window as an application quit.

Qt answers ``QEvent::Quit`` with ``closeAllWindows()``, whose close events are not
spontaneous, so the window used to read them as a programmatic close: it asked
nothing and deleted the autosave draft on the way out. The quit is now confirmed
before any window is closed.

A real quit would end the test session's event loop, so ``quit_barrier`` stands in
for ``QApplication::event``: installed before the window, it runs after the window's
guard, closes the windows the way Qt would and swallows the event.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui.project import dialogs


class _QuitBarrier(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.reached = 0

    def eventFilter(self, watched, event) -> bool:
        if event.type() != QEvent.Type.Quit:
            return False
        self.reached += 1
        QApplication.closeAllWindows()
        return True


class _RunningController(QObject):
    running_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.is_running = True
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1

    def force_stop(self) -> None:
        raise AssertionError("an exit confirmation never force-stops")

    def set_running(self, running: bool) -> None:
        self.is_running = running
        self.running_changed.emit(running)


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


@pytest.fixture
def quit_barrier():
    application = QApplication.instance()
    barrier = _QuitBarrier()
    application.installEventFilter(barrier)
    yield barrier
    application.removeEventFilter(barrier)


@pytest.fixture
def prompts(monkeypatch):
    """Record each exit prompt; the answers are set per test before quitting."""
    asked: list[str] = []
    answers = {"unsaved": None, "stop": None}

    def unsaved(_parent):
        asked.append("unsaved")
        return answers["unsaved"]

    def stop(_parent):
        asked.append("stop")
        return answers["stop"]

    monkeypatch.setattr(dialogs, "ask_unsaved_changes", unsaved, raising=False)
    monkeypatch.setattr(dialogs, "confirm_stop_for_exit", stop, raising=False)
    # The old yes/no discard box must not be what answers any more; recording it keeps a
    # regression from blocking the run on a real modal.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda _parent, title, *_args, **_kwargs: (asked.append(title), QMessageBox.StandardButton.No)[1],
    )
    return asked, answers


def _shown_window(qtbot, *, controller=None):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(operation_controller=controller)
    qtbot.addWidget(window)
    window.show()
    return window


def _dirty_with_draft(window, tmp_path):
    path = tmp_path / "quit.xrrproj.json"
    window.save_project(path)
    window.document.replace_project(api.set_expert_mode(window.document.project, True))
    assert window.autosave.autosave_now()
    draft = window.autosave.draft_path()
    assert draft is not None and draft.exists()
    return path, draft


def _quit() -> None:
    QCoreApplication.sendEvent(QApplication.instance(), QEvent(QEvent.Type.Quit))


def _answer_when_shown(answer, seen: list) -> None:
    """Answer the next modal from inside its own ``exec()`` loop."""

    def visit() -> None:
        box = QApplication.activeModalWidget()
        if not isinstance(box, QMessageBox):
            QTimer.singleShot(10, visit)
            return
        seen.append(
            (
                box.objectName(),
                [button.text() for button in box.buttons()],
                box.defaultButton().text(),
                box.escapeButton().text(),
            )
        )
        answer(box)

    QTimer.singleShot(0, visit)


def _click(text):
    return lambda box: next(button for button in box.buttons() if button.text() == text).click()


def _escape(box) -> None:
    QTest.keyClick(box, Qt.Key.Key_Escape)


@pytest.mark.parametrize(
    ("answer", "choice"),
    (
        (_click("保存"), dialogs.UnsavedChoice.SAVE),
        (_click("不保存"), dialogs.UnsavedChoice.DISCARD),
        (_click("取消"), dialogs.UnsavedChoice.CANCEL),
        (_escape, dialogs.UnsavedChoice.CANCEL),
    ),
)
def test_unsaved_changes_prompt_offers_save_first_in_chinese(qtbot, answer, choice) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    seen: list = []
    _answer_when_shown(answer, seen)

    assert dialogs.ask_unsaved_changes(host) is choice
    name, buttons, default, escape = seen[0]
    assert name == "unsavedChangesPrompt"
    assert sorted(buttons) == sorted(["保存", "不保存", "取消"])
    assert (default, escape) == ("保存", "取消")


@pytest.mark.parametrize(
    ("answer", "stopping"), ((_click("停止并退出"), True), (_click("继续运行"), False), (_escape, False))
)
def test_stop_for_exit_prompt_defaults_to_keep_running(qtbot, answer, stopping) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    seen: list = []
    _answer_when_shown(answer, seen)

    assert dialogs.confirm_stop_for_exit(host) is stopping
    name, buttons, default, escape = seen[0]
    assert name == "stopForExitPrompt"
    assert sorted(buttons) == sorted(["停止并退出", "继续运行"])
    assert (default, escape) == ("继续运行", "继续运行")


def test_quit_with_unsaved_changes_asks_and_cancel_keeps_window_and_draft(
    qtbot, tmp_path, quit_barrier, prompts
) -> None:
    window = _shown_window(qtbot)
    _path, draft = _dirty_with_draft(window, tmp_path)
    asked, answers = prompts
    answers["unsaved"] = dialogs.UnsavedChoice.CANCEL

    _quit()

    assert asked == ["unsaved"]
    assert quit_barrier.reached == 0
    assert window.isVisible()
    assert window.document.is_dirty
    assert draft.exists()


def test_quit_and_discard_closes_without_saving(qtbot, tmp_path, quit_barrier, prompts) -> None:
    window = _shown_window(qtbot)
    path, draft = _dirty_with_draft(window, tmp_path)
    saved = path.read_bytes()
    asked, answers = prompts
    answers["unsaved"] = dialogs.UnsavedChoice.DISCARD

    _quit()

    assert asked == ["unsaved"]
    assert quit_barrier.reached == 1
    assert not window.isVisible()
    assert path.read_bytes() == saved
    # 「不保存」是明确放弃，草稿跟着走；只有没走到这一步就断掉的会话才该留下草稿。
    assert not draft.exists()


def test_quit_and_save_writes_the_project_before_closing(qtbot, tmp_path, quit_barrier, prompts) -> None:
    window = _shown_window(qtbot)
    path, draft = _dirty_with_draft(window, tmp_path)
    asked, answers = prompts
    answers["unsaved"] = dialogs.UnsavedChoice.SAVE

    _quit()

    assert asked == ["unsaved"]
    assert quit_barrier.reached == 1
    assert not window.isVisible()
    assert api.load_project(path).ui_state.expert_mode is True
    assert not draft.exists()


def test_quit_and_save_of_an_unnamed_project_stays_when_the_file_dialog_is_cancelled(
    qtbot, quit_barrier, prompts, monkeypatch
) -> None:
    window = _shown_window(qtbot)
    window.document.replace_project(api.set_expert_mode(window.document.project, True))
    asked, answers = prompts
    answers["unsaved"] = dialogs.UnsavedChoice.SAVE
    monkeypatch.setattr(dialogs, "choose_project_to_save", lambda *_args, **_kwargs: None)

    _quit()

    assert asked == ["unsaved"]
    assert quit_barrier.reached == 0
    assert window.isVisible()
    assert window.document.is_dirty


def test_quit_of_a_clean_project_asks_nothing(qtbot, quit_barrier, prompts) -> None:
    window = _shown_window(qtbot)
    asked, _answers = prompts

    _quit()

    assert asked == []
    assert quit_barrier.reached == 1
    assert not window.isVisible()


@pytest.mark.parametrize("confirm", (False, True))
def test_quit_while_running_asks_before_stopping(qtbot, quit_barrier, prompts, confirm) -> None:
    controller = _RunningController()
    window = _shown_window(qtbot, controller=controller)
    asked, answers = prompts
    answers["stop"] = confirm

    _quit()

    assert asked == ["stop"]
    assert controller.cancel_calls == int(confirm)
    assert quit_barrier.reached == int(confirm)
    # 确认后也要等后台真正停下才关；窗口先留着，停下后由原有的挂起关闭接着走。
    assert window.isVisible()
    assert window.close_pending is confirm
    controller.set_running(False)


@pytest.mark.parametrize("choice", (dialogs.UnsavedChoice.CANCEL, dialogs.UnsavedChoice.DISCARD))
def test_quit_while_running_asks_about_unsaved_changes_once_stopped(
    qtbot, tmp_path, quit_barrier, prompts, choice
) -> None:
    controller = _RunningController()
    controller.is_running = False
    window = _shown_window(qtbot, controller=controller)
    _path, draft = _dirty_with_draft(window, tmp_path)
    controller.set_running(True)
    asked, answers = prompts
    answers.update(stop=True, unsaved=choice)

    _quit()
    controller.set_running(False)

    # 「停止并退出」只答了停不停；停下后接着走的那次关闭照用户关窗算，未保存的改动还要再问，
    # 这时点「取消」，窗口和草稿都留下。
    qtbot.waitUntil(lambda: asked == ["stop", "unsaved"], timeout=3000)
    kept = choice is dialogs.UnsavedChoice.CANCEL
    assert window.isVisible() is kept
    assert draft.exists() is kept


@pytest.mark.parametrize("confirm", (False, True))
def test_user_close_while_running_asks_before_stopping(qtbot, prompts, confirm) -> None:
    controller = _RunningController()
    window = _shown_window(qtbot, controller=controller)
    asked, answers = prompts
    answers["stop"] = confirm
    event = _UserCloseEvent()

    window.closeEvent(event)

    assert asked == ["stop"]
    assert event.ignored and not event.accepted
    assert controller.cancel_calls == int(confirm)
    assert window.close_pending is confirm
    controller.set_running(False)


def test_programmatic_close_while_running_stops_without_asking(qtbot, prompts) -> None:
    controller = _RunningController()
    window = _shown_window(qtbot, controller=controller)
    asked, _answers = prompts

    window.close()

    assert asked == []
    assert controller.cancel_calls == 1
    assert window.close_pending is True
    controller.set_running(False)
