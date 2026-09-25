from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def test_module_help_exits_before_importing_pyside6(tmp_path: Path) -> None:
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import sys\n"
        "class Guard:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'PySide6' or fullname.startswith('PySide6.'):\n"
        "            raise RuntimeError('PySide6 imported while rendering --help')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Guard())\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(guard), str(ROOT / "src")))

    result = subprocess.run(
        (sys.executable, "-m", "xrr_fitter", "--help"),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage: xrr-fitter" in result.stdout
    assert "PySide6 imported" not in result.stdout + result.stderr


def test_module_main_freezes_support_before_gui_launch(monkeypatch) -> None:
    import xrr_fitter.__main__ as entrypoint

    events: list[object] = []
    monkeypatch.setattr(entrypoint, "freeze_support", lambda: events.append("freeze"))
    monkeypatch.setattr(
        entrypoint,
        "_launch",
        lambda argv: (events.append(("launch", tuple(argv))), 23)[1],
    )

    assert entrypoint.main([]) == 23
    assert events == ["freeze", ("launch", (sys.argv[0],))]


def test_gui_launch_creates_application_then_window_and_propagates_exit(
    monkeypatch,
) -> None:
    import xrr_fitter.__main__ as entrypoint

    events: list[object] = []

    class FakeWindow:
        def __init__(self) -> None:
            events.append("window")

        def show(self) -> None:
            events.append("show")

    def run_application() -> int:
        events.append("exec")
        return 29

    fake_application = SimpleNamespace(processEvents=lambda: None, exec=run_application)

    monkeypatch.setitem(
        sys.modules,
        "xrr_fitter.gui.application",
        SimpleNamespace(
            create_application=lambda argv: (
                events.append(("application", tuple(argv))),
                fake_application,
            )[1],
            show_splash=lambda app: SimpleNamespace(showMessage=lambda *a: None, finish=lambda w: None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "xrr_fitter.gui.main_window",
        SimpleNamespace(MainWindow=FakeWindow),
    )

    assert entrypoint._launch(["xrr-fitter", "--fixture"]) == 29
    assert events == [
        ("application", ("xrr-fitter", "--fixture")),
        "window",
        "show",
        "exec",
    ]


def test_splash_is_up_before_the_heavy_window_module_is_imported(monkeypatch) -> None:
    """启动画面要在最重的那段等待之前出现，并逐段报告进度。

    ``main_window`` 牵出 pyqtgraph / matplotlib / 拟合内核，是冷启动里最长的一段。
    启动画面若在它导入之后才显示，用户看到的是一段空白，画面只在等待结束时闪一下。
    """
    import types

    import xrr_fitter.__main__ as entrypoint

    events: list[object] = []

    class FakeWindow:
        def __init__(self) -> None:
            events.append("window")

        def show(self) -> None:
            events.append("show")

    def run_application() -> int:
        events.append("exec")
        return 0

    fake_application = SimpleNamespace(processEvents=lambda: None, exec=run_application)

    def show_splash(_app):
        events.append("splash")
        return SimpleNamespace(
            showMessage=lambda message, *_args: events.append(("message", message)),
            finish=lambda _window: events.append("finish"),
        )

    window_module = types.ModuleType("xrr_fitter.gui.main_window")

    def _lazy(name):
        if name == "MainWindow":
            events.append("import main_window")
            return FakeWindow
        raise AttributeError(name)

    window_module.__getattr__ = _lazy
    monkeypatch.setitem(
        sys.modules,
        "xrr_fitter.gui.application",
        SimpleNamespace(create_application=lambda argv: fake_application, show_splash=show_splash),
    )
    monkeypatch.setitem(sys.modules, "xrr_fitter.gui.main_window", window_module)

    assert entrypoint._launch(["xrr-fitter"]) == 0

    assert events.index("splash") < events.index("import main_window")
    messages = [event[1] for event in events if isinstance(event, tuple)]
    assert len(messages) >= 2, messages
    assert events[-3:] == ["show", "finish", "exec"]
