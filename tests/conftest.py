from __future__ import annotations

import faulthandler
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# 卡住的用例（没人应答的模态框、不再回报的 worker）原本要静默等到 CI 的 60 分钟上限，日志里
# 什么也不说；超过这个秒数就把所有线程的栈打出来，让日志点名卡住的那一行。只报告，不判失败。
# 不放进 pyproject 的 ``faulthandler_timeout``：包清单绑定了 pyproject.toml 的哈希。
STALL_SECONDS = 300
# 慢用例与失败当场写进同一条流：CI 的 job 一旦到点被取消，pytest 收尾时那份失败清单和
# 耗时表就没机会印出来了。
SLOW_SECONDS = 30
STALL_STREAM = pytest.StashKey()
PROGRESS_LOG = pytest.StashKey()


class ProgressLog:
    """逐条记下开跑的用例；一条用例卡住超过 ``stall_seconds``，再追加所有线程的栈。

    上面那条 stderr 流在覆盖率审计里留不下来：``tools/audit_reports.py`` 把 pytest 的输出整个收在
    内存里，job 到点被取消，一个字也落不了盘。这份日志写在会话的 basetemp 下，审计的 basetemp 在
    报告目录里，取消了也照样上传。每行直接 ``os.write``：进程挨了 SIGKILL 也不丢，守望线程也不必
    去抢 Python 文件对象的缓冲锁。慢用例与卡住的表头都带上开跑以来的进程 CPU 时间：CPU 跟着墙钟涨
    是在算，墙钟涨 CPU 不涨是在等锁、等子进程，或者主机被限流。
    """

    def __init__(self, stall_seconds: float, slow_seconds: float) -> None:
        self.stall_seconds = stall_seconds
        self.slow_seconds = slow_seconds
        self.descriptor: int | None = None
        # 第一条用例开跑时，会话夹具还没把 basetemp 告诉这里；先攒着，打开时补写。关了以后再写的丢掉。
        self.backlog: list[str] | None = []
        self.watcher: tuple[str, tuple[float, float], threading.Thread, threading.Event] | None = None

    def open(self, path: Path) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        for line in self.backlog or ():
            os.write(descriptor, line.encode("utf-8"))
        self.backlog = None
        self.descriptor = descriptor

    def close(self) -> None:
        descriptor, self.descriptor, self.backlog = self.descriptor, None, None
        if descriptor is not None:
            os.close(descriptor)

    def write(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {text}\n"
        if self.descriptor is not None:
            os.write(self.descriptor, line.encode("utf-8"))
        elif self.backlog is not None:
            self.backlog.append(line)

    def start(self, nodeid: str) -> None:
        self.write(f"[start] {nodeid}")
        began = time.monotonic(), time.process_time()
        finished = threading.Event()
        thread = threading.Thread(
            target=self._watch, args=(nodeid, began, finished), name="xrr-stall-watch", daemon=True
        )
        thread.start()
        self.watcher = nodeid, began, thread, finished

    def finish(self) -> None:
        if self.watcher is None:
            return
        nodeid, began, thread, finished = self.watcher
        self.watcher = None
        finished.set()
        thread.join()
        if time.monotonic() - began[0] > self.slow_seconds:
            self.write(f"[slow {self._elapsed(began)}] {nodeid}")

    def _watch(self, nodeid: str, began: tuple[float, float], finished: threading.Event) -> None:
        while not finished.wait(self.stall_seconds):
            descriptor = self.descriptor
            if descriptor is None:
                continue
            self.write(f"[stalled {self._elapsed(began)}] {nodeid}")
            faulthandler.dump_traceback(descriptor, all_threads=True)

    @staticmethod
    def _elapsed(began: tuple[float, float]) -> str:
        wall, cpu = time.monotonic() - began[0], time.process_time() - began[1]
        return f"{wall:.0f}s, cpu {cpu:.0f}s, {threading.active_count()} threads"


def pytest_configure(config):
    # 配置阶段 stderr 还没被逐用例捕获；复制一份，栈才会落到终端，而不是落进某条用例的捕获里。
    config.stash[STALL_STREAM] = os.fdopen(os.dup(sys.stderr.fileno()), "w")
    config.stash[PROGRESS_LOG] = ProgressLog(STALL_SECONDS, SLOW_SECONDS)


def pytest_unconfigure(config):
    config.stash[STALL_STREAM].close()
    config.stash[PROGRESS_LOG].close()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    stream = item.config.stash[STALL_STREAM]
    if report.duration > SLOW_SECONDS:
        stream.write(f"\n[slow {report.duration:.0f}s] {report.when} {report.nodeid}\n")
    if report.failed:
        detail = "\n".join(str(report.longrepr).splitlines()[-30:])
        note = f"[failed] {report.when} {report.nodeid}\n{detail}"
        stream.write(f"\n{note}\n")
        item.config.stash[PROGRESS_LOG].write(note)
    stream.flush()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    progress = item.config.stash[PROGRESS_LOG]
    progress.start(item.nodeid)
    faulthandler.dump_traceback_later(STALL_SECONDS, file=item.config.stash[STALL_STREAM])
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        progress.finish()


@pytest.fixture(scope="session", autouse=True)
def _progress_log(request, tmp_path_factory):
    """进度日志落在本次会话的 basetemp 下；审计与 verify 的 basetemp 都在报告目录里，随报告上传。"""
    request.config.stash[PROGRESS_LOG].open(tmp_path_factory.getbasetemp() / "progress.log")


@pytest.fixture(scope="session", autouse=True)
def _isolated_user_settings(tmp_path_factory):
    """``QSettings`` 写到本次会话的临时目录，而不是开发机的用户配置。

    明暗偏好（F52）存在 ``QSettings`` 里；不隔开的话，一条点了深色开关的用例会改掉开发者自己
    的外观，而开发机上存着的深色又会让后面的用例换一套外观去量。
    """
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path_factory.mktemp("qsettings")))


def _running_application():
    """The session's ``QApplication``, without importing Qt widgets into a non-GUI run."""
    widgets = sys.modules.get("PySide6.QtWidgets")
    return None if widgets is None else widgets.QApplication.instance()


def _appearance_snapshot():
    application = _running_application()
    if application is None:
        return None
    from PySide6.QtGui import QPalette

    from xrr_fitter.gui import theme

    return application.styleSheet(), QPalette(application.palette()), theme.LIGHT_PALETTE


def _restore_appearance(snapshot) -> None:
    application = _running_application()
    if application is None:
        return
    from PySide6.QtCore import QSettings

    from xrr_fitter.gui import theme

    sheet, palette, light = ("", None, None) if snapshot is None else snapshot
    if application.styleSheet() != sheet:
        application.setStyleSheet(sheet)
    if palette is not None and application.palette() != palette:
        application.setPalette(palette)
    # 深色切换会记住「来时的亮色板」；留着它，下一条用例切回亮色时拿到的是别人的板。
    theme.LIGHT_PALETTE = light
    # 明暗偏好落在 QSettings（已由 ``_isolated_user_settings`` 重定向到临时目录）；点过开关的
    # 用例不清掉它，后面每一扇新窗口都会按深色开。
    QSettings().clear()


@pytest.fixture(autouse=True)
def _application_appearance_is_restored():
    """把用例留在会话唯一那个 ``QApplication`` 上的外观还原回去。

    ``create_application`` 会装上 ``theme.apply_theme`` 的样式表，多数 GUI 用例却不走它。命令栏
    明暗开关那条用例（``test_full_control_operations``）点两下之后样式表就留下了，排在后面的用例
    量到的是发货外观，单跑量到的是裸界面：眉标字体与导入对话框尺寸两条用例单跑能过、宽跑失败。
    要量发货外观的用例自己装、自己卸（见 ``test_shipped_theme_contracts``）。没有起过 Qt 的用例
    什么也不做。
    """
    snapshot = _appearance_snapshot()
    yield
    _restore_appearance(snapshot)


@pytest.fixture
def load_tool_module():
    def load(name: str):
        path = ROOT / "tools" / f"{name}.py"
        if not path.is_file():
            pytest.fail(f"missing implementation: tools/{name}.py", pytrace=False)
        spec = importlib.util.spec_from_file_location(f"r23_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    return load
