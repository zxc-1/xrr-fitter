"""Command-line entry point for the desktop application."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from multiprocessing import freeze_support


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="xrr-fitter",
        description="X-ray reflectivity fitting desktop application",
    )


def _launch(argv: Sequence[str]) -> int:
    from xrr_fitter.gui.application import create_application, show_splash

    application = create_application(list(argv))
    splash = show_splash(application)

    def stage(message: str) -> None:
        splash.showMessage(message)
        application.processEvents()

    # main_window 牵出 pyqtgraph / matplotlib / 拟合内核，是冷启动最长的一段；
    # 放在启动画面之后导入，等待期间画面才看得见，并能逐段报告进度。
    stage("正在载入绘图与拟合模块…")
    from xrr_fitter.gui.main_window import MainWindow

    stage("正在构建工作区…")
    window = MainWindow()
    window.show()
    splash.finish(window)
    return application.exec()


def main(argv: Sequence[str] | None = None) -> int:
    """Enable frozen workers, parse arguments, and launch the Qt event loop."""
    freeze_support()
    arguments = list(sys.argv if argv is None else (sys.argv[0], *argv))
    _parser().parse_args(arguments[1:])
    return _launch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
