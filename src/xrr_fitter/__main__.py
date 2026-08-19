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
    from xrr_fitter.gui.application import create_application
    from xrr_fitter.gui.main_window import MainWindow

    application = create_application(list(argv))
    window = MainWindow()
    window.show()
    return application.exec()


def main(argv: Sequence[str] | None = None) -> int:
    """Enable frozen workers, parse arguments, and launch the Qt event loop."""
    freeze_support()
    arguments = list(sys.argv if argv is None else (sys.argv[0], *argv))
    _parser().parse_args(arguments[1:])
    return _launch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
