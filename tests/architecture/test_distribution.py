from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_gui_entrypoint_and_shell_modules_are_explicit() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["gui-scripts"] == {
        "xrr-fitter": "xrr_fitter.__main__:main"
    }
    required = {
        ROOT / "src/xrr_fitter/__main__.py",
        ROOT / "src/xrr_fitter/gui/__init__.py",
        ROOT / "src/xrr_fitter/gui/application.py",
        ROOT / "src/xrr_fitter/gui/document.py",
        ROOT / "src/xrr_fitter/gui/main_window.py",
    }
    assert all(path.is_file() and not path.is_symlink() for path in required)
