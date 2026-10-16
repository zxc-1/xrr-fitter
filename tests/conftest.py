from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


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


@pytest.fixture
def controls(qtbot, request):
    """The real-input control driver every control sweep shares.

    It lives here instead of being imported into each sweep module: an imported fixture reads
    as an unused import, and ruff's autofix in the pre-commit hook deletes it, which leaves the
    sweep without its driver. Qt is imported only when a test asks for this fixture.
    """
    from tests.gui.control_operation_driver import driven_controls

    yield from driven_controls(qtbot, request)
