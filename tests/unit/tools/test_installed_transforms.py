from __future__ import annotations

import io
import types
import zipfile

import pytest
from tests.support.installed_wheels import SCRIPT_TEMPLATE


def _pip(source):
    return types.SimpleNamespace(record={"name": "pip", "version": "26.1.2"}, read=lambda name: source.encode())


@pytest.mark.parametrize("call", ["unknown", "other.dedent", "textwrap.indent"])
def test_pip_template_rejects_a_different_transform(load_tool_module, call):
    module = load_tool_module("installed_transforms")
    wheel = _pip(f"class PipScriptMaker:\n    script_template = {call}({SCRIPT_TEMPLATE!r})\n")
    with pytest.raises(ValueError, match="template|transform"):
        module.pip_template(wheel)


def test_pip_template_rejects_keyword_arguments(load_tool_module):
    module = load_tool_module("installed_transforms")
    wheel = _pip(f"class PipScriptMaker:\n    script_template = textwrap.dedent({SCRIPT_TEMPLATE!r}, extra=True)\n")
    with pytest.raises(ValueError, match="template|transform"):
        module.pip_template(wheel)


def _launcher(body=b"entry point\n", *, extra=False):
    buffer = io.BytesIO()
    info = zipfile.ZipInfo("__main__.py", (2026, 9, 11, 12, 0, 0))
    info.create_system = 0
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(info, body)
        if extra:
            archive.writestr("extra.py", b"not allowed")
    return b"MZ pinned launcher" + b"#!python.exe\n" + buffer.getvalue()


def test_windows_launcher_uses_the_pinned_image_and_exact_zip_framing(load_tool_module):
    module = load_tool_module("installed_transforms")
    actual = _launcher()
    assert module.windows_launcher(actual, b"MZ pinned launcher", b"#!python.exe\n", b"entry point\n") == actual


@pytest.mark.parametrize("change", ["prefix", "suffix", "body", "extra"])
def test_windows_launcher_does_not_authorize_extra_or_changed_bytes(load_tool_module, change):
    module = load_tool_module("installed_transforms")
    actual = _launcher(extra=change == "extra")
    if change == "prefix":
        actual = b"different prefix" + actual
    elif change == "suffix":
        actual += b"extra payload"
    elif change == "body":
        actual = _launcher(b"different body\n")
    if change == "extra":
        with pytest.raises(ValueError, match="members"):
            module.windows_launcher(actual, b"MZ pinned launcher", b"#!python.exe\n", b"entry point\n")
    else:
        expected = module.windows_launcher(actual, b"MZ pinned launcher", b"#!python.exe\n", b"entry point\n")
        assert expected != actual


def test_windows_gui_interpreter_changes_only_the_filename(load_tool_module, tmp_path, monkeypatch):
    module = load_tool_module("installed_transforms")
    layout = types.SimpleNamespace(
        target="windows-x64-py312", interpreter=r"C:\python-tools\Scripts\python.exe", scripts=tmp_path
    )
    seen = []
    monkeypatch.setattr(module, "windows_launcher", lambda actual, launcher, header, body: seen.append(header) or b"")
    pip = types.SimpleNamespace(read=lambda name: b"pinned launcher")
    snapshot = types.SimpleNamespace(read=lambda path: b"launcher")
    module._entry_script("hello", "sample:main", "gui_scripts", layout, pip, SCRIPT_TEMPLATE, snapshot)
    assert seen == [b"#!C:\\python-tools\\Scripts\\pythonw.exe\n"]
