from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/lock_audit_tools.py").is_file(), "missing hash-pinned audit tool lock generator"
    return load_tool_module("lock_audit_tools")


def _report():
    return {
        "version": "1",
        "pip_version": "26.1.2",
        "environment": {"python_version": "3.12", "sys_platform": "darwin", "platform_machine": "arm64"},
        "install": [
            {
                "is_direct": False,
                "is_yanked": False,
                "metadata": {"name": "sample", "version": "1.0", "requires_dist": []},
                "download_info": {
                    "url": "https://files.pythonhosted.org/packages/sample-1.0-py3-none-any.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                },
            }
        ],
    }


def test_lock_uses_selected_wheel_hash_with_canonical_bytes(load_tool_module):
    module = _module(load_tool_module)
    report = _report()
    assert module.render_lock(report, ("sample==1.0",), ()) == f"sample==1.0 --hash=sha256:{'a' * 64}\n".encode()


@pytest.mark.parametrize("mutation", ["hash", "duplicate", "windows", "url", "wheel", "pin", "closure", "yanked"])
def test_lock_rejects_unverifiable_resolution(load_tool_module, mutation):
    module = _module(load_tool_module)
    report = _report()
    item = report["install"][0]
    if mutation == "hash":
        item["download_info"]["archive_info"]["hashes"].clear()
    elif mutation == "duplicate":
        report["install"].append(deepcopy(item))
    elif mutation == "windows":
        report["environment"]["sys_platform"] = "win32"
    elif mutation == "url":
        item["download_info"]["url"] = "file:///sample-1.0-py3-none-any.whl"
    elif mutation == "wheel":
        item["download_info"]["url"] = "https://files.pythonhosted.org/packages/other-1.0-py3-none-any.whl"
    elif mutation == "pin":
        item["metadata"]["version"] = "2"
    elif mutation == "closure":
        item["metadata"]["requires_dist"] = ["missing>=1"]
    else:
        item["is_yanked"] = True
    with pytest.raises(ValueError):
        module.render_lock(report, ("sample==1.0",), ())


def test_tool_lock_respects_shared_application_pin(load_tool_module):
    module = _module(load_tool_module)
    with pytest.raises(ValueError, match="constraint"):
        module.render_lock(_report(), ("sample==1.0",), ("sample==2.0",))


def test_lock_requires_every_direct_tool(load_tool_module):
    module = _module(load_tool_module)
    with pytest.raises(ValueError, match="missing"):
        module.render_lock(_report(), ("sample==1.0", "missing==2"), ())


def test_lock_tracks_extras_in_transitive_dependency_closure(load_tool_module):
    module = _module(load_tool_module)
    report = _report()
    report["install"][0]["metadata"]["requires_dist"] = ['extra-dependency>=1; extra == "check"']
    with pytest.raises(ValueError, match="missing"):
        module.render_lock(report, ("sample[check]==1.0",), ())
