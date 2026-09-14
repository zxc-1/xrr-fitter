from __future__ import annotations

import hashlib
import json

import pytest


def _evidence():
    return {
        "schema": "xrr-headless-executable-evidence-v1",
        "source_commit": "a" * 40,
        "sha256": hashlib.sha256(b"exe bytes").hexdigest(),
        "size": len(b"exe bytes"),
        "help": "PASS",
        "validate": "PASS",
        "export": "PASS",
        "gui_tested": False,
    }


def test_executable_evidence_is_bound_to_the_exact_run_source_and_binary(load_tool_module):
    module = load_tool_module("executable_binding")
    module.verify_executable_evidence(_evidence(), b"exe bytes", {"source_commit": "a" * 40})


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_commit", "b" * 40),
        ("sha256", "c" * 64),
        ("size", 1),
        ("help", "FAIL"),
        ("validate", "FAIL"),
        ("export", "FAIL"),
        ("gui_tested", True),
        ("gui_tested", 0),
        ("schema", "unknown"),
    ],
)
def test_executable_evidence_rejects_mismatched_or_unsuccessful_runs(load_tool_module, field, value):
    module = load_tool_module("executable_binding")
    evidence = {**_evidence(), field: value}
    with pytest.raises(ValueError, match="evidence|source|bytes"):
        module.verify_executable_evidence(evidence, b"exe bytes", {"source_commit": "a" * 40})


def test_executable_receipt_rejects_duplicate_keys_before_source_inspection(tmp_path, load_tool_module, monkeypatch):
    module = load_tool_module("executable_binding")
    path, receipt = tmp_path / "example.exe", tmp_path / "receipt.json"
    path.write_bytes(b"exe bytes")
    receipt.write_text('{"gui_tested":true,' + json.dumps(_evidence())[1:])

    def unexpected(*args):
        raise AssertionError("duplicate evidence must fail before source inspection")

    monkeypatch.setattr(module, "committed_tree_files", unexpected)
    with pytest.raises(ValueError, match="duplicate"):
        module.bind_executable(
            tmp_path, path, receipt, [], {"target": "windows-x64-py312"}, {"source_commit": "a" * 40}
        )
