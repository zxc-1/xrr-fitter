from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest


HEADER = (
    "source_tree",
    "source_nodeid",
    "contract_id",
    "action",
    "target_nodeids",
    "reason",
)
SOURCE_HASH = "b" * 64


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _seal(payload: dict[str, object]) -> dict[str, object]:
    base = {key: value for key, value in payload.items() if key != "collection_sha256"}
    return {**base, "collection_sha256": hashlib.sha256(_canonical(base)).hexdigest()}


def _manifest_payload(suite: str, nodeids: tuple[str, ...]) -> dict[str, object]:
    paths = sorted({nodeid.split("::", 1)[0] for nodeid in nodeids})
    return _seal(
        {
            "schema": "xrr-test-manifest-v1",
            "source_commit": "c" * 40,
            "suite": suite,
            "test_tree": [
                {"path": path, "size": 1, "sha256": SOURCE_HASH} for path in paths
            ],
            "node_count": len(nodeids),
            "nodes": [{"nodeid": nodeid, "markers": []} for nodeid in sorted(nodeids)],
            "python_version": "3.12.13",
            "platform": "macOS-arm64",
            "lock_sha256": "d" * 64,
        }
    )


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(_canonical(payload))


def _ledger_bytes(rows: list[dict[str, str]], header: tuple[str, ...] = HEADER) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _write_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_bytes(_ledger_bytes(rows))


def _row(tree: str, nodeid: str, contract: str, target: str) -> dict[str, str]:
    return {
        "source_tree": tree,
        "source_nodeid": nodeid,
        "contract_id": contract,
        "action": "rewrite",
        "target_nodeids": json.dumps([target], ensure_ascii=False, separators=(",", ":")),
        "reason": "preserve behavior",
    }


def _draft_case(tmp_path: Path) -> tuple[Path, Path, Path, list[dict[str, str]]]:
    active = tmp_path / "active.json"
    r21 = tmp_path / "r21.json"
    ledger = tmp_path / "ledger.csv"
    _write_manifest(active, _manifest_payload("tests", ("tests/test_a.py::test_a",)))
    _write_manifest(r21, _manifest_payload("tests_r21", ("tests_r21/test_b.py::test_b",)))
    rows = [
        _row("tests", "tests/test_a.py::test_a", "core.a", "tests/unit/test_a.py::test_a"),
        _row(
            "tests_r21",
            "tests_r21/test_b.py::test_b",
            "core.b",
            "tests/unit/test_b.py::test_b",
        ),
    ]
    _write_ledger(ledger, rows)
    return active, r21, ledger, rows


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", "wrong", "schema"),
        ("source_commit", "C" * 40, "source commit"),
        ("source_commit", "c" * 39, "source commit"),
        ("python_version", "", "python_version"),
        ("platform", 12, "platform"),
        ("lock_sha256", "D" * 64, "lock hash"),
        ("node_count", True, "node count"),
    ],
)
def test_manifest_rejects_invalid_schema_fields(
    tmp_path: Path, load_tool_module, field: str, value: object, match: str
) -> None:
    module = load_tool_module("validate_test_ledger")
    path = tmp_path / "manifest.json"
    payload = _manifest_payload("tests", ("tests/test_a.py::test_a",))
    payload[field] = value
    _write_manifest(path, _seal(payload))
    with pytest.raises(ValueError, match=match):
        module._manifest(path, "tests")


def test_manifest_requires_exact_keys_and_expected_suite(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("validate_test_ledger")
    path = tmp_path / "manifest.json"
    payload = _manifest_payload("tests", ("tests/test_a.py::test_a",))
    payload["unexpected"] = "value"
    _write_manifest(path, _seal(payload))
    with pytest.raises(ValueError, match="schema"):
        module._manifest(path, "tests")
    del payload["unexpected"]
    _write_manifest(path, _seal(payload))
    with pytest.raises(ValueError, match="suite"):
        module._manifest(path, "tests_r21")


def test_manifest_rejects_noncanonical_duplicate_and_hash_drift(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("validate_test_ledger")
    path = tmp_path / "manifest.json"
    payload = _manifest_payload("tests", ("tests/test_a.py::test_a",))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        module._manifest(path, "tests")
    encoded = _canonical(payload).replace(
        b'"schema":"xrr-test-manifest-v1"',
        b'"schema":"xrr-test-manifest-v1","schema":"xrr-test-manifest-v1"',
    )
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        module._manifest(path, "tests")
    payload["collection_sha256"] = "0" * 64
    _write_manifest(path, payload)
    with pytest.raises(ValueError, match="collection hash"):
        module._manifest(path, "tests")


@pytest.mark.parametrize(
    "tree",
    [
        {},
        [{"path": "tests/test_a.py", "size": 1}],
        [{"path": "tests_r21/test_a.py", "size": 1, "sha256": SOURCE_HASH}],
        [{"path": "tests/../test_a.py", "size": 1, "sha256": SOURCE_HASH}],
        [{"path": "tests/test_a.py", "size": True, "sha256": SOURCE_HASH}],
        [{"path": "tests/test_a.py", "size": -1, "sha256": SOURCE_HASH}],
        [{"path": "tests/test_a.py", "size": 1, "sha256": "B" * 64}],
        [
            {"path": "tests/test_b.py", "size": 1, "sha256": SOURCE_HASH},
            {"path": "tests/test_a.py", "size": 1, "sha256": SOURCE_HASH},
        ],
        [
            {"path": "tests/test_a.py", "size": 1, "sha256": SOURCE_HASH},
            {"path": "tests/test_a.py", "size": 1, "sha256": SOURCE_HASH},
        ],
    ],
)
def test_manifest_rejects_invalid_tree_records(tmp_path: Path, load_tool_module, tree: object) -> None:
    module = load_tool_module("validate_test_ledger")
    path = tmp_path / "manifest.json"
    payload = _manifest_payload("tests", ("tests/test_a.py::test_a",))
    payload["test_tree"] = tree
    _write_manifest(path, _seal(payload))
    with pytest.raises(ValueError, match="test_tree|path|size|hash|record|canonical"):
        module._manifest(path, "tests")


@pytest.mark.parametrize(
    "nodes",
    [
        {},
        [{"nodeid": "tests/test_a.py::test_a"}],
        [{"nodeid": "tests_r21/test_a.py::test_a", "markers": []}],
        [{"nodeid": "tests/../test_a.py::test_a", "markers": []}],
        [{"nodeid": "tests/test_a.py::test_a", "markers": "slow"}],
        [{"nodeid": "tests/test_a.py::test_a", "markers": [[]]}],
        [{"nodeid": "tests/test_a.py::test_a", "markers": ["slow", "fast"]}],
        [{"nodeid": "tests/test_a.py::test_a", "markers": ["slow", "slow"]}],
        [
            {"nodeid": "tests/test_b.py::test_b", "markers": []},
            {"nodeid": "tests/test_a.py::test_a", "markers": []},
        ],
        [
            {"nodeid": "tests/test_a.py::test_a", "markers": []},
            {"nodeid": "tests/test_a.py::test_a", "markers": []},
        ],
    ],
)
def test_manifest_rejects_invalid_node_records(
    tmp_path: Path, load_tool_module, nodes: object
) -> None:
    module = load_tool_module("validate_test_ledger")
    path = tmp_path / "manifest.json"
    payload = _manifest_payload("tests", ("tests/test_a.py::test_a",))
    payload["nodes"] = nodes
    payload["node_count"] = len(nodes) if isinstance(nodes, list) else 0
    _write_manifest(path, _seal(payload))
    with pytest.raises(ValueError, match="node|marker|canonical|record"):
        module._manifest(path, "tests")


def test_manifest_requires_every_node_file_in_test_tree(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("validate_test_ledger")
    path = tmp_path / "manifest.json"
    payload = _manifest_payload("tests", ("tests/test_a.py::test_a",))
    payload["test_tree"] = []
    _write_manifest(path, _seal(payload))
    with pytest.raises(ValueError, match="test.tree|node file"):
        module._manifest(path, "tests")


@pytest.mark.parametrize("value", ["tests", "tests/../test_a.py", "tests//test_a.py"])
def test_manifest_paths_must_be_canonical_suite_descendants(
    load_tool_module, value: str
) -> None:
    module = load_tool_module("validate_test_ledger")
    with pytest.raises(ValueError, match="path"):
        module._canonical_path(value, "tests", "node")


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (b"not,the,header\n", "header"),
        (b"\xff\n", "UTF-8"),
        ((",".join(HEADER) + "\r\n").encode(), "LF"),
        ((",".join(HEADER)).encode(), "LF"),
        ((",".join(HEADER) + "\na,b,c\n").encode(), "width"),
        ((",".join(HEADER) + "\na,b,c,d,e,f,g,h\n").encode(), "width"),
    ],
)
def test_ledger_rejects_invalid_encoding_header_and_width(
    tmp_path: Path, load_tool_module, content: bytes, match: str
) -> None:
    module = load_tool_module("validate_test_ledger")
    ledger = tmp_path / "ledger.csv"
    ledger.write_bytes(content)
    with pytest.raises(ValueError, match=match):
        module._read_ledger(ledger)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_tree", "legacy", "source_tree"),
        ("source_nodeid", "tests/test_a.py::test_other", "coverage"),
        ("contract_id", "Core.a", "contract_id"),
        ("contract_id", "core a", "contract_id"),
        ("contract_id", "", "contract_id"),
        ("action", "skip", "action"),
        ("reason", "", "reason"),
        ("reason", " padded", "reason"),
        ("target_nodeids", "not-json", "canonical JSON"),
        ("target_nodeids", "[]", "non-empty"),
        ("target_nodeids", '["tests/z.py::test_z","tests/z.py::test_z"]', "sorted and unique"),
        ("target_nodeids", '["tests/z.py::test_z","tests/a.py::test_a"]', "sorted and unique"),
        ("target_nodeids", '[ "tests/a.py::test_a" ]', "canonical JSON"),
        ("target_nodeids", '["legacy/test_a.py::test_a"]', "target nodeid"),
        ("target_nodeids", '["tests/../test_a.py::test_a"]', "target nodeid"),
        ("target_nodeids", "[1]", "target nodeid"),
    ],
)
def test_source_draft_rejects_invalid_row_contract(
    tmp_path: Path, load_tool_module, field: str, value: str, match: str
) -> None:
    module = load_tool_module("validate_test_ledger")
    active, r21, ledger, rows = _draft_case(tmp_path)
    rows[0][field] = value
    _write_ledger(ledger, rows)
    with pytest.raises(ValueError, match=match):
        module.validate_source_draft(active, r21, ledger)


@pytest.mark.parametrize("action", ["port", "rewrite", "merge", "delete_layout_only"])
def test_source_draft_accepts_every_action_without_claiming_target_existence(
    tmp_path: Path, load_tool_module, action: str
) -> None:
    module = load_tool_module("validate_test_ledger")
    active, r21, ledger, rows = _draft_case(tmp_path)
    rows[0]["action"] = action
    rows[0]["target_nodeids"] = '["tests/not_created_yet.py::test_future"]'
    _write_ledger(ledger, rows)
    result = module.validate_source_draft(active, r21, ledger)
    assert result["phase"] == "source-draft"
    assert result["source_count"] == 2


def test_source_draft_requires_exact_once_only_coverage(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("validate_test_ledger")
    active, r21, ledger, rows = _draft_case(tmp_path)
    _write_ledger(ledger, rows[:1])
    with pytest.raises(ValueError, match="coverage"):
        module.validate_source_draft(active, r21, ledger)
    _write_ledger(ledger, [*rows, rows[0]])
    with pytest.raises(ValueError, match="duplicate source coverage"):
        module.validate_source_draft(active, r21, ledger)


def test_final_requires_every_target_to_exist(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("validate_test_ledger")
    active, r21, ledger, _ = _draft_case(tmp_path)
    target = tmp_path / "target.json"
    _write_manifest(target, _manifest_payload("tests", ("tests/unit/test_a.py::test_a",)))
    with pytest.raises(ValueError, match="target nodeid does not exist"):
        module.validate_final(active, r21, ledger, target)


def test_final_restricts_every_delete_layout_target_to_architecture(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("validate_test_ledger")
    active, r21, ledger, rows = _draft_case(tmp_path)
    architecture = "tests/architecture/test_layout.py::test_removed"
    unit = "tests/unit/test_layout.py::test_removed"
    rows[0]["action"] = "delete_layout_only"
    rows[0]["target_nodeids"] = json.dumps([architecture, unit], separators=(",", ":"))
    rows[1]["target_nodeids"] = f'["{architecture}"]'
    _write_ledger(ledger, rows)
    target = tmp_path / "target.json"
    _write_manifest(target, _manifest_payload("tests", (architecture, unit)))
    with pytest.raises(ValueError, match="architecture"):
        module.validate_final(active, r21, ledger, target)


def test_final_accepts_existing_targets_and_reports_final_phase(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("validate_test_ledger")
    active, r21, ledger, rows = _draft_case(tmp_path)
    architecture = "tests/architecture/test_layout.py::test_removed"
    unit = "tests/unit/test_a.py::test_a"
    rows[0]["target_nodeids"] = f'["{unit}"]'
    rows[1]["action"] = "delete_layout_only"
    rows[1]["target_nodeids"] = f'["{architecture}"]'
    _write_ledger(ledger, rows)
    target = tmp_path / "target.json"
    _write_manifest(target, _manifest_payload("tests", (architecture, unit)))
    result = module.validate_final(active, r21, ledger, target)
    assert result["phase"] == "final"
    assert result["source_count"] == 2
    assert result["target_count"] == 2


def test_cli_exposes_only_source_draft_and_final_phases(
    tmp_path: Path, load_tool_module, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_tool_module("validate_test_ledger")
    active, r21, ledger, _ = _draft_case(tmp_path)
    common = [
        "--active-manifest",
        str(active),
        "--r21-manifest",
        str(r21),
        "--ledger",
        str(ledger),
    ]
    with pytest.raises(SystemExit):
        module.main(["--phase", "draft", *common])
    with pytest.raises(SystemExit):
        module.main(["--phase", "final", *common])
    with pytest.raises(SystemExit):
        module.main(["--phase", "source-draft", *common, "--target-manifest", str(active)])
    assert module.main(["--phase", "source-draft", *common]) == 0
    assert json.loads(capsys.readouterr().out)["phase"] == "source-draft"
