from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


CASE_IDS = (
    "known_single_layer",
    "unstable_multilayer",
    "workable_mo_si_multilayer",
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _file_record(path: str, content: bytes) -> dict[str, object]:
    return {
        "path": path,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _candidate(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    raw = tmp_path / "raw"
    raw.mkdir()
    cases = []
    for index, case_id in enumerate(CASE_IDS):
        source_content = f"{case_id}\n".encode()
        source_path = f"inputs/{case_id}.xy"
        target = raw / source_path
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(source_content)
        runs = []
        for ordinal in range(1, 5):
            prefix = f"{case_id}/run-{ordinal}"
            runs.append(
                {
                    "ordinal": ordinal,
                    "seed": 4200 + index if ordinal < 4 else 5200 + index,
                    "project": _file_record(f"{prefix}/result.xrrproj.json", b"project"),
                    "exports": [_file_record(f"{prefix}/fit.csv", b"export")],
                    "plots": [_file_record(f"{prefix}/fit.png", b"plot")],
                    "normalized_result": {
                        "confidence": "trusted" if index < 2 else "untrusted",
                        "metrics": {"objective": 0.01 + index},
                        "warnings": [] if index < 2 else ["unstable basin"],
                    },
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "source": _file_record(source_path, source_content),
                "configuration_sha256": hashlib.sha256(f"config-{index}".encode()).hexdigest(),
                "operations": ["import", "fit", "save", "export", "plot", "reopen"],
                "runs": runs,
                "normalized_result": {
                    "confidence": "trusted" if index < 2 else "untrusted",
                    "metrics": {"repeatability": 0.001 + index},
                    "warnings": [] if index < 2 else ["owner review required"],
                },
                "conclusion": f"approved conclusion for {case_id}",
            }
        )
    value = {
        "schema": "xrr-r23-approved-data-candidate-v1",
        "environment": {
            "python_version": "3.12.13",
            "platform": "macos-arm64",
            "dependency_lock_sha256": "1" * 64,
            "production_tree_sha256": "2" * 64,
            "acceptance_test_tree_sha256": "3" * 64,
            "qt_runtime_identity": "PySide6 6.9.1 / Qt 6.9.1",
        },
        "workflow_contract_sha256": "4" * 64,
        "cases": cases,
    }
    return raw, value


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    raw, candidate = _candidate(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_bytes(_canonical(candidate))
    candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    signoff = {
        "schema": "xrr-r23-domain-signoff-v1",
        "reviewer": "Domain Owner",
        "role": "XRR acceptance owner",
        "candidate_report_sha256": candidate_hash,
        "cases": [
            {
                "case_id": case["case_id"],
                "approved": True,
                "conclusion": case["conclusion"],
            }
            for case in candidate["cases"]
        ],
    }
    signoff_path = tmp_path / "signoff.json"
    signoff_path.write_bytes(_canonical(signoff))
    reference = tmp_path / "r22-reference.json"
    reference.write_bytes(b'{"schema":"frozen-reference"}\n')
    output = tmp_path / "approved-data"
    return raw, candidate_path, signoff_path, reference, candidate


def _freeze(module, tmp_path: Path):
    raw, candidate, signoff, reference, _value = _write_inputs(tmp_path)
    output = tmp_path / "approved-data"
    module.freeze_approved_data(candidate, signoff, raw, reference, output)
    return raw, candidate, signoff, reference, output


def test_check_candidate_validates_without_writing(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("freeze_approved_data")
    raw, candidate, _signoff, reference, _value = _write_inputs(tmp_path)

    checked = module.check_candidate(candidate, raw, reference)

    assert tuple(case.case_id for case in checked.cases) == CASE_IDS
    assert set(tmp_path.iterdir()) == {raw, candidate, _signoff, reference}


def test_freeze_atomically_publishes_manifest_and_three_records(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("freeze_approved_data")
    raw, candidate, signoff, reference, output = _freeze(module, tmp_path)

    assert {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()} == {
        "manifest.json",
        *(f"records/{case_id}.json" for case_id in CASE_IDS),
    }
    binding = module.calculate_approved_data_binding(output, raw)
    assert binding.status == "PASS"
    assert binding.candidate_report_sha256 == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert binding.domain_signoff_sha256 == hashlib.sha256(signoff.read_bytes()).hexdigest()
    assert binding.manifest.path == "verification/approved-data/manifest.json"
    module.validate_approved_data(output, raw, reference)


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate-case",
        "missing-case",
        "extra-case",
        "path-traversal",
        "bad-source-hash",
        "bad-run-order",
        "same-fourth-seed",
        "empty-operations",
        "noncanonical-export-order",
    ),
)
def test_candidate_rejects_case_path_source_run_and_operation_drift(
    tmp_path: Path,
    load_tool_module,
    mutation: str,
) -> None:
    module = load_tool_module("freeze_approved_data")
    raw, value = _candidate(tmp_path)
    if mutation == "duplicate-case":
        value["cases"][1] = copy.deepcopy(value["cases"][0])
    elif mutation == "missing-case":
        value["cases"].pop()
    elif mutation == "extra-case":
        value["cases"].append(copy.deepcopy(value["cases"][0]))
        value["cases"][-1]["case_id"] = "extra"
    elif mutation == "path-traversal":
        value["cases"][0]["source"]["path"] = "../outside.xy"
    elif mutation == "bad-source-hash":
        value["cases"][0]["source"]["sha256"] = "f" * 64
    elif mutation == "bad-run-order":
        value["cases"][0]["runs"] = list(reversed(value["cases"][0]["runs"]))
    elif mutation == "same-fourth-seed":
        value["cases"][0]["runs"][3]["seed"] = value["cases"][0]["runs"][0]["seed"]
    elif mutation == "empty-operations":
        value["cases"][0]["operations"] = []
    else:
        value["cases"][0]["runs"][0]["exports"] = [
            _file_record("z.csv", b"z"),
            _file_record("a.csv", b"a"),
        ]
    path = tmp_path / "candidate.json"
    path.write_bytes(_canonical(value))

    with pytest.raises(ValueError):
        module.check_candidate(path, raw, tmp_path / "reference.json")


@pytest.mark.parametrize(
    "mutation",
    ("project-export", "export-plot", "different-runs", "different-cases"),
)
def test_candidate_parser_rejects_duplicate_evidence_paths(
    tmp_path: Path,
    load_tool_module,
    mutation: str,
) -> None:
    module = load_tool_module("freeze_approved_data")
    _raw, value = _candidate(tmp_path)
    first_run = value["cases"][0]["runs"][0]
    if mutation == "project-export":
        first_run["exports"][0]["path"] = first_run["project"]["path"]
    elif mutation == "export-plot":
        first_run["plots"][0]["path"] = first_run["exports"][0]["path"]
    elif mutation == "different-runs":
        value["cases"][0]["runs"][1]["project"]["path"] = first_run["project"]["path"]
    else:
        value["cases"][1]["runs"][0]["project"]["path"] = first_run["project"]["path"]

    with pytest.raises(ValueError, match="evidence paths"):
        module.parse_candidate_report(_canonical(value))


def test_committed_record_parser_rejects_duplicate_evidence_paths(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("freeze_approved_data")
    _raw, _candidate_path, _signoff, _reference, output = _freeze(module, tmp_path)
    record = output / "records" / f"{CASE_IDS[0]}.json"
    value = json.loads(record.read_text(encoding="utf-8"))
    value["runs"][0]["plots"][0]["path"] = value["runs"][0]["project"]["path"]

    with pytest.raises(ValueError, match="evidence paths"):
        module.parse_approved_case_record(_canonical(value))


@pytest.mark.parametrize(
    "mutation",
    (
        "reviewer",
        "role",
        "candidate-hash",
        "approval",
        "conclusion",
        "case-order",
    ),
)
def test_freeze_rejects_signoff_projection_drift(
    tmp_path: Path,
    load_tool_module,
    mutation: str,
) -> None:
    module = load_tool_module("freeze_approved_data")
    raw, candidate, signoff, reference, _value = _write_inputs(tmp_path)
    payload = json.loads(signoff.read_text(encoding="utf-8"))
    if mutation in {"reviewer", "role"}:
        payload[mutation] = ""
    elif mutation == "candidate-hash":
        payload["candidate_report_sha256"] = "9" * 64
    elif mutation == "approval":
        payload["cases"][0]["approved"] = False
    elif mutation == "conclusion":
        payload["cases"][0]["conclusion"] = "different"
    else:
        payload["cases"] = list(reversed(payload["cases"]))
    signoff.write_bytes(_canonical(payload))

    with pytest.raises(ValueError):
        module.freeze_approved_data(
            candidate,
            signoff,
            raw,
            reference,
            tmp_path / "approved-data",
        )


def _committed_projection_path(output: Path, target: str) -> Path:
    if target == "environment":
        return output / "manifest.json"
    return output / "records" / f"{CASE_IDS[0]}.json"


def _tamper_normalized_result(result: dict[str, object], target: str) -> None:
    if target == "warning":
        result["warnings"].append("tampered")
    elif target == "confidence":
        result["confidence"] = "tampered"
    else:
        result["metrics"]["repeatability"] = 99


def _tamper_committed_projection(value: dict[str, object], target: str) -> None:
    if target == "environment":
        value["environment"]["qt_runtime_identity"] += " tampered"
    elif target == "operations":
        value["operations"].append("tampered")
    elif target == "run":
        value["runs"][0]["seed"] += 1
    elif target == "project":
        value["runs"][0]["project"]["sha256"] = "a" * 64
    elif target == "export":
        value["runs"][0]["exports"][0]["sha256"] = "a" * 64
    elif target == "plot":
        value["runs"][0]["plots"][0]["sha256"] = "a" * 64
    elif target in {"warning", "confidence", "metric"}:
        _tamper_normalized_result(value["normalized_result"], target)
    else:
        value["signoff"]["conclusion"] = "tampered"


@pytest.mark.parametrize(
    "target",
    (
        "environment",
        "operations",
        "run",
        "project",
        "export",
        "plot",
        "warning",
        "confidence",
        "metric",
        "conclusion",
    ),
)
def test_committed_projection_detects_every_candidate_field_tamper(
    tmp_path: Path,
    load_tool_module,
    target: str,
) -> None:
    module = load_tool_module("freeze_approved_data")
    raw, _candidate_path, _signoff, reference, output = _freeze(module, tmp_path)
    path = _committed_projection_path(output, target)
    value = json.loads(path.read_text(encoding="utf-8"))
    _tamper_committed_projection(value, target)
    path.write_bytes(_canonical(value))

    with pytest.raises(ValueError):
        module.calculate_approved_data_binding(output, raw)
    with pytest.raises(ValueError):
        module.validate_approved_data(output, raw, reference)


def test_parser_rejects_duplicate_keys_and_noncanonical_json(
    load_tool_module,
) -> None:
    module = load_tool_module("freeze_approved_data")
    for content in (
        b'{"schema":"x","schema":"y"}\n',
        b"{}",
        b"{}\r\n",
        b"{ }\n",
        b"[]\n",
        b"\xff",
    ):
        with pytest.raises(ValueError):
            module.parse_candidate_report(content)


def test_source_drift_symlink_and_existing_output_are_rejected(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("freeze_approved_data")
    raw, candidate, signoff, reference, _value = _write_inputs(tmp_path)
    source = raw / "inputs" / f"{CASE_IDS[0]}.xy"
    source.write_bytes(b"drift")
    with pytest.raises(ValueError):
        module.freeze_approved_data(candidate, signoff, raw, reference, tmp_path / "out-a")

    source.unlink()
    source.symlink_to(raw / "inputs" / f"{CASE_IDS[1]}.xy")
    with pytest.raises(ValueError):
        module.freeze_approved_data(candidate, signoff, raw, reference, tmp_path / "out-b")

    source.unlink()
    source.write_bytes(f"{CASE_IDS[0]}\n".encode())
    output = tmp_path / "out-c"
    output.mkdir()
    with pytest.raises(ValueError, match="exist"):
        module.freeze_approved_data(candidate, signoff, raw, reference, output)


def test_atomic_failure_leaves_no_partial_output(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("freeze_approved_data")
    raw, candidate, signoff, reference, _value = _write_inputs(tmp_path)
    output = tmp_path / "approved-data"
    monkeypatch.setattr(
        module.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        module.freeze_approved_data(candidate, signoff, raw, reference, output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".approved-data.*"))
