from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tests.support import approved_workflows as workflows

CASE_IDS = (
    "known_single_layer",
    "unstable_multilayer",
    "workable_mo_si_multilayer",
)


def _owner_root(tmp_path: Path) -> Path:
    root = tmp_path / "owner-data"
    root.mkdir()
    for case_id in CASE_IDS:
        (root / f"{case_id}.dat").write_text(f"fixture {case_id}\n", encoding="utf-8")
    return root


def _environment() -> dict[str, object]:
    return {
        "python_version": "3.12.13",
        "platform": "macos-arm64",
        "dependency_lock_sha256": "1" * 64,
        "production_tree_sha256": "2" * 64,
        "acceptance_test_tree_sha256": "3" * 64,
        "qt_runtime_identity": "PySide6 6.11.1 / Qt 6.11.1",
    }


def _record(path: Path, relative: str) -> workflows.FileRecord:
    content = path.read_bytes()
    return workflows.FileRecord(relative, len(content), hashlib.sha256(content).hexdigest())


def _fake_run(
    spec: workflows.CaseSpec,
    source: Path,
    seed: int,
    ordinal: int,
    staging: Path,
) -> workflows.RunRecord:
    directory = staging / spec.case_id / f"run-{ordinal}"
    directory.mkdir(parents=True)
    project = directory / "project.xrrproj.json"
    exported = directory / "fit_result.json"
    plot = directory / "fit_overview.png"
    project.write_text(f"project {spec.case_id} {seed}\n", encoding="utf-8")
    exported.write_text(f"export {spec.case_id} {seed}\n", encoding="utf-8")
    plot.write_bytes(f"plot {spec.case_id} {seed}\n".encode())
    prefix = f"runs/{spec.case_id}/run-{ordinal}"
    normalized = {
        "case_id": spec.case_id,
        "confidence": "fixture-only",
        "seed_class": "same" if ordinal < 4 else "fresh",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    return workflows.RunRecord(
        ordinal,
        seed,
        _record(project, f"{prefix}/project.xrrproj.json"),
        (_record(exported, f"{prefix}/fit_result.json"),),
        (_record(plot, f"{prefix}/fit_overview.png"),),
        normalized,
    )


def test_api_capture_writes_exact_canonical_four_run_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _owner_root(tmp_path)
    report = tmp_path / "report"
    monkeypatch.setattr(workflows, "_candidate_environment", _environment)
    monkeypatch.setattr(workflows, "_execute_run", _fake_run)

    records = workflows.run_api_acceptance(root, report)

    assert tuple(record.case_id for record in records) == CASE_IDS
    assert all(tuple(run.ordinal for run in record.runs) == (1, 2, 3, 4) for record in records)
    assert all(len({run.seed for run in record.runs[:3]}) == 1 for record in records)
    assert all(record.runs[3].seed != record.runs[0].seed for record in records)
    candidate = report / "approved-data-candidate.json"
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    assert tuple(case["case_id"] for case in payload["cases"]) == CASE_IDS
    assert tuple(case["source"]["path"] for case in payload["cases"]) == tuple(f"{case_id}.dat" for case_id in CASE_IDS)
    assert candidate.read_bytes() == workflows.canonical_json_bytes(payload)
    assert not any(path.name.startswith(".capture-") for path in report.iterdir())


def test_api_capture_failure_leaves_no_candidate_or_published_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _owner_root(tmp_path)
    report = tmp_path / "report"
    monkeypatch.setattr(workflows, "_candidate_environment", _environment)

    def fail_on_second(*args, **kwargs):
        if kwargs["ordinal"] == 2:
            raise RuntimeError("fit failed")
        return _fake_run(*args, **kwargs)

    monkeypatch.setattr(workflows, "_execute_run", fail_on_second)

    with pytest.raises(RuntimeError, match="fit failed"):
        workflows.run_api_acceptance(root, report)

    assert not (report / "approved-data-candidate.json").exists()
    assert not (report / "runs").exists()
    assert not tuple(report.glob(".capture-*"))


@pytest.mark.parametrize("failure", ("missing", "symlink"))
def test_api_capture_rejects_missing_or_linked_owner_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = _owner_root(tmp_path)
    source = root / "known_single_layer.dat"
    if failure == "missing":
        source.unlink()
    else:
        outside = tmp_path / "outside.dat"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
    monkeypatch.setattr(workflows, "_candidate_environment", _environment)

    with pytest.raises((FileNotFoundError, ValueError), match="source|regular"):
        workflows.run_api_acceptance(root, tmp_path / "report")


def test_gui_acceptance_consumes_only_published_candidate_projects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _owner_root(tmp_path)
    report = tmp_path / "report"
    monkeypatch.setattr(workflows, "_candidate_environment", _environment)
    monkeypatch.setattr(workflows, "_execute_run", _fake_run)
    workflows.run_api_acceptance(root, report)

    observed: list[str] = []

    def verify(case_id: str, project: Path, output: Path) -> workflows.GuiResult:
        observed.append(case_id)
        assert project.is_file()
        assert project.resolve().is_relative_to(report.resolve())
        output.mkdir(parents=True)
        return workflows.GuiResult(case_id, True, True, True)

    monkeypatch.setattr(workflows, "_verify_gui_case", verify)

    results = workflows.verify_gui_acceptance(root, report)

    assert tuple(observed) == CASE_IDS
    assert tuple(result.case_id for result in results) == CASE_IDS
    assert all(result.project_reopened for result in results)
