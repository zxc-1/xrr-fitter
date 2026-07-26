from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest


def _git(root: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True, capture_output=True)


def _write(root: Path, relative: str, source: str = "VALUE = 1\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _repo(tmp_path: Path, *, populate: bool = True) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Quality Test")
    _git(root, "config", "user.email", "quality@example.invalid")
    if populate:
        for relative in (
            "src/pkg/a.py",
            "tests/test_a.py",
            "tools/a.py",
            "examples/a.py",
        ):
            _write(root, relative)
    return root


def _issue_kinds(report: dict[str, object]) -> set[str]:
    issues = report["issues"]
    assert isinstance(issues, list)
    return {str(item["kind"]) for item in issues}


def test_discovery_and_json_report_cover_every_managed_root(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("check_radon")
    root = _repo(tmp_path)
    paths, issues = module.discover_python_files(root)
    expected = {"examples/a.py", "src/pkg/a.py", "tests/test_a.py", "tools/a.py"}
    assert {path.as_posix() for path in paths} == expected
    assert issues == ()
    report = module.build_report(root)
    assert {item["path"] for item in report["files"]} == expected
    assert json.loads(module._canonical(report)) == report


@pytest.mark.parametrize(
    ("relative", "kind"),
    [("outside.py", "ownership"), ("tests/ignored.py", "ignore-policy")],
)
def test_discovery_rejects_wrong_owner_and_ignored_python(
    tmp_path: Path, load_tool_module, relative: str, kind: str
) -> None:
    module = load_tool_module("check_radon")
    root = _repo(tmp_path)
    _write(root, relative)
    if kind == "ignore-policy":
        _write(root, ".gitignore", "tests/ignored.py\n")
    paths, issues = module.discover_python_files(root)
    assert relative in {path.as_posix() for path in paths}
    assert kind in {issue.kind for issue in issues}


def test_average_a_with_highest_b_passes_and_reports_ranks(load_tool_module) -> None:
    module = load_tool_module("check_radon")
    report = module.evaluate_metrics(
        [{"path": "tests/a.py", "mi": 90.0, "blocks": [4, 6]}],
        radon_version="6.0.1",
    )
    assert report["status"] == "PASS"
    assert report["repository_cc_average"] == 5.0
    assert report["repository_cc_rank"] == "A"
    assert report["files"][0]["cc_rank"] == "A"


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        ([{"path": "tests/a.py", "mi": 90.0, "blocks": [11]}], "block-cc"),
        ([{"path": "tests/a.py", "mi": 90.0, "blocks": [6, 6]}], "file-average"),
        ([{"path": "tests/a.py", "mi": 19.0, "blocks": [1]}], "mi-rank"),
    ],
)
def test_complexity_policy_rejects_each_hard_limit(
    load_tool_module, payload: list[dict[str, object]], kind: str
) -> None:
    module = load_tool_module("check_radon")
    report = module.evaluate_metrics(payload, radon_version="6.0.1")
    assert report["status"] == "FAIL"
    assert kind in _issue_kinds(report)


def test_repository_average_uses_every_block_exactly(load_tool_module) -> None:
    module = load_tool_module("check_radon")
    report = module.evaluate_metrics(
        [
            {"path": "tests/a.py", "mi": 90.0, "blocks": [5, 5]},
            {"path": "tests/b.py", "mi": 90.0, "blocks": [6]},
        ],
        radon_version="6.0.1",
    )
    assert report["repository_cc_average"] == pytest.approx(16 / 3)
    assert "repository-average" in _issue_kinds(report)


def test_version_mismatch_is_a_failed_report(load_tool_module) -> None:
    module = load_tool_module("check_radon")
    report = module.evaluate_metrics([], radon_version="6.0.0")
    assert report["status"] == "FAIL"
    assert _issue_kinds(report) == {"radon-version"}


def test_nested_closure_is_counted_and_class_method_is_not_duplicated(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("check_radon")
    root = _repo(tmp_path, populate=False)
    source = (
        "class Sample:\n"
        "    def method(self, value):\n"
        "        return value\n\n"
        "def outer(value):\n"
        "    def inner(item):\n"
        "        if item:\n"
        "            return 1\n"
        "        return 0\n"
        "    return inner(value)\n"
    )
    path = _write(root, "tools/sample.py", source)
    metrics = module._source_metrics(path, Path("tools/sample.py"))
    names = [item["name"] for item in metrics["symbols"]]
    assert names.count("method") == 1
    assert names.count("inner") == 1
    assert set(names) == {"Sample", "method", "outer", "inner"}


def test_assertion_complexity_uses_radon_default_behavior(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("check_radon")
    root = _repo(tmp_path, populate=False)
    assertions = "".join(f"    assert value != {index}\n" for index in range(10))
    _write(root, "tests/test_assertions.py", "def test_many(value):\n" + assertions)
    report = module.build_report(root)
    assert "block-cc" in _issue_kinds(report)
    assert report["files"][0]["symbols"][0]["complexity"] == 11.0


@pytest.mark.parametrize(
    ("content", "detail"),
    [(b"def broken(:\n", "syntax"), (b"\xff\xfe", "decode")],
)
def test_invalid_python_is_a_reported_failure(
    tmp_path: Path, load_tool_module, content: bytes, detail: str
) -> None:
    module = load_tool_module("check_radon")
    root = _repo(tmp_path, populate=False)
    path = root / "tools/broken.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    report = module.build_report(root)
    assert report["status"] == "FAIL"
    assert "analysis" in _issue_kinds(report)
    assert detail in str(report["issues"][0]["detail"]).lower()


def test_missing_tracked_python_is_a_git_set_failure(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("check_radon")
    root = _repo(tmp_path, populate=False)
    path = _write(root, "tests/deleted.py")
    _git(root, "add", "tests/deleted.py")
    path.unlink()
    report = module.build_report(root)
    assert report["status"] == "FAIL"
    assert "git-set" in _issue_kinds(report)


def test_discovery_empty_after_generated_directories_are_pruned(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("check_radon")
    root = _repo(tmp_path, populate=False)
    _write(root, "build/generated.py")
    _write(root, "src/pkg.egg-info/generated.py")
    report = module.build_report(root)
    assert report["files"] == []
    assert "empty-discovery" in _issue_kinds(report)


def test_main_writes_full_json_and_prints_concise_failures(
    tmp_path: Path, load_tool_module, monkeypatch, capsys
) -> None:
    module = load_tool_module("check_radon")
    report = {
        "schema": "xrr-r23-radon-report-v1",
        "radon_version": "6.0.1",
        "status": "FAIL",
        "repository_cc_average": 11.0,
        "repository_cc_rank": "C",
        "files": [],
        "issues": [
            {
                "kind": "block-cc",
                "path": "tests/a.py",
                "detail": "work line 4: CC 11 rank C exceeds 10",
            }
        ],
    }
    monkeypatch.setattr(module, "build_report", lambda _root: report)
    output = tmp_path / "radon.json"
    assert module.main(["--repo-root", str(tmp_path), "--output", str(output)]) == 1
    assert json.loads(output.read_bytes()) == report
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "block-cc: tests/a.py: work line 4: CC 11 rank C exceeds 10\n"
