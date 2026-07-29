from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _collection_repo(root: Path, *, unknown_marker: bool = False) -> tuple[Path, str]:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src/source_identity.py").write_text('VALUE = "repo"\n', encoding="utf-8")
    marker = "unknown" if unknown_marker else "slow"
    (root / "tests/test_collection.py").write_text(
        "import source_identity\n"
        "import pytest\n\n"
        'assert source_identity.VALUE == "repo"\n\n'
        f"@pytest.mark.{marker}\n"
        f"def test_{marker}():\n"
        "    pass\n\n"
        "def test_fast():\n"
        "    pass\n",
        encoding="utf-8",
    )
    markers = 'markers = ["slow: deliberately collected slow test"]\n' if not unknown_marker else ""
    (root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        'addopts = "-m \'not slow\'"\n'
        f"{markers}",
        encoding="utf-8",
    )
    lock = root / "requirements.lock"
    lock.write_text("pytest==8.4.2\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Collector")
    _git(root, "config", "user.email", "collector@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return lock, _git(root, "rev-parse", "HEAD")


def _run_collector(
    module,
    root: Path,
    lock: Path,
    source_commit: str,
    output: Path,
    *,
    python_path: Path,
    pytest_addopts: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(python_path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if pytest_addopts is not None:
        environment["PYTEST_ADDOPTS"] = pytest_addopts
    return subprocess.run(
        (
            sys.executable,
            str(Path(module.__file__).resolve()),
            "--repo-root",
            str(root),
            "--source-commit",
            source_commit,
            "--lock-file",
            str(lock),
            "--suite",
            "tests",
            "--output",
            str(output),
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_manifest_is_path_free_sorted_hash_bound_and_canonical(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("collect_test_manifest")
    root = tmp_path / "repo"
    tests = root / "tests"
    tests.mkdir(parents=True)
    source = tests / "test_sample.py"
    source.write_text("def test_ok(): pass\n", encoding="utf-8")
    lock = root / "requirements.lock"
    lock.write_text("pytest==8.3.5\n", encoding="utf-8")
    records = (
        {"nodeid": "tests/test_sample.py::test_slow", "markers": ["slow"]},
        {"nodeid": "tests/test_sample.py::test_ok", "markers": []},
    )
    manifest = module.build_manifest(
        repo_root=root,
        source_commit="a" * 40,
        suite="tests",
        lock_file=lock,
        records=records,
        python_version="3.12.11",
        platform="macOS-arm64",
    )
    assert [item["nodeid"] for item in manifest["nodes"]] == sorted(
        item["nodeid"] for item in manifest["nodes"]
    )
    assert manifest["test_tree"][0] == {
        "path": "tests/test_sample.py",
        "size": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert "/" not in manifest["python_version"]
    assert all("tmp" not in str(value) for value in manifest.values())
    encoded = module.canonical_json_bytes(manifest)
    assert encoded == _canonical(manifest)
    assert module.canonical_json_bytes(manifest) == encoded


def test_manifest_rejects_duplicate_nodes_unknown_suite_and_bad_output_parent(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("collect_test_manifest")
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests/test_a.py").write_text("def test_a(): pass\n", encoding="utf-8")
    lock = root / "lock"
    lock.write_text("pytest==8.3.5\n", encoding="utf-8")
    kwargs = dict(
        repo_root=root,
        source_commit="b" * 40,
        suite="tests",
        lock_file=lock,
        python_version="3.12.11",
        platform="macOS-arm64",
    )
    with pytest.raises(ValueError, match="duplicate"):
        module.build_manifest(records=({"nodeid": "x", "markers": []},) * 2, **kwargs)
    with pytest.raises(ValueError, match="suite"):
        module.build_manifest(records=(), **{**kwargs, "suite": "unknown"})
    with pytest.raises(ValueError, match="parent"):
        module.write_manifest(tmp_path / "missing/output.json", {})


def test_pytest_collection_receives_mutable_argument_list(
    tmp_path: Path, load_tool_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_tool_module("collect_test_manifest")
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)

    class FakePytest:
        class ExitCode:
            OK = 0

        @staticmethod
        def main(arguments, *, plugins):
            assert isinstance(arguments, list)
            assert plugins
            return 0

    monkeypatch.setitem(sys.modules, "pytest", FakePytest)
    assert module.collect_records(root, "tests", root) == ()


def test_collection_isolates_pythonpath_clears_default_addopts_and_collects_slow(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("collect_test_manifest")
    root = tmp_path / "repo"
    lock, source_commit = _collection_repo(root)
    polluted = tmp_path / "polluted"
    polluted.mkdir()
    (polluted / "source_identity.py").write_text('VALUE = "polluted"\n', encoding="utf-8")
    output = tmp_path / "manifest.json"

    result = _run_collector(
        module,
        root,
        lock,
        source_commit,
        output,
        python_path=polluted,
        pytest_addopts="-p no:terminal -k fast",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["nodes"] == [
        {"markers": [], "nodeid": "tests/test_collection.py::test_fast"},
        {"markers": ["slow"], "nodeid": "tests/test_collection.py::test_slow"},
    ]


def test_collection_accepts_duplicate_module_basenames_in_distinct_directories(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("collect_test_manifest")
    root = tmp_path / "repo"
    lock, _source_commit = _collection_repo(root)
    for owner in ("model", "services"):
        directory = root / "tests/unit" / owner
        directory.mkdir(parents=True)
        (directory / "test_parameters.py").write_text(
            f"def test_{owner}_parameters(): pass\n",
            encoding="utf-8",
        )
    _git(root, "add", "tests")
    _git(root, "commit", "-qm", "add duplicate basenames")
    source_commit = _git(root, "rev-parse", "HEAD")
    output = tmp_path / "manifest.json"

    result = _run_collector(
        module,
        root,
        lock,
        source_commit,
        output,
        python_path=tmp_path / "polluted",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    nodeids = {
        item["nodeid"]
        for item in json.loads(output.read_text(encoding="utf-8"))["nodes"]
    }
    assert {
        "tests/unit/model/test_parameters.py::test_model_parameters",
        "tests/unit/services/test_parameters.py::test_services_parameters",
    } <= nodeids


def test_collection_allows_explicit_repository_tool_imports(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("collect_test_manifest")
    root = tmp_path / "repo"
    lock, _source_commit = _collection_repo(root)
    tools = root / "tools"
    tools.mkdir()
    (tools / "__init__.py").write_bytes(b"")
    (tools / "fixture_helper.py").write_text("VALUE = 'tool'\n", encoding="utf-8")
    (root / "tests/test_tool_import.py").write_text(
        "from tools.fixture_helper import VALUE\n\n"
        "def test_tool_import():\n"
        "    assert VALUE == 'tool'\n",
        encoding="utf-8",
    )
    _git(root, "add", "tests", "tools")
    _git(root, "commit", "-qm", "add repository tool import")
    source_commit = _git(root, "rev-parse", "HEAD")
    output = tmp_path / "manifest.json"

    result = _run_collector(
        module,
        root,
        lock,
        source_commit,
        output,
        python_path=tmp_path / "polluted",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    nodeids = {
        item["nodeid"]
        for item in json.loads(output.read_text(encoding="utf-8"))["nodes"]
    }
    assert "tests/test_tool_import.py::test_tool_import" in nodeids


def test_collection_rejects_unknown_marker_without_writing_output(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("collect_test_manifest")
    root = tmp_path / "repo"
    lock, source_commit = _collection_repo(root, unknown_marker=True)
    output = tmp_path / "manifest.json"

    result = _run_collector(
        module,
        root,
        lock,
        source_commit,
        output,
        python_path=tmp_path,
    )

    assert result.returncode != 0
    assert "unknown" in result.stdout + result.stderr
    assert not output.exists()


def test_collection_restores_caller_process_state(
    tmp_path: Path, load_tool_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_tool_module("collect_test_manifest")
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    import_root = root / "src"
    import_root.mkdir()
    polluted = str(tmp_path / "polluted")
    monkeypatch.setenv("PYTHONPATH", polluted)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-p no:terminal -k missing")
    monkeypatch.setenv("PYTEST_PLUGINS", "caller_plugin")
    monkeypatch.setattr(sys, "path", [polluted, *sys.path])
    original_path = list(sys.path)
    original_bytecode = sys.dont_write_bytecode

    class FakePytest:
        class ExitCode:
            OK = 0

        @staticmethod
        def main(arguments, *, plugins):
            assert sys.path[0] == str(import_root)
            assert polluted not in sys.path
            assert os.environ["PYTHONPATH"] == str(import_root)
            assert not any(name.startswith("PYTEST_") for name in os.environ)
            assert sys.dont_write_bytecode is True
            return 0

    monkeypatch.setitem(sys.modules, "pytest", FakePytest)
    assert module.collect_records(root, "tests", import_root) == ()
    assert sys.path == original_path
    assert os.environ["PYTHONPATH"] == polluted
    assert os.environ["PYTEST_ADDOPTS"] == "-p no:terminal -k missing"
    assert os.environ["PYTEST_PLUGINS"] == "caller_plugin"
    assert sys.dont_write_bytecode is original_bytecode


def test_manifest_bytes_do_not_depend_on_repository_path(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("collect_test_manifest")
    first = tmp_path / "first/repo"
    lock, source_commit = _collection_repo(first)
    second = tmp_path / "second/repo"
    second.parent.mkdir()
    shutil.copytree(first, second)
    outputs = (tmp_path / "first.json", tmp_path / "second.json")

    for root, output in zip((first, second), outputs, strict=True):
        result = _run_collector(
            module,
            root,
            root / lock.relative_to(first),
            source_commit,
            output,
            python_path=tmp_path / "polluted",
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_r23_source_commit_must_be_an_ancestor_with_an_unchanged_test_tree(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("collect_test_manifest")
    root = tmp_path / "repo"
    _collection_repo(root)
    source_commit = _git(root, "rev-parse", "HEAD")
    (root / "README.md").write_text("metadata only\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-qm", "metadata")
    assert module._assert_r23_source(root, source_commit, "tests") == source_commit

    tree = _git(root, "rev-parse", f"{source_commit}^{{tree}}")
    unrelated = _git(root, "commit-tree", tree, "-m", "unrelated")
    with pytest.raises(ValueError, match="ancestor"):
        module._assert_r23_source(root, unrelated, "tests")

    (root / "tests/test_collection.py").write_text("def test_changed(): pass\n", encoding="utf-8")
    _git(root, "add", "tests/test_collection.py")
    _git(root, "commit", "-qm", "change tests")
    with pytest.raises(ValueError, match="differs"):
        module._assert_r23_source(root, source_commit, "tests")


def test_cli_rejects_symlink_output_parent(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("collect_test_manifest")
    root = tmp_path / "repo"
    lock, source_commit = _collection_repo(root)
    actual_parent = tmp_path / "actual"
    actual_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="parent"):
        module.main(
            [
                "--repo-root",
                str(root),
                "--source-commit",
                source_commit,
                "--lock-file",
                str(lock),
                "--suite",
                "tests",
                "--output",
                str(linked_parent / "manifest.json"),
            ]
        )
    assert not (actual_parent / "manifest.json").exists()
