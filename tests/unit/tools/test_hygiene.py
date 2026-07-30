from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ALLOWED_ROOTS = (
    ".github",
    "docs",
    "examples",
    "packaging",
    "src",
    "tests",
    "tools",
    "verification",
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True, capture_output=True)


def _commit(root: Path, message: str = "base") -> None:
    _git(root, "add", ".")
    _git(root, "commit", "-qm", message)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Hygiene")
    _git(root, "config", "user.email", "hygiene@example.invalid")
    for directory in ALLOWED_ROOTS:
        (root / directory).mkdir()
    (root / "README.md").write_text("ok\n", encoding="utf-8")
    (root / "requirements-windows-x64-py312.lock").write_text(
        "pyinstaller==6.21.0\n",
        encoding="utf-8",
    )
    _commit(root)
    return root


def _write(root: Path, relative: str, content: str = "x\n") -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _ignore(root: Path, relative: str) -> None:
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write(relative + "\n")


def _kinds(module, root: Path, *, strict: bool = False) -> set[str]:
    return {
        issue.kind
        for issue in module.inspect_repository(root, require_git_clean=strict)
    }


@pytest.mark.parametrize("ignored", [False, True], ids=("ordinary", "ignored"))
@pytest.mark.parametrize(
    ("relative", "kind"),
    [
        ("tools/.venv/bin/python", "generated"),
        ("tools/venv/bin/python", "generated"),
        ("tools/custom-env/pyvenv.cfg", "generated"),
        ("tests/__pycache__/x.pyc", "generated"),
        ("tests/.pytest_cache/state", "generated"),
        ("src/x.pyc", "generated"),
        ("src/x.pyo", "generated"),
        ("src/pkg.egg-info/PKG-INFO", "generated"),
        ("tools/build/output", "generated"),
        ("tools/dist/output", "generated"),
        ("docs/.coverage", "generated"),
        ("docs/.coverage.worker", "generated"),
        ("docs/.DS_Store", "generated"),
        ("docs/Thumbs.db", "generated"),
        ("docs/file.tmp", "partial"),
        ("docs/file.partial", "partial"),
        ("docs/file.part", "partial"),
        ("docs/file.bak", "partial"),
        ("docs/file~", "partial"),
        ("artifacts/output", "generated"),
        ("reports/output", "generated"),
        ("exports/output", "generated"),
        ("output/result", "generated"),
        ("tmp/result", "generated"),
        ("unknown/file.txt", "ownership"),
    ],
)
def test_hygiene_rejects_generated_partial_and_unowned_paths_even_when_ignored(
    tmp_path: Path, load_tool_module, relative: str, kind: str, ignored: bool
) -> None:
    module = load_tool_module("check_hygiene")
    root = _repo(tmp_path)
    _write(root, relative)
    if ignored:
        _ignore(root, relative.split("/", maxsplit=1)[0] + "/")
    assert kind in _kinds(module, root)


@pytest.mark.parametrize("ignored", [False, True], ids=("ordinary", "ignored"))
def test_hygiene_rejects_symlink_even_when_ignored(
    tmp_path: Path, load_tool_module, ignored: bool
) -> None:
    module = load_tool_module("check_hygiene")
    root = _repo(tmp_path)
    link = root / "docs/link"
    link.symlink_to(root / "README.md")
    if ignored:
        _ignore(root, "docs/link")
    assert "symlink" in _kinds(module, root)


@pytest.mark.parametrize("state", ["modified", "untracked"])
@pytest.mark.parametrize("directory", ALLOWED_ROOTS)
def test_each_allowed_root_accepts_review_files_but_strict_requires_clean_git(
    tmp_path: Path, load_tool_module, directory: str, state: str
) -> None:
    module = load_tool_module("check_hygiene")
    root = _repo(tmp_path)
    tracked = _write(root, f"{directory}/tracked.txt", "base\n")
    _commit(root, "tracked")
    if state == "modified":
        tracked.write_text("changed\n", encoding="utf-8")
    else:
        _write(root, f"{directory}/untracked.txt")
    assert module.inspect_repository(root, require_git_clean=False) == ()
    assert _kinds(module, root, strict=True) == {"git-dirty"}


def test_clean_independent_repository_passes_both_modes(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("check_hygiene")
    root = _repo(tmp_path)
    assert module.inspect_repository(root) == ()
    assert module.inspect_repository(root, require_git_clean=True) == ()


@pytest.mark.parametrize("control_type", ["gitfile", "symlink", "fifo"])
def test_git_control_must_be_a_normal_directory(
    tmp_path: Path, load_tool_module, control_type: str
) -> None:
    module = load_tool_module("check_hygiene")
    root = _repo(tmp_path)
    git_dir = root / ".git"
    moved = tmp_path / "git-dir"
    git_dir.rename(moved)
    if control_type == "gitfile":
        git_dir.write_text(f"gitdir: {moved}\n", encoding="utf-8")
    elif control_type == "symlink":
        git_dir.symlink_to(moved, target_is_directory=True)
    else:
        os.mkfifo(git_dir)
    assert "git-control" in _kinds(module, root)


def test_nonregular_file_type_is_rejected(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("check_hygiene")
    root = _repo(tmp_path)
    os.mkfifo(root / "docs/pipe")
    assert "file-type" in _kinds(module, root)


def test_wrong_repository_root_is_rejected(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("check_hygiene")
    parent = _repo(tmp_path)
    nested = parent / "docs/nested"
    nested.mkdir()
    (nested / ".git").mkdir()
    assert "git-root" in _kinds(module, nested)


@pytest.mark.parametrize("ignored", [False, True], ids=("ordinary", "ignored"))
def test_nested_git_control_directory_is_rejected_even_when_ignored(
    tmp_path: Path, load_tool_module, ignored: bool
) -> None:
    module = load_tool_module("check_hygiene")
    root = _repo(tmp_path)
    (root / "docs/nested/.git").mkdir(parents=True)
    if ignored:
        _ignore(root, "docs/nested/")
    issues = module.inspect_repository(root)
    assert any(issue.kind == "git-control" and issue.path == "docs/nested/.git" for issue in issues)
