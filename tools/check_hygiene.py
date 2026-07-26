#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
from pathlib import Path
import subprocess
from typing import Iterable, Sequence


ALLOWED_DIRS = {".github", "docs", "examples", "src", "tests", "tools", "verification"}
ALLOWED_FILES = {
    ".gitignore",
    "AGENTS.md",
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
    "requirements-macos-arm64-py312.lock",
}
FORBIDDEN_ROOT_DIRS = {"artifacts", "exports", "output", "reports", "tmp"}
FORBIDDEN_DIR_NAMES = {".pytest_cache", "__pycache__", "build", "dist", ".venv", "venv"}
PARTIAL_PATTERNS = ("*.bak", "*.part", "*.partial", "*.tmp", "*~")


@dataclass(frozen=True)
class HygieneIssue:
    kind: str
    path: str
    detail: str


def _issue(kind: str, relative: Path, detail: str) -> HygieneIssue:
    return HygieneIssue(kind, relative.as_posix(), detail)


def _generated_issue(relative: Path) -> HygieneIssue | None:
    name = relative.name
    if name == "pyvenv.cfg":
        return _issue("generated", relative, "virtual environment is inside repository")
    if any(part in FORBIDDEN_DIR_NAMES or part.endswith(".egg-info") for part in relative.parts):
        return _issue("generated", relative, "generated directory is inside repository")
    generated_names = {".DS_Store", "Thumbs.db"}
    if name in generated_names or name.endswith((".pyc", ".pyo")) or name.startswith(".coverage"):
        return _issue("generated", relative, "generated file is inside repository")
    if any(fnmatch.fnmatch(name, pattern) for pattern in PARTIAL_PATTERNS):
        return _issue("partial", relative, "partial file is inside repository")
    return None


def _git_root(repository: Path) -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty(repository: Path) -> bool:
    result = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode != 0 or bool(result.stdout)


def _git_control_issues(repository: Path) -> list[HygieneIssue]:
    issues: list[HygieneIssue] = []
    control = repository / ".git"
    if control.is_symlink() or not control.is_dir():
        issues.append(_issue("git-control", Path(".git"), ".git must be a normal directory"))
    if _git_root(repository) != str(repository):
        detail = "git top-level does not match repository root"
        issues.append(_issue("git-root", Path("."), detail))
    return issues


def _root_child_issue(child: Path, relative: Path) -> HygieneIssue | None:
    if child.is_symlink():
        return _issue("symlink", relative, "symlinks are forbidden")
    if child.is_dir():
        if child.name in FORBIDDEN_ROOT_DIRS:
            return _issue("generated", relative, "publication directory must be external")
        if child.name not in ALLOWED_DIRS:
            return _issue("ownership", relative, "top-level directory is not owned")
        return None
    if child.is_file():
        if child.name not in ALLOWED_FILES:
            return _issue("ownership", relative, "top-level file is not owned")
        return None
    return _issue("file-type", relative, "only regular files and directories are allowed")


def _root_issues(repository: Path) -> list[HygieneIssue]:
    issues: list[HygieneIssue] = []
    for child in repository.iterdir():
        if child.name == ".git":
            continue
        issue = _root_child_issue(child, child.relative_to(repository))
        if issue is not None:
            issues.append(issue)
    return issues


def _tree_path_issue(path: Path, relative: Path) -> HygieneIssue | None:
    if relative.name == ".git":
        return _issue("git-control", relative, "nested Git control paths are forbidden")
    if path.is_symlink():
        return _issue("symlink", relative, "symlinks are forbidden")
    if not path.is_dir() and not path.is_file():
        return _issue("file-type", relative, "only regular files and directories are allowed")
    return _generated_issue(relative)


def _tree_issues(repository: Path) -> list[HygieneIssue]:
    issues: list[HygieneIssue] = []
    for path in repository.rglob("*"):
        relative = path.relative_to(repository)
        if relative.parts[0] == ".git":
            continue
        issue = _tree_path_issue(path, relative)
        if issue is not None:
            issues.append(issue)
    return issues


def _cleanliness_issues(repository: Path, required: bool) -> list[HygieneIssue]:
    if required and _git_dirty(repository):
        return [_issue("git-dirty", Path("."), "Git worktree is not clean")]
    return []


def _unique(issues: Iterable[HygieneIssue]) -> tuple[HygieneIssue, ...]:
    indexed = {(item.kind, item.path, item.detail): item for item in issues}
    return tuple(indexed[key] for key in sorted(indexed))


def inspect_repository(root: str | Path, *, require_git_clean: bool = False) -> tuple[HygieneIssue, ...]:
    repository = Path(root).resolve()
    issues = _git_control_issues(repository)
    issues.extend(_root_issues(repository))
    issues.extend(_tree_issues(repository))
    issues.extend(_cleanliness_issues(repository, require_git_clean))
    return _unique(issues)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-git-clean", action="store_true")
    args = parser.parse_args(argv)
    issues = inspect_repository(args.repo_root, require_git_clean=args.require_git_clean)
    for item in issues:
        print(f"{item.kind}: {item.path}: {item.detail}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
