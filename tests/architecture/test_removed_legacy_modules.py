from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ROOT_TREE = {
    ".gitignore",
    "docs/acceptance/real-data-template.md",
    "docs/algorithm.md",
    "docs/architecture/r23-clean-break.md",
    "docs/images/gui-dark-1280x760.png",
    "docs/images/gui-dark-1600x900-expert.png",
    "docs/images/gui-light-1280x760.png",
    "docs/images/gui-light-1600x900-expert.png",
    "docs/user-guide.md",
}


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def test_repository_has_one_parentless_r23_root_commit() -> None:
    roots = _git("rev-list", "--max-parents=0", "--all").splitlines()
    assert len(roots) == 1
    root = roots[0]
    assert _git("rev-list", "--parents", "-n", "1", root).split() == [root]
    observed = set(_git("ls-tree", "-r", "--name-only", root).splitlines())
    assert observed == ROOT_TREE
    history = _git("rev-list", "--all").splitlines()
    assert history
    for commit in history:
        _git("merge-base", "--is-ancestor", root, commit)
    assert (ROOT / ".git").is_dir() and not (ROOT / ".git").is_symlink()


def test_legacy_layout_is_absent_from_filesystem_and_history() -> None:
    forbidden = (
        "xrr_fitter",
        "xrr_core.py",
        "xrr_app.py",
        "xrr",
        "gui",
        "tests_r21",
        ".integration",
        ".superpowers",
        "docs/superpowers",
    )
    assert all(not (ROOT / path).exists() for path in forbidden)
    historical_paths = _git(
        "log",
        "--all",
        "--name-only",
        "--format=",
        "--",
        *forbidden,
    ).splitlines()
    assert not historical_paths
