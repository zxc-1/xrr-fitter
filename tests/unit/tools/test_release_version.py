from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def _fixture_repo(tmp_path: Path, version: str = "0.2.3") -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f"[project]\nname = 'xrr-fitter'\nversion = '{version}'\n",
        encoding="utf-8",
    )
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(("git", "add", "pyproject.toml"), cwd=root, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "version fixture",
        ),
        cwd=root,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-08-06T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-08-06T10:00:00+00:00",
        },
    )
    return root, _git(root, "rev-parse", "HEAD")


def test_validate_release_tag_requires_annotated_tag_matching_project_version(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("release_version")
    root, head = _fixture_repo(tmp_path)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "tag",
            "-a",
            "v0.2.3",
            "-m",
            "release v0.2.3",
        ),
        cwd=root,
        check=True,
    )

    assert module.validate_release_tag(root, "v0.2.3", expected_commit=head) == "0.2.3"


@pytest.mark.parametrize("tag", ("R23-final", "0.2.3", "v0.2", "v0.2.3-rc.1"))
def test_validate_release_tag_rejects_non_stable_version_tag(
    tmp_path: Path,
    load_tool_module,
    tag: str,
) -> None:
    module = load_tool_module("release_version")
    root, _head = _fixture_repo(tmp_path)

    with pytest.raises(ValueError, match="vMAJOR.MINOR.PATCH"):
        module.validate_release_tag(root, tag)


def test_validate_release_tag_rejects_project_version_mismatch(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("release_version")
    root, _head = _fixture_repo(tmp_path, version="0.2.2")

    with pytest.raises(ValueError, match="does not match project version"):
        module.validate_release_tag(root, "v0.2.3")


def test_validate_release_tag_rejects_lightweight_tag(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("release_version")
    root, _head = _fixture_repo(tmp_path)
    subprocess.run(("git", "tag", "v0.2.3"), cwd=root, check=True)

    with pytest.raises(ValueError, match="annotated"):
        module.validate_release_tag(root, "v0.2.3")


def test_validate_release_tag_rejects_wrong_commit(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("release_version")
    root, head = _fixture_repo(tmp_path)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "tag",
            "-a",
            "v0.2.3",
            "-m",
            "release v0.2.3",
        ),
        cwd=root,
        check=True,
    )

    with pytest.raises(ValueError, match="different commit"):
        module.validate_release_tag(root, "v0.2.3", expected_commit="0" * len(head))
