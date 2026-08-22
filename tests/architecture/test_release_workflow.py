from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "verify.yml"


def _payload() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_release_trigger_uses_stable_version_tags_only() -> None:
    payload = _payload()
    assert payload["on"] == {
        "push": {
            "branches": ["main"],
            "tags": ["v*"],
        }
    }


def test_candidate_readiness_validates_the_version_tag() -> None:
    job = _payload()["jobs"]["candidate-readiness"]
    validation = next(step for step in job["steps"] if step.get("name") == "Validate release version tag")
    assert validation["if"] == "startsWith(github.ref, 'refs/tags/')"
    assert "tools/release_version.py" in validation["run"]
    assert '"$GITHUB_REF_NAME"' in validation["run"]


def test_candidate_readiness_restores_the_remote_annotated_tag_before_validation() -> None:
    job = _payload()["jobs"]["candidate-readiness"]
    validation = next(step for step in job["steps"] if step.get("name") == "Validate release version tag")
    run = validation["run"]
    restore = 'git fetch --force --no-tags origin "refs/tags/$GITHUB_REF_NAME:refs/tags/$GITHUB_REF_NAME"'

    assert restore in run
    assert run.index(restore) < run.index("python3.12 tools/release_version.py")


def test_version_tag_release_builds_windows_after_release_gates() -> None:
    windows = _payload()["jobs"]["windows"]
    assert windows["needs"] == ["release"]
    assert windows["if"] == ("startsWith(github.ref, 'refs/tags/') && needs.release.result == 'success'")
    assert windows["uses"] == "./.github/workflows/windows-executable.yml"
    assert windows["with"] == {
        "source_ref": "${{ github.ref_name }}",
        "expected_commit": "${{ github.sha }}",
    }


def test_version_tag_release_drafts_release_after_windows_build() -> None:
    draft = _payload()["jobs"]["draft-release"]
    assert draft["needs"] == ["release", "windows"]
    assert draft["if"] == (
        "startsWith(github.ref, 'refs/tags/') && needs.release.result == 'success' && needs.windows.result == 'success'"
    )
    assert draft["permissions"] == {"contents": "write"}
    draft_run = draft["steps"][-1]["run"]
    required_snippets = (
        'gh release view "$TAG" --repo "$GITHUB_REPOSITORY" --json isDraft,assets',
        "existing GitHub Release is not a draft",
        "release-bundle/artifact-manifest.json",
        "release-bundle/release-identity.json",
        'windows-release/"*.exe',
        "gh release download",
        'cmp "$ASSET" "$REMOTE_ASSET_DIR/$(basename "$ASSET")"',
        "gh release create",
    )
    assert "find " not in draft_run
    assert all(snippet in draft_run for snippet in required_snippets)
    assert all(snippet not in draft_run for snippet in ("gh release upload", "--clobber"))


def test_draft_release_rechecks_downloaded_asset_names_after_download() -> None:
    draft_run = _payload()["jobs"]["draft-release"]["steps"][-1]["run"]
    download_command = 'gh release download "$TAG" --repo "$GITHUB_REPOSITORY" --dir "$REMOTE_ASSET_DIR"'
    download_offset = draft_run.index(download_command)
    cmp_offset = draft_run.index('cmp "$ASSET" "$REMOTE_ASSET_DIR/$(basename "$ASSET")"')
    post_download_pre_cmp = draft_run[download_offset:cmp_offset]

    assert 'python3 - "$REMOTE_ASSET_DIR" "${ASSETS[@]}" <<' in post_download_pre_cmp
    assert "downloaded_names = sorted(os.listdir(sys.argv[1]))" in post_download_pre_cmp
    assert "local_names = sorted(os.path.basename(path) for path in sys.argv[2:])" in post_download_pre_cmp
    assert "downloaded release asset names differ from local bundle" in post_download_pre_cmp


def test_release_checkpoint_requires_windows_and_draft_release_for_tags() -> None:
    checkpoint = _payload()["jobs"]["checkpoint"]
    assert "windows" in checkpoint["needs"]
    assert "draft-release" in checkpoint["needs"]
    step = checkpoint["steps"][0]
    assert step["env"]["WINDOWS_RESULT"] == "${{ needs.windows.result }}"
    assert step["env"]["DRAFT_RELEASE_RESULT"] == "${{ needs.draft-release.result }}"
    assert 'test "$WINDOWS_RESULT" = success' in step["run"]
    assert 'test "$DRAFT_RELEASE_RESULT" = success' in step["run"]
