"""Expected version-tag release jobs shared by workflow contract tests."""

from __future__ import annotations

CHECKOUT = "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
DOWNLOAD_ARTIFACT = "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"


def expected_windows_job() -> dict[str, object]:
    return {
        "needs": ["release"],
        "if": "startsWith(github.ref, 'refs/tags/') && needs.release.result == 'success'",
        "uses": "./.github/workflows/windows-executable.yml",
        "with": {
            "source_ref": "${{ github.ref_name }}",
            "expected_commit": "${{ github.sha }}",
        },
    }


def _draft_release_run() -> str:
    return "\n".join(
        (
            "set -euo pipefail",
            'TAG="$GITHUB_REF_NAME"',
            "shopt -s nullglob",
            "ASSETS=(",
            '  "$RUNNER_TEMP/release-bundle/artifact-manifest.json"',
            '  "$RUNNER_TEMP/release-bundle/release-identity.json"',
            '  "$RUNNER_TEMP/release-bundle/artifacts/"*',
            '  "$RUNNER_TEMP/windows-release/"*.exe',
            '  "$RUNNER_TEMP/windows-release/"*.json',
            ")",
            'test "${#ASSETS[@]}" -eq 6',
            'for ASSET in "${ASSETS[@]}"; do',
            '  test -f "$ASSET"',
            "done",
            'RELEASE_JSON="$RUNNER_TEMP/release.json"',
            'RELEASE_VIEW_ERROR="$RUNNER_TEMP/release-view.err"',
            'if gh release view "$TAG" --repo "$GITHUB_REPOSITORY" --json isDraft,assets \\',
            '  >"$RELEASE_JSON" 2>"$RELEASE_VIEW_ERROR"; then',
            '  python3 - "$RELEASE_JSON" "${ASSETS[@]}" <<\'PY\'',
            "import json",
            "import os",
            "import sys",
            "",
            'with open(sys.argv[1], encoding="utf-8") as stream:',
            "    payload = json.load(stream)",
            'if payload.get("isDraft") is not True:',
            '    raise SystemExit("existing GitHub Release is not a draft")',
            "local_names = [os.path.basename(path) for path in sys.argv[2:]]",
            "if len(set(local_names)) != len(local_names):",
            '    raise SystemExit(f"local release asset names are not unique: {sorted(local_names)!r}")',
            'remote_names = [asset.get("name") for asset in payload.get("assets", [])]',
            "if sorted(remote_names) != sorted(local_names):",
            '    raise SystemExit("remote release asset names differ from local bundle: "',
            '                     f"remote={sorted(remote_names)!r} local={sorted(local_names)!r}")',
            "PY",
            '  REMOTE_ASSET_DIR="$RUNNER_TEMP/release-assets"',
            '  test ! -e "$REMOTE_ASSET_DIR"',
            '  mkdir "$REMOTE_ASSET_DIR"',
            '  gh release download "$TAG" --repo "$GITHUB_REPOSITORY" --dir "$REMOTE_ASSET_DIR"',
            '  python3 - "$REMOTE_ASSET_DIR" "${ASSETS[@]}" <<\'PY\'',
            "import os",
            "import sys",
            "",
            "downloaded_names = sorted(os.listdir(sys.argv[1]))",
            "local_names = sorted(os.path.basename(path) for path in sys.argv[2:])",
            "if downloaded_names != local_names:",
            '    raise SystemExit("downloaded release asset names differ from local bundle: "',
            '                     f"downloaded={downloaded_names!r} local={local_names!r}")',
            "PY",
            '  for ASSET in "${ASSETS[@]}"; do',
            '    cmp "$ASSET" "$REMOTE_ASSET_DIR/$(basename "$ASSET")"',
            "  done",
            "else",
            '  if ! grep -Eiq "not found|HTTP 404" "$RELEASE_VIEW_ERROR"; then',
            '    cat "$RELEASE_VIEW_ERROR" >&2',
            "    exit 1",
            "  fi",
            '  gh release create "$TAG" "${ASSETS[@]}" \\',
            '    --repo "$GITHUB_REPOSITORY" \\',
            "    --verify-tag \\",
            '    --title "XRR $TAG" \\',
            "    --notes-file docs/acceptance/r23-release-acceptance.md \\",
            "    --draft \\",
            "    --latest=false",
            "fi",
            "",
        )
    )


def expected_draft_release_job() -> dict[str, object]:
    return {
        "needs": ["release", "windows"],
        "if": (
            "startsWith(github.ref, 'refs/tags/') && "
            "needs.release.result == 'success' && needs.windows.result == 'success'"
        ),
        "runs-on": "ubuntu-latest",
        "timeout-minutes": 15,
        "permissions": {"contents": "write"},
        "steps": [
            {
                "uses": CHECKOUT,
                "with": {"persist-credentials": False, "fetch-depth": 1},
            },
            {
                "name": "Download canonical release bundle",
                "uses": DOWNLOAD_ARTIFACT,
                "with": {
                    "name": "xrr-release-${{ github.ref_name }}-${{ github.sha }}",
                    "path": "${{ runner.temp }}/release-bundle",
                },
            },
            {
                "name": "Download Windows release bundle",
                "uses": DOWNLOAD_ARTIFACT,
                "with": {
                    "name": "xrr-windows-executable-${{ github.sha }}",
                    "path": "${{ runner.temp }}/windows-release",
                },
            },
            {
                "name": "Create or update draft GitHub Release",
                "shell": "bash",
                "env": {"GH_TOKEN": "${{ github.token }}"},
                "run": _draft_release_run(),
            },
        ],
    }
