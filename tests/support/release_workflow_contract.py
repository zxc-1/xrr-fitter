"""Expected version-tag release jobs shared by workflow contract tests."""

from __future__ import annotations


CHECKOUT = "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683"
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
            'if gh release view "$TAG" --repo "$GITHUB_REPOSITORY" >/dev/null 2>&1; then',
            '  gh release upload "$TAG" "${ASSETS[@]}" \\',
            '    --repo "$GITHUB_REPOSITORY" --clobber',
            "else",
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
