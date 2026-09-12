"""Same-run statistical artifact transfer expected by both release workflows."""

from __future__ import annotations

from tests.support.release_workflow_contract import DOWNLOAD_ARTIFACT


def statistical_download_step() -> dict:
    return {
        "name": "Download statistical shard evidence",
        "uses": DOWNLOAD_ARTIFACT,
        "with": {
            "pattern": "statistical-shard-${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}-*",
            "path": "${{ runner.temp }}/statistical-inputs",
            "merge-multiple": False,
        },
    }
