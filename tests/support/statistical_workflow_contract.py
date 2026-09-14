"""Explicit statistical authorization and provenance-preserving transfer contracts."""

from __future__ import annotations

from tests.support.release_workflow_contract import DOWNLOAD_ARTIFACT


def statistical_inputs() -> dict:
    return {
        "compute": {
            "description": "Explicitly authorize new fits on the first manual attempt only",
            "type": "boolean",
            "required": False,
            "default": False,
        },
        "producer": {
            "description": "Exact producer descriptor JSON for revalidation without new fits",
            "type": "string",
            "required": False,
            "default": "",
        },
    }


def statistical_call_inputs() -> dict:
    return {"compute": "${{ inputs.compute || false }}", "producer": "${{ inputs.producer || '' }}"}


def statistical_environment() -> dict:
    return {
        "PYTHON": "${{ steps.python.outputs.python }}",
        "PRODUCER": "${{ inputs.producer }}",
        "GH_TOKEN": "${{ github.token }}",
    }


def replay_arguments() -> tuple[str, ...]:
    return (
        'STATISTICAL_ARGS=(--statistical-results "$RUNNER_TEMP/statistical-inputs")',
        'if test -n "$PRODUCER"; then',
        '  "$PYTHON" tools/statistical_handoff.py --producer-json "$PRODUCER" --report-dir "$RUNNER_TEMP/statistical-handoff"',
        '  STATISTICAL_ARGS=(--statistical-results "$RUNNER_TEMP/statistical-handoff/shards" --statistical-producer "$RUNNER_TEMP/statistical-handoff/producer.json")',
        "fi",
    )


def statistical_download_step() -> dict:
    return {
        "name": "Download statistical shard evidence",
        "if": "${{ inputs.producer == '' }}",
        "uses": DOWNLOAD_ARTIFACT,
        "with": {
            "pattern": "statistical-shard-${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}-*",
            "path": "${{ runner.temp }}/statistical-inputs",
            "merge-multiple": False,
        },
    }
