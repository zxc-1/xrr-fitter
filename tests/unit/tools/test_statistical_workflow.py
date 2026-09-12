from __future__ import annotations

from pathlib import Path

import yaml
from tests.support.macos_action_contract import cleanup_step
from tests.support.release_workflow_contract import CHECKOUT
from tests.support.statistical_workflow_contract import statistical_download_step
from tests.support.verify_workflow_contract import UPLOAD_ARTIFACT, setup_step

ROOT = Path(__file__).resolve().parents[3]


def _workflow():
    path = ROOT / ".github/workflows/statistical.yml"
    assert path.is_file(), "missing reusable full statistical workflow"
    return yaml.safe_load(path.read_text())


def test_statistical_workflow_is_reusable_read_only_and_exactly_two_stages() -> None:
    payload = _workflow()
    assert set(payload) == {"name", "on", "permissions", "jobs"}
    assert payload["name"] == "statistical"
    assert payload["on"] == {"workflow_call": {}}
    assert payload["permissions"] == {"contents": "read"}
    assert set(payload["jobs"]) == {"shards", "aggregate"}


def test_all_eight_shards_use_standard_runners_without_cancelling_other_evidence() -> None:
    job = _workflow()["jobs"]["shards"]
    assert set(job) == {"runs-on", "timeout-minutes", "strategy", "steps"}
    assert job["runs-on"] == "macos-15"
    assert job["timeout-minutes"] == 360
    assert job["strategy"] == {
        "fail-fast": False,
        "max-parallel": 8,
        "matrix": {"shard": list(range(8))},
    }


def test_every_statistical_job_uses_locked_setup_and_owned_cleanup() -> None:
    for job in _workflow()["jobs"].values():
        steps = job["steps"]
        assert steps[0] == {"uses": CHECKOUT, "with": {"persist-credentials": False, "fetch-depth": 0}}
        assert steps.count(setup_step()) == steps.count(cleanup_step()) == 1
        assert steps.index(setup_step()) < steps.index(cleanup_step())
        assert len(steps) == 6


def test_shard_command_computes_only_the_canonical_index_and_checks_hygiene() -> None:
    step = _workflow()["jobs"]["shards"]["steps"][2]
    assert step == {
        "name": "Compute statistical shard",
        "env": {"PYTHON": "${{ steps.python.outputs.python }}", "SHARD": "${{ matrix.shard }}"},
        "shell": "bash",
        "run": "\n".join(
            (
                "set -euo pipefail",
                'test "$PYTHON" = "${{ steps.python.outputs.python }}"',
                '"$PYTHON" tools/check_hygiene.py --require-git-clean',
                'PYTHONPATH="$GITHUB_WORKSPACE/src" "$PYTHON" tools/statistical_shards.py --shard-index "$SHARD" --report-dir "$RUNNER_TEMP/statistical-shard-$SHARD" 2>&1 | tee "$RUNNER_TEMP/statistical-shard-$SHARD.log"',
                '"$PYTHON" tools/check_hygiene.py --require-git-clean',
                "",
            )
        ),
    }


def test_shard_bytes_and_failure_logs_are_retained_separately_with_exact_run_attempt_names() -> None:
    steps = _workflow()["jobs"]["shards"]["steps"]
    expected = (
        (
            "Retain statistical shard evidence",
            "statistical-shard",
            "${{ runner.temp }}/statistical-shard-${{ matrix.shard }}/",
        ),
        (
            "Retain statistical shard log",
            "statistical-shard-log",
            "${{ runner.temp }}/statistical-shard-${{ matrix.shard }}.log",
        ),
    )
    for step, (name, prefix, path) in zip(steps[-2:], expected, strict=True):
        assert step == {
            "name": name,
            "if": "${{ always() }}",
            "uses": UPLOAD_ARTIFACT,
            "with": {
                "name": prefix
                + "-${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.shard }}",
                "path": path,
                "if-no-files-found": "error",
                "retention-days": 14,
                "compression-level": 0,
            },
        }


def test_aggregate_requires_every_shard_and_reruns_the_original_statistical_gate() -> None:
    job = _workflow()["jobs"]["aggregate"]
    assert set(job) == {"needs", "runs-on", "timeout-minutes", "steps"}
    assert job["needs"] == ["shards"]
    assert job["runs-on"] == "macos-15"
    assert job["timeout-minutes"] == 60
    assert job["steps"][1] == statistical_download_step()
    step = job["steps"][3]
    assert step == {
        "name": "Verify complete statistical corpus",
        "env": {"PYTHON": "${{ steps.python.outputs.python }}"},
        "shell": "bash",
        "run": "\n".join(
            (
                "set -euo pipefail",
                'test "$PYTHON" = "${{ steps.python.outputs.python }}"',
                '"$PYTHON" tools/verify.py statistical --statistical-results "$RUNNER_TEMP/statistical-inputs" --report-dir "$RUNNER_TEMP/statistical-aggregate" 2>&1 | tee "$RUNNER_TEMP/statistical-aggregate.log"',
                "",
            )
        ),
    }


def test_aggregate_retains_the_full_summary_and_failure_log() -> None:
    step = _workflow()["jobs"]["aggregate"]["steps"][-1]
    assert step == {
        "name": "Retain statistical aggregation evidence",
        "if": "${{ always() }}",
        "uses": UPLOAD_ARTIFACT,
        "with": {
            "name": "statistical-aggregate-${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}",
            "path": "${{ runner.temp }}/statistical-aggregate/statistical-evidence.json\n${{ runner.temp }}/statistical-aggregate.log\n",
            "if-no-files-found": "error",
            "retention-days": 14,
            "compression-level": 0,
        },
    }


def test_both_release_workflows_require_the_same_shards_and_revalidate_raw_evidence() -> None:
    for name in ("verify.yml", "hosted-release-verify.yml"):
        payload = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        statistical = payload["jobs"].get("statistical")
        assert statistical is not None, "release lacks a mandatory statistical prerequisite"
        assert statistical["uses"] == "./.github/workflows/statistical.yml"
        release = payload["jobs"]["release"]
        assert "statistical" in release["needs"]
        assert statistical_download_step() in release["steps"]
        step = next(step for step in release["steps"] if step.get("name") == "Verify release")
        assert "tools/verify.py release" in step["run"]
        assert '--statistical-results "$RUNNER_TEMP/statistical-inputs"' in step["run"]
