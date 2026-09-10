"""Independent contracts for verified macOS inputs and job-private environments."""

TRUST = "${{ github.event_name == 'pull_request' && format('pr-{0}', github.event.pull_request.number) || 'trusted' }}"
CACHE = "${{ runner.temp }}/xrr-macos-input-cache/${{ steps.cache-key.outputs.digest }}/"
RESTORE = "actions/cache/restore@0400d5f644dc74513175e3cd8d07132dd4860809"
SAVE = "actions/cache/save@0400d5f644dc74513175e3cd8d07132dd4860809"
UPLOAD = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"


def _run(*lines):
    return "\n".join((*lines, ""))


def bootstrap_run():
    return _run(
        "set -euo pipefail",
        "export PYTHONDONTWRITEBYTECODE=1",
        'python3.12 -c \'import platform, sys; assert sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info[:2] == (3, 12)\'',
        'JOB_ROOT=$(mktemp -d "$RUNNER_TEMP/xrr-macos-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-$GITHUB_JOB.XXXXXXXX")',
        'python3.12 -m venv "$JOB_ROOT/venv"',
        'python3.12 tools/macos_environment.py own --job-root "$JOB_ROOT"',
        'printf \'root=%s\\nevidence-name=%s\\n\' "$JOB_ROOT" "${JOB_ROOT##*/}" >> "$GITHUB_OUTPUT"',
        'printf \'PYTHONDONTWRITEBYTECODE=1\\nMPLCONFIGDIR=%s/mpl-cache\\nXDG_CACHE_HOME=%s/xdg-cache\\n\' "$JOB_ROOT" "$JOB_ROOT" >> "$GITHUB_ENV"',
        'PYTHON="$JOB_ROOT/venv/bin/python"',
        '"$PYTHON" -m pip --isolated install --no-cache-dir --require-hashes --no-deps --only-binary=:all: -r tools/bootstrap-requirements.lock',
    )


def environment_run():
    return _run(
        "set -euo pipefail",
        'PYTHON="$JOB_ROOT/venv/bin/python"',
        '"$PYTHON" tools/macos_environment.py setup --job-root "$JOB_ROOT" --cache-dir "$CACHE_DIR" --trust-domain "$CACHE_TRUST_DOMAIN"',
        '"$PYTHON" -m pip check',
        '"$PYTHON" tools/check_hygiene.py --require-git-clean',
        'printf \'python=%s\\n\' "$PYTHON" >> "$GITHUB_OUTPUT"',
    )


def expected_action():
    return {
        "name": "Set up locked macOS Python",
        "description": "Create and validate the macOS ARM64 Python 3.12 verification environment",
        "outputs": {
            "python": {
                "description": "Validated Python executable",
                "value": "${{ steps.environment.outputs.python }}",
            },
            "job-root": {
                "description": "Owned external job directory for bounded cleanup",
                "value": "${{ steps.bootstrap.outputs.root }}",
            },
        },
        "runs": {
            "using": "composite",
            "steps": [
                {"id": "bootstrap", "shell": "bash", "run": bootstrap_run()},
                {
                    "id": "cache-key",
                    "shell": "bash",
                    "env": {
                        "PYTHON": "${{ steps.bootstrap.outputs.root }}/venv/bin/python",
                        "CACHE_TRUST_DOMAIN": TRUST,
                    },
                    "run": _run(
                        "set -euo pipefail",
                        '"$PYTHON" tools/macos_environment.py key --trust-domain "$CACHE_TRUST_DOMAIN" >> "$GITHUB_OUTPUT"',
                    ),
                },
                {
                    "id": "input-cache",
                    "uses": RESTORE,
                    "with": {"key": "${{ steps.cache-key.outputs.key }}", "path": CACHE},
                },
                {
                    "id": "environment",
                    "shell": "bash",
                    "env": {
                        "JOB_ROOT": "${{ steps.bootstrap.outputs.root }}",
                        "CACHE_DIR": CACHE,
                        "CACHE_TRUST_DOMAIN": TRUST,
                    },
                    "run": environment_run(),
                },
                {
                    "if": "${{ success() && steps.input-cache.outputs.cache-hit != 'true' }}",
                    "uses": SAVE,
                    "with": {"key": "${{ steps.cache-key.outputs.key }}", "path": CACHE},
                },
                {
                    "if": "${{ always() && steps.bootstrap.outputs.root != '' }}",
                    "uses": UPLOAD,
                    "with": {
                        "name": "${{ steps.bootstrap.outputs.evidence-name }}",
                        "path": _run(
                            "${{ steps.bootstrap.outputs.root }}/owner.json",
                            "${{ steps.bootstrap.outputs.root }}/reports/",
                            "!${{ steps.bootstrap.outputs.root }}/reports/**/wheels/**",
                            "!${{ steps.bootstrap.outputs.root }}/reports/**/inputs/**",
                        ),
                        "include-hidden-files": True,
                        "if-no-files-found": "error",
                        "retention-days": 14,
                    },
                },
            ],
        },
    }


def cleanup_step():
    return {
        "name": "Clean the owned macOS environment",
        "if": "${{ always() && steps.python.outputs.job-root != '' }}",
        "uses": "./.github/actions/cleanup-macos-python",
        "with": {"job-root": "${{ steps.python.outputs.job-root }}"},
    }
