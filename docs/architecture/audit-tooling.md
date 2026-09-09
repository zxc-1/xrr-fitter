# Non-GUI Audit Tooling

The independent `audit.yml` workflow runs coverage, incremental static types
and ordinary Python-package advisories on read-only hosted macOS runners.
It does not change the standard or release verification order. Workflow YAML
and local tests are not evidence of a successful GitHub Actions run.

## Locked Tools

`[tool.xrr.audit]` declares development-only tools; it is not a wheel extra or
a runtime dependency group. `tools/audit-requirements.lock` pins the complete
macOS ARM64/Python 3.12 tool wheel closure by SHA-256, including shared versions
from the application lock. It is not a Windows tool environment or an
application-package hash lock.

Use the existing locked application environment, then install the audit lock:

```bash
python3.12 -m pip --isolated install --no-cache-dir --require-hashes --only-binary=:all: -r tools/audit-requirements.lock
python3.12 -m pip check
```

The generator `tools/lock_audit_tools.py --report REPORT --output LOCK --check`
checks an existing lock against a pip 26.1.2 JSON resolution. Generation omits
`--check`. Resolve all direct `[tool.xrr.audit].requires` with
`--dry-run --ignore-installed --only-binary=:all: --report REPORT`, constrained
by `requirements-macos-arm64-py312.lock`. Only non-yanked HTTPS PyPI wheels are
accepted; wheel identity, tags, SHA-256, dependency markers, extras and exact
closure are checked. Preserve the original resolver report outside Git.

## Verification

Run from a clean ordinary checkout with Python 3.12. Reports must use distinct
new external directories; replace the example paths for subsequent runs.

```bash
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/verify.py coverage --report-dir /tmp/xrr-coverage-run
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/verify.py typing --report-dir /tmp/xrr-types-run
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/verify.py advisories --report-dir /tmp/xrr-advisories-run
```

The verifier writes its normal outcome record. The `audit/` child directory
contains input hashes, tool versions, exact commands, exit codes, stdout/stderr
and a summary. Existing reports are never overwritten. Input changes or missing
tools fail without publishing a success summary.

- Coverage reuses exactly unit/integration/spawn/regression and the existing
  outcome enforcement. Branch tracing includes multiprocessing/subprocesses.
  Raw `.coverage*` files, JSON/XML and HTML are retained even when tests fail.
  Report generation never cancels a failing test exit code. There is no new
  percentage threshold and no merged-retry all-green claim.
- Types check the seven modules listed in `[tool.mypy].files`, with imported
  types analyzed and no blanket missing-import or error-code suppression.
  The selected modules must be diagnostic-free; remaining project-wide type
  debt is not represented as repaired by this gate.
- Advisories query exact ordinary pins from both validated platform locks via
  PyPI. Missing results, skipped dependencies, database/tool errors and findings
  all fail. The `refnx` Git source and bundled native libraries are explicitly
  outside this lookup; no released PyPI version is invented for Git source.
  Each target uses its own explicitly selected cache under the new report
  directory; an old pip HTTP cache is not used as fresh advisory evidence.

CI retains reports on failure with hidden coverage data included. Reports from
an untrusted PR remain PR evidence, not trusted release attestations. Package
archive verification, artifact/native SBOM, verified caches and target-platform
execution remain separate acceptance requirements.

## Package Bytes

`tools/package-manifests/` contains deterministic identities from pinned-pip
resolution for 44 macOS and 38 Windows ordinary wheels. Each manifest binds the
exact lock and project bytes, target wheel tags, HTTPS URL and SHA-256. These
files are reviewable pins, not a claim that every wheel has been installed on
both targets. Refresh manifests whenever a bound declaration changes.

```bash
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/package_downloads.py resolve --target macos-arm64-py312 --report-dir /tmp/xrr-package-resolution
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/package_downloads.py download --manifest tools/package-manifests/macos-arm64-py312.json --report-dir /tmp/xrr-package-download
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/package_downloads.py verify --manifest tools/package-manifests/macos-arm64-py312.json --wheel-dir /tmp/xrr-package-download/wheels
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/package_downloads.py install --manifest tools/package-manifests/macos-arm64-py312.json --wheel-dir /tmp/xrr-package-download/wheels --report-dir /tmp/xrr-package-install
```

Installation requires an external virtual environment on the declared target.
The tools reject missing/extra wheels, symlinks, hash mismatches and changed
input files. Pip is explicitly cache-disabled: `--isolated` would otherwise
ignore an environment-only `PIP_CACHE_DIR`. Installation is local, hash-required
and dependency-resolution-free; package versions and `pip check` are checked
afterward. Failed downloads remove owned partial wheels while retaining logs.
No command installs the separately recorded VCS source or a compiler toolchain.

`tools/vcs_source.py` verifies the `refnx` archive against `git archive` of the
exact declared commit, honoring the commit's export rules and rejecting local
attribute overrides. `refnx-source.json` binds 303 exported files and the actual
archive SHA-256. This is source identity, not a compiled wheel hash or a claim
that its isolated build dependencies are locked.

```bash
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/vcs_source.py verify --manifest tools/package-manifests/refnx-source.json --archive /tmp/refnx-source.tar.gz
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/package_sbom.py --manifest tools/package-manifests/macos-arm64-py312.json --wheel-dir /tmp/xrr-package-download/wheels
```

The CycloneDX 1.6 package SBOM reads actual wheel metadata and hashes every
archive file without extraction or package imports. Top-level dependency edges
use explicit target markers and required extras. Vendored distribution metadata
is recorded separately, with its own name/version/licenses and metadata hash;
it is never assigned a fabricated standalone wheel hash. Native files are
identified by headers or native suffixes and retain their byte identities.

Composition remains `incomplete`: native and vendored linking relationships,
installed/archive differences, the VCS wheel and the final executable are not
fully established by this wheel archive scan. `--require-complete` retains the
incomplete report but exits nonzero. Do not use a successful inventory exit as
evidence of a complete native SBOM or vulnerability-free distribution.

## Windows Audit

`audit-windows.yml` is separate from the existing GUI executable release job.
It runs on a read-only hosted Windows 2025 runner, checks out the exact event
commit, creates a unique external environment and disables shared pip caching.
The bootstrap and pytest additions reuse existing universal hash-pinned wheels.
It verifies the Windows wheel bytes before installation, runs existing CLI and
export tests, and builds a headless executable for help/validation/export
checks. Reports bind the executable hash to the source commit and explicitly
state that the GUI was not tested. The job-owned environment is cleaned after
evidence upload; the release asset set is unchanged.
