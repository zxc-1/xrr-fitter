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

The package SBOM also inspects PE normal/delay imports and export forwarders,
Mach-O load commands across every universal slice, and GNU/BSD/Microsoft
archive members. Image hashes, architecture, member lineage and unparsed
content remain bound to the containing wheel. Recognized malformed loader
tables fail; unsupported content remains explicit. Native reads are bounded,
and file/descriptor identity checks reject a wheel rewritten during inspection.

Native dependency edges describe only unambiguous, architecture-matching
wheel-local `@loader_path` and loader-relative declared `@rpath` paths. They are
not observed runtime loads. Unknown rpath/executable context, Windows loader
search, relocated wheel paths, system libraries and unsupported COFF objects
remain unresolved. Foreign-platform or foreign-architecture payloads are
retained and labeled rather than dropped from the target wheel inventory.

Composition remains `incomplete`: native and vendored linking relationships,
installed/archive differences, the VCS wheel and the final executable are not
fully established by this wheel archive scan. `--require-complete` retains the
incomplete report but exits nonzero. Do not use a successful inventory exit as
evidence of a complete native SBOM or vulnerability-free distribution.

`tools/artifact_sbom.py` binds the built `artifact-manifest.json` and its exact
wheel/sdist bytes to a second CycloneDX 1.6 report. It inventories every sdist
member and every wheel member, preserves native headers and file hashes, and
records the captured source commit/tree plus artifact-manifest SHA-256:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/artifact_sbom.py \
  --repo-root . \
  --artifact-manifest /tmp/xrr-distribution/artifact-manifest.json \
  --artifact-dir /tmp/xrr-distribution/artifacts \
  > /tmp/xrr-distribution/artifacts.cdx.json
```

The report intentionally declares `compositions: incomplete`: the built
wheel/sdist pair does not prove installed dependency closure, native dynamic
link relationships, or a final executable. `--require-complete` retains no
false success and exits nonzero until those boundaries are supplied. The
artifact manifest remains the authoritative exact release identity; this report
does not replace it or add a release asset.

## Installed and Frozen Artifact Evidence

`tools/installed_sbom.py` checks the actual target venv against independently
hash-verified wheel inputs. Run it before installing the separate audit-tool
closure. It reconstructs relocation, pip scripts/launchers and metadata, RECORD,
and CPython 3.12 bytecode without executing inspected package source. Installed
RECORD hashes are not trusted as proof of file contents. Identical shared files
retain every owner; conflicting, missing or unowned site-packages files fail.
Input/output paths must have regular, unchanged ancestors, not symlink aliases.
Bootstrap installation runs twice with the same hash lock: the second invocation
uses the pinned pip to recreate its own scripts and metadata, so ensurepip's seed
version cannot determine those bytes. Bytecode reconstruction preserves pip's
join of the library directory with its slash-separated RECORD path on Windows.

The macOS scope is 44 ordinary wheels plus pinned bootstrap pip and the separately
verified refnx build. Windows retains six auxiliary input wheels from the
bootstrap/test locks, including packaging already in the 38-wheel ordinary
manifest; its installed scope is 43 distributions. The ordinary manifest is not
expanded to include these auxiliary packages. Interpreter and venv-bootstrap
members outside wheel ownership are reported separately.

The optional `--artifact-manifest` / `--artifact-dir` pair binds real wheel/sdist
members to captured Git blobs, then selects the wheel-metadata runtime dependency
closure rather than claiming every development package is shipped. The optional
`--executable` / `--executable-evidence` pair verifies the successful headless
execution receipt and inventories the unsigned Windows PyInstaller 6.21.0
CArchive, PYZ and nested base-library ZIP without loading their code objects.
Payloads retain full byte hashes and all matching installed-input candidates.
Python sources use complete module paths and the declared CLI entry point,
never basename-only attribution. Recompiled code must match all code fields;
valid marshal sharing/interning layout differences are explicitly distinguished
from exact wire-byte matches and retain both wire hashes and a code-field digest.

Reports include `inventory.json`, `installed.cdx.json`, optional artifact/frozen
reports, and a `summary.json` binding every report hash to the source commit/tree.
Failed installed-byte checks retain package, source, transformation and anchored
byte diagnostics in `failure.json`, never a success summary. Small byte samples
are capped at 64 KiB per side; bytecode diagnostics use the bounded code-field
parser, not `marshal.loads`, and do not relax exact installed-byte comparison.
`--require-complete` retains the evidence but exits 2: modified bootloader bytes,
unmatched OS/CPython files, native/vendored resolution and observed runtime loader
closure are still incomplete. A byte match proves an input witness, not that an
entire package or its complete runtime closure is bundled.

`tools/audit_bundle.py` links these reports to a successful `advisories` report:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/audit_bundle.py --repo-root . --installed-report /private/tmp/xrr-installed-run --advisory-report /private/tmp/xrr-advisories-run/audit --report-dir /private/tmp/xrr-audit-binding-run
```

It verifies source/input identities, report hashes and inner SBOM bindings,
both exact-pin scan scopes and tool versions, zero findings, and the actual
ordinary wheel records including SHA-256. Bootstrap, VCS and Windows test-only
inputs remain explicitly unscanned. Windows pin lookup is produced by the macOS
advisory job; attaching it to Windows artifact evidence does not claim Windows
executed pip-audit. This receipt is neither a signature nor a complete native
vulnerability scan. Keep the original reports alongside it, and do not promote
PR evidence into trusted release attestations.

## Windows Audit

`audit-windows.yml` is separate from the existing GUI executable release job.
It runs on a read-only hosted Windows 2025 runner, checks out the exact event
commit, creates a unique external environment and disables shared pip caching.
The bootstrap and pytest additions reuse existing universal hash-pinned wheels.
It verifies the Windows wheel bytes before installation, runs existing CLI and
export tests, and builds a headless executable for help/validation/export
checks. Installed/frozen byte verification follows those execution checks and
precedes wheel-cache publication. Reports bind the executable hash to the source
commit and explicitly state that the GUI was not tested. The job-owned environment is cleaned after
evidence upload; the release asset set is unchanged.

## Verified Wheel Cache

`tools/package_cache.py` provides a wheel-only cache for either manifest target.
The exact key includes the trust domain (`trusted` or `pr-N`), target,
Python/pip versions, lock SHA-256 and complete canonical manifest SHA-256.
Each supplied external cache directory is dedicated to that key. Its on-disk
entry uses the key's SHA-256 to avoid unnecessarily long Windows paths.

```bash
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/package_cache.py key --manifest tools/package-manifests/windows-x64-py312.json --trust-domain trusted
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/package_cache.py fetch --manifest tools/package-manifests/windows-x64-py312.json --trust-domain trusted --cache-dir /tmp/xrr-windows-wheel-cache --report-dir /tmp/xrr-cached-download
```

A cold cache uses the existing hash-required, cache-disabled downloader. A warm
cache verifies every wheel before and after copying it to the new report.
Missing/extra files, symlinks, unrelated cache members and byte mismatches fail;
they do not trigger a silent download fallback. Failed copies remove only their
owned temporary output; old evidence and unrelated paths are preserved.
`cache.json` records the exact key and whether the operation was a verified hit.
Installation still independently verifies the copied bytes.

The Windows audit uses pinned `actions/cache` restore/save actions with no
restore prefixes. PR keys include the PR number; trusted events use a separate
key domain. GitHub's branch/ref-scoped cache storage remains part of the trust
boundary. Only the dedicated package cache is uploaded, never a venv, report,
credential or pip HTTP cache. Cache saving requires successful validation.
Actual hosted cold/warm runs remain a separate verification requirement.

Lock-resolver caches are separate from these verified inputs. Neither cache
establishes account isolation for a self-hosted runner sharing the developer's
account.

## Verified macOS Build Inputs

The shared macOS setup uses one exact, trust-scoped cache key for ordinary
wheel inputs and `tools/refnx-build-requirements.lock`. The dedicated builder
manifest also binds the application lock, direct builder requirements and
`refnx-source.json`; it does not enlarge the application wheel manifest or
published runtime dependencies. Every restore rechecks all source/wheel bytes.

`tools/refnx_build.py` extracts only the verified source archive and builds in
a disposable external venv using hash-pinned pip, Meson, Cython, Ninja and their
closure. Pip builds with `--no-index --no-deps --no-build-isolation`; inherited
compiler/cache flags are excluded. The report records the actual compiler,
SDK, Python and CPU, the derived wheel hash, metadata and installed native
bytes. Optional reduction imports are not part of the reflect smoke test;
their native files are still byte-checked. This is not a claim of cross-host
bit-reproducibility or a network sandbox around the build backend.

The provider cache path is stable across jobs and run attempts, while venvs
and reports have unique private job roots. `tools/macos_environment.py` binds
ownership to the current user and directory identities. The paired cleanup
action rechecks those identities, removes only the job venv and uploads its
receipt. Setup retains reports and the derived refnx wheel as build evidence,
but excludes downloaded input copies from the artifact. The derived wheel is
never saved to the shared cache or treated as independent trusted input.

Native dependency/installed/executable SBOM completeness and actual hosted
cold/warm behavior remain separate acceptance checks. Main/tag runner
selection is unchanged; changing the account or runner requires approval.
