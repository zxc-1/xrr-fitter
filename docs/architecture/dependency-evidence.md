# Dependency and release evidence

## Locked environments

The macOS ARM64/Python 3.12 verification environment uses
`requirements-macos-arm64-py312.lock`. The Windows x64/Python 3.12 executable
environment uses `requirements-windows-x64-py312.lock`; it includes packaging
tools, not the macOS test closure. Do not interchange these files or resolve
Windows environment markers using the host platform.

The local composite action `.github/actions/setup-macos-python/action.yml`
owns the macOS runner assertion, external virtual environment, pinned pip,
lock installation, `pip check`, and repository hygiene check. PR jobs remain on
isolated GitHub-hosted runners; main/release jobs retain their existing runner
and permission contracts. The action does not cache or reuse a virtual
environment. Candidate readiness still checks prerequisite files before
installing dependencies.

`pip check` verifies installed dependency metadata, not package integrity or
known vulnerabilities. Exact version pins and a full Git commit for `refnx`
do not constitute a hash-locked download policy. Release wheel/sdist byte
identities remain owned by `tools/verify_distribution.py` and its
`artifact-manifest.json`; release/source binding remains owned by
`tools/release_identity.py`. The inventory below does not replace either.

## Offline lock inventory

`tools/lock_sbom.py` emits deterministic CycloneDX 1.6 JSON to stdout without
installing packages, importing application code, querying an index, or reading
the caller's installed environment. It reuses the platform lock validators and
records the exact lock and `pyproject.toml` SHA-256 values. Repeating a command
on identical inputs in a different checkout produces identical bytes.

From the repository root, with its existing Python 3.12 tool dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/lock_sbom.py --target macos-arm64-py312
PYTHONDONTWRITEBYTECODE=1 python3.12 tools/lock_sbom.py --target windows-x64-py312
```

To retain reports outside the repository, use a new directory and stop on
failure before using the files:

```bash
set -e; OUT=$(mktemp -d /tmp/xrr-lock-inventory.XXXXXX); PYTHONDONTWRITEBYTECODE=1 python3.12 tools/lock_sbom.py --target macos-arm64-py312 > "$OUT/macos.cdx.json"; PYTHONDONTWRITEBYTECODE=1 python3.12 tools/lock_sbom.py --target windows-x64-py312 > "$OUT/windows.cdx.json"; shasum -a 256 "$OUT/"*.json
```

Malformed, noncanonical, undeclared, or changed lock inputs fail with exit code
2 and no JSON on stdout. The tool deliberately does not modify the release
bundle, publish an artifact, or introduce another required CI job.

### Evidence limits

- This is a **locked Python environment inventory**, not a complete SBOM of
  installed wheels or a PyInstaller executable. macOS entries include test and
  build packages; Windows entries include build/packaging packages.
- Exact PyPI pins have versioned package URLs. VCS entries retain their Git URL
  and full commit; no PyPI version is invented for `refnx`.
- Flat lock files do not prove dependency edges, package archive hashes,
  licenses, or the native libraries bundled inside NumPy/Qt/PyInstaller.
  Those fields are absent and the composition is explicitly `incomplete`.
- Input hashes identify the inventory's declarations, not downloaded package
  bytes. A report is not a vulnerability scan, signature, or release approval.
- Format reference: [CycloneDX 1.6 JSON schema](https://github.com/CycloneDX/specification/blob/1.6/schema/bom-1.6.schema.json).

## Changes requiring a separate rollout decision

The following are deliberately not enabled by this audit:

- **Shared download cache:** first define trust-separated PR/release caches,
  platform/Python/pip/lock-hash keys, and package-byte verification. Do not
  restore an environment or release evidence across trust boundaries.
- **Coverage and static typing:** select and pin development tools, measure a
  baseline, and introduce reporting before negotiating incremental thresholds.
  Annotation-contract tests are not a substitute for a static type checker.
- **Vulnerability scanning and complete artifact SBOM:** approve the tool and
  database/network policy, bind reports to downloaded/bundled bytes, and decide
  how findings affect releases before adding a CI gate or changing release
  assets. Existing exact release asset sets and schemas stay unchanged.
