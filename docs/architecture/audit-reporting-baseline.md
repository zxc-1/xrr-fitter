# Non-GUI audit reporting baseline

These historical measurements are advisory reporting, not CI gate results. The reporting
tools were installed in an external environment and were not added to project
dependencies or release assets. Reports distinguish test execution, dependency
metadata and artifact integrity; none substitutes for the others.

The later permanent development-only tooling is described in `audit-tooling.md`.
It does not retroactively convert this baseline into a passing single-run gate.

## Coverage

Source snapshot: `5a47af1`, coverage.py `7.16.0`, Python 3.12. The application
source is unchanged by the subsequent build/test-toolchain security migration.
Selections come directly from the existing `unit`, `integration`, `spawn` and
`regression` verifier modes. GUI and the 220-case statistical corpus are excluded.

- Statements: **16028 / 17356, 92.35%**.
- Branches: **4140 / 5304, 78.05%**.
- Combined statement/branch score: **89.00%**, across 137 non-GUI source files.

The unified traced run had 1771 passes and one real-worker timeout at the unchanged
30-second deadline while the statistical workload was also running. That node
passed on a traced isolated retry; the ordinary untraced spawn gate also passed.
The combined measurements include both attempts and are not presented as a
single all-green coverage run. No timeout or assertion was relaxed.

Remaining branch gaps are concentrated in model validation, candidate handling,
profile analysis, export publication, batch operations and dataset services.
Prioritize failure-path tests there before negotiating a coverage threshold.

For multiprocessing coverage, retain per-process data and combine only after all
writers exit. A later `coverage combine` without `--append` rebuilds the aggregate;
reporting commands in 7.16.0 can also combine and delete raw inputs unless
`--keep-combined` is set. Preserve raw files and separate incomplete experiments.

## Static Types

mypy `2.3.1`, `check_untyped_defs=True`, Python 3.12, with GUI modules excluded:

| Source | Diagnostics | Files with diagnostics |
| --- | ---: | ---: |
| `c369ed0` baseline | 833 | 88 |
| `5a47af1` audit implementation | 808 | 88 |

Both snapshots used the same checker and dependency environment. No blanket
missing-import suppression was used; 18 `import-untyped` diagnostics remain.
The largest classes are `attr-defined` (468), `arg-type` (160), `var-annotated`
(45) and `type-var` (40). These are type diagnostics, not 808 demonstrated runtime
bugs. Annotation contract tests alone do not establish static type safety.

The next useful boundary is concrete input/output containers in fit, analysis
and services. Narrow one boundary with its callers and focused tests; do not
enable a failing project-wide hard gate or silence all diagnostics.

## Dependency Advisories

pip-audit `2.10.1` queried PyPI with exact ordinary pins, `--disable-pip`,
`--no-deps` and `--strict`, without fixes or ignored advisory IDs. It checked 44
macOS and 38 Windows pins. The `refnx` Git commit is explicitly outside that
lookup; no released PyPI version is invented for it. Native bundled libraries
and archive byte integrity require different evidence.

The initial audit produced six macOS and five Windows records, including
duplicates. After deduplication these were four unique CVEs in three packages:

| Tool | Advisory | Migrated version |
| --- | --- | --- |
| pytest | CVE-2025-71176 | 9.0.3 |
| setuptools | CVE-2025-47273, CVE-2026-59890 | 83.0.0 |
| wheel | CVE-2026-24049 | 0.46.2 |

The ordinary-pin rescan on 2026-09-08 found **zero known advisories** after the
migration. This time-bound result is neither a VCS/native scan nor a guarantee
that future advisory data will remain unchanged. The migrated builder puts the
license at `.dist-info/licenses/LICENSE`; the exact archive allowlist and a
license-byte test track that layout without permitting both old and new paths.

See `dependency-evidence.md` for SBOM limits and the rollout decisions still
required for shared caches, permanent tooling gates and additional release assets.
