# Export Integrity And Scaling Design

## Goal

Make one exported run independently auditable, prevent evidence from one fit
candidate being attributed to another, keep batch export storage and peak
serialization memory linear in the number of datasets, and make parameter
comparisons dimensionally honest.

This is a deliberate export-format revision. Newly emitted dataset JSON uses
`export_schema_version: 2`; the application no longer emits the historical
unversioned document that embedded the complete project in every dataset file.

## Scope

The change covers result publication through `api.export_result()`, including
the dataset JSON/workbook/log, exported plots, optional ORSO documents, GUI
completion summary, and CLI export output. The public Python API signature and
the existing timestamped directory publication model remain unchanged.

The change does not add an export importer, archive the run into one container,
add a production dependency, retain an export-format-v1 writer, or make the
checksum manifest self-authenticating. A recipient still needs an externally
communicated hash or signature to authenticate the manifest itself.

## Run Layout

Every run contains these root files:

- `project_snapshot.xrrproj.json`: one complete project document produced by
  the authoritative project codec. It retains all datasets, results, candidate
  ownership, configuration, seed state, and UI candidate selections and must
  round-trip through `api.load_project()`.
- `export_manifest.json`: one canonical UTF-8/LF JSON checksum manifest.
- `compatibility_summary.xlsx`.
- For multi-dataset runs only, `batch_summary.xlsx` and
  `parameter_trends.png`.

Each dataset directory retains the existing fixed artifact names:

- `fit_result.json`
- `fit_result.xlsx`
- `fit_overview.png`
- `sld_profile.png`
- `residuals.png`
- `run_log.txt`
- Optional `fit_result.ort`

Dataset directory allocation, hostile-ID slugging, timestamp/token run names,
and no-replace publication remain unchanged.

## Dataset JSON Version 2

`fit_result.json` keeps the existing result-oriented top-level sections so it
remains convenient to inspect, but adds `export_schema_version` as the first
field and changes `project` from an embedded project object to this reference:

```json
{
  "path": "project_snapshot.xrrproj.json",
  "size": 12345,
  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

The reference path is run-root-relative, not relative to the dataset
directory. The size and SHA-256 must exactly match both the published snapshot
and its entry in `export_manifest.json`.

The remaining top-level result, candidates, raw data, model residuals,
convergence, and run information describe only the owning dataset. The
authoritative project snapshot appears once per run, so total serialized data
grows linearly rather than embedding every dataset result in every dataset
JSON.

## Candidate-Owned Evidence

`DatasetExportData` owns the single candidate-evidence decision used by every
serializer:

- `selected_uncertainty` is the persisted uncertainty report only when its
  non-null `candidate_id` exactly equals `selected.candidate_id`.
- A missing report, an unowned legacy report, and a report owned by a different
  candidate all produce no selected uncertainty.
- `uncertainty_absent_reason` distinguishes those cases and names the selected
  and owning candidate IDs for a mismatch.

All exported uncertainty-derived evidence must use that decision:

- `Correlation` and `Profiles` workbook sheets are empty on absence.
- `RunInfo.mcmc_child_seed` is empty on absence and RunInfo records the reason.
- `run_log.txt` excludes mismatched uncertainty diagnostics and MCMC seed and
  records the reason.
- `sld_profile.png` does not draw mismatched credible bands.
- ORSO error bars, covariance, and confidence extensions use the same report
  or record the same absence reason.

Fit-wide warnings and stage summaries remain fit-wide. Candidate diagnostics
continue to come from the selected candidate. The complete project snapshot
retains other candidates and their owned evidence for audit without presenting
it as selected-candidate evidence.

## Persistent Manifest

`export_manifest.json` uses this canonical schema:

```json
{
  "schema": "xrr-fitter-export-manifest-v2",
  "export_schema_version": 2,
  "project_snapshot": {
    "path": "project_snapshot.xrrproj.json",
    "size": 12345,
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  },
  "datasets": [
    {"dataset_id": "sample", "directory": "001-sample-aaaaaaaa"}
  ],
  "files": [
    {
      "path": "compatibility_summary.xlsx",
      "size": 123,
      "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    }
  ]
}
```

The file list is sorted by normalized POSIX path and contains every published
regular file except `export_manifest.json` itself. Excluding the manifest
avoids a recursive self-hash. The returned in-memory `ExportManifest` includes
the manifest's own `ExportFileRecord`, allowing the GUI and CLI to display an
external anchor for it.

Manifest serialization happens inside the private publication directory after
all other files have been written and hashed but before tree `fsync` and the
exclusive rename. Manifest creation, writing, hashing, or syncing failure
therefore fails the complete run and cleans only that call's private partial
directory.

The GUI completion summary lists every path, byte size, and SHA-256. CLI output
keeps the run directory as its first line for script compatibility, then prints
the manifest path and manifest SHA-256 on named lines.

## Linear-Memory Publication

Run publication separates fixed byte payloads used by exact-tree publication
from lazy export producers. An internal immutable `ArtifactProducer` contains a
normalized relative path and a zero-argument renderer returning bytes.

The export service constructs producers without invoking dataset workbook,
JSON, plot, log, or ORSO serializers. The publisher invokes one producer,
writes its bytes, validates and hashes the resulting regular nonempty file, and
then releases that byte object before invoking the next producer. File hashing
reads bounded chunks rather than `Path.read_bytes()`.

The project snapshot is serialized once before dataset contexts are created so
its exact size and digest can be placed in every dataset reference. Holding
that one shared snapshot plus the currently rendered artifact makes peak
serialization memory linear in project size instead of retaining the complete
batch artifact tree.

Existing exact-tree publication keeps `ArtifactPayload` and its byte-for-byte
idempotence behavior. It does not accept lazy producers.

## Dimensionally Honest Batch Output

The `Parameters` sheet in `batch_summary.xlsx` has this stable column order:

1. `dataset_id`
2. `parameter_name`
3. `display_name`
4. `category`
5. `value`
6. `lower`
7. `upper`
8. `unit`

Definition metadata comes from the owning result's
`parameter_definitions`. A selected parameter without a matching definition is
an invalid export context and fails before publication rather than silently
inventing a unit.

`parameter_trends.png` compares only parameter names present in every dataset
with exactly the same unit in every owning definition. It creates one subplot
per unit, uses `dimensionless` for the empty unit, and labels each y-axis with
that unit. Parameter series with different units never share an axis. Dataset
order stays on the x-axis, preserving deterministic rendering and avoiding
font-dependent dataset-name substitutions.

## Determinism And Error Handling

- JSON uses UTF-8, LF, `allow_nan=False`, fixed field insertion order, and a
  trailing newline.
- The export manifest is canonical and deterministic for identical artifact
  records. The randomized run directory name is not embedded in it.
- Workbook formula/URL conversion remains disabled.
- PNG metadata and global Matplotlib style isolation remain unchanged.
- Source identity is checked before snapshot creation and again while prepared
  data is restored, preserving the existing source-race guard.
- Serialization, schema validation, write, hash, `fsync`, manifest, or rename
  failure publishes no final run.
- Existing published runs are never replaced or deleted.

## Compatibility

- `api.export_result(result, output_dir, *, include_ort=False)` is unchanged.
- `ExportManifest` and its file records remain the public return value.
- Existing artifact names remain, with two new root artifacts.
- GUI defaults and CLI `--ort` semantics remain unchanged.
- Export-format-v1 byte equality and frozen field-order assertions are replaced
  by explicit version-2 schema assertions. No dual writer or silent fallback is
  retained.
- `project_snapshot.xrrproj.json` uses the current project schema independently
  of export schema version 2.

## Verification

Implementation follows RED -> GREEN for each behavior group.

1. Candidate ownership tests select a different candidate from the uncertainty
   owner and assert empty correlation/profile evidence, no credible bands, no
   leaked MCMC seed or diagnostic, and one explicit absence reason across JSON,
   workbook, log, PNG, and ORSO.
2. Manifest tests independently parse the published file, recompute every
   listed size and SHA-256, verify exact tree coverage, verify the project
   reference, and confirm manifest failure leaves no final or partial run.
3. Scaling tests assert dataset JSON contains only a project reference, the
   snapshot round-trips, serializers are not called while producers are being
   assembled, producers execute one at a time, and file hashing is chunked.
4. Batch tests assert unit metadata columns, reject missing definitions, omit
   cross-dataset unit mismatches from trends, and render one labeled subplot per
   comparable unit.
5. Existing export model, I/O, service, CLI, GUI, integration, ORSO, source-race,
   hostile-path, determinism, cleanup, and exclusive-publication tests remain
   green after their version-2 expectations are updated.
6. Repository verification runs the relevant focused tests first, then
   `python tools/verify.py unit`, `integration`, `gui`, and `quality`, followed
   by `python tools/check_radon.py`.
