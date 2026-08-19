# Export Integrity And Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish export-format-v2 runs that keep candidate-owned evidence aligned, persist a verifiable checksum manifest, scale linearly across datasets, and compare batch parameters only within a shared unit.

**Architecture:** One complete loadable project snapshot is shared by all dataset JSON documents through a size/hash reference. Export artifacts become lazy producers consumed one at a time by the existing atomic publisher, which hashes files in chunks and writes a canonical manifest before `fsync` and exclusive rename. `DatasetExportData` becomes the single owner of uncertainty attribution and parameter-definition lookup.

**Tech Stack:** Python 3.12, dataclasses, NumPy, pandas/XlsxWriter, Matplotlib Agg, orsopy 1.2.3, PySide6, pytest.

## Global Constraints

- Use Python 3.12 and the `src/` package layout.
- Keep `xrr_fitter.api` as the only supported Python API; do not change `export_result(result, output_dir, *, include_ort=False)`.
- Follow the dependency graph in `docs/architecture/r23-clean-break.md`.
- Do not add production dependencies, compatibility writers, import shims, dual implementations, or silent fallbacks.
- Preserve source revalidation, hostile-path rejection, exact-tree publication, private-partial cleanup, `fsync`, and exclusive rename behavior.
- Every production behavior change requires a witnessed RED test before implementation and a focused GREEN run after implementation.
- Use `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=src`, `-p no:cacheprovider`, and `--import-mode=importlib` for focused pytest commands.
- Do not overwrite unrelated working-tree changes.

---

### Task 1: Centralize Selected-Candidate Evidence Ownership

**Files:**
- Modify: `src/xrr_fitter/io/export_tables.py:149-179,262-302,531-571,785-830`
- Modify: `src/xrr_fitter/io/export_plots.py:149-174`
- Modify: `src/xrr_fitter/io/orso.py:37-58,165-245`
- Modify: `src/xrr_fitter/services/exports.py:133-149`
- Test: `tests/unit/io/test_export_tables.py`
- Test: `tests/unit/io/test_export_plots.py`
- Test: `tests/unit/io/test_orso_export.py`
- Test: `tests/unit/services/test_exports.py`

**Interfaces:**
- Produces: `DatasetExportData.selected_uncertainty -> UncertaintyReport | None`.
- Produces: `DatasetExportData.uncertainty_absent_reason -> str | None`.
- Consumes: persisted `UncertaintyReport.candidate_id` and `FitCandidate.candidate_id`.
- Preserves: `orso_bytes(context, *, covariance)` test boundary while production passes covariance only from `selected_uncertainty`.

- [ ] **Step 1: Write the failing table/log ownership test**

Add a helper that takes `_context_with_mcmc_diagnostics()`, retains its original candidate as the uncertainty owner, adds a valid `candidate-b`, and constructs `DatasetExportData` with `candidate-b` selected. Add this test:

```python
def test_export_omits_uncertainty_owned_by_another_selected_candidate() -> None:
    context = _context_with_mismatched_uncertainty()

    payload = json.loads(dataset_json_bytes(context))
    workbook = pd.ExcelFile(BytesIO(dataset_workbook_bytes(context)))
    correlation = pd.read_excel(BytesIO(dataset_workbook_bytes(context)), sheet_name="Correlation")
    profiles = pd.read_excel(BytesIO(dataset_workbook_bytes(context)), sheet_name="Profiles")
    log = run_log_bytes(context).decode("utf-8")

    assert context.selected.candidate_id == "candidate-b"
    assert context.result.uncertainty.candidate_id != context.selected.candidate_id
    assert payload["run_info"]["mcmc_child_seed"] is None
    assert "uncertainty candidate mismatch" in payload["run_info"]["uncertainty_absent_reason"]
    assert correlation.empty
    assert profiles.empty
    assert "mcmc_child_seed: None" in log
    assert "uncertainty-code" not in log
    assert "uncertainty candidate mismatch" in log
    assert workbook.sheet_names == [
        "Parameters", "Candidates", "RawData", "ModelResiduals",
        "Correlation", "Profiles", "RunInfo",
    ]
```

- [ ] **Step 2: Write the failing SLD-band ownership test**

Construct two otherwise identical contexts around `_context_with_bands()`: one selects a candidate different from the report owner, and one has `uncertainty=None`. Assert their exported SLD PNG bytes are identical:

```python
def test_sld_profile_omits_bands_owned_by_another_candidate() -> None:
    mismatched, absent = _mismatched_and_absent_band_contexts()

    assert mismatched.result.uncertainty.candidate_id != mismatched.selected.candidate_id
    assert sld_profile_png(mismatched) == sld_profile_png(absent)
```

- [ ] **Step 3: Extend the existing ORSO mismatch test**

Require `xrr_fitter.confidence.covariance_absent_reason` to equal the context-level reason and require `error_bars == []` when another candidate owns the report:

```python
assert confidence["covariance_absent_reason"] == context.uncertainty_absent_reason
assert confidence["error_bars"] == []
```

- [ ] **Step 4: Run the ownership tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/unit/io/test_export_tables.py::test_export_omits_uncertainty_owned_by_another_selected_candidate tests/unit/io/test_export_plots.py::test_sld_profile_omits_bands_owned_by_another_candidate tests/unit/io/test_orso_export.py::test_orso_bytes_omits_uncertainty_owned_by_another_candidate`

Expected: FAIL because table/log/plot serializers still read `result.uncertainty` directly and `DatasetExportData` has no shared ownership properties.

- [ ] **Step 5: Implement the shared ownership properties**

Add these properties to `DatasetExportData`, using the exact stable reason strings in all consumers:

```python
@property
def selected_uncertainty(self) -> UncertaintyReport | None:
    report = self.result.uncertainty
    if report is None or report.candidate_id != self.selected.candidate_id:
        return None
    return report

@property
def uncertainty_absent_reason(self) -> str | None:
    report = self.result.uncertainty
    selected = self.selected.candidate_id
    if report is None:
        return "uncertainty not estimated for this fit result"
    if report.candidate_id is None:
        return f"uncertainty candidate is unowned: selected={selected}"
    if report.candidate_id != selected:
        return f"uncertainty candidate mismatch: selected={selected}, owner={report.candidate_id}"
    return None
```

Change `_run_info_payload`, `_correlation_frame`, `_profiles_frame`, `_persisted_diagnostics`, and `run_log_bytes` to read `context.selected_uncertainty`. Add `uncertainty_absent_reason` immediately after `selected_candidate_id` in JSON RunInfo and workbook RunInfo. Append `uncertainty_absent_reason: ...` to the log only when non-null.

- [ ] **Step 6: Apply the same decision to plots, ORSO, and the service**

Make `_selected_bands()` return bands only from `context.selected_uncertainty`. In ORSO, replace the local candidate-owner decision with the context properties. In the service, derive covariance only from the selected report:

```python
report = context.selected_uncertainty
covariance = None if report is None else report.covariance
files.append(ArtifactPayload("fit_result.ort", orso_bytes(context, covariance=covariance)))
```

- [ ] **Step 7: Run focused ownership tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/unit/io/test_export_tables.py tests/unit/io/test_export_plots.py tests/unit/io/test_orso_export.py tests/unit/services/test_exports.py`

Expected: all selected tests PASS with no warnings.

- [ ] **Step 8: Commit candidate ownership**

```bash
git add src/xrr_fitter/io/export_tables.py src/xrr_fitter/io/export_plots.py src/xrr_fitter/io/orso.py src/xrr_fitter/services/exports.py tests/unit/io/test_export_tables.py tests/unit/io/test_export_plots.py tests/unit/io/test_orso_export.py tests/unit/services/test_exports.py
git commit -m "fix: keep exported evidence candidate-owned"
```

---

### Task 2: Introduce Export JSON Version 2 And One Shared Project Snapshot

**Files:**
- Modify: `src/xrr_fitter/io/export_tables.py:49-55,149-180,188-192,328-380`
- Modify: `src/xrr_fitter/services/exports.py:5-34,108-190`
- Test: `tests/unit/io/test_export_tables.py`
- Test: `tests/unit/services/test_exports.py`
- Test: `tests/integration/test_export_workflow.py`

**Interfaces:**
- Produces: `EXPORT_SCHEMA_VERSION = 2`.
- Produces: `PROJECT_SNAPSHOT_PATH = "project_snapshot.xrrproj.json"`.
- Produces: required `DatasetExportData.project_reference: ExportFileRecord`.
- Consumes: `project_to_bytes()`, `resolve_source_path()`, and existing `ExportFileRecord` validation.

- [ ] **Step 1: Write the failing version-2 dataset JSON test**

Replace the frozen R22 shape/order tests with an explicit v2 contract. Update context helpers to carry a deterministic `ExportFileRecord` and add:

```python
def test_export_json_v2_references_one_shared_project_snapshot() -> None:
    context = _context()

    payload = json.loads(dataset_json_bytes(context))

    assert tuple(payload) == (
        "export_schema_version", "dataset_id", "source_path", "source_sha256",
        "beam", "instrument", "scale_prior", "structure_evidence",
        "oxide_decisions", "raw_data", "model_residuals", "fit_result",
        "project", "candidates", "convergence", "run_info",
    )
    assert payload["export_schema_version"] == 2
    assert payload["project"] == {
        "path": context.project_reference.path,
        "size": context.project_reference.size,
        "sha256": context.project_reference.sha256,
    }
    assert "datasets" not in payload["project"]
```

- [ ] **Step 2: Write the failing snapshot publication and round-trip test**

Extend the integration export tree test to find `project_snapshot.xrrproj.json`, load it through `api.load_project()`, and assert IDs, results, and selected candidate IDs match the fitted project:

```python
snapshot = manifest.run_directory / "project_snapshot.xrrproj.json"
reopened = api.load_project(snapshot)
assert tuple(item.dataset_id for item in reopened.datasets) == tuple(item.dataset_id for item in value.datasets)
assert all(item.last_valid_result is not None for item in reopened.datasets)
assert reopened.ui_state.selected_candidate_ids == value.ui_state.selected_candidate_ids
```

Also assert all serialized snapshot source paths are absolute so loading from the run directory does not reinterpret previous relative declarations.

- [ ] **Step 3: Run the v2 and snapshot tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/unit/io/test_export_tables.py::test_export_json_v2_references_one_shared_project_snapshot tests/integration/test_export_workflow.py::test_export_multi_dataset_writes_complete_atomic_artifact_tree`

Expected: FAIL because dataset JSON still embeds the project and no snapshot is published.

- [ ] **Step 4: Add and validate the project reference**

Import `ExportFileRecord`, add `project_reference` to `DatasetExportData`, validate its type and exact path, and emit the reference:

```python
PROJECT_SNAPSHOT_PATH = "project_snapshot.xrrproj.json"
EXPORT_SCHEMA_VERSION = 2

def _project_reference(record: ExportFileRecord) -> dict[str, object]:
    return {"path": record.path, "size": record.size, "sha256": record.sha256}

return {
    "export_schema_version": EXPORT_SCHEMA_VERSION,
    # existing fields in their documented order
    "project": _project_reference(context.project_reference),
}
```

Delete `_export_project_document()` because v2 no longer embeds a modified codec document.

- [ ] **Step 5: Build the loadable snapshot once in the service**

Resolve every source declaration before serialization and compute one shared record:

```python
def _snapshot_project(project: XrrProject) -> XrrProject:
    datasets = tuple(
        replace(dataset, source_path=str(resolve_source_path(project, dataset).resolve()))
        for dataset in project.datasets
    )
    return replace(project, datasets=datasets, base_directory=None)

snapshot = project_to_bytes(_snapshot_project(project))
reference = ExportFileRecord(
    PROJECT_SNAPSHOT_PATH,
    len(snapshot),
    sha256(snapshot).hexdigest(),
)
contexts = _contexts(project, reference)
root_files = (ArtifactPayload(PROJECT_SNAPSHOT_PATH, snapshot), *_root_artifacts(contexts))
```

Change `_contexts()` to require the reference and pass the same immutable object to every context.

- [ ] **Step 6: Run v2 unit, service, and integration tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/unit/io/test_export_tables.py tests/unit/services/test_exports.py tests/integration/test_export_workflow.py`

Expected: all selected tests PASS; root files now include the snapshot and per-dataset JSON no longer contains project datasets.

- [ ] **Step 7: Commit export schema v2**

```bash
git add src/xrr_fitter/io/export_tables.py src/xrr_fitter/services/exports.py tests/unit/io/test_export_tables.py tests/unit/services/test_exports.py tests/integration/test_export_workflow.py
git commit -m "feat: share one project snapshot across exports"
```

---

### Task 3: Publish Lazy Artifacts And A Canonical Checksum Manifest

**Files:**
- Modify: `src/xrr_fitter/io/export_run.py:27-347`
- Modify: `src/xrr_fitter/services/exports.py:7-190`
- Test: `tests/unit/io/test_export_run.py`
- Test: `tests/unit/services/test_exports.py`
- Test: `tests/integration/test_export_workflow.py`

**Interfaces:**
- Produces: `ArtifactProducer(path: str, renderer: Callable[[], bytes])`.
- Changes internal `DatasetArtifacts.files` to `tuple[ArtifactProducer, ...]`.
- Changes internal `publish_export_run()` root/dataset inputs from eager `ArtifactPayload` to lazy producers.
- Preserves `ArtifactPayload` and `publish_exact_tree()` for immutable exact-tree publication.
- Produces reserved `EXPORT_MANIFEST_PATH = "export_manifest.json"` and schema `xrr-fitter-export-manifest-v2`.

- [ ] **Step 1: Write failing producer validation and laziness tests**

Add tests that prove renderers are callable, are not invoked during construction, execute in path order during publication, and reject non-bytes or empty bytes only when rendered:

```python
def test_export_producers_render_lazily_in_sorted_path_order(tmp_path: Path) -> None:
    calls: list[str] = []
    producers = (
        ArtifactProducer("z.txt", lambda: (calls.append("z"), b"z")[1]),
        ArtifactProducer("a.txt", lambda: (calls.append("a"), b"a")[1]),
    )
    assert calls == []

    manifest = publish_export_run(
        tmp_path,
        (DatasetArtifacts("sample", producers),),
        run_timestamp="20260715T120000",
    )

    assert calls == ["a", "z"]
    assert all((manifest.run_directory / item.path).is_file() for item in manifest.files)
```

- [ ] **Step 2: Write the failing canonical manifest coverage test**

Publish root and dataset producers, parse `export_manifest.json`, and independently recompute every listed entry:

```python
def test_export_persists_canonical_manifest_covering_every_other_file(tmp_path: Path) -> None:
    manifest = publish_export_run(
        tmp_path,
        (_producer_dataset("sample"),),
        (ArtifactProducer("project_snapshot.xrrproj.json", lambda: b"project"),),
        run_timestamp="20260715T120000",
    )
    path = manifest.run_directory / "export_manifest.json"
    content = path.read_bytes()
    payload = json.loads(content)
    listed = {item["path"]: item for item in payload["files"]}
    observed = {
        item.path for item in manifest.files if item.path != "export_manifest.json"
    }

    assert content.endswith(b"\n")
    assert payload["schema"] == "xrr-fitter-export-manifest-v2"
    assert set(listed) == observed
    for relative, record in listed.items():
        artifact = manifest.run_directory / relative
        body = artifact.read_bytes()
        assert record == {
            "path": relative,
            "size": len(body),
            "sha256": sha256(body).hexdigest(),
        }
```

- [ ] **Step 3: Write failing manifest-failure cleanup and chunked-hash tests**

Monkeypatch manifest serialization to raise and assert no final/partial run. Monkeypatch a file object's `read()` to record requested sizes and assert `_record()` never requests an unbounded read:

```python
assert read_sizes and all(size == HASH_CHUNK_SIZE for size in read_sizes)
```

- [ ] **Step 4: Run producer/manifest tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/unit/io/test_export_run.py`

Expected: FAIL because `ArtifactProducer`, canonical manifest publication, and chunked hashing do not exist.

- [ ] **Step 5: Implement `ArtifactProducer` without changing exact-tree payloads**

Add the internal value and renderer validation:

```python
@dataclass(frozen=True, slots=True)
class ArtifactProducer:
    path: str
    renderer: Callable[[], bytes]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        if not callable(self.renderer):
            raise TypeError("artifact renderer must be callable")

    def render(self) -> bytes:
        content = self.renderer()
        if not isinstance(content, bytes):
            raise TypeError("artifact renderer must return bytes")
        if not content:
            raise ValueError("artifact content must not be empty")
        return content
```

Use separate `_producers()` validation for run export. Leave `_payloads()` and exact-tree matching byte-only.

- [ ] **Step 6: Render one artifact at a time and hash in chunks**

Replace bulk `_write_artifacts()` on the run path with a loop that renders, writes with `xb`, records, and drops the local bytes before the next iteration. Replace `_record()` with `stat().st_size` plus chunked SHA-256:

```python
HASH_CHUNK_SIZE = 1024 * 1024

def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
```

- [ ] **Step 7: Generate and publish the reserved manifest**

After all caller producers are recorded, build canonical bytes from the sorted records and dataset manifests:

```python
def _manifest_bytes(
    datasets: tuple[DatasetExportManifest, ...],
    root_files: tuple[ExportFileRecord, ...],
) -> bytes:
    files = tuple(sorted(
        (*root_files, *(record for dataset in datasets for record in dataset.files)),
        key=lambda item: item.path,
    ))
    snapshot = next(item for item in root_files if item.path == PROJECT_SNAPSHOT_PATH)
    payload = {
        "schema": "xrr-fitter-export-manifest-v2",
        "export_schema_version": 2,
        "project_snapshot": _record_value(snapshot),
        "datasets": [
            {"dataset_id": item.dataset_id, "directory": item.directory}
            for item in datasets
        ],
        "files": [_record_value(item) for item in files],
    }
    return (json.dumps(payload, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")
```

Reserve `export_manifest.json` in layout validation, write it last, append its record to returned root files, then include it in `_sync_tree()` before exclusive rename.

- [ ] **Step 8: Convert the service to lazy producers**

Wrap every serializer without invoking it:

```python
ArtifactProducer("fit_result.xlsx", lambda: dataset_workbook_bytes(context))
ArtifactProducer("fit_result.json", lambda: dataset_json_bytes(context))
ArtifactProducer("fit_overview.png", lambda: fit_overview_png(context))
```

Capture loop variables through function-local context or default arguments. Snapshot bytes may be captured once by its producer. Update the service test so `publish_export_run` receives producers while the serializer call log remains empty; invoke captured renderers to prove they delegate correctly.

- [ ] **Step 9: Run run/service/integration tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/unit/io/test_export_run.py tests/unit/services/test_exports.py tests/integration/test_export_workflow.py`

Expected: all selected tests PASS; every run contains a valid checksum manifest and failed publication leaves no partial directory.

- [ ] **Step 10: Commit lazy manifested publication**

```bash
git add src/xrr_fitter/io/export_run.py src/xrr_fitter/services/exports.py tests/unit/io/test_export_run.py tests/unit/services/test_exports.py tests/integration/test_export_workflow.py
git commit -m "feat: publish lazy exports with checksum manifest"
```

---

### Task 4: Make Batch Parameter Output Unit-Aware

**Files:**
- Modify: `src/xrr_fitter/io/export_tables.py:712-761`
- Modify: `src/xrr_fitter/io/export_plots.py:213-263`
- Test: `tests/unit/io/test_export_tables.py`
- Test: `tests/unit/io/test_export_plots.py`

**Interfaces:**
- Produces: `DatasetExportData.parameter_definition(name: str) -> ParameterDefinition`.
- Produces internal comparable series grouped as `dict[str, tuple[str, ...]]`, keyed by exact unit; empty unit renders as `dimensionless`.
- Rejects selected parameters absent from `result.parameter_definitions`.

- [ ] **Step 1: Write failing batch workbook metadata tests**

Update the expected stable columns and assert exact metadata values:

```python
assert parameters.columns.tolist() == [
    "dataset_id", "parameter_name", "display_name", "category",
    "value", "lower", "upper", "unit",
]
assert parameters.loc[parameters["parameter_name"] == "scale", "unit"].eq("").all()
```

Add a malformed context with a selected parameter absent from definitions and assert `ValueError("selected parameter has no unique definition: missing-name")`.

- [ ] **Step 2: Write failing unit-grouped trend tests**

Build two contexts with common dimensionless and Angstrom parameters, monkeypatch `Axes.plot`, and assert series are routed to distinct axes whose y labels include `dimensionless` and `angstrom`. Add a same-name/different-unit pair and assert the name is not plotted.

```python
assert set(axis.get_ylabel() for axis in axes) == {
    "Selected value (dimensionless)",
    "Selected value (angstrom)",
}
assert "mixed-unit" not in plotted_names
```

- [ ] **Step 3: Run batch tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/unit/io/test_export_tables.py::test_export_batch_parameters_put_dataset_identity_first tests/unit/io/test_export_plots.py::test_export_parameter_trends_group_common_parameters_by_unit`

Expected: FAIL because batch rows omit metadata and the trend renderer has one mixed-unit axis.

- [ ] **Step 4: Implement strict parameter-definition lookup**

Add a context method that requires exactly one matching definition:

```python
def parameter_definition(self, name: str) -> ParameterDefinition:
    matches = tuple(item for item in self.result.parameter_definitions if item.name == name)
    if len(matches) != 1:
        raise ValueError(f"selected parameter has no unique definition: {name}")
    return matches[0]
```

Use it when building batch workbook rows and include `display_name`, `category`, and `unit` in the specified column order.

- [ ] **Step 5: Implement one trend subplot per exact unit**

Compute common names, retain only names whose definition units are identical across all contexts, group stable sorted names by unit, and create vertically stacked axes:

```python
unit_groups = _common_parameter_units(values)
figure = Figure(figsize=(7.2, max(4.2, 2.8 * max(1, len(unit_groups)))), layout="constrained")
axes = figure.subplots(max(1, len(unit_groups)), 1, squeeze=False).ravel()
for axis, (unit, names) in zip(axes, unit_groups.items(), strict=True):
    _plot_parameter_lines(axis, values, names, positions)
    label = unit or "dimensionless"
    axis.set_ylabel(f"Selected value ({label})")
```

Apply dataset-order ticks and x labels to every used axis. Preserve isolated Matplotlib defaults and deterministic metadata.

- [ ] **Step 6: Run table/plot tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/unit/io/test_export_tables.py tests/unit/io/test_export_plots.py`

Expected: all selected tests PASS, including deterministic PNG assertions after intentional baselines are updated only where the changed trend image is covered.

- [ ] **Step 7: Commit unit-aware batch output**

```bash
git add src/xrr_fitter/io/export_tables.py src/xrr_fitter/io/export_plots.py tests/unit/io/test_export_tables.py tests/unit/io/test_export_plots.py
git commit -m "fix: separate exported parameter trends by unit"
```

---

### Task 5: Surface Manifest Evidence In GUI And CLI

**Files:**
- Modify: `src/xrr_fitter/gui/export/dialog.py:64-68`
- Modify: `src/xrr_fitter/cli/commands.py:102-111`
- Modify: `tests/gui/test_export_dialog.py`
- Modify: `tests/unit/cli/test_dispatch.py`
- Modify: `tests/integration/test_cli_workflow.py`
- Modify: `docs/algorithm.md:470-484,672-679`

**Interfaces:**
- Consumes the `export_manifest.json` record from `ExportManifest.root_files`.
- Preserves the CLI run directory as stdout line 1.
- Produces GUI file lines containing path, size, and full lowercase SHA-256.

- [ ] **Step 1: Write failing GUI summary test**

Extend the manifest fixture with an `export_manifest.json` root record and assert exact evidence text:

```python
assert (
    f"{manifest.run_directory / record.path} "
    f"({record.size} bytes, sha256 {record.sha256})"
) in export_summary(manifest)
```

- [ ] **Step 2: Write failing CLI manifest output test**

Return a fake manifest containing a manifest record and assert stdout lines:

```python
assert capsys.readouterr().out.splitlines() == [
    str(manifest.run_directory),
    f"manifest: {manifest.run_directory / record.path}",
    f"manifest_sha256: {record.sha256}",
]
```

- [ ] **Step 3: Run GUI/CLI tests and verify RED**

Run: `QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/gui/test_export_dialog.py tests/unit/cli/test_dispatch.py`

Expected: FAIL because summaries currently print paths only and CLI prints only the run directory.

- [ ] **Step 4: Implement evidence-rich output**

Render GUI records as:

```python
files = "\n".join(
    f"{manifest.run_directory / record.path} "
    f"({record.size} bytes, sha256 {record.sha256})"
    for record in manifest.files
)
```

In CLI, find exactly one root manifest record, print the run directory first, then its path and hash. Raise `RuntimeError` if a returned manifest violates that internal invariant; add `RuntimeError` to the expected CLI export error mapping.

- [ ] **Step 5: Update algorithm documentation**

Document export schema version 2, the shared project snapshot, candidate-owned uncertainty rule, lazy publication, manifest self-hash exclusion, and the CLI three-line output. Remove statements that imply each dataset JSON embeds a complete project.

- [ ] **Step 6: Run GUI, CLI, and CLI integration tests and verify GREEN**

Run: `QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/gui/test_export_dialog.py tests/unit/cli/test_dispatch.py tests/integration/test_cli_workflow.py`

Expected: all selected tests PASS.

- [ ] **Step 7: Commit user-facing audit evidence**

```bash
git add src/xrr_fitter/gui/export/dialog.py src/xrr_fitter/cli/commands.py tests/gui/test_export_dialog.py tests/unit/cli/test_dispatch.py tests/integration/test_cli_workflow.py docs/algorithm.md
git commit -m "feat: expose export manifest evidence"
```

---

### Task 6: End-To-End Verification And Test Manifest Refresh

**Files:**
- Modify: `verification/r23/tests.json`
- Verify: all production and test files changed in Tasks 1-5

**Interfaces:**
- Consumes the final committed test collection and locked macOS Python 3.12 environment.
- Produces a source-bound `verification/r23/tests.json` whose `source_commit` is the implementation HEAD immediately before the manifest refresh commit.

- [ ] **Step 1: Run the complete focused export suite**

Run: `QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= -p no:cacheprovider --import-mode=importlib -q tests/unit/model/test_export_values.py tests/unit/io/test_export_run.py tests/unit/io/test_export_tables.py tests/unit/io/test_export_plots.py tests/unit/io/test_orso_export.py tests/unit/services/test_exports.py tests/unit/cli/test_dispatch.py tests/integration/test_export_workflow.py tests/integration/test_cli_workflow.py tests/gui/test_export_dialog.py tests/regression/test_orso_validation.py`

Expected: all tests PASS with no skips, xfails, or warnings.

- [ ] **Step 2: Run a real two-dataset export audit**

Use the existing integration fitted-project helper in a temporary directory, export with `include_ort=True`, and assert in one Python command that:

- `export_manifest.json` covers all other files and every size/hash matches.
- `project_snapshot.xrrproj.json` reloads with both fitted results and selections.
- Both dataset JSON documents use export schema 2 and reference the same snapshot record.
- Excel contains the new metadata columns.
- ORSO loads through orsopy.
- All PNG files have a nonempty bounding box and nonzero pixel variance.

Expected: the command exits 0 and its `TemporaryDirectory` leaves no artifact behind.

- [ ] **Step 3: Run repository verification modes**

Run: `PYTHONDONTWRITEBYTECODE=1 ../venvs/repo/bin/python tools/verify.py unit`

Run: `PYTHONDONTWRITEBYTECODE=1 ../venvs/repo/bin/python tools/verify.py integration`

Run: `QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 ../venvs/repo/bin/python tools/verify.py gui`

Run: `PYTHONDONTWRITEBYTECODE=1 ../venvs/repo/bin/python tools/verify.py quality`

Run: `PYTHONDONTWRITEBYTECODE=1 ../venvs/repo/bin/python tools/check_radon.py`

Expected: every command exits 0.

- [ ] **Step 4: Check formatting, repository hygiene, and accidental artifacts**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors and only intentional tracked changes; no `.pytest_cache`, `__pycache__`, build directory, exported run, or temporary report remains.

- [ ] **Step 5: Commit any final test/doc correction before freezing the collection**

If verification required a correction, inspect `git diff --name-only`, list each intentional corrected source or test path explicitly in `git add`, and commit them with `git commit -m "test: close export integrity verification gaps"`. Skip this commit when verification required no correction. Do not include `verification/r23/tests.json` yet.

- [ ] **Step 6: Regenerate the source-bound test manifest**

Run: `TEST_SOURCE_COMMIT=$(git rev-parse HEAD) && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python tools/collect_test_manifest.py --repo-root . --source-commit "$TEST_SOURCE_COMMIT" --lock-file requirements-macos-arm64-py312.lock --suite tests --output verification/r23/tests.json`

Expected: collection succeeds with the new test node count and a valid `collection_sha256`.

- [ ] **Step 7: Verify and commit the refreshed manifest**

Run: `git diff --check -- verification/r23/tests.json`

```bash
git add verification/r23/tests.json
git commit -m "test: refresh export integrity test manifest"
```

- [ ] **Step 8: Run final source-binding and clean-tree checks**

Run: `PYTHONDONTWRITEBYTECODE=1 ../venvs/repo/bin/python tools/verify.py quality`

Run: `git status --short --branch`

Expected: quality passes and the branch is clean except for the committed implementation being ahead of `origin/main`.
