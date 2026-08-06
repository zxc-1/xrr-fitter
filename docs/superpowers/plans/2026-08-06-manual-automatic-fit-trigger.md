# Manual Automatic Fit Trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make automatic imports stop at an inspectable `PENDING` project and require the operator to click the existing automatic-fit button before any worker starts.

**Architecture:** Remove the import-to-fit Qt signal and workspace connection while preserving the existing import transaction and explicit `FitPanel.start_automatic_fit()` path. The button continues to call the public API without a batch filter, so one operator action runs all currently runnable automatic datasets across imported batches.

**Tech Stack:** Python 3.12, PySide6, pytest/pytest-qt, existing `xrr_fitter.api` GUI boundary.

## Global Constraints

- Do not change filename parsing, material recognition, substrate selection, automatic SiO2, batch routing, fitting budgets, or numerical results.
- Do not add a confirmation dialog, automatic-start preference, persistence field, or production dependency.
- GUI code continues to import domain behavior only through `xrr_fitter.api`.
- Keep `FitPanel.start_automatic_fit()`, worker protocols, checkpoint behavior, expert fitting, and exports unchanged.
- Obtain fresh RED then GREEN evidence before claiming the runtime behavior changed.
- Do not stage or modify `.claude/` or root-level probe files.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/xrr_fitter/gui/data/panel.py` | Publish imported datasets without requesting a fit. |
| `src/xrr_fitter/gui/workspace.py` | Configure workspace layout without wiring import completion to fitting. |
| `src/xrr_fitter/gui/fitting/panel.py` | Keep the explicit automatic-fit button and describe its all-runnable-datasets scope. |
| `tests/gui/test_data_import.py` | Prove successful imports persist as pending automatic datasets. |
| `tests/integration/test_gui_automatic_workflow.py` | Prove no worker starts at import and one starts after an explicit button click. |
| `docs/user-guide.md` | Document inspection-before-fit workflow and button scope. |

### Task 1: Require Explicit Automatic-Fit Action

**Files:**
- Modify: `tests/gui/test_data_import.py:190-207`
- Modify: `tests/integration/test_gui_automatic_workflow.py:75-111`
- Modify: `src/xrr_fitter/gui/data/panel.py:45-53,216-254`
- Modify: `src/xrr_fitter/gui/workspace.py:123-149`
- Modify: `src/xrr_fitter/gui/fitting/panel.py:57-63`
- Modify: `docs/user-guide.md:585-604`

**Interfaces:**
- Consumes: `DataPanel.import_paths(paths) -> api.ProjectImportResult` and the existing `FitPanel.automatic_button` click path.
- Preserves: `FitPanel.start_automatic_fit(import_batch_id: str | None = None, checkpoint_path=None) -> bool`.
- Produces: imports whose datasets retain `automation.status.value == "pending"` without calling `api.start_automatic_fit_job()`.
- Removes: the GUI-internal `DataPanel.automatic_fit_requested` signal and `_connect_automatic_workflow()` workspace helper.

- [ ] **Step 1: Replace the import-start assertions with manual-trigger assertions**

Replace `test_successful_automatic_import_emits_batch_request` in `tests/gui/test_data_import.py` with:

```python
def test_successful_automatic_import_keeps_dataset_pending(
    qtbot,
    tmp_path,
) -> None:
    from dataclasses import replace

    from xrr_fitter.gui.document import ProjectDocument

    project = replace(api.new_project(), measurement_preset=_saved_preset())
    panel = _panel(qtbot, ProjectDocument(project))

    result = panel.import_paths((_write_curve(tmp_path / "P1 Zr.xy"),))

    dataset = panel.document.project.datasets[0]
    assert result.imported_dataset_ids == ("P1",)
    assert dataset.automation.import_batch_id == result.import_batch_id
    assert dataset.automation.status.value == "pending"
    assert dataset.last_valid_result is None
    assert panel.document.project.batch_mode == "independent"
```

Replace the body and name of the automatic GUI integration test with the following decisive flow while retaining its existing fixture setup and failure-table assertions:

```python
def test_partial_import_waits_for_manual_fit_keeps_failure_recovery_and_publishes_curve(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    preset = api.MeasurementPreset(
        "integration-lab",
        api.BeamSpec("monochromatic", wavelength_a=1.5406),
        api.InstrumentSpec(instrument_id="integration-lab"),
    )
    project = replace(api.new_project(), measurement_preset=preset)
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    starts: list[tuple[api.XrrProject, str | None]] = []

    def start(value, import_batch_id=None, checkpoint_path=None):
        del checkpoint_path
        starts.append((value, import_batch_id))
        return FakeJob(value)

    monkeypatch.setattr(api, "start_automatic_fit_job", start, raising=False)
    valid = _write_curve(tmp_path / "P1 Zr.xy")
    bad = _write_curve(tmp_path / "bad-name.xy")

    result = window.data_panel.import_paths((valid, bad))

    assert starts == []
    assert window.document.project.datasets[0].automation.status.value == "pending"
    assert window.document.project.datasets[0].last_valid_result is None
    assert window.fit_panel.automatic_button.isEnabled()

    window.fit_panel.automatic_button.click()

    assert len(starts) == 1
    assert starts[0][0] is result.updated_project
    assert starts[0][1] is None

    window.fit_panel.controller.poll_now()

    assert window.document.project.datasets[0].last_valid_result is not None
    failures = window.data_panel.findChild(QTableWidget, "importFailureTable")
    assert failures.rowCount() == 1
    assert failures.item(0, 0).text() == "bad-name.xy"
    assert failures.item(0, 2).text()
```

- [ ] **Step 2: Run the focused tests and record RED**

Run:

```bash
QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -o addopts= -p no:cacheprovider tests/gui/test_data_import.py::test_successful_automatic_import_keeps_dataset_pending tests/integration/test_gui_automatic_workflow.py::test_partial_import_waits_for_manual_fit_keeps_failure_recovery_and_publishes_curve -q
```

Expected: the data-panel state test passes, while the integration test fails at `assert starts == []` because the current workspace connection starts one automatic worker during import.

- [ ] **Step 3: Remove the automatic import trigger and clarify the button**

In `src/xrr_fitter/gui/data/panel.py`, remove:

```python
automatic_fit_requested = Signal(str)
```

and remove this line from the successful-import branch:

```python
self.automatic_fit_requested.emit(result.import_batch_id)
```

In `src/xrr_fitter/gui/workspace.py`, make `configure_splitters()` begin directly with layout configuration:

```python
def configure_splitters(view: WorkspaceView) -> None:
    workspace = view.workspace_splitter
```

Delete `_connect_automatic_workflow()` entirely. Do not replace it with a confirmation or deferred signal.

In `src/xrr_fitter/gui/fitting/panel.py`, retain the existing button and change only its tooltip:

```python
self.automatic_button.setToolTip("运行项目中所有待拟合的自动数据集")
```

- [ ] **Step 4: Update the stable user workflow**

Replace step 4 under `docs/user-guide.md` section 13.1 with:

```markdown
4. 有效 dataset 导入后保持 `PENDING`，工作台不会自动启动拟合。先检查或修改膜层、
   基底、自动 SiO2 和参数，再点击“自动拟合”；该按钮统一运行项目中全部可运行的
   自动数据集。路由仍以各自 `import_batch_id` 为边界：单条物理签名走单样品路径，
   同批同签名多点先并行预拟合，再联合精修。
```

- [ ] **Step 5: Run focused GREEN**

Run the exact command from Step 2.

Expected: `2 passed`; the fake worker is absent after import and starts exactly once after the explicit button click.

- [ ] **Step 6: Run affected repository gates**

Run each command separately:

```bash
QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 python tools/verify.py gui
QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 python tools/verify.py integration
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py quality
PYTHONDONTWRITEBYTECODE=1 python tools/check_radon.py
git diff --check
```

Expected: every command exits 0. The GUI and integration registries cover the changed files; quality and radon confirm removal did not leave dead imports or degrade maintainability.

- [ ] **Step 7: Commit the behavior change**

```bash
git add src/xrr_fitter/gui/data/panel.py src/xrr_fitter/gui/workspace.py src/xrr_fitter/gui/fitting/panel.py tests/gui/test_data_import.py tests/integration/test_gui_automatic_workflow.py docs/user-guide.md
git commit -m "fix: require manual automatic fit start"
```
