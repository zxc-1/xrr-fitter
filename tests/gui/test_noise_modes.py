"""The visible noise declaration reaches validation, persistence, and real jobs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox, QDialog, QLabel
from tests.support.model_cases import simple_structure

import xrr_fitter.api as api
from xrr_fitter.gui.data.import_dialog import ImportDialog
from xrr_fitter.gui.data.panel import DataPanel
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.fitting.panel import FitPanel

MODE_LABELS = (
    ("稳健对数（探索）", "robust_log"),
    ("Gaussian（已知标准差）", "gaussian"),
    ("Poisson（原始整数计数）", "poisson"),
)


def _source(tmp_path, *, sigma=True, fractional=False):
    path = tmp_path / "curve.xy"
    rows = []
    for index in range(40):
        intensity = 1200 - index * 25 + (0.5 if fractional else 0)
        suffix = " 5" if sigma else ""
        rows.append(f"{0.05 + index * 0.03:.6f} {intensity}{suffix}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _project(tmp_path, *, sigma=True, fractional=False):
    source = _source(tmp_path, sigma=sigma, fractional=fractional)
    value = replace(api.new_project(), fit_config=replace(api.FitConfig.fast(71), local_workers=1))
    mapping = api.DataColumnMapping(intensity_sigma=2) if sigma else None
    value = api.add_dataset(value, source, api.InstrumentSpec(), column_mapping=mapping)
    value = api.set_structure(value, "curve", simple_structure())
    return api.set_expert_mode(value, True)


def _panel(qtbot, project):
    panel = FitPanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    return panel


def _selector(panel):
    selector = panel.findChild(QComboBox, "noiseModelSelector")
    assert selector is not None, "拟合面板必须提供实际噪声模式选择"
    return selector


def _select(panel, mode):
    selector = _selector(panel)
    index = selector.findData(mode)
    assert index >= 0
    selector.setCurrentIndex(index)
    return selector


def test_noise_selector_exposes_all_three_exact_mode_labels(qtbot):
    panel = _panel(qtbot, api.new_project())
    selector = _selector(panel)
    assert tuple((selector.itemText(i), selector.itemData(i)) for i in range(selector.count())) == MODE_LABELS
    assert selector.isHidden() is False


@pytest.mark.parametrize("mode", ("gaussian", "poisson"))
def test_gui_mode_selection_roundtrips_the_config_and_user_mask(qtbot, tmp_path, mode):
    value = _project(tmp_path)
    mask = (False,) + value.datasets[0].fit_mask[1:]
    value = api.set_fit_mask(value, "curve", np.asarray(mask, dtype=bool))
    panel = _panel(qtbot, value)

    _select(panel, mode)

    updated = panel.document.project
    assert updated.fit_config.noise_model == mode
    assert updated.datasets[0].fit_mask == mask
    path = tmp_path / "selected.xrr.json"
    api.save_project(updated, path)
    loaded = api.load_project(path)
    assert loaded.fit_config == updated.fit_config
    assert loaded.schema_version == 5
    assert loaded.algorithm_version == "xrr-fit-v2-poisson-5"
    restored = _panel(qtbot, loaded)
    assert _selector(restored).currentData() == mode


@pytest.mark.parametrize(
    ("mode", "sigma", "fractional", "error"),
    (("gaussian", False, False, "sigma"), ("poisson", True, True, "integer")),
)
def test_invalid_noise_declaration_blocks_before_worker_launch(
    qtbot, tmp_path, monkeypatch, mode, sigma, fractional, error
):
    panel = _panel(qtbot, _project(tmp_path, sigma=sigma, fractional=fractional))
    started = []
    monkeypatch.setattr(api, "start_fit_job", started.append)

    _select(panel, mode)

    assert panel.start_button.isEnabled() is False
    assert error in panel.status_text().lower()
    assert panel.start_fit() is False
    assert started == []


def test_poisson_selection_is_an_explicit_raw_counts_declaration(qtbot):
    panel = _panel(qtbot, api.new_project())
    _select(panel, "poisson")
    label = panel.findChild(QLabel, "noiseModelRequirements")
    assert label is not None
    assert "声明" in label.text()
    assert "原始" in label.text()
    assert "整数" in label.text()
    assert "counts" in label.text()
    assert "计数率" in label.text()


@pytest.mark.parametrize("mode", ("gaussian", "poisson"))
def test_gui_selection_reaches_a_real_fit_operation(qtbot, tmp_path, mode):
    value = _project(tmp_path)
    budget = replace(
        value.fit_config.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=2,
        local_nfev_per_parameter=1,
        bootstrap_samples=1,
    )
    value = api.set_fit_config(value, replace(value.fit_config, budget=budget, profile_steps=5))
    panel = _panel(qtbot, value)
    _select(panel, mode)
    failures = []
    results = []
    panel.operation_failed.connect(failures.append)
    panel.result_published.connect(results.append)

    assert panel.start_fit() is True
    assert _selector(panel).isEnabled() is False
    qtbot.waitUntil(lambda: not panel.is_running, timeout=60000)

    assert failures == []
    assert len(results) == 1
    candidates = results[0].datasets[0].fit_result.candidates
    assert candidates
    assert {candidate.noise_model for candidate in candidates} == {mode}
    assert _selector(panel).isEnabled() is True
    _assert_mode_change_invalidates_result(panel)


def _assert_mode_change_invalidates_result(panel):
    assert panel.document.project.datasets[0].last_valid_result is not None
    _select(panel, "robust_log")
    dataset = panel.document.project.datasets[0]
    assert dataset.last_valid_result is None
    assert dataset.checkpoint is None
    assert panel.document.project.ui_state.selected_candidate_ids == ()


@pytest.mark.parametrize(("mode", "offset"), (("gaussian", -2), ("poisson", 0)))
def test_import_caller_previews_actual_mode_without_dropping_valid_observations(
    qtbot, tmp_path, monkeypatch, mode, offset
):
    value = replace(api.new_project(), fit_config=replace(api.FitConfig.fast(71), noise_model=mode))
    panel = DataPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    source = tmp_path / "curve.xy"
    source.write_text("\n".join(f"{0.05 + i * 0.03} {i + offset} 1" for i in range(40)) + "\n", encoding="utf-8")
    inspected = []
    modes = []
    import_data = api.import_data

    def import_with_record(*args, **kwargs):
        modes.append(kwargs["noise_model"])
        return import_data(*args, **kwargs)

    def inspect_dialog(dialog):
        dialog.select_beam_kind("monochromatic")
        dialog.set_column_mapping(intensity_sigma=2)
        inspected.extend(dialog._preview_curve.getData()[1][:3])
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(api, "import_data", import_with_record)
    monkeypatch.setattr(ImportDialog, "exec", inspect_dialog)
    panel._confirm_import((source,), folder=False)

    assert modes and set(modes) == {mode}
    assert inspected == [offset, offset + 1, offset + 2]
    assert panel.document.project is value


def test_gaussian_column_mapping_revalidates_the_visible_preview(qtbot, tmp_path, monkeypatch):
    source = _source(tmp_path)
    value = replace(api.new_project(), fit_config=replace(api.FitConfig.fast(71), noise_model="gaussian"))
    panel = DataPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    states = []

    def inspect_dialog(dialog):
        dialog.select_beam_kind("monochromatic")
        states.append(float(dialog._preview_curve.getData()[1][0]))
        dialog.set_column_mapping(intensity=2, intensity_sigma=1)
        states.append(float(dialog._preview_curve.getData()[1][0]))
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(ImportDialog, "exec", inspect_dialog)
    panel._confirm_import((source,), folder=False)

    assert states == [1200.0, 5.0]


def test_unavailable_first_preview_does_not_block_other_batch_files(qtbot, tmp_path, monkeypatch):
    broken = tmp_path / "P1 Zr.xy"
    broken.write_text("no numeric observations\n", encoding="utf-8")
    good = _source(tmp_path).rename(tmp_path / "P2 Zr.xy")
    panel = DataPanel(ProjectDocument())
    qtbot.addWidget(panel)
    observed = []

    def accept_available(dialog):
        dialog.select_beam_kind("monochromatic")
        observed.append(dialog._preview_error.text())
        return QDialog.DialogCode.Accepted if dialog.import_button().isEnabled() else QDialog.DialogCode.Rejected

    monkeypatch.setattr(ImportDialog, "exec", accept_available)
    panel._confirm_import((broken, good), folder=False)

    assert "预览不可用" in observed[0]
    assert tuple(item.dataset_id for item in panel.document.project.datasets) == ("P2",)
    assert panel.failure_table.rowCount() == 1
    assert panel.failure_table.item(0, 0).text() == broken.name


def test_gaussian_mask_editor_can_restore_negative_observations(qtbot, tmp_path):
    source = tmp_path / "gaussian.xy"
    source.write_text("\n".join(f"{0.05 + i * 0.03} {i - 2} 1" for i in range(40)) + "\n", encoding="utf-8")
    value = api.set_fit_config(api.new_project(), replace(api.FitConfig.fast(71), noise_model="gaussian"))
    value = api.add_dataset(
        value, source, api.InstrumentSpec(), column_mapping=api.DataColumnMapping(intensity_sigma=2)
    )
    panel = DataPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    panel.set_fit_range("gaussian", 0.05, 1.22)
    panel.set_point_enabled("gaussian", 7, False)
    panel.set_point_enabled("gaussian", 0, False)

    panel.set_point_enabled("gaussian", 0, True)

    mask = panel.document.project.datasets[0].fit_mask
    assert mask[:3] == (True, True, True)
    assert mask[7] is False
