"""Placement of the MCMC sampling controls.

MCMC is an opt-in deep dive rather than part of the fit loop, so its seven
inputs must not occupy the analysis column for every project. The panel keeps
ownership of the controls, which is what lets candidate configuration and
operation state keep tracking the selection, but an on-demand dialog holds them.
"""

from __future__ import annotations

from dataclasses import replace
from math import log

import numpy as np
import pytest
from PySide6.QtCore import Qt
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

import xrr_fitter.api as api


def _uncertainty(candidate_id: str = "candidate-a") -> api.UncertaintyReport:
    return api.UncertaintyReport(
        correlation_names=("scale",),
        correlation_matrix=np.array([[1.0]]),
        profiles=(),
        bootstrap_intervals=(("scale", 0.8, 1.2),),
        bootstrap_failure_rate=0.125,
        boundary_hits=("scale",),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate_id,
    )


def _result():
    first = replace(
        fit_candidate("candidate-a", 0.2),
        ranking_objective=0.4,
        unit_vector=np.zeros(17),
    )
    second = replace(
        fit_candidate("candidate-b", 0.3),
        ranking_objective=0.8,
        unit_vector=np.zeros(3),
    )
    return replace(final_fit_result(first, second), uncertainty=_uncertainty())


def _panel(qtbot, *, expert: bool):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    value = project(dataset_project(result=_result()))
    value = replace(value, base_directory="/private/tmp")
    value = api.select_active_dataset(value, "curve")
    if expert:
        value = api.set_expert_mode(value, True)
    panel = ResultsPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    panel.show()
    return panel


def test_mcmc_controls_stay_out_of_the_panel_layout(qtbot) -> None:
    panel = _panel(qtbot, expert=True)

    assert panel.mcmc_group.isVisibleTo(panel) is False
    for control in (panel.walkers, panel.burn_in, panel.production, panel.thin):
        assert control.isVisibleTo(panel) is False


def test_uncertainty_dialog_button_is_expert_only(qtbot) -> None:
    """One entry point replaces the inline group, and only in expert mode."""
    expert = _panel(qtbot, expert=True)
    assert expert.uncertainty_button.isVisibleTo(expert) is True

    plain = _panel(qtbot, expert=False)
    assert plain.uncertainty_button.isVisibleTo(plain) is False


def test_uncertainty_dialog_hosts_the_mcmc_controls(qtbot) -> None:
    """Opening the entry point reveals the same owned controls, not copies."""
    panel = _panel(qtbot, expert=True)

    dialog = panel.open_uncertainty_dialog()
    qtbot.addWidget(dialog)

    assert panel.mcmc_group.isVisibleTo(dialog) is True
    assert panel.mcmc_group.window() is dialog
    assert panel.mcmc_config() == api.McmcConfig.standard(17)


def _mcmc_report(**changes) -> api.McmcReport:
    values = {
        "config": api.McmcConfig(walkers=4, burn_in=2, production_steps=4),
        "child_seed": 7,
        "parameter_names": ("component.0.thickness_a", "instrument.scale"),
        "samples_physical": np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]]),
        "log_probability": np.zeros(4),
        "acceptance_fraction": np.array([0.2, 0.6, 0.4, 0.8]),
        "split_rhat": np.array([1.05, 1.12]),
        "effective_sample_size": np.array([120.0, 80.0]),
        "boundary_hits": (),
        "candidate_id": "candidate-a",
    }
    values.update(changes)
    return api.McmcReport(**values)


def test_report_lines_show_prior_conflicts() -> None:
    from xrr_fitter.gui.results.uncertainty import _report_lines

    report = replace(_uncertainty(), prior_conflicts=("slab1.thickness",))
    text = "\n".join(_report_lines(report))

    assert "先验冲突" in text
    assert "slab1.thickness" in text


def test_report_lines_show_no_conflict_when_empty() -> None:
    from xrr_fitter.gui.results.uncertainty import _report_lines

    report = replace(_uncertainty(), prior_conflicts=())
    conflict_line = next(line for line in _report_lines(report) if "先验冲突" in line)

    assert conflict_line.endswith("无")


def test_mcmc_lines_show_prior_conflicts() -> None:
    from xrr_fitter.gui.results.uncertainty import _mcmc_lines

    mcmc = _mcmc_report(prior_conflicts=("component.0.thickness_a",))
    report = replace(_uncertainty("candidate-a"), mcmc=mcmc)
    text = "\n".join(_mcmc_lines(report, "candidate-a"))

    assert "MCMC 先验冲突" in text
    assert "component.0.thickness_a" in text


def test_boundary_and_prior_conflict_are_distinct_lines() -> None:
    from xrr_fitter.gui.results.uncertainty import _report_lines

    report = replace(
        _uncertainty(),
        boundary_hits=("scale",),
        prior_conflicts=("slab1.thickness",),
    )
    lines = _report_lines(report)
    boundary_line = next(line for line in lines if line.startswith("边界命中"))
    conflict_line = next(line for line in lines if line.startswith("先验冲突"))

    assert boundary_line != conflict_line
    assert boundary_line.startswith("边界命中（可疑）：")
    assert conflict_line.startswith("先验冲突（信息）：")
    assert "scale" in boundary_line
    assert "slab1.thickness" in conflict_line


def test_mcmc_boundary_and_prior_conflict_explain_signal_nature() -> None:
    from xrr_fitter.gui.results.uncertainty import _mcmc_lines

    mcmc = _mcmc_report(
        boundary_hits=("instrument.scale",),
        prior_conflicts=("component.0.thickness_a",),
    )
    report = replace(_uncertainty("candidate-a"), mcmc=mcmc)
    lines = _mcmc_lines(report, "candidate-a")

    assert any(line.startswith("MCMC 边界命中（可疑）：") for line in lines)
    assert any(line.startswith("MCMC 先验冲突（信息）：") for line in lines)


def _prior_definition(
    name: str = "component.0.thickness_a",
    **changes,
) -> api.ParameterDefinition:
    values = {
        "name": name,
        "display_name": name,
        "unit": "Å",
        "category": "structure",
        "initial": 40.0,
        "lower": 10.0,
        "upper": 100.0,
        "transform": "log",
        "locked": False,
    }
    values.update(changes)
    return api.ParameterDefinition(**values)


def _prior_dialog(
    qtbot,
    definition: api.ParameterDefinition | None = None,
    existing_prior: api.PriorSpec | None = None,
):
    from xrr_fitter.gui.parameters.dialogs import PriorDialog

    dialog = PriorDialog(
        definition or _prior_definition(),
        existing_prior=existing_prior,
    )
    qtbot.addWidget(dialog)
    return dialog


def test_prior_dialog_builds_spec_from_selection(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox

    dialog = _prior_dialog(qtbot)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("normal")
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(1.0)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.2)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() == api.PriorSpec("normal", (10.0, 2.0))


def test_prior_dialog_rejects_invalid_and_stays_open(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox, QLabel

    dialog = _prior_dialog(qtbot)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("normal")
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(1.0)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.0)  # sigma must be positive

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() is None
    error = dialog.findChild(QLabel, "priorDialogError")
    assert error.isVisibleTo(dialog)


def test_prior_dialog_commit_callback_receives_validated_core_unit_spec(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox

    from xrr_fitter.gui.parameters.dialogs import PriorDialog

    committed: list[api.PriorSpec] = []
    dialog = PriorDialog(_prior_definition(), commit_spec=committed.append)
    qtbot.addWidget(dialog)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("normal")
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(4.0)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.5)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert committed == [api.PriorSpec("normal", (40.0, 5.0))]


@pytest.mark.parametrize(
    ("kind", "display_values", "persisted_values"),
    (
        ("normal", (4.0, 0.5), (40.0, 5.0)),
        ("soft_range", (2.0, 8.0, 0.5), (20.0, 80.0, 5.0)),
        ("lognormal", (log(4.0), 0.2), (log(40.0), 0.2)),
    ),
)
def test_prior_dialog_converts_length_values_from_nm_to_core_units(
    qtbot,
    kind: str,
    display_values: tuple[float, ...],
    persisted_values: tuple[float, ...],
) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox

    dialog = _prior_dialog(qtbot)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText(kind)
    for index, value in enumerate(display_values):
        dialog.findChild(QDoubleSpinBox, f"priorParam{index}").setValue(value)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() is not None
    assert dialog.spec().parameters == pytest.approx(persisted_values)


@pytest.mark.parametrize(
    ("existing", "display_values"),
    (
        (api.PriorSpec("normal", (40.0, 5.0)), (4.0, 0.5)),
        (api.PriorSpec("soft_range", (20.0, 80.0, 5.0)), (2.0, 8.0, 0.5)),
        (api.PriorSpec("lognormal", (log(40.0), 0.2)), (log(4.0), 0.2)),
    ),
)
def test_prior_dialog_prefills_existing_length_prior_in_display_units(
    qtbot,
    existing: api.PriorSpec,
    display_values: tuple[float, ...],
) -> None:
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

    dialog = _prior_dialog(qtbot, existing_prior=existing)

    assert dialog.findChild(QComboBox, "priorKindSelect").currentText() == existing.kind
    actual = tuple(
        dialog.findChild(QDoubleSpinBox, f"priorParam{index}").value() for index in range(len(display_values))
    )
    assert actual == pytest.approx(display_values)


def test_prior_dialog_roughness_fraction_uses_unscaled_fraction_fields(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox, QLabel

    definition = _prior_definition(
        "component.0.roughness_a",
        initial=3.0,
        lower=0.0,
        upper=50.0,
        transform="roughness_fraction",
    )
    existing = api.PriorSpec("normal", (0.5, 0.1))
    dialog = _prior_dialog(qtbot, definition, existing)

    assert dialog.findChild(QDoubleSpinBox, "priorParam0").value() == pytest.approx(0.5)
    assert "分数" in dialog.findChild(QLabel, "priorParamLabel0").text()
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("normal")
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(0.6)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.2)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() == api.PriorSpec("normal", (0.6, 0.2))


def test_prior_dialog_validates_converted_spec_against_definition(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox, QLabel

    dialog = _prior_dialog(qtbot)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("normal")
    # 11 nm becomes 110 Å, beyond the declaration's 100 Å upper bound.
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(11.0)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.2)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() is None
    error = dialog.findChild(QLabel, "priorDialogError")
    assert error.isVisibleTo(dialog)
    assert "within bounds" in error.text()


def test_prior_dialog_rejects_lognormal_for_zero_lower_bound(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox, QLabel

    definition = _prior_definition(
        "instrument.scale",
        unit="1",
        initial=1.0,
        lower=0.0,
        upper=2.0,
        transform="linear",
    )
    dialog = _prior_dialog(qtbot, definition)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("lognormal")
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(0.0)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.2)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() is None
    assert "positive lower bound" in dialog.findChild(QLabel, "priorDialogError").text()


def test_parameter_panel_opens_prior_dialog_with_definition_and_existing_prior(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters import dialogs
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    source = tmp_path / "sample.xy"
    source.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    value = api.add_dataset(api.new_project(), source, api.InstrumentSpec())
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", api.MaterialSpec("SiO2", "SiO2", 2.2), 40.0),),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    value = api.set_structure(value, "sample", structure)
    name = "component.0.thickness_a"
    existing = api.PriorSpec("normal", (40.0, 5.0))
    value = api.set_parameter_priors(value, "sample", (api.ParameterPrior(name, existing),))
    panel = ParametersPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    captured: list[tuple[object, object]] = []

    class StubDialog:
        def __init__(self, definition, parent, *, existing_prior):
            captured.append((definition, existing_prior))

        def exec(self):
            return 0

    monkeypatch.setattr(dialogs, "PriorDialog", StubDialog)

    panel._edit_prior_row(name)

    definition, prior = captured[0]
    assert definition.name == name
    assert prior == existing
