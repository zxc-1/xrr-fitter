from dataclasses import replace

import pytest
from tests.support.model_cases import dataset_project, project

from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    MeasurementPreset,
)
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec


def test_measurement_preset_owns_beam_instrument_and_angle_offset() -> None:
    preset = MeasurementPreset(
        "lab-cu-kalpha",
        BeamSpec(kind="monochromatic", wavelength_a=1.5406),
        InstrumentSpec(instrument_id="lab", footprint_mode="fit"),
        0.012,
    )
    assert preset.preset_id == "lab-cu-kalpha"
    assert preset.import_angle_offset_deg == 0.012


def test_automatic_state_requires_group_identity_and_review_reason() -> None:
    with pytest.raises(ValueError, match="fit_group_id"):
        DatasetAutomation(role=AutomaticRole.JOINT, status=AutomaticStatus.PENDING)
    with pytest.raises(ValueError, match="reason"):
        DatasetAutomation(
            import_batch_id="batch-1",
            fit_group_id="group-1",
            role=AutomaticRole.SINGLE,
            status=AutomaticStatus.REVIEW,
        )


def test_only_passed_automatic_results_can_be_statistics_members() -> None:
    with pytest.raises(ValueError, match="statistics_member"):
        DatasetAutomation(
            import_batch_id="batch-1",
            fit_group_id="group-1",
            role=AutomaticRole.JOINT,
            status=AutomaticStatus.REFINING,
            statistics_member=True,
        )


def test_project_validates_dataset_automation_values() -> None:
    value = project(dataset_project("sample"))
    state = DatasetAutomation(
        import_batch_id="batch-1",
        fit_group_id="group-1",
        role=AutomaticRole.SINGLE,
        status=AutomaticStatus.PENDING,
    )
    updated = replace(value, datasets=(replace(value.datasets[0], automation=state),))
    assert updated.datasets[0].automation is state
