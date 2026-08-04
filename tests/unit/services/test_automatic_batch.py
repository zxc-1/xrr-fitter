from dataclasses import replace
from threading import Event
from types import SimpleNamespace

from tests.support.model_cases import dataset_project, final_fit_result, project

from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    MeasurementPreset,
)
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.fitting import FitCheckpoint
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterDefinition, ParameterSetting
from xrr_fitter.services import batch
from xrr_fitter.services.fitting import (
    AutomaticPreparedResult,
    validate_parameter_setting_declarations,
)
from xrr_fitter.services.materials import automatic_structure


def _preset() -> MeasurementPreset:
    return MeasurementPreset(
        "lab",
        BeamSpec("monochromatic", wavelength_a=1.5406),
        InstrumentSpec(instrument_id="lab"),
    )


def _automatic_dataset(dataset_id: str, layers: tuple[str, ...]):
    structure, settings = automatic_structure(layers, "Si")
    return replace(
        dataset_project(dataset_id),
        beam=_preset().beam,
        instrument=_preset().instrument,
        structure=structure,
        parameter_settings=settings,
        automation=DatasetAutomation(
            import_batch_id="batch-1",
            role=AutomaticRole.UNROUTED,
            status=AutomaticStatus.PENDING,
        ),
    )


class RecordingAutomaticFits:
    def __init__(self) -> None:
        self.prefit_dataset_ids: set[str] = set()
        self.checkpoints = []

    def seeds(self, value):
        return (
            {dataset.dataset_id: index + 1 for index, dataset in enumerate(value.datasets)},
            101,
            102,
        )

    def prepare(self, value, dataset_id, _seed):
        index = next(
            index
            for index, dataset in enumerate(value.datasets)
            if dataset.dataset_id == dataset_id
        )
        self.prefit_dataset_ids.add(dataset_id)
        return SimpleNamespace(
            dataset_id=dataset_id,
            dataset_index=index,
            updated_dataset=value.datasets[index],
        )

    def fit_dataset(self, prepared, **_kwargs):
        return AutomaticPreparedResult(prepared, final_fit_result(), True, None)

    def checkpoint(self, value) -> None:
        self.checkpoints.append(value)


def test_mixed_import_batch_routes_singletons_and_matching_points_separately(
    monkeypatch,
) -> None:
    other_batch = replace(
        _automatic_dataset("d", ("Zr",)),
        automation=replace(
            _automatic_dataset("d", ("Zr",)).automation,
            import_batch_id="batch-2",
        ),
    )
    value = replace(
        project(
            _automatic_dataset("a", ("Zr",)),
            _automatic_dataset("b", ("Zr",)),
            _automatic_dataset("c", ("TaN",)),
            other_batch,
        ),
        measurement_preset=_preset(),
    )
    records = tuple(
        SimpleNamespace(dataset_id=item.dataset_id, status=SimpleNamespace(value="ok"))
        for item in value.datasets
    )
    monkeypatch.setattr(
        batch,
        "inspect_sources",
        lambda _value: SimpleNamespace(valid=True, issues=(), datasets=records),
    )
    calls = RecordingAutomaticFits()

    result = batch.fit_automatic_transaction(
        value,
        None,
        None,
        calls.checkpoint,
        None,
        seed_branches=calls.seeds,
        prepare_dataset=calls.prepare,
        fit_dataset=calls.fit_dataset,
    )

    assert calls.prefit_dataset_ids == {"a", "b", "c", "d"}
    assert tuple(item.dataset_id for item in result.datasets) == ("a", "b", "c", "d")
    by_id = {item.dataset_id: item.automation for item in result.updated_project.datasets}
    assert by_id["a"].role is AutomaticRole.JOINT
    assert by_id["b"].fit_group_id == by_id["a"].fit_group_id
    assert by_id["a"].status is AutomaticStatus.REFINING
    assert by_id["b"].status is AutomaticStatus.REFINING
    assert by_id["c"].role is AutomaticRole.SINGLE
    assert by_id["c"].status is AutomaticStatus.PASSED
    assert by_id["d"].role is AutomaticRole.SINGLE
    assert by_id["d"].fit_group_id != by_id["a"].fit_group_id
    assert result.mode == "automatic"


def test_physical_signature_separates_backing_and_beam() -> None:
    preset = _preset()
    first = _automatic_dataset("a", ("Zr",))
    different_backing = replace(
        first,
        structure=replace(
            first.structure,
            backing=first.structure.components[0].material,
        ),
    )
    different_beam = replace(
        first,
        beam=BeamSpec("monochromatic", wavelength_a=0.7093),
    )
    signatures = {
        batch.automatic_physical_signature(value, preset)
        for value in (first, different_backing, different_beam)
    }
    assert len(signatures) == 3


def test_prefits_share_one_worker_budget_and_publish_in_completion_order(
    monkeypatch,
) -> None:
    value = replace(
        project(
            _automatic_dataset("slow", ("Zr",)),
            _automatic_dataset("fast", ("TaN",)),
        ),
        measurement_preset=_preset(),
        fit_config=replace(project().fit_config, local_workers=4),
    )
    records = tuple(
        SimpleNamespace(dataset_id=item.dataset_id, status=SimpleNamespace(value="ok"))
        for item in value.datasets
    )
    monkeypatch.setattr(
        batch,
        "inspect_sources",
        lambda _value: SimpleNamespace(valid=True, issues=(), datasets=records),
    )
    calls = RecordingAutomaticFits()
    release_slow = Event()
    allocations: list[tuple[str, int]] = []
    published: list[tuple[AutomaticStatus, AutomaticStatus]] = []

    def fit_dataset(prepared, *, local_workers, **_kwargs):
        allocations.append((prepared.dataset_id, local_workers))
        if prepared.dataset_id == "slow":
            assert release_slow.wait(timeout=2.0)
        return AutomaticPreparedResult(prepared, final_fit_result(), True, None)

    def checkpoint(updated) -> None:
        statuses = tuple(dataset.automation.status for dataset in updated.datasets)
        published.append(statuses)
        if statuses[1] is AutomaticStatus.PASSED:
            release_slow.set()

    result = batch.fit_automatic_transaction(
        value,
        None,
        None,
        checkpoint,
        None,
        seed_branches=calls.seeds,
        prepare_dataset=calls.prepare,
        fit_dataset=fit_dataset,
    )

    assert sorted(allocations) == [("fast", 2), ("slow", 2)]
    assert published[0] == (AutomaticStatus.REFINING, AutomaticStatus.PASSED)
    assert all(
        dataset.automation.status is AutomaticStatus.PASSED
        for dataset in result.updated_project.datasets
    )


def test_source_and_preparation_failures_are_isolated_and_all_results_publish(
    monkeypatch,
) -> None:
    value = replace(
        project(
            _automatic_dataset("source", ("Zr",)),
            _automatic_dataset("prepare", ("TaN",)),
            _automatic_dataset("fit", ("Al2O3",)),
        ),
        measurement_preset=_preset(),
    )
    records = (
        SimpleNamespace(
            dataset_id="source",
            status=SimpleNamespace(value="missing"),
            message="source disappeared",
        ),
        *tuple(
            SimpleNamespace(
                dataset_id=dataset_id,
                status=SimpleNamespace(value="ok"),
                message="",
            )
            for dataset_id in ("prepare", "fit")
        ),
    )
    monkeypatch.setattr(
        batch,
        "inspect_sources",
        lambda _value: SimpleNamespace(valid=False, issues=(), datasets=records),
    )
    calls = RecordingAutomaticFits()
    fitted: list[str] = []

    def prepare(updated, dataset_id, seed):
        if dataset_id == "prepare":
            raise ValueError("compile failed")
        return calls.prepare(updated, dataset_id, seed)

    def fit_dataset(prepared, **_kwargs):
        fitted.append(prepared.dataset_id)
        return AutomaticPreparedResult(prepared, final_fit_result(), True, None)

    result = batch.fit_automatic_transaction(
        value,
        None,
        None,
        calls.checkpoint,
        None,
        seed_branches=calls.seeds,
        prepare_dataset=prepare,
        fit_dataset=fit_dataset,
    )

    assert tuple(item.dataset_id for item in result.datasets) == (
        "source",
        "prepare",
        "fit",
    )
    assert fitted == ["fit"]
    by_id = {dataset.dataset_id: dataset for dataset in result.updated_project.datasets}
    assert by_id["source"].automation.status is AutomaticStatus.FAILED
    assert "source disappeared" in by_id["source"].automation.reason
    assert by_id["prepare"].automation.status is AutomaticStatus.FAILED
    assert "compile failed" in by_id["prepare"].automation.reason
    assert by_id["fit"].automation.status is AutomaticStatus.PASSED


def test_completed_prefit_persists_winner_settings_and_checkpoint(monkeypatch) -> None:
    definition = ParameterDefinition(
        "scale",
        "Scale",
        "",
        "scale",
        0.75,
        0.5,
        1.5,
        "linear",
        False,
    )
    fitted = replace(final_fit_result(), parameter_definitions=(definition,))
    saved_checkpoint = FitCheckpoint(
        data_sha256="a" * 64,
        structure_fingerprint="b" * 64,
        config_fingerprint="c" * 64,
        stage="E",
        candidates=fitted.candidates,
        child_seeds=(101,),
    )
    dataset = replace(
        _automatic_dataset("fit", ("Zr",)),
        parameter_settings=(ParameterSetting("scale", 0.75, 0.5, 1.5),),
    )
    value = replace(project(dataset), measurement_preset=_preset())
    records = (
        SimpleNamespace(
            dataset_id="fit",
            status=SimpleNamespace(value="ok"),
            message="",
        ),
    )
    monkeypatch.setattr(
        batch,
        "inspect_sources",
        lambda _value: SimpleNamespace(valid=True, issues=(), datasets=records),
    )
    calls = RecordingAutomaticFits()

    def fit_dataset(prepared, *, checkpoint, **_kwargs):
        checkpoint(saved_checkpoint)
        return AutomaticPreparedResult(prepared, fitted, True, None)

    result = batch.fit_automatic_transaction(
        value,
        None,
        None,
        calls.checkpoint,
        None,
        seed_branches=calls.seeds,
        prepare_dataset=calls.prepare,
        fit_dataset=fit_dataset,
    )

    updated = result.updated_project.datasets[0]
    assert updated.parameter_settings == (
        ParameterSetting("scale", 1.0, 0.5, 1.5),
    )
    validate_parameter_setting_declarations(
        fitted.parameter_definitions,
        updated.parameter_settings,
    )
    assert updated.checkpoint is not None
    assert updated.checkpoint.stage == saved_checkpoint.stage
    assert tuple(candidate.candidate_id for candidate in updated.checkpoint.candidates) == (
        "candidate-0",
    )
    assert updated.last_valid_result is fitted
