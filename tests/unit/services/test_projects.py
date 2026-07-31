from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    project,
    simple_structure,
)
from xrr_fitter.io.project_codec import (
    load_project as load_project_payload,
    save_project as save_project_payload,
)
from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.analysis import StructureEvidence
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.project import ProjectUiState, ScalePriorState
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services import projects as project_service
from xrr_fitter.services.projects import (
    clear_fit_results,
    inspect_sources,
    load_project,
    new_project,
    save_project,
    select_active_dataset,
    select_candidate,
    set_batch_mode,
    set_expert_mode,
    set_workspace_state,
)


def _source(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0.1, 3.2, 32)
    path.write_bytes(xy_bytes(angles, np.geomspace(1.0, 1e-4, angles.size)))
    return path


def test_new_project_is_empty_versioned_and_has_a_persisted_seed() -> None:
    first = new_project()
    second = new_project()

    assert (
        first.datasets,
        first.batch_mode,
        isinstance(first.master_seed, int),
        0 <= first.master_seed < 2**64,
        first.master_seed != second.master_seed,
    ) == ((), "independent", True, True, True)


def test_new_project_defaults_to_fast_interactive_fit_budget() -> None:
    value = new_project()

    assert value.fit_config == FitConfig.fast(value.master_seed)


def test_save_load_round_trip_preserves_allocated_id_and_custom_display_name(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "sample.xy")
    original = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="round-trip"),
        display_name="measured wafer",
    )
    target = tmp_path / "project.xrrproj.json"

    save_project(original, target)
    loaded = load_project(target)

    assert (
        loaded.datasets[0].dataset_id,
        loaded.datasets[0].display_name,
        loaded.base_directory,
        loaded,
    ) == ("sample", "measured wafer", str(tmp_path), original)


def test_save_as_rebases_relative_sources_without_changing_their_identity(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "sources" / "sample.xy")
    original = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="save-as"),
    )
    original = replace(
        original,
        datasets=(replace(original.datasets[0], source_path="sample.xy"),),
        base_directory=str(source.parent),
    )
    target = tmp_path / "moved" / "workspace.xrrproj.json"
    target.parent.mkdir()

    save_project(original, target)
    loaded = load_project(target)

    assert Path(loaded.datasets[0].source_path).is_absolute() is False
    assert loaded.datasets[0].source_path != original.datasets[0].source_path
    assert inspect_sources(loaded).valid is True
    assert original.datasets[0].source_path == "sample.xy"


def test_normal_save_preserves_relative_source_declaration_byte_for_byte(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "sample.xy")
    original = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="normal-save"),
    )
    original = replace(
        original,
        datasets=(replace(original.datasets[0], source_path="./sample.xy"),),
        base_directory=str(tmp_path),
    )
    target = tmp_path / "workspace.xrrproj.json"

    save_project(original, target)
    save_project(load_project(target), target)

    assert load_project(target).datasets[0].source_path == "./sample.xy"


def test_load_project_clears_source_derived_state_after_hash_mismatch(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "stale.xy")
    original = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="stale-load"),
    )
    result = final_fit_result()
    dataset = replace(
        original.datasets[0],
        structure=simple_structure(),
        structure_evidence=StructureEvidence(1, 1, None, (20.0,)),
        scale_prior=ScalePriorState(enabled=True, s_hat=1.0, tau_s_decades=0.1),
        last_valid_result=result,
    )
    original = replace(
        original,
        datasets=(dataset,),
        ui_state=replace(
            original.ui_state,
            selected_candidate_ids=((dataset.dataset_id, result.candidates[0].candidate_id),),
        ),
    )
    target = tmp_path / "stale.xrrproj.json"
    save_project(original, target)
    source.write_bytes(source.read_bytes() + b"# changed\n")

    loaded = load_project(target)

    changed = loaded.datasets[0]
    assert changed.structure is not None
    assert changed.structure_evidence is None
    assert changed.scale_prior == ScalePriorState(enabled=False)
    assert changed.last_valid_result is None
    assert changed.checkpoint is None
    assert loaded.ui_state.selected_candidate_ids == ()
    assert inspect_sources(loaded).valid is False


def test_save_project_revalidates_and_clears_stale_source_results(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "stale-save.xy")
    original = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="stale-save"),
    )
    result = final_fit_result()
    dataset = replace(
        original.datasets[0],
        structure=simple_structure(),
        structure_evidence=StructureEvidence(1, 1, None, (20.0,)),
        scale_prior=ScalePriorState(enabled=True, s_hat=1.0, tau_s_decades=0.1),
        last_valid_result=result,
    )
    original = replace(original, datasets=(dataset,))
    target = tmp_path / "stale-save.xrrproj.json"
    save_project(original, target)
    source.write_bytes(source.read_bytes() + b"# stale before save\n")

    save_project(original, target)
    persisted = load_project_payload(target)

    assert persisted.datasets[0].structure_evidence is None
    assert persisted.datasets[0].scale_prior == ScalePriorState(enabled=False)
    assert persisted.datasets[0].last_valid_result is None
    assert persisted.datasets[0].checkpoint is None


def test_load_project_rejects_persisted_mask_shape_mismatch(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "mask.xy")
    original = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="mask-load"),
    )
    corrupted = replace(
        original,
        datasets=(replace(original.datasets[0], fit_mask=(True,)),),
    )
    target = tmp_path / "invalid-mask.xrrproj.json"
    save_project_payload(corrupted, target)

    try:
        load_project(target)
    except ValueError as error:
        assert "fit mask must match derived data length" in str(error)
    else:
        raise AssertionError("persisted mask shape mismatch was accepted")


def test_load_project_revalidation_restores_persisted_mask_after_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source(tmp_path / "retry-mask.xy")
    original = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="retry-mask"),
    )
    mask = list(original.datasets[0].fit_mask)
    mask[8] = False
    original = replace(
        original,
        datasets=(replace(original.datasets[0], fit_mask=tuple(mask)),),
    )
    target = tmp_path / "retry-mask.xrrproj.json"
    save_project(original, target)
    read_xy = project_service.read_xy
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient source read failure")
        return read_xy(*args, **kwargs)

    monkeypatch.setattr(project_service, "read_xy", fail_once)

    loaded = load_project(target)

    assert attempts == 2
    assert loaded.datasets[0].fit_mask[8] is False


def test_load_project_revalidation_has_a_bounded_retry_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source(tmp_path / "unstable.xy")
    original = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="unstable"),
    )
    target = tmp_path / "unstable.xrrproj.json"
    save_project(original, target)
    attempts = 0

    def fail_read(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError("source keeps racing")

    monkeypatch.setattr(project_service, "read_xy", fail_read)

    with pytest.raises(
        RuntimeError,
        match="source changed repeatedly during project restore",
    ):
        load_project(target)

    assert attempts == 2


def test_load_project_revalidation_reports_source_missing_during_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source(tmp_path / "missing-during-retry.xy")
    original = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="missing-during-retry"),
    )
    target = tmp_path / "missing-during-retry.xrrproj.json"
    save_project(original, target)
    attempts = 0

    def fail_then_remove(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            source.unlink()
        raise OSError("source read raced validation")

    monkeypatch.setattr(project_service, "read_xy", fail_then_remove)

    loaded = load_project(target)

    assert attempts == 2
    assert inspect_sources(loaded).datasets[0].status.value == "missing"


def test_workspace_mutations_and_independent_result_clearing_preserve_unrelated_state() -> None:
    result = final_fit_result()
    first = dataset_project("first", result=result)
    second = dataset_project("second", result=result)
    value = project(first, second)

    value = select_active_dataset(value, "second")
    value = select_candidate(value, "first", result.candidates[0].candidate_id)
    value = set_expert_mode(value, True)

    assert (
        value.ui_state.active_dataset_id,
        value.ui_state.selected_candidate_ids,
        value.ui_state.expert_mode,
        value.batch_mode,
    ) == ("second", (("first", "candidate-0"),), True, "independent")

    custom = replace(value.ui_state, workspace_splitter_sizes=(1, 2, 3))
    value = set_workspace_state(value, custom)
    value = clear_fit_results(value, ("first",))

    assert (
        value.datasets[0].last_valid_result,
        value.datasets[1].last_valid_result is result,
        value.ui_state.selected_candidate_ids,
        value.ui_state.active_dataset_id,
        value.ui_state.workspace_splitter_sizes,
    ) == (None, True, (), "second", (1, 2, 3))


def test_batch_mode_change_invalidates_the_complete_result_graph() -> None:
    result = final_fit_result()
    value = project(
        dataset_project("first", result=result),
        dataset_project("second", result=result),
    )
    value = select_candidate(value, "first", result.candidates[0].candidate_id)

    updated = set_batch_mode(value, "joint")

    assert updated.batch_mode == "joint"
    assert all(dataset.last_valid_result is None for dataset in updated.datasets)
    assert all(dataset.checkpoint is None for dataset in updated.datasets)
    assert updated.ui_state.selected_candidate_ids == ()


def test_workspace_state_rejects_a_foreign_dataset_reference() -> None:
    value = project(dataset_project("known"))
    state = ProjectUiState(active_dataset_id="foreign")

    try:
        set_workspace_state(value, state)
    except ValueError as error:
        assert "active_dataset_id" in str(error)
    else:
        raise AssertionError("foreign workspace state was accepted")
