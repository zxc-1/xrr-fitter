"""A2: the returned manifest must describe the run, not just list its files.

The export dialog's manifest preview promises four run-level facts before the
user commits: which batch mode ran, which stages the search actually completed,
what confidence the published candidates earned, and whether the run can be
replayed bit-identically. Every one of those is already owned by the model
layer, so the manifest must carry them through rather than force each caller to
re-derive them from the project.

These fields describe the *returned value* only. ``export_manifest.json`` on
disk is written by ``io.export_run._manifest_bytes`` and is deliberately left
untouched, because the published tree must stay byte-for-byte identical to a
run without this feature.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support.model_cases import fit_candidate, simple_structure

from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.fitting import FitSearchResult, FitStageSummary
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.services import exports
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.projects import new_project, set_batch_mode
from xrr_fitter.services.structures import set_structure

STAGES = ("A", "B", "E")


def _staged_result(
    candidate,
    *,
    confidence: ConfidenceClass,
    stages: tuple[str, ...] = STAGES,
) -> FitResult:
    """A search whose stage history is non-empty, unlike the shared double."""
    summaries = tuple(
        FitStageSummary(
            stage=stage,
            candidate_ids=(candidate.candidate_id,),
            best_objective=1.0,
            total_nfev=12,
            stop_reasons=("converged",),
        )
        for stage in stages
    )
    search = FitSearchResult(
        parameter_definitions=(),
        candidates=(candidate,),
        best_index=0,
        warnings=(),
        child_seeds=(101,),
        stage_summaries=summaries,
        region_labels=np.zeros(4, dtype=int),
        region_weights=np.ones(4),
    )
    return FitResult.from_search(
        search,
        confidence=confidence,
        uncertainty=None,
        classification_evidence=(),
    )


def _source(tmp_path: Path, name: str) -> Path:
    source = tmp_path / f"{name}.xy"
    angles = np.linspace(0.1, 3.2, 40)
    source.write_bytes(xy_bytes(angles, np.geomspace(1.0, 1e-5, angles.size)))
    return source


def _candidate_for(project, dataset, *, ranking_objective: float | None = None):
    data = exports.load_export_data(project, dataset)
    return replace(
        fit_candidate(),
        qz_a_inv=data.qz_a_inv,
        model_normalized=data.intensity_normalized,
        log_residuals_decades=np.zeros(data.qz_a_inv.size),
        weighted_residuals=np.zeros(data.qz_a_inv.size),
        ranking_objective=ranking_objective,
    )


def _with_results(project, results: tuple[FitResult, ...]):
    return replace(
        project,
        datasets=tuple(
            replace(dataset, last_valid_result=result)
            for dataset, result in zip(project.datasets, results, strict=True)
        ),
    )


def _fitted_project(tmp_path: Path, *, confidence: ConfidenceClass = ConfidenceClass.TRUSTED):
    value = add_dataset(
        new_project(),
        _source(tmp_path, "curve"),
        InstrumentSpec(instrument_id="export-service", footprint_mode="none"),
    )
    value = set_structure(value, "curve", simple_structure())
    result = _staged_result(_candidate_for(value, value.datasets[0]), confidence=confidence)
    return _with_results(value, (result,))


def _two_dataset_project(tmp_path: Path):
    instrument = InstrumentSpec(instrument_id="export-service", footprint_mode="none")
    value = add_dataset(new_project(), _source(tmp_path, "curve_a"), instrument)
    value = add_dataset(value, _source(tmp_path, "curve_b"), instrument)
    for dataset in value.datasets:
        value = set_structure(value, dataset.dataset_id, simple_structure())
    return value


def _stub_serializers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "dataset_json_bytes",
        "dataset_workbook_bytes",
        "fit_overview_png",
        "sld_profile_png",
        "residuals_png",
        "run_log_bytes",
        "compatibility_workbook_bytes",
        "batch_workbook_bytes",
        "parameter_trends_png",
    ):
        monkeypatch.setattr(exports, name, lambda *_args, _name=name: _name.encode())


def test_manifest_reports_batch_mode_stages_confidence_and_reproducibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path, confidence=ConfidenceClass.CORRELATED)
    _stub_serializers(monkeypatch)

    manifest = exports.export_result(value, tmp_path / "exports")

    assert manifest.mode == "independent"
    # Stage order is the search's own history, not a hardcoded ladder.
    assert manifest.stages == STAGES
    assert manifest.confidence == ConfidenceClass.CORRELATED
    # A published run pins its seed and re-verified every source, so the claim
    # must be actively established by the export, not left at a default.
    assert manifest.reproducible is True


def test_manifest_mode_follows_the_project_batch_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A joint run must not be published as if each dataset ran alone."""
    value = set_batch_mode(_two_dataset_project(tmp_path), "joint")
    # Joint projections share candidate lineage and rank; only arrays are local.
    results = tuple(
        _staged_result(
            _candidate_for(value, dataset, ranking_objective=1.0),
            confidence=ConfidenceClass.TRUSTED,
        )
        for dataset in value.datasets
    )
    value = _with_results(value, results)
    _stub_serializers(monkeypatch)

    manifest = exports.export_result(value, tmp_path / "exports")

    assert manifest.mode == "joint"


def test_manifest_stages_deduplicate_repeated_stage_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage E can run twice; the descriptor names each stage once, in order."""
    value = _fitted_project(tmp_path)
    result = value.datasets[0].last_valid_result
    repeated = (*result.stage_summaries, result.stage_summaries[-1])
    value = _with_results(value, (replace(result, stage_summaries=repeated),))
    _stub_serializers(monkeypatch)

    manifest = exports.export_result(value, tmp_path / "exports")

    assert manifest.stages == STAGES


def test_manifest_confidence_reports_the_weakest_published_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One untrustworthy dataset must not hide behind a trusted sibling."""
    value = _two_dataset_project(tmp_path)
    results = (
        _staged_result(
            _candidate_for(value, value.datasets[0]),
            confidence=ConfidenceClass.TRUSTED,
        ),
        _staged_result(
            _candidate_for(value, value.datasets[1]),
            confidence=ConfidenceClass.MULTIPLE,
        ),
    )
    value = _with_results(value, results)
    _stub_serializers(monkeypatch)

    manifest = exports.export_result(value, tmp_path / "exports")

    assert manifest.confidence == ConfidenceClass.MULTIPLE


def test_manifest_stages_union_covers_every_published_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dataset that needed an extra stage must still see it recorded."""
    value = _two_dataset_project(tmp_path)
    results = (
        _staged_result(
            _candidate_for(value, value.datasets[0]),
            confidence=ConfidenceClass.TRUSTED,
            stages=("A", "B"),
        ),
        _staged_result(
            _candidate_for(value, value.datasets[1]),
            confidence=ConfidenceClass.TRUSTED,
            stages=("A", "B", "bootstrap", "E"),
        ),
    )
    value = _with_results(value, results)
    _stub_serializers(monkeypatch)

    manifest = exports.export_result(value, tmp_path / "exports")

    assert manifest.stages == ("A", "B", "bootstrap", "E")


ARCHIVED_ID = "candidate-archived"


def _result_with_archive(candidate) -> FitResult:
    """A published graph that still holds a Stage-B archive.

    ``fit.candidates._archived_candidate`` deliberately rewrites the archived
    seed index to ``-1`` (see the comment on ``model.fitting._archived_seed``),
    so that candidate can no longer be traced back to a ``child_seeds`` entry —
    ``fit.stages`` hands out ``seed_index`` by ``enumerate(child_seeds)``. The
    archive stays addressable and ``_validate_ui`` accepts it as a selection, so
    an export can legitimately be asked to publish it.
    """
    archived = replace(
        candidate,
        candidate_id=ARCHIVED_ID,
        seed_index=-1,
        stop_reason="early_eliminated",
    )
    summaries = tuple(
        FitStageSummary(
            stage=stage,
            candidate_ids=((candidate.candidate_id, ARCHIVED_ID) if stage == "B" else (candidate.candidate_id,)),
            best_objective=1.0,
            total_nfev=12,
            stop_reasons=(("converged", "early_eliminated") if stage == "B" else ("converged",)),
        )
        for stage in STAGES
    )
    search = FitSearchResult(
        parameter_definitions=(),
        candidates=(candidate, archived),
        best_index=0,
        warnings=(),
        child_seeds=(101,),
        stage_summaries=summaries,
        region_labels=np.zeros(4, dtype=int),
        region_weights=np.ones(4),
    )
    return FitResult.from_search(
        search,
        confidence=ConfidenceClass.TRUSTED,
        uncertainty=None,
        classification_evidence=(),
    )


def test_manifest_reproducibility_is_false_when_the_published_seed_is_unrecoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publishing a seed-erased archive cannot claim a replayable run."""
    value = _fitted_project(tmp_path)
    dataset = value.datasets[0]
    value = _with_results(value, (_result_with_archive(_candidate_for(value, dataset)),))
    value = replace(
        value,
        ui_state=replace(
            value.ui_state,
            selected_candidate_ids=((dataset.dataset_id, ARCHIVED_ID),),
        ),
    )
    _stub_serializers(monkeypatch)

    manifest = exports.export_result(value, tmp_path / "exports")

    assert manifest.reproducible is False


def test_manifest_reproducibility_survives_an_unpublished_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An archive that lost its seed is harmless while it stays unpublished."""
    value = _fitted_project(tmp_path)
    dataset = value.datasets[0]
    value = _with_results(value, (_result_with_archive(_candidate_for(value, dataset)),))
    _stub_serializers(monkeypatch)

    manifest = exports.export_result(value, tmp_path / "exports")

    assert manifest.reproducible is True


def test_describe_export_plan_reports_the_same_conclusions_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The export dialog must state the run's verdict before it publishes anything.

    ``export_result`` only learns ``mode``/``stages``/``confidence``/``reproducible``
    on the way out, but the preview has to show them on the way in. Recomputing
    them inside the GUI would fork the derivation, so the service layer answers
    the same question twice from one implementation -- and the planning answer
    must be reachable without writing a single byte.
    """
    value = _fitted_project(tmp_path, confidence=ConfidenceClass.CORRELATED)
    destination = tmp_path / "exports"

    def refuse(*_args, **_kwargs):
        raise AssertionError("describe_export_plan must not publish")

    monkeypatch.setattr(exports, "publish_export_run", refuse)
    _stub_serializers(monkeypatch)

    plan = exports.describe_export_plan(value)

    assert plan.mode == "independent"
    assert plan.stages == STAGES
    assert plan.confidence == ConfidenceClass.CORRELATED
    assert plan.reproducible is True
    # A plan describes a run that has not happened. ``ExportManifest`` requires a
    # ``run_directory`` and carries file records, so reusing it here would force a
    # placeholder path and let a caller mistake the plan for a publication. The
    # plan type deliberately owns neither.
    assert not hasattr(plan, "run_directory")
    assert not hasattr(plan, "files")
    assert not destination.exists()


def test_describe_export_plan_counts_datasets_and_follows_joint_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preview's dataset count comes from the plan, not from the file tree."""
    value = set_batch_mode(_two_dataset_project(tmp_path), "joint")
    results = tuple(
        _staged_result(
            _candidate_for(value, dataset, ranking_objective=1.0),
            confidence=ConfidenceClass.TRUSTED,
        )
        for dataset in value.datasets
    )
    value = _with_results(value, results)
    _stub_serializers(monkeypatch)

    plan = exports.describe_export_plan(value)

    assert plan.mode == "joint"
    assert plan.dataset_ids == ("curve_a", "curve_b")


def test_describe_export_plan_reports_an_unreplayable_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seed-erased archive must fail the claim in the preview, not after export."""
    value = _fitted_project(tmp_path)
    dataset = value.datasets[0]
    value = _with_results(value, (_result_with_archive(_candidate_for(value, dataset)),))
    value = replace(
        value,
        ui_state=replace(
            value.ui_state,
            selected_candidate_ids=((dataset.dataset_id, ARCHIVED_ID),),
        ),
    )
    _stub_serializers(monkeypatch)

    assert exports.describe_export_plan(value).reproducible is False
