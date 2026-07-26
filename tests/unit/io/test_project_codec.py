from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.support.model_cases import dataset_project, project
from xrr_fitter.io.project_codec import (
    ProjectSchemaError,
    ProjectVersionError,
    fit_result_from_dict,
    fit_result_to_dict,
    load_project,
    project_from_bytes,
    project_to_dict,
    save_project,
)
from xrr_fitter.model.analysis import (
    ConfidenceClass,
    FitResult,
    McmcConfig,
    McmcReport,
    ParameterProfile,
    UncertaintyReport,
)
from xrr_fitter.model.fitting import FitCandidate, FitCheckpoint, FitStageSummary
from xrr_fitter.model.instrument import PhysicsDiagnostic
from xrr_fitter.model.parameters import ParameterDefinition, ParameterValue
from xrr_fitter.model.project import ProjectUiState
from xrr_fitter.model.structure import PeriodicSpan, SlabStack


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_INPUTS = ROOT / "verification/r22/reference/xrr_fitter/examples"


def _simple_project():
    return project(dataset_project("sample-1"))


def _manual_result_graph() -> tuple[FitResult, FitCheckpoint]:
    definition = ParameterDefinition(
        name="layer.0.thickness_a",
        display_name="thickness",
        unit="A",
        category="structure",
        initial=10.0,
        lower=2.0,
        upper=50.0,
        transform="linear",
        locked=False,
        sharing_key="thickness:film",
    )
    parameter = ParameterValue("layer.0.thickness_a", 10.25, 2.0, 50.0)
    stack = SlabStack(
        thickness_a=np.array([0.0, 10.0, 10.0, 0.0]),
        sld_a2=np.array(
            [0.0j, 2.0e-6 + 0.1e-6j, 2.0e-6 + 0.1e-6j, 3.0e-6j]
        ),
        roughness_a=np.array([1.0, 1.0, 1.0]),
        periodic_spans=(PeriodicSpan(1, 1, 2),),
    )
    diagnostic = PhysicsDiagnostic("finite", "model finite", (1,))
    candidate = FitCandidate(
        candidate_id="candidate-1",
        seed_index=3,
        unit_vector=np.array([0.25]),
        parameters=(parameter,),
        objective=0.125,
        valid=True,
        stop_reason="converged",
        nfev=17,
        qz_a_inv=np.array([0.01, 0.02, 0.03]),
        model_normalized=np.array([1.0, 0.5, 0.25]),
        log_residuals_decades=np.array([0.0, 0.01, -0.02]),
        weighted_residuals=np.array([0.0, 0.5, -1.0]),
        expanded_stack=stack,
        sld_depth_a=np.array([0.0, 5.0, 10.0]),
        sld_profile_a2=np.array(
            [0.0j, 1.0e-6 + 0.2e-6j, 2.0e-6 + 0.1e-6j]
        ),
        diagnostics=(diagnostic,),
    )
    stage = FitStageSummary(
        "stage-e",
        ("candidate-1",),
        0.125,
        17,
        ("converged",),
    )
    profile = ParameterProfile(
        "layer.0.thickness_a",
        np.array([9.0, 10.25, 11.0]),
        np.array([0.2, 0.125, 0.22]),
        False,
        True,
    )
    mcmc = McmcReport(
        config=McmcConfig(walkers=2, burn_in=1, production_steps=3),
        child_seed=99,
        parameter_names=("layer.0.thickness_a",),
        samples_physical=np.array([[10.0], [10.1], [10.2]]),
        log_probability=np.array([-1.0, -0.9, -1.1]),
        acceptance_fraction=np.array([0.4, 0.5]),
        split_rhat=np.array([1.01]),
        effective_sample_size=np.array([23.0]),
        boundary_hits=("upper",),
        warnings=("short chain",),
        candidate_id="candidate-1",
    )
    uncertainty = UncertaintyReport(
        correlation_names=("layer.0.thickness_a",),
        correlation_matrix=np.array([[1.0]]),
        profiles=(profile,),
        bootstrap_intervals=(("layer.0.thickness_a", 9.5, 10.8),),
        bootstrap_failure_rate=0.1,
        boundary_hits=("layer.0.thickness_a",),
        strong_correlations=(("a", "b", 0.97),),
        systematic_residual=True,
        diagnostics=(diagnostic,),
        residual_autocorrelation=True,
        mcmc=mcmc,
        candidate_id="candidate-1",
    )
    result = FitResult(
        parameter_definitions=(definition,),
        candidates=(candidate,),
        best_index=0,
        confidence=ConfidenceClass.CORRELATED,
        warnings=("correlated",),
        child_seeds=(99,),
        stage_summaries=(stage,),
        region_labels=np.array([0, 1, 1]),
        region_weights=np.array([1.0, 2.0, 2.0]),
        uncertainty=uncertainty,
    )
    checkpoint = FitCheckpoint(
        data_sha256="a" * 64,
        structure_fingerprint="b" * 64,
        config_fingerprint="c" * 64,
        stage="stage-e",
        candidates=(candidate,),
        child_seeds=(99,),
        instrument_fingerprint="d" * 64,
        parameter_settings_fingerprint="e" * 64,
        runtime_warnings=("runtime warning",),
        stage_summaries=(stage,),
    )
    return result, checkpoint


def _project_with_legal_json_sentinels():
    result, checkpoint = _manual_result_graph()
    source = result.candidates[0]
    valid = replace(
        source,
        model_normalized=np.array([1.0, np.nan, 0.25]),
        log_residuals_decades=np.array([0.0, np.nan, -0.02]),
        weighted_residuals=np.array([0.0, np.nan, -1.0]),
        sld_profile_a2=np.array(
            [0.0j, complex(np.nan, np.nan), 2.0e-6 + 0.1e-6j]
        ),
    )
    invalid = replace(
        valid,
        candidate_id="candidate-invalid",
        seed_index=4,
        objective=float("inf"),
        valid=False,
        stop_reason="invalid_model",
    )
    invalid_stage = FitStageSummary(
        "screen",
        ("candidate-invalid",),
        float("inf"),
        4,
        ("invalid_model",),
    )
    result = replace(
        result,
        candidates=(valid, invalid),
        stage_summaries=(result.stage_summaries[0], invalid_stage),
    )
    checkpoint = replace(
        checkpoint,
        candidates=(valid, invalid),
        stage_summaries=(result.stage_summaries[0], invalid_stage),
    )
    dataset = replace(
        dataset_project("sample-1"),
        last_valid_result=result,
        checkpoint=checkpoint,
    )
    return replace(
        project(dataset),
        ui_state=ProjectUiState(
            selected_candidate_ids=(("sample-1", "candidate-1"),)
        ),
    )


def _saved_project_path(tmp_path: Path) -> Path:
    path = tmp_path / "saved.xrrproj.json"
    save_project(_simple_project(), path)
    return path


def _rewrite(path: Path, mutation) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def test_r22_example_projects_round_trip_through_the_only_codec() -> None:
    for stem in ("single-layer", "mo-si-periodic"):
        content = (REFERENCE_INPUTS / f"{stem}.xrrproj.json").read_bytes()
        expected = json.loads(content)

        loaded = project_from_bytes(content)

        assert project_to_dict(loaded) == expected


def test_save_load_round_trip_sets_runtime_base_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested/project.xrrproj.json"
    path.parent.mkdir()
    original = _simple_project()

    save_project(original, path)
    loaded = load_project(path)

    assert loaded == original
    assert loaded.base_directory == str(path.resolve().parent)
    assert "base_directory" not in json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("field", ("beam", "instrument"))
def test_project_missing_required_dataset_field_fails_loudly(
    tmp_path: Path,
    field: str,
) -> None:
    path = _saved_project_path(tmp_path)
    _rewrite(path, lambda payload: payload["datasets"][0].pop(field))

    with pytest.raises(ProjectSchemaError, match=rf"sample-1.*{field}"):
        load_project(path)


def test_project_missing_beam_kind_fails_loudly(tmp_path: Path) -> None:
    path = _saved_project_path(tmp_path)
    _rewrite(path, lambda payload: payload["datasets"][0]["beam"].pop("kind"))

    with pytest.raises(ProjectSchemaError, match=r"sample-1.*beam.*kind"):
        load_project(path)


@pytest.mark.parametrize("token", ("NaN", "Infinity", "-Infinity"))
def test_project_rejects_nonstandard_json_constants_on_load(
    tmp_path: Path,
    token: str,
) -> None:
    path = _saved_project_path(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        '"master_seed": 1201',
        f'"master_seed": {token}',
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ProjectSchemaError, match="nonstandard JSON"):
        load_project(path)


def test_project_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    path = _saved_project_path(tmp_path)
    _rewrite(path, lambda payload: payload.update(schema_version=999))

    with pytest.raises(ProjectVersionError, match="unsupported project schema: 999"):
        load_project(path)


@pytest.mark.parametrize("schema_version", (True, 1.0))
def test_project_rejects_noninteger_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    path = _saved_project_path(tmp_path)
    _rewrite(path, lambda payload: payload.update(schema_version=schema_version))

    with pytest.raises(ProjectVersionError, match="unsupported project schema"):
        load_project(path)


def test_project_rejects_unknown_structure_discriminator(tmp_path: Path) -> None:
    path = _saved_project_path(tmp_path)
    _rewrite(
        path,
        lambda payload: payload["datasets"][0]["structure"]["components"][0].update(
            kind="mystery"
        ),
    )

    with pytest.raises(ProjectSchemaError, match="structure.*mystery"):
        load_project(path)


@pytest.mark.parametrize(
    ("candidate_ids", "best_index", "match"),
    (
        (["same", "same"], 0, "candidate_id"),
        (["only"], 3, "best_index"),
        ([""], 0, "candidate_id"),
    ),
)
def test_project_rejects_invalid_result_identity_graph(
    tmp_path: Path,
    candidate_ids: list[str],
    best_index: int,
    match: str,
) -> None:
    path = _saved_project_path(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["datasets"][0]["last_valid_result"] = {
            "candidates": [{"candidate_id": value} for value in candidate_ids],
            "best_index": best_index,
        }

    _rewrite(path, mutate)
    with pytest.raises(ProjectSchemaError, match=match):
        load_project(path)


@pytest.mark.parametrize(
    "raw_evidence",
    ("profile_path_merge_failed", ["profile_path_merge_failed", 7], {"reason": "x"}),
)
def test_fit_result_codec_rejects_non_string_classification_evidence(
    raw_evidence: object,
) -> None:
    source, _checkpoint = _manual_result_graph()
    payload = fit_result_to_dict(source)
    assert payload is not None
    payload["classification_evidence"] = raw_evidence

    with pytest.raises(ProjectSchemaError, match="classification_evidence"):
        fit_result_from_dict(payload)


def _assert_encoded_sentinels(text: str, payload: dict[str, object]) -> None:
    assert "NaN" not in text
    assert "Infinity" not in text
    assert "-Infinity" not in text
    encoded = payload["datasets"][0]["last_valid_result"]
    assert encoded["candidates"][0]["model_normalized"][1] is None
    assert encoded["candidates"][0]["sld_profile_a2"][1] == {
        "real": None,
        "imag": None,
    }
    assert encoded["candidates"][1]["objective"] is None
    assert encoded["stage_summaries"][1]["best_objective"] is None


def _assert_restored_sentinels(loaded) -> None:
    restored = loaded.datasets[0].last_valid_result
    assert restored is not None
    assert np.isnan(restored.candidates[0].model_normalized[1])
    assert np.isnan(restored.candidates[0].sld_profile_a2[1].real)
    assert np.isinf(restored.candidates[1].objective)
    assert np.isinf(restored.stage_summaries[1].best_objective)


def test_project_writes_standard_json_and_restores_array_and_objective_sentinels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legal-json.xrrproj.json"
    save_project(_project_with_legal_json_sentinels(), path)

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    loaded = load_project(path)

    _assert_encoded_sentinels(text, payload)
    _assert_restored_sentinels(loaded)


@pytest.mark.parametrize("field", ("candidate", "stage"))
def test_project_rejects_null_objective_for_valid_state(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / f"bad-{field}-objective.xrrproj.json"
    save_project(_project_with_legal_json_sentinels(), path)

    def mutate(payload: dict[str, object]) -> None:
        result = payload["datasets"][0]["last_valid_result"]
        if field == "candidate":
            result["candidates"][0]["objective"] = None
        else:
            result["stage_summaries"][0]["best_objective"] = None

    _rewrite(path, mutate)
    with pytest.raises(ProjectSchemaError, match="objective"):
        load_project(path)


@pytest.mark.parametrize("failure_point", ("write", "fsync", "replace"))
def test_atomic_save_failure_preserves_previous_bytes_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    path = tmp_path / "atomic.xrrproj.json"
    previous = b"previous-project-bytes"
    path.write_bytes(previous)

    def fail(*_args, **_kwargs):
        raise OSError(f"injected {failure_point} failure")

    with monkeypatch.context() as context:
        context.setattr(os, failure_point, fail)
        with pytest.raises(OSError, match=rf"injected {failure_point} failure"):
            save_project(_simple_project(), path)

    assert path.read_bytes() == previous
    assert tuple(tmp_path.glob(f".{path.name}.*.tmp")) == ()


def test_atomic_save_close_failure_still_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "atomic-close.xrrproj.json"
    previous = b"previous-project-bytes"
    path.write_bytes(previous)
    descriptors: list[int] = []

    def fail_close(descriptor: int) -> None:
        descriptors.append(descriptor)
        raise OSError("injected close failure")

    with monkeypatch.context() as context:
        context.setattr(os, "close", fail_close)
        with pytest.raises(OSError, match="injected close failure"):
            save_project(_simple_project(), path)

    for descriptor in set(descriptors):
        os.close(descriptor)
    assert path.read_bytes() == previous
    assert tuple(tmp_path.glob(f".{path.name}.*.tmp")) == ()


@pytest.mark.parametrize(
    "path",
    (
        ("fit_config", "c_decades"),
        ("fit_config", "budget", "short_de_maxiter"),
        ("datasets", 0, "import_angle_offset_deg"),
        ("datasets", 0, "fit_range_two_theta_deg", 0),
        ("datasets", 0, "last_valid_result", "parameter_definitions", 0, "initial"),
        ("datasets", 0, "last_valid_result", "candidates", 0, "seed_index"),
        (
            "datasets",
            0,
            "last_valid_result",
            "candidates",
            0,
            "parameters",
            0,
            "value",
        ),
        ("datasets", 0, "last_valid_result", "stage_summaries", 0, "total_nfev"),
        ("datasets", 0, "last_valid_result", "uncertainty", "bootstrap_failure_rate"),
        ("datasets", 0, "last_valid_result", "uncertainty", "mcmc", "child_seed"),
        ("datasets", 0, "checkpoint", "child_seeds", 0),
    ),
)
def test_project_rejects_null_required_numeric_scalar(
    tmp_path: Path,
    path: tuple[str | int, ...],
) -> None:
    project_path = tmp_path / "null-required-scalar.xrrproj.json"
    save_project(_project_with_legal_json_sentinels(), project_path)

    def mutate(payload: dict[str, object]) -> None:
        target = payload
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = None

    _rewrite(project_path, mutate)
    with pytest.raises(ProjectSchemaError, match="null"):
        load_project(project_path)


def test_project_rejects_null_required_scalar_before_save(tmp_path: Path) -> None:
    current = _simple_project()
    object.__setattr__(current.fit_config, "c_decades", None)
    path = tmp_path / "typed-null.xrrproj.json"

    with pytest.raises(ProjectSchemaError, match="null"):
        save_project(current, path)

    assert not path.exists()


def test_project_rejects_duplicate_json_keys_and_extra_fields(tmp_path: Path) -> None:
    path = _saved_project_path(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("{", '{"schema_version": 1,', 1), encoding="utf-8")
    with pytest.raises(ProjectSchemaError, match="duplicate"):
        load_project(path)

    path = _saved_project_path(tmp_path)
    _rewrite(path, lambda payload: payload.update(extra=True))
    with pytest.raises(ProjectSchemaError, match="project.*field"):
        load_project(path)
