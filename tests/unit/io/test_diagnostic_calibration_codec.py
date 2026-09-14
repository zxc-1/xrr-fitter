"""Strict current-schema persistence of the saved Poisson diagnostic decision.

These are value-boundary tests, not coverage or Monte Carlo experiments. A
small sealed family makes raw advisory, effective conclusion, and incomplete
work distinguishable in every export. No codec or display operation is allowed
to refit the source or regenerate the null axis. Missing keys are different
from explicit null values; no omitted-field default is a migration path.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest
from tests.support.diagnostic_work_cases import completed_work, failed_work
from tests.support.model_cases import project
from tests.unit.io.test_export_v2_evidence import _metadata, evidence_context

from xrr_fitter.io.codec_declarations import _fit_config_from_dict, _fit_config_to_dict
from xrr_fitter.io.codec_inference import (
    bootstrap_from_dict,
    bootstrap_to_dict,
    residual_from_dict,
    residual_to_dict,
)
from xrr_fitter.io.export_tables import dataset_json_bytes
from xrr_fitter.io.project_codec import project_from_bytes, project_to_bytes, project_to_dict
from xrr_fitter.model.bootstrap import BootstrapResult
from xrr_fitter.model.diagnostic_calibration import DiagnosticCalibration, DiagnosticStatistic
from xrr_fitter.model.diagnostic_work import combine_refit_work
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.inference import ResidualEvidence
from xrr_fitter.model.instrument import PhysicsDiagnostic
from xrr_fitter.model.provenance import diagnostic_calibration_sha256

CALIBRATION_SUMMARIES = {"p_value", "rejected", "resolution"}
RESIDUAL_NEW_FIELDS = {"raw_systematic", "raw_autocorrelation", "advisories", "calibration", "owner_sha256"}


def _codec():
    name = "xrr_fitter.io.codec_diagnostic_calibration"
    assert find_spec(name) is not None, "saved calibration needs a strict current-schema codec"
    return import_module(name)


def _calibration(status="available", *, p_value=0.2, dataset_id="curve", **changes):
    values = dict(
        status=status,
        sample_count=999,
        child_seed=123,
        owner_sha256="a" * 64,
        statistics=(DiagnosticStatistic(dataset_id, "background", 2.0),),
    )
    if status == "available":
        values.update(
            attempted_count=999,
            successful_count=999,
            statistics=(
                DiagnosticStatistic(dataset_id, "background", 2.0, 0.0, 1.0, p_value),
                DiagnosticStatistic(dataset_id, "acf", 0.4, 0.0, 1.0, 0.8),
            ),
            tail_count=round(1000 * p_value),
            tie_count=1,
            observed_score=2.0,
            null_statistics_sha256="b" * 64,
            refit_nfev=12004,
            observed_work=completed_work(nfev=16),
            null_work=completed_work(3996),
            refit_discrepancy=(0.0, 1e-9, 1e-9),
        )
    else:
        values.update(
            attempted_count=3,
            successful_count=2,
            failure_reasons=((2, "diagnostic_refit_failed"),),
            unavailable_reason="diagnostic_refit_failed",
            refit_nfev=37,
            observed_work=completed_work(),
            null_work=combine_refit_work(completed_work(8), failed_work()),
        )
    evidence = DiagnosticCalibration(**(values | changes))
    return replace(evidence, provenance_sha256=diagnostic_calibration_sha256(evidence))


def _residual(calibration, *, point_indices=()):
    available = calibration.status == "available"
    return ResidualEvidence(
        "curve",
        available,
        calibration.rejected if available else None,
        False if available else None,
        32,
        unavailable_reason=calibration.unavailable_reason,
        raw_systematic=True,
        raw_autocorrelation=False,
        advisories=(PhysicsDiagnostic("suspected_diffuse_background", "raw tail screen", point_indices),),
        calibration=calibration,
        owner_sha256=calibration.owner_sha256,
    )


def _context(calibration, *, point_indices=()):
    context = evidence_context("poisson")
    member = _residual(calibration, point_indices=point_indices)
    report = replace(
        context.selected_uncertainty,
        member_residuals=(member,),
        systematic_residual=member.systematic,
        residual_autocorrelation=member.autocorrelation,
    )
    dataset = replace(context.dataset, last_valid_result=replace(context.result, uncertainty=report))
    return replace(context, project=replace(context.project, datasets=(dataset,)), dataset=dataset)


def _forbid_recalculation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("persistence/export must not execute physics, refitting, or Monte Carlo")

    for name, attribute in (
        ("xrr_fitter.evaluation", "evaluate_model"),
        ("xrr_fitter.evaluation", "evaluate_model_jacobian"),
        ("xrr_fitter.fit.diagnostic_refit", "refit_diagnostic_single"),
        ("xrr_fitter.fit.diagnostic_refit", "refit_diagnostic_joint"),
        ("xrr_fitter.analysis.residual_calibration", "calibrate_residuals"),
    ):
        monkeypatch.setattr(import_module(name), attribute, unexpected)


@pytest.mark.parametrize("status", ("available", "unavailable"))
def test_saved_calibration_roundtrips_without_recomputation(status, monkeypatch):
    evidence = _calibration(status)
    _forbid_recalculation(monkeypatch)
    codec = _codec()
    payload = codec.calibration_to_dict(evidence)
    restored = codec.calibration_from_dict(json.loads(json.dumps(payload, allow_nan=False)))
    assert restored == evidence
    assert payload["p_value"] == evidence.p_value
    assert payload["rejected"] == evidence.rejected
    assert payload["resolution"] == evidence.resolution


@pytest.mark.parametrize(
    "field", [field.name for field in fields(DiagnosticCalibration)] + sorted(CALIBRATION_SUMMARIES)
)
def test_calibration_requires_every_current_key(field):
    codec = _codec()
    payload = codec.calibration_to_dict(_calibration())
    del payload[field]
    with pytest.raises(ValueError, match="field|missing"):
        codec.calibration_from_dict(payload)


@pytest.mark.parametrize("field", [field.name for field in fields(DiagnosticStatistic)])
def test_statistic_requires_every_current_key(field):
    codec = _codec()
    payload = codec.calibration_to_dict(_calibration())
    del payload["statistics"][0][field]
    with pytest.raises(ValueError, match="field|missing"):
        codec.calibration_from_dict(payload)


@pytest.mark.parametrize("nested", (False, True))
def test_calibration_and_statistic_reject_unknown_fields(nested):
    codec = _codec()
    payload = codec.calibration_to_dict(_calibration())
    target = payload["statistics"][0] if nested else payload
    target["invented"] = 1
    with pytest.raises(ValueError, match="field|extra"):
        codec.calibration_from_dict(payload)


@pytest.mark.parametrize(
    "field,value", (("child_seed", 124), ("owner_sha256", "c" * 64), ("refit_discrepancy", [0.0, 1e-8, 1e-8]))
)
def test_calibration_rejects_value_tampering_even_with_valid_dataclass_shape(field, value):
    codec = _codec()
    payload = codec.calibration_to_dict(_calibration())
    payload[field] = value
    with pytest.raises(ValueError, match="seal|provenance"):
        codec.calibration_from_dict(payload)


@pytest.mark.parametrize("field,value", (("p_value", 0.3), ("rejected", 0), ("resolution", 0.1)))
def test_calibration_rejects_false_or_mistyped_derived_summaries(field, value):
    codec = _codec()
    payload = codec.calibration_to_dict(_calibration())
    payload[field] = value
    with pytest.raises(ValueError, match="metadata|summary"):
        codec.calibration_from_dict(payload)


@pytest.mark.parametrize("change", ({"provenance_sha256": None}, {"child_seed": 9}))
def test_encoder_does_not_silently_create_or_repair_a_seal(change):
    evidence = replace(_calibration(), **change)
    with pytest.raises(ValueError, match="seal|provenance"):
        _codec().calibration_to_dict(evidence)


@pytest.mark.parametrize("changes", ({"successful_count": 998}, {"status": "unavailable"}, {"tail_count": 0}))
def test_calibration_decoder_rejects_contradictory_states(changes):
    codec = _codec()
    payload = codec.calibration_to_dict(_calibration()) | changes
    with pytest.raises(ValueError):
        codec.calibration_from_dict(payload)


@pytest.mark.parametrize("status", ("available", "unavailable"))
def test_current_residual_roundtrip_preserves_raw_and_effective_evidence(status):
    evidence = _residual(_calibration(status))
    restored = residual_from_dict(residual_to_dict(evidence))
    assert restored == evidence
    assert restored.advisories and not restored.diagnostics


@pytest.mark.parametrize("field", sorted(RESIDUAL_NEW_FIELDS))
def test_residual_requires_new_fields_including_nullable_keys(field):
    payload = residual_to_dict(ResidualEvidence("curve", True, False, False, 32))
    assert field in payload, "nullable current fields still require a JSON key"
    del payload[field]
    with pytest.raises(ValueError, match="field|missing"):
        residual_from_dict(payload)


@pytest.mark.parametrize("owner", (None, "c" * 64))
def test_residual_encoder_rejects_calibration_from_a_different_owner(owner):
    evidence = replace(_residual(_calibration()), owner_sha256=owner)
    with pytest.raises(ValueError, match="owner"):
        residual_to_dict(evidence)


@pytest.mark.parametrize("owner", (None, "c" * 64))
def test_residual_decoder_rejects_calibration_from_a_different_owner(owner):
    payload = residual_to_dict(_residual(_calibration()))
    assert payload["calibration"]["owner_sha256"] == "a" * 64
    payload["owner_sha256"] = owner
    with pytest.raises(ValueError, match="owner"):
        residual_from_dict(payload)


def test_current_file_identity_and_diagnostic_configuration_are_explicit():
    payload = project_to_dict(project())
    assert payload["schema_version"] == 5
    assert payload["algorithm_version"] == "xrr-fit-v2-poisson-5"
    assert payload["fit_config"]["objective_version"] == "2"
    assert payload["fit_config"]["diagnostic_version"] == "poisson-refit-null-v4"
    assert payload["fit_config"]["budget"]["diagnostic_samples"] == 999


def test_fit_config_requires_diagnostic_version_not_a_dataclass_default():
    payload = _fit_config_to_dict(FitConfig.fast(7))
    payload.pop("diagnostic_version", None)
    with pytest.raises(ValueError, match="diagnostic_version"):
        _fit_config_from_dict(payload)


def test_search_budget_requires_diagnostic_sample_count():
    payload = _fit_config_to_dict(FitConfig.fast(7))
    del payload["budget"]["diagnostic_samples"]
    with pytest.raises(ValueError, match="diagnostic_samples"):
        _fit_config_from_dict(payload)


@pytest.mark.parametrize("reason", (None, "poisson_diagnostic_calibration_required"))
def test_bootstrap_roundtrip_keeps_quantiles_and_diagnostic_publication_gate(reason):
    evidence = BootstrapResult(
        ("scale",),
        np.ones((200, 1)),
        (("scale", 0.9, 1.1),),
        0.0,
        200,
        method="poisson_parametric",
        unavailable_reason=reason,
        diagnostic_unavailable_reason=reason,
    )
    payload = bootstrap_to_dict(evidence)
    assert payload["diagnostic_unavailable_reason"] == reason
    restored = bootstrap_from_dict(payload)
    np.testing.assert_array_equal(restored.samples, evidence.samples)
    assert restored.intervals == evidence.intervals
    assert restored.confidence_level == (0.95 if reason is None else None)
    assert restored.diagnostic_unavailable_reason == reason


@pytest.mark.parametrize("status", ("available", "unavailable"))
def test_project_bytes_accept_explicit_nullable_calibration_values_without_physics(status, monkeypatch):
    context = _context(_calibration(status))
    _forbid_recalculation(monkeypatch)
    encoded = project_to_bytes(context.project)
    restored = project_from_bytes(encoded)
    assert project_to_bytes(restored) == encoded
    restored_residuals = restored.datasets[0].last_valid_result.uncertainty.member_residuals
    assert restored_residuals == context.selected_uncertainty.member_residuals


@pytest.mark.parametrize("status", ("available", "unavailable"))
@pytest.mark.parametrize("kind", ("json", "csv", "workbook", "log", "orso"))
def test_all_exports_expose_the_same_saved_raw_effective_calibration(status, kind, monkeypatch):
    calibration = _calibration(status)
    context = _context(calibration)
    _forbid_recalculation(monkeypatch)
    member = _metadata(context, kind)["member_residuals"][0]
    assert member["raw_systematic"] is True
    assert member["systematic"] is calibration.rejected
    assert member["advisories"][0]["code"] == "suspected_diffuse_background"
    assert member["calibration"]["status"] == status
    assert member["calibration"]["p_value"] == calibration.p_value
    assert member["calibration"]["successful_count"] == calibration.successful_count
    assert member["calibration"]["unavailable_reason"] == calibration.unavailable_reason


def test_dense_raw_advisory_is_summarized_but_complete_json_preserves_point_identity():
    context = _context(_calibration(), point_indices=tuple(range(10000)))
    summary = _metadata(context, "workbook")["member_residuals"][0]
    full = json.loads(dataset_json_bytes(context))["fit_result"]["uncertainty"]["member_residuals"][0]
    assert summary["advisories"][0]["point_count"] == 10000
    assert "point_indices" not in summary["advisories"][0]
    assert full["advisories"][0]["point_indices"] == list(range(10000))


def test_residual_encoder_revalidates_effective_flags_before_export():
    evidence = _residual(_calibration())
    object.__setattr__(evidence, "systematic", True)
    with pytest.raises(ValueError, match="effective flags"):
        residual_to_dict(evidence)


def test_residual_decoder_rejects_contradictory_effective_flags():
    payload = residual_to_dict(_residual(_calibration()))
    payload["systematic"] = True
    with pytest.raises(ValueError, match="effective flags"):
        residual_from_dict(payload)


@pytest.mark.parametrize("field", ("statistics", "failure_reasons", "refit_discrepancy"))
def test_calibration_sequences_require_json_arrays(field):
    codec = _codec()
    payload = codec.calibration_to_dict(_calibration())
    payload[field] = tuple(payload[field])
    with pytest.raises(ValueError, match="JSON array"):
        codec.calibration_from_dict(payload)


@pytest.mark.parametrize("seal", (None, "0" * 64))
def test_calibration_decoder_rejects_missing_or_wrong_seal(seal):
    codec = _codec()
    payload = codec.calibration_to_dict(_calibration())
    payload["provenance_sha256"] = seal
    with pytest.raises(ValueError, match="provenance seal"):
        codec.calibration_from_dict(payload)


def test_bootstrap_diagnostic_reason_key_is_required_even_when_null():
    payload = bootstrap_to_dict(evidence_context().selected_uncertainty.bootstrap_evidence)
    del payload["diagnostic_unavailable_reason"]
    with pytest.raises(ValueError, match="diagnostic_unavailable_reason"):
        bootstrap_from_dict(payload)
