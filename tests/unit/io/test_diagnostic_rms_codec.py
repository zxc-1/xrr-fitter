"""Persist the actual RMS calibration without recalculating its inference."""

import json

import pytest
from tests.unit.io.test_diagnostic_calibration_codec import _calibration, _codec, _forbid_recalculation

from xrr_fitter.model.diagnostic_calibration import DiagnosticStatistic


def _rms_evidence():
    values = (
        DiagnosticStatistic("curve", "background", 2.0, 2.0, 0.0, 1.0),
        DiagnosticStatistic("curve", "acf", 0.5, 0.0, 0.25, 0.2),
    )
    return _calibration(statistics=values)


def test_fractional_and_constant_rms_evidence_roundtrips_without_recalculation(monkeypatch):
    _forbid_recalculation(monkeypatch)
    value = _rms_evidence()
    payload = _codec().calibration_to_dict(value)
    restored = _codec().calibration_from_dict(json.loads(json.dumps(payload, allow_nan=False)))
    assert restored == value
    assert payload["statistics"][0]["scale"] == 0.0
    assert payload["statistics"][1]["scale"] == 0.25


@pytest.mark.parametrize("changes", [{"observed": 3.0}, {"adjusted_p_value": 0.5}, {"scale": -0.1}])
def test_decoder_rejects_false_constant_claim_before_accepting_a_seal(changes):
    payload = _codec().calibration_to_dict(_rms_evidence())
    payload["statistics"][0].update(changes)
    with pytest.raises(ValueError, match="scale|constant"):
        _codec().calibration_from_dict(payload)


def test_decoder_still_rejects_rms_scale_tampering_that_preserves_value_shape():
    payload = _codec().calibration_to_dict(_rms_evidence())
    payload["statistics"][1]["scale"] = 0.3
    with pytest.raises(ValueError, match="seal|provenance"):
        _codec().calibration_from_dict(payload)
