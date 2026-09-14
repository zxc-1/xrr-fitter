"""RMS scale evidence must remain explicit, immutable, and self-consistent."""

import pickle

import numpy as np
import pytest
from tests.unit.model.test_diagnostic_calibration import _available

from xrr_fitter.model.diagnostic_calibration import DiagnosticStatistic


@pytest.mark.parametrize("scale", [0.99, 0.125, np.nextafter(0.0, 1.0)])
def test_positive_fractional_rms_scale_roundtrips(scale):
    value = DiagnosticStatistic("curve", "acf", 0.5, 0.0, scale, 0.2)
    assert value.scale == scale
    assert pickle.loads(pickle.dumps(value)) == value


def test_constant_scale_is_explicit_and_neutral():
    value = DiagnosticStatistic("curve", "acf", 2.0, 2.0, 0.0, 1.0)
    assert value.scale == 0
    assert pickle.loads(pickle.dumps(value)) == value
    result = _available(statistics=(value,), observed_score=0.0, tail_count=1000, tie_count=1000)
    assert result.p_value == 1
    assert result.rejected is False
    assert pickle.loads(pickle.dumps(result)) == result


@pytest.mark.parametrize("observed,center,probability", [(2.0, 0.0, 1.0), (2.0, 2.0, 0.5)])
def test_zero_scale_cannot_describe_a_nonconstant_or_significant_observation(observed, center, probability):
    with pytest.raises(ValueError, match="scale|constant"):
        DiagnosticStatistic("curve", "acf", observed, center, 0.0, probability)


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf"), True])
def test_invalid_rms_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        DiagnosticStatistic("curve", "acf", 2.0, 0.0, scale, 0.2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale", -1.0),
        ("scale", 0.0),
        ("scale", float("nan")),
        ("center", float("nan")),
        ("observed", float("nan")),
        ("kind", "unexpected"),
    ],
)
def test_parent_revalidates_forged_nested_statistic(field, value):
    statistic = DiagnosticStatistic("curve", "acf", 2.0, 0.0, 1.0, 0.008)
    object.__setattr__(statistic, field, value)
    with pytest.raises(ValueError):
        _available(statistics=(statistic,))


def test_parent_owns_revalidated_statistic_instead_of_external_slot_state():
    statistic = DiagnosticStatistic("curve", "acf", 2.0, 0.0, 1.0, 0.008)
    result = _available(statistics=(statistic,))
    object.__setattr__(statistic, "scale", -1.0)
    assert result.statistics[0].scale == 1.0
    assert result.statistics[0] is not statistic


@pytest.mark.parametrize("changes", [{"observed_score": 1.0}, {"tie_count": 1}])
def test_all_constant_family_cannot_publish_nonzero_score_or_partial_ties(changes):
    statistic = DiagnosticStatistic("curve", "acf", 2.0, 2.0, 1.0, 1.0)
    object.__setattr__(statistic, "scale", 0.0)
    values = dict(statistics=(statistic,), observed_score=0.0, tail_count=1000, tie_count=1000)
    with pytest.raises(ValueError, match="constant"):
        _available(**(values | changes))


def test_mixed_constant_and_nonconstant_family_preserves_nonconstant_decision():
    values = (
        DiagnosticStatistic("curve", "background", 2.0, 2.0, 0.0, 1.0),
        DiagnosticStatistic("curve", "acf", 0.5, 0.0, 0.25, 0.008),
    )
    result = _available(statistics=values)
    assert result.p_value == 0.008
    assert result.rejected is True
