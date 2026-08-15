import pytest
from tests.support.drift_cases import make_layer

from xrr_fitter.io.codec_declarations import _drift_from_dict, _periodic_from_dict, _periodic_to_dict
from xrr_fitter.model.structure import DriftSpec, PeriodicBlock


def test_periodic_without_drift_omits_key() -> None:
    block = PeriodicBlock(name="p", layers=(make_layer(),), repeats=2)
    assert "drift" not in _periodic_to_dict(block)


def test_periodic_drift_round_trips() -> None:
    drift = DriftSpec(kind="sine", target="thickness", amount=0.1, period=4.0, phase=0.5)
    block = PeriodicBlock(name="p", layers=(make_layer(),), repeats=5, drift=drift)
    assert _periodic_from_dict(_periodic_to_dict(block)) == block


def test_periodic_drift_round_trips_unused_fields_as_well() -> None:
    drift = DriftSpec(kind="random", target="roughness", amount=0.2, period=7.0, phase=0.5, seed=11)
    block = PeriodicBlock(name="p", layers=(make_layer(),), repeats=5, drift=drift)
    assert _periodic_from_dict(_periodic_to_dict(block)) == block


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    (
        pytest.param("period", float("nan"), ValueError, id="period-nan"),
        pytest.param("seed", "11", TypeError, id="seed-string"),
    ),
)
def test_drift_codec_rejects_invalid_unused_fields(field: str, value: object, expected: type[Exception]) -> None:
    payload = {
        "kind": "linear",
        "target": "thickness",
        "amount": 0.1,
        "period": 0.0,
        "phase": 0.0,
        "seed": 0,
    }
    payload[field] = value

    with pytest.raises(expected, match="drift\\."):
        _drift_from_dict(payload)
