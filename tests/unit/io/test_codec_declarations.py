from tests.support.drift_cases import make_layer

from xrr_fitter.io.codec_declarations import _periodic_from_dict, _periodic_to_dict
from xrr_fitter.model.structure import DriftSpec, PeriodicBlock


def test_periodic_without_drift_omits_key() -> None:
    block = PeriodicBlock(name="p", layers=(make_layer(),), repeats=2)
    assert "drift" not in _periodic_to_dict(block)


def test_periodic_drift_round_trips() -> None:
    drift = DriftSpec(kind="sine", target="thickness", amount=0.1, period=4.0, phase=0.5)
    block = PeriodicBlock(name="p", layers=(make_layer(),), repeats=5, drift=drift)
    assert _periodic_from_dict(_periodic_to_dict(block)) == block
