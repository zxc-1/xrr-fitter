import math

from xrr_fitter.fit.drift import drift_coefficients
from xrr_fitter.model.structure import DriftSpec


def test_linear_coefficients():
    d = DriftSpec(kind="linear", target="thickness", amount=0.1)
    assert drift_coefficients(d, 4) == (0.0, 1.0, 2.0, 3.0)


def test_sine_coefficients_zero_at_copy0():
    d = DriftSpec(kind="sine", target="thickness", amount=0.1, period=4.0, phase=0.0)
    c = drift_coefficients(d, 3)
    assert c[0] == 0.0 and math.isclose(c[1], math.sin(2 * math.pi / 4))


def test_random_is_deterministic_bitwise():
    d = DriftSpec(kind="random", target="roughness", amount=0.2, seed=7)
    a = drift_coefficients(d, 6)
    b = drift_coefficients(d, 6)
    assert a == b and a[0] == 0.0 and all(-1.0 <= v <= 1.0 for v in a[1:])
