from tests.support.drift_cases import two_layer_block, two_layer_block_with_thickness_drift

from xrr_fitter.fit.parameters import _periodic_definitions


def _names(defs):
    return [d.name for d in defs]


def test_no_drift_definitions_unchanged():
    defs = _periodic_definitions("component.0", two_layer_block(), (2.0, 500.0))
    assert not any(".drift_scale" in n or ".repeat." in n for n in _names(defs))


def test_drift_adds_scale_and_percopy():  # repeats=3, 2 layers
    block = two_layer_block_with_thickness_drift()
    defs = _periodic_definitions("component.0", block, (2.0, 500.0))
    names = _names(defs)
    assert "component.0.drift_scale" in names
    for k in (1, 2):
        for i in (0, 1):
            assert f"component.0.repeat.{k}.layer.{i}.thickness_a" in names
    # 非目标族（roughness）不发逐副本
    assert not any(".repeat." in n and n.endswith("roughness_a") for n in names)
