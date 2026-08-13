"""Public invariants for the immutable SLD uncertainty-band value."""

from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.model.sld_bands import SldUncertaintyBands


def _bands(**overrides: object) -> SldUncertaintyBands:
    values: dict[str, object] = {
        "depth_a": np.array([-1.0, 0.0, 1.0]),
        "quantiles": (0.16, 0.5, 0.84),
        "real": np.zeros((3, 3)),
        "imaginary": np.zeros((3, 3)),
        "align_label": "基底界面",
        "sample_count": 10,
        "total_samples": 10,
        "failure_rate": 0.0,
    }
    values.update(overrides)
    return SldUncertaintyBands(**values)


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), float("-inf")))
def test_sld_bands_reject_nonfinite_depth_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="depth_a.*finite"):
        _bands(depth_a=np.array([-1.0, bad_value, 1.0]))


@pytest.mark.parametrize(
    "depth",
    (
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.0, 1.0, 0.0]),
    ),
)
def test_sld_bands_require_a_strictly_increasing_depth_axis(depth: np.ndarray) -> None:
    with pytest.raises(ValueError, match="depth_a.*strictly increasing"):
        _bands(depth_a=depth)


@pytest.mark.parametrize("field", ("real", "imaginary"))
def test_sld_bands_reject_nonfinite_quantile_curves(field: str) -> None:
    curve = np.zeros((3, 3))
    curve[1, 1] = np.nan

    with pytest.raises(ValueError, match=rf"{field}.*finite"):
        _bands(**{field: curve})


@pytest.mark.parametrize("label", ("", " \t\n"))
def test_sld_bands_reject_an_empty_alignment_label(label: str) -> None:
    with pytest.raises(ValueError, match="align_label.*empty"):
        _bands(align_label=label)
