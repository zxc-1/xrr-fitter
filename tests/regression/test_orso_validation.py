"""Community ORSO validation suite parity for the unpolarised specular path.

Layer tables are always four columns: thickness, real SLD, imaginary SLD, and
the roughness of that row's top interface. Data files carry two to four
columns; a fourth column is a pointwise 1-sigma dQ and selects the smeared
comparison at the suite's looser tolerance. The suite is an additional, looser
oracle: tests/regression/test_numerical_reference.py remains the strict gate.

Measured worst relative deviation, this frozen suite against this kernel:
strict tier 6.3854e-05 over six cases, worst at test1.txt q=0.57205; smeared
tier 1.3366e-04 over two cases, worst at test4.txt q=0.013916.

The smeared tier reproduces the suite's own generation convention, which
truncates the resolution kernel at +-3.5 sigma and normalises analytically
rather than by summed weights. Every reference implementation in the suite
truncates the same way: refnx's ``_INTLIMIT``, refl1d's
``linspace(q - 3.5 dQ, q + 3.5 dQ)``, BornAgain's
``DistributionGaussian(0, 1, 21, 3.5)``, and GenX's ``resintrange = 3.5``.

The manifests record generation at 10001 quadrature points. At that order both
cases land within 5.6e-13, but each costs about 80 s. test4.txt converges
slowly because its layer table absorbs and the integrand has a branch point at
the critical edge near q=0.0139, where Gauss-Legendre degrades to algebraic
convergence; 401 points reach 1.3366e-04 in about 34 ms, a 224x margin under
the suite's published rtol. That tolerance exists precisely to absorb
convolution-scheme differences: BornAgain validates at 21 points against the
same 0.03.

Production smearing in xrr_fitter.physics.resolution is deliberately
untruncated, so it disagrees with the suite by up to 9.0070e-02 at deep
interference minima, where the tails beyond 3.5 sigma sample regions one to two
orders of magnitude brighter than the minimum itself. That is a convention
boundary, not a kernel defect;
test_untruncated_production_smearing_agrees_away_from_minima pins the
production path against the same data wherever the conventions do agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xrr_fitter.model.slab_stack import SlabStack
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.resolution import gaussian_smear

SUITE = Path(__file__).resolve().parents[1] / "fixtures/orso"
UNPOLARISED = SUITE / "unpolarised"
SLD_SCALE = 1e-6
KERNEL_RTOL = 8e-5
SMEARED_RTOL = 0.03
ROUGHNESS_COLUMN = 3
# Row N's roughness belongs to interface [N-1, N]; the fronting row's value is
# unused. This mirrors the suite's own reference construction and the existing
# layers[1:, 3] slice in test_numerical_reference.py.
FIRST_INTERFACE_ROW = 1
# The suite's shared truncation limit, in sigma. See the module docstring for
# the four reference implementations that all pin this same value.
SUITE_SIGMA_LIMIT = 3.5
SUITE_QUAD_ORDER = 401


def _frozen_index() -> dict[str, object]:
    return json.loads((SUITE / "index.json").read_text(encoding="utf-8"))


def _manifest_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            Path(str(record["path"])).name
            for record in _frozen_index()["files"]
            if Path(str(record["path"])).suffix == ".txt"
        )
    )


def _references(manifest: Path) -> tuple[str, str]:
    lines = tuple(
        stripped
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    )
    assert len(lines) == 2, f"{manifest.name} must name a layer table and a data file"
    return lines[0], lines[1]


def _load_case(name: str) -> tuple[np.ndarray, np.ndarray]:
    layers_reference, data_reference = _references(UNPOLARISED / name)
    layers_path = UNPOLARISED / layers_reference
    data_path = UNPOLARISED / data_reference
    assert layers_path.is_file(), f"{name} references a missing layer table: {layers_reference}"
    assert data_path.is_file(), f"{name} references a missing data file: {data_reference}"
    layers = np.atleast_2d(np.loadtxt(layers_path))
    data = np.atleast_2d(np.loadtxt(data_path))
    assert layers.shape[1] == 4, f"{name} layer table must be four columns, got {layers.shape[1]}"
    assert data.shape[1] in (2, 3, 4), f"{name} data file has unregistered column count {data.shape[1]}"
    return layers, data


def _stack(layers: np.ndarray) -> SlabStack:
    return SlabStack(
        layers[:, 0],
        (layers[:, 1] + 1j * layers[:, 2]) * SLD_SCALE,
        layers[FIRST_INTERFACE_ROW:, ROUGHNESS_COLUMN],
    )


def _suite_smeared(stack: SlabStack, qz: np.ndarray, sigma_q: np.ndarray) -> np.ndarray:
    """Smear with the convention the suite's own reference data was generated under.

    A transcription of refnx's ``_smeared_kernel_pointwise``: fixed order
    Gauss-Legendre across +-SUITE_SIGMA_LIMIT sigma, weighted by the analytic
    standard normal density and scaled by the half width, with no
    renormalisation of the truncated weights.
    """
    abscissa, weights = np.polynomial.legendre.leggauss(SUITE_QUAD_ORDER)
    offsets = abscissa * SUITE_SIGMA_LIMIT
    density = np.exp(-0.5 * offsets**2) / np.sqrt(2.0 * np.pi)
    query = qz[:, None] + offsets[None, :] * sigma_q[:, None]
    assert np.all(query >= 0.0), "a smeared case reaches negative q; the suite data changed"
    values = parratt_reflectivity(query, stack)
    return np.sum(values * (density * weights)[None, :], axis=-1) * SUITE_SIGMA_LIMIT


def _production_smeared(stack: SlabStack, qz: np.ndarray, sigma_q: np.ndarray) -> np.ndarray:
    return gaussian_smear(
        qz,
        lambda query: parratt_reflectivity(query, stack),
        sigma_q_a_inv=sigma_q,
        emit_warning=False,
    )


def _local_minima(values: np.ndarray) -> np.ndarray:
    interior = np.zeros(values.size, dtype=bool)
    interior[1:-1] = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    return interior


def _actual(stack: SlabStack, data: np.ndarray) -> tuple[np.ndarray, float]:
    qz = data[:, 0]
    if data.shape[1] < 4:
        return parratt_reflectivity(qz, stack), KERNEL_RTOL
    return _suite_smeared(stack, qz, data[:, 3]), SMEARED_RTOL


@pytest.mark.parametrize("name", _manifest_names())
def test_unpolarised_case_matches_the_orso_suite(name: str) -> None:
    layers, data = _load_case(name)
    expected = data[:, 1]
    actual, rtol = _actual(_stack(layers), data)

    deviation = np.abs(actual - expected) / np.abs(expected)
    worst = int(np.argmax(deviation))
    assert np.all(deviation <= rtol), (
        f"{name}: worst relative deviation {deviation[worst]:.3e} exceeds rtol {rtol:g} "
        f"at q={data[worst, 0]:.6g} (columns={data.shape[1]})"
    )


def test_every_frozen_case_is_covered_and_both_tolerances_are_exercised() -> None:
    names = _manifest_names()
    assert len(names) == 8
    widths = {_load_case(name)[1].shape[1] for name in names}
    assert any(width < 4 for width in widths), "no case exercises the strict kernel tolerance"
    assert any(width == 4 for width in widths), "no case exercises the smeared tolerance"


def test_reversed_roughness_attribution_breaks_at_least_one_case() -> None:
    broken: list[str] = []
    for name in _manifest_names():
        layers, data = _load_case(name)
        if not np.any(layers[:, ROUGHNESS_COLUMN]):
            continue
        reversed_stack = SlabStack(
            layers[:, 0],
            (layers[:, 1] + 1j * layers[:, 2]) * SLD_SCALE,
            layers[:-1, ROUGHNESS_COLUMN],
        )
        actual, rtol = _actual(reversed_stack, data)
        if not np.all(np.abs(actual - data[:, 1]) / np.abs(data[:, 1]) <= rtol):
            broken.append(name)
    assert broken, "roughness attribution is untested: no case has a discriminating roughness column"


def test_untruncated_production_smearing_agrees_away_from_minima() -> None:
    """The production kernel must match the suite wherever the two conventions agree.

    Truncation only matters where the Gaussian tails outweigh the point itself,
    which happens exactly at the interference minima. Away from them the
    production path is held to the suite's own tolerance, so this covers
    xrr_fitter.physics.resolution.gaussian_smear against real reference data
    rather than leaving the smeared tier to the transcribed kernel alone.
    """
    checked = 0
    for name in _manifest_names():
        layers, data = _load_case(name)
        if data.shape[1] != 4:
            continue
        checked += 1
        expected = data[:, 1]
        actual = _production_smeared(_stack(layers), data[:, 0], data[:, 3])
        deviation = np.abs(actual - expected) / np.abs(expected)
        off_minimum = ~_local_minima(expected)
        worst = int(np.argmax(np.where(off_minimum, deviation, -1.0)))
        assert np.all(deviation[off_minimum] <= SMEARED_RTOL), (
            f"{name}: untruncated smearing deviates {deviation[worst]:.3e} at a "
            f"non-minimum q={data[worst, 0]:.6g}, above rtol {SMEARED_RTOL:g}"
        )
        # Truncation discards outer tails, which at a minimum are brighter than
        # the minimum itself, so the untruncated result can only sit higher.
        assert np.all(actual >= _suite_smeared(_stack(layers), data[:, 0], data[:, 3])), (
            f"{name}: untruncated smearing fell below the truncated convention; the tail contribution changed sign"
        )
    assert checked == 2, f"expected two smeared cases, covered {checked}"


def test_frozen_suite_content_is_hash_bound() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "tools/sync_orso_suite.py"
    spec = importlib.util.spec_from_file_location("orso_suite_sync_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_index(SUITE)
