"""Shared stack-drift test builders (module-level functions; model_cases idiom)."""

from __future__ import annotations

from dataclasses import replace

from tests.support.model_cases import prepared_data
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import (
    DriftSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)

_AIR = MaterialSpec("Air", None, None, 0.0j)
_SILICON = MaterialSpec("Si", "Si", 2.329)
_SILICA = MaterialSpec("SiO2", "SiO2", 2.2)


def media() -> MaterialSpec:
    return _AIR


def make_layer(name="film", thickness_a=20.0, roughness_a=2.0) -> LayerSpec:
    return LayerSpec(name, _SILICA, thickness_a, roughness_a=roughness_a)


def two_layer_block(repeats=3) -> PeriodicBlock:
    return PeriodicBlock(
        name="p",
        layers=(
            LayerSpec("a", _SILICA, 20.0, roughness_a=2.0),
            LayerSpec("b", _SILICON, 500.0, roughness_a=3.0),
        ),
        repeats=repeats,
    )


def two_layer_block_with_thickness_drift(repeats=3, amount=0.1) -> PeriodicBlock:
    return replace(
        two_layer_block(repeats),
        drift=DriftSpec(kind="linear", target="thickness", amount=amount),
    )


def drift_block() -> PeriodicBlock:
    return two_layer_block_with_thickness_drift()


def one_drift_block_structure() -> StructureSpec:
    return StructureSpec(
        fronting=_AIR,
        components=(drift_block(),),
        backing=_SILICON,
        backing_roughness_a=3.0,
    )


def plain_periodic_structure() -> StructureSpec:
    return StructureSpec(
        fronting=_AIR,
        components=(two_layer_block(),),
        backing=_SILICON,
        backing_roughness_a=3.0,
    )


def drift_structure() -> StructureSpec:
    return one_drift_block_structure()


def wavelength() -> float:
    return 1.5406


def drift_case() -> tuple:
    return (
        prepared_data(),
        drift_structure(),
        InstrumentSpec(instrument_id="lab"),
        FitConfig.standard(11),
    )


def drift_values(structure) -> dict[str, float]:
    from xrr_fitter.fit.drift import drift_coefficients

    block = structure.components[0]
    drift = block.drift
    coeffs = drift_coefficients(drift, block.repeats)
    prefix = "component.0"
    values = {f"{prefix}.drift_scale": drift.amount}
    for index, layer in enumerate(block.layers):
        base = f"{prefix}.layer.{index}"
        values[f"{base}.thickness_a"] = layer.thickness_a
        values[f"{base}.roughness_a"] = layer.roughness_a
        values[f"{base}.density_scale"] = layer.density_scale
        for k in range(1, block.repeats):
            values[f"{prefix}.repeat.{k}.layer.{index}.thickness_a"] = layer.thickness_a * (
                1.0 + drift.amount * coeffs[k]
            )
    return values
