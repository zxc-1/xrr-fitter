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

AIR = MaterialSpec("Air", None, None, 0.0j)
SILICON = MaterialSpec("Si", "Si", 2.329)
SILICA = MaterialSpec("SiO2", "SiO2", 2.2)


def media() -> MaterialSpec:
    return AIR


def make_layer(name="film", thickness_a=20.0, roughness_a=2.0) -> LayerSpec:
    return LayerSpec(name, SILICA, thickness_a, roughness_a=roughness_a)


def two_layer_block(repeats=3) -> PeriodicBlock:
    return PeriodicBlock(
        name="p",
        layers=(
            LayerSpec("a", SILICA, 20.0, roughness_a=2.0),
            LayerSpec("b", SILICON, 500.0, roughness_a=3.0),
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
        fronting=AIR,
        components=(drift_block(),),
        backing=SILICON,
        backing_roughness_a=3.0,
    )


def plain_periodic_structure() -> StructureSpec:
    return StructureSpec(
        fronting=AIR,
        components=(two_layer_block(),),
        backing=SILICON,
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
    """Emit the complete physical map ``rebuild_structure`` reads for a drift block.

    The map covers the free ``drift_scale``, every base-cell coordinate, the
    per-copy ``.repeat.{k}`` coordinate for whichever family the block drifts, and
    the backing roughness -- rebuilding requires all of them by name, since it
    bakes drift into per-copy layers rather than reading the declaration's values.
    """
    from xrr_fitter.fit.drift import drift_coefficients

    block = structure.components[0]
    drift = block.drift
    coeffs = drift_coefficients(drift, block.repeats)
    prefix = "component.0"
    values = {
        f"{prefix}.drift_scale": drift.amount,
        "backing.roughness_a": structure.backing_roughness_a,
    }
    for index, layer in enumerate(block.layers):
        base = f"{prefix}.layer.{index}"
        values[f"{base}.thickness_a"] = layer.thickness_a
        values[f"{base}.roughness_a"] = layer.roughness_a
        values[f"{base}.density_scale"] = layer.density_scale
        base_target = layer.thickness_a if drift.target == "thickness" else layer.roughness_a
        for k in range(1, block.repeats):
            copy = f"{prefix}.repeat.{k}.layer.{index}.{drift.target}_a"
            values[copy] = base_target * (1.0 + drift.amount * coeffs[k])
    return values
