"""Deterministic routing identity for automatic dataset fitting.

Automatic transaction orchestration consumes these values to decide which
datasets can share a fit.  The routing identity intentionally records only
topology and acquisition settings that change the physical fitting problem;
fitted parameter values remain outside the grouping contract.
"""

from __future__ import annotations

import hashlib
import json

from xrr_fitter.model.automation import MeasurementPreset
from xrr_fitter.model.structure import GradientLayerSpec, LayerSpec, PeriodicBlock


def _material_signature(material) -> tuple[object, ...]:
    """Return the material identity relevant to automatic sharing."""

    return (
        material.name,
        material.formula,
        material.sld_override_a2 is not None,
    )


def _component_signature(component) -> tuple[object, ...]:
    """Describe one structure component for physical-route hashing."""

    if isinstance(component, LayerSpec):
        return (component.name, *_material_signature(component.material))
    if isinstance(component, PeriodicBlock):
        return (
            component.name,
            tuple((layer.name, *_material_signature(layer.material)) for layer in component.layers),
            component.repeats,
            component.top_roughness_a is not None,
        )
    if isinstance(component, GradientLayerSpec):
        return (
            component.name,
            (component.upper_sld_a2.real, component.upper_sld_a2.imag),
            (component.lower_sld_a2.real, component.lower_sld_a2.imag),
            component.microslab_max_a,
        )
    raise TypeError(f"unsupported automatic structure component: {type(component).__name__}")


def _dataclass_values(value) -> tuple[object, ...]:
    """Read declared dataclass fields in stable definition order."""

    return tuple(getattr(value, field) for field in value.__dataclass_fields__)


def _canonical_json(value: object) -> str:
    """Serialize a signature payload with deterministic key ordering."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def automatic_physical_signature(dataset, preset: MeasurementPreset) -> str:
    """Hash the behavior-changing automatic grouping contract."""

    if dataset.structure is None:
        raise ValueError(f"dataset {dataset.dataset_id} has no structure")
    if not isinstance(preset, MeasurementPreset):
        raise TypeError("preset must be MeasurementPreset")
    structure = dataset.structure
    payload = {
        "layers": tuple(_component_signature(component) for component in structure.components),
        "backing": _material_signature(structure.backing),
        "beam": _dataclass_values(dataset.beam),
        "import_angle_offset_deg": dataset.import_angle_offset_deg,
        "instrument": _dataclass_values(dataset.instrument),
        "preset": (
            _dataclass_values(preset.beam),
            _dataclass_values(preset.instrument),
            preset.import_angle_offset_deg,
        ),
        "structure_modes": tuple(type(component).__name__ for component in structure.components),
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def automatic_group_id(import_batch_id: str, signature: str) -> str:
    """Derive a stable group identifier from batch and physical identity."""

    payload = {"import_batch_id": import_batch_id, "physical_signature": signature}
    digest = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    return f"automatic-{digest[:24]}"
