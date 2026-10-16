"""A value edit on one layer must not discard the parameter work on the others.

Structure edits used to be judged "topology changed" whenever ``components``
differed at all, so typing one thickness threw away every ``component.*``
setting, prior, sharing rule and constraint in the dataset.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from tests.support.model_cases import simple_structure

from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterFreedom,
    ParameterReference,
    ParameterSetting,
    SharingRule,
)
from xrr_fitter.model.structure import LayerSpec, MaterialSpec, StructureSpec
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.parameters import (
    describe_parameters,
    set_constraint_rules,
    set_parameter_settings,
    set_sharing_rules,
)
from xrr_fitter.services.projects import new_project
from xrr_fitter.services.structures import set_structure

SILICA = MaterialSpec("SiO2", "SiO2", 2.2)
ALUMINA = MaterialSpec("Al2O3", "Al2O3", 3.95)


def _source(path: Path, floor: float = 1e-5) -> Path:
    angles = np.linspace(0.1, 3.2, 64)
    path.write_bytes(xy_bytes(angles, np.geomspace(1.0, floor, angles.size)))
    return path


def _two_layer_structure() -> StructureSpec:
    base = simple_structure()
    top = LayerSpec("cap", ALUMINA, 30.0, roughness_a=2.0)
    return replace(base, components=(top, *base.components))


def _project(tmp_path: Path, structure: StructureSpec, dataset_id: str = "curve"):
    value = add_dataset(
        new_project(),
        _source(tmp_path / f"{dataset_id}.xy"),
        InstrumentSpec(instrument_id="value-edit"),
    )
    return set_structure(value, dataset_id, structure)


def _definition(value, name: str, dataset_id: str = "curve"):
    return next(item for item in describe_parameters(value, dataset_id) if item.name == name)


def _fixed(value, name: str, dataset_id: str = "curve") -> ParameterSetting:
    definition = _definition(value, name, dataset_id)
    return ParameterSetting(name, definition.initial, definition.lower, definition.upper, ParameterFreedom.FIXED)


def _with_layer(structure: StructureSpec, index: int, **changes) -> StructureSpec:
    components = list(structure.components)
    components[index] = replace(components[index], **changes)
    return replace(structure, components=tuple(components))


def _settings(value, dataset_id: str = "curve") -> dict[str, ParameterSetting]:
    dataset = next(item for item in value.datasets if item.dataset_id == dataset_id)
    return {setting.name: setting for setting in dataset.parameter_settings}


def test_editing_one_layer_keeps_a_fixed_setting_on_another_layer(tmp_path: Path) -> None:
    structure = _two_layer_structure()
    value = _project(tmp_path, structure)
    fixed = _fixed(value, "component.1.density_scale")
    value = set_parameter_settings(value, "curve", (fixed,))

    updated = set_structure(value, "curve", _with_layer(structure, 0, thickness_a=30.3))

    assert _settings(updated) == {fixed.name: fixed}


def test_editing_a_fixed_value_moves_the_setting_and_keeps_it_fixed(tmp_path: Path) -> None:
    structure = _two_layer_structure()
    value = _project(tmp_path, structure)
    value = set_parameter_settings(value, "curve", (_fixed(value, "component.1.thickness_a"),))

    updated = set_structure(value, "curve", _with_layer(structure, 1, thickness_a=25.0))

    setting = _settings(updated)["component.1.thickness_a"]
    assert setting.freedom is ParameterFreedom.FIXED
    assert setting.initial == 25.0
    assert setting.lower <= 25.0 <= setting.upper


def test_a_free_interval_follows_a_value_edit_that_stays_inside_it(tmp_path: Path) -> None:
    structure = _two_layer_structure()
    value = _project(tmp_path, structure)
    value = set_parameter_settings(value, "curve", (ParameterSetting("component.1.thickness_a", 20.0, 18.0, 30.0),))

    updated = set_structure(value, "curve", _with_layer(structure, 1, thickness_a=25.0))

    assert _settings(updated)["component.1.thickness_a"] == ParameterSetting(
        "component.1.thickness_a", 25.0, 18.0, 30.0
    )


def test_a_value_edit_outside_a_custom_interval_drops_only_that_setting(tmp_path: Path) -> None:
    structure = _two_layer_structure()
    value = _project(tmp_path, structure)
    fixed = _fixed(value, "component.0.roughness_a")
    interval = ParameterSetting("component.1.thickness_a", 20.0, 18.0, 30.0)
    value = set_parameter_settings(value, "curve", (fixed, interval))

    updated = set_structure(value, "curve", _with_layer(structure, 1, thickness_a=40.0))

    assert _settings(updated) == {fixed.name: fixed}


def test_a_value_edit_keeps_sharing_rules_on_the_edited_dataset(tmp_path: Path) -> None:
    structure = simple_structure()
    value = _project(tmp_path, structure, "first")
    value = add_dataset(value, _source(tmp_path / "second.xy", 1e-4), InstrumentSpec(instrument_id="value-edit"))
    value = set_structure(value, "second", structure)
    name = "component.0.thickness_a"
    rule = SharingRule("shared-film", (ParameterReference("first", name), ParameterReference("second", name)))
    value = set_sharing_rules(value, (rule,))

    updated = set_structure(value, "first", _with_layer(structure, 0, roughness_a=2.5))

    assert updated.sharing_rules == (rule,)


def test_a_value_edit_keeps_constraints_between_untouched_parameters(tmp_path: Path) -> None:
    structure = _two_layer_structure()
    value = _project(tmp_path, structure)
    rule = ConstraintRule(
        ParameterReference("curve", "component.1.density_scale"),
        ConstraintNode("ref", reference=ParameterReference("curve", "component.0.density_scale")),
    )
    value = set_constraint_rules(value, (rule,))

    updated = set_structure(value, "curve", _with_layer(structure, 0, thickness_a=31.0))

    assert updated.constraint_rules == (rule,)


def test_swapping_one_material_drops_only_that_layer_settings(tmp_path: Path) -> None:
    structure = _two_layer_structure()
    value = _project(tmp_path, structure)
    kept = _fixed(value, "component.1.density_scale")
    replaced = _fixed(value, "component.0.thickness_a")
    value = set_parameter_settings(value, "curve", (kept, replaced))
    nitride = MaterialSpec("Si3N4", "Si3N4", 3.17)

    updated = set_structure(value, "curve", _with_layer(structure, 0, name="Si3N4", material=nitride))

    assert _settings(updated) == {kept.name: kept}


def test_inserting_a_layer_still_drops_index_bound_component_settings(tmp_path: Path) -> None:
    structure = simple_structure()
    value = _project(tmp_path, structure)
    value = set_parameter_settings(value, "curve", (_fixed(value, "component.0.thickness_a"),))
    cap = LayerSpec("cap", ALUMINA, 30.0, roughness_a=2.0)

    updated = set_structure(value, "curve", replace(structure, components=(cap, *structure.components)))

    assert _settings(updated) == {}
