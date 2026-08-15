from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.model.parameters import PhysicalValueError
from xrr_fitter.model.structure import (
    DriftSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.stack import expand_structure, rebuild_structure

VAC = MaterialSpec("vacuum", None, None, sld_override_a2=0j)
MO = MaterialSpec("Mo", "Mo", 10.28)
SI = MaterialSpec("Si", "Si", 2.329)
WL = 1.5406
QZ = np.linspace(0.01, 0.30, 64)


def test_non_drift_periodic_keeps_matrix_power_fast_path() -> None:
    """无漂移块仍登记 PeriodicSpan（逐位不变的物理侧代理，spec line 103）。"""
    block = PeriodicBlock("mirror", (LayerSpec("Mo", MO, 50.0, roughness_a=3.0),), 3)
    stack = expand_structure(StructureSpec(VAC, (block,), SI, backing_roughness_a=2.0), WL)
    assert block.drift is None
    assert len(stack.periodic_spans) == 1 and stack.periodic_spans[0].repeats == 3


def test_drift_thickness_expansion_equals_hand_written_layers() -> None:
    """线性厚度漂移(amount=0.1) 逐副本厚度 50/55/60，与手工三层结构逐位一致，
    且 periodic_spans 为空——慢路径与手工展开必须给同一答案（spec line 101-102）。"""
    drifted = PeriodicBlock(
        "grade",
        (LayerSpec("Mo", MO, 50.0, roughness_a=3.0),),
        3,
        drift=DriftSpec(kind="linear", target="thickness", amount=0.1),
    )
    values = {
        "component.0.layer.0.thickness_a": 50.0,
        "component.0.layer.0.density_scale": 1.0,
        "component.0.layer.0.roughness_a": 3.0,
        "component.0.repeat.1.layer.0.thickness_a": 55.0,
        "component.0.repeat.2.layer.0.thickness_a": 60.0,
        "backing.roughness_a": 2.0,
    }
    rebuilt = rebuild_structure(StructureSpec(VAC, (drifted,), SI, backing_roughness_a=2.0), values)
    drift_stack = expand_structure(rebuilt, WL)

    hand = StructureSpec(
        VAC,
        tuple(LayerSpec("Mo", MO, thickness, roughness_a=3.0) for thickness in (50.0, 55.0, 60.0)),
        SI,
        backing_roughness_a=2.0,
    )
    hand_stack = expand_structure(hand, WL)

    assert drift_stack.periodic_spans == ()
    assert np.array_equal(drift_stack.thickness_a, hand_stack.thickness_a)
    assert np.array_equal(drift_stack.roughness_a, hand_stack.roughness_a)
    assert np.array_equal(drift_stack.sld_a2, hand_stack.sld_a2)
    assert np.array_equal(
        parratt_reflectivity(QZ, drift_stack),
        parratt_reflectivity(QZ, hand_stack),
    )


def test_roughness_drift_over_neighbor_limit_raises_physical_value_error() -> None:
    """粗糙度目标漂移把某副本推过 _validate_roughness 邻层上限时报错（spec line 90-91）。"""
    drifted = PeriodicBlock(
        "grade",
        (LayerSpec("Mo", MO, 50.0, roughness_a=20.0),),
        3,
        drift=DriftSpec(kind="linear", target="roughness", amount=0.5),
    )
    values = {
        "component.0.layer.0.thickness_a": 50.0,
        "component.0.layer.0.density_scale": 1.0,
        "component.0.layer.0.roughness_a": 20.0,
        "component.0.repeat.1.layer.0.roughness_a": 30.0,
        "component.0.repeat.2.layer.0.roughness_a": 40.0,
        "backing.roughness_a": 2.0,
    }
    rebuilt = rebuild_structure(StructureSpec(VAC, (drifted,), SI, backing_roughness_a=2.0), values)
    with pytest.raises(PhysicalValueError):
        expand_structure(rebuilt, WL)
