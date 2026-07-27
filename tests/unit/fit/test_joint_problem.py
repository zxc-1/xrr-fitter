from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterReference, SharingRule


SHARED_NAME = "component.0.density_scale"
LOCAL_NAME = "component.0.thickness_a"


def _problem(*, seed: int, size: int = 40):
    config = replace(
        FitConfig.fast(seed),
        final_seed_count=4,
        budget=SearchBudget(0, 0, 5, 1, 1),
        local_workers=1,
        scale_prior_enabled=False,
    )
    return compile_fit_problem(
        prepared_data(size=size),
        simple_structure(),
        InstrumentSpec(footprint_mode="none", instrument_id="shared-lab"),
        config,
    )


def _rules() -> tuple[SharingRule, ...]:
    return (
        SharingRule(
            "film-thickness",
            (
                ParameterReference("left", SHARED_NAME),
                ParameterReference("right", SHARED_NAME),
            ),
        ),
    )


def _compile():
    api = import_module("xrr_fitter.fit.joint_problem")
    return api.compile_joint_problem(
        ("left", "right"),
        (_problem(seed=809), _problem(seed=809, size=48)),
        _rules(),
    )


def _coordinate_index(problem, name: str) -> int:
    return next(
        index for index, coordinate in enumerate(problem.variables)
        if coordinate.name == name
    )


def test_joint_problem_has_stable_global_layout_and_shared_scatter_identity() -> None:
    joint = _compile()
    left_index = _coordinate_index(joint.problems[0], SHARED_NAME)
    right_index = _coordinate_index(joint.problems[1], SHARED_NAME)

    assert joint.dataset_ids == ("left", "right")
    assert joint.scatter_maps[0][left_index] == joint.scatter_maps[1][right_index]
    shared_global = joint.scatter_maps[0][left_index]
    assert joint.global_variables[shared_global].sharing_key == "film-thickness"
    assert joint.global_variables[shared_global].members == (
        ParameterReference("left", SHARED_NAME),
        ParameterReference("right", SHARED_NAME),
    )
    assert len(joint.layout_fingerprint) == 64


def test_joint_problem_requires_strict_dataset_problem_pairing() -> None:
    api = import_module("xrr_fitter.fit.joint_problem")

    with pytest.raises(ValueError, match="dataset|problem|length|pair"):
        api.compile_joint_problem(
            ("left", "right"),
            (_problem(seed=811),),
            (),
        )


def test_joint_problem_shares_density_but_keeps_thickness_dataset_local() -> None:
    joint = _compile()
    left_density = _coordinate_index(joint.problems[0], SHARED_NAME)
    right_density = _coordinate_index(joint.problems[1], SHARED_NAME)
    left_thickness = _coordinate_index(joint.problems[0], LOCAL_NAME)
    right_thickness = _coordinate_index(joint.problems[1], LOCAL_NAME)

    assert joint.scatter_maps[0][left_density] == joint.scatter_maps[1][right_density]
    assert joint.scatter_maps[0][left_thickness] != joint.scatter_maps[1][right_thickness]


def test_joint_scatter_copies_shared_and_local_coordinates_without_aliasing() -> None:
    api = import_module("xrr_fitter.fit.joint_sharing")
    joint = _compile()
    global_unit = np.linspace(0.1, 0.9, len(joint.global_variables))

    local = api.scatter_joint_vector(joint, global_unit)

    assert len(local) == 2
    for vector, problem in zip(local, joint.problems, strict=True):
        assert vector.shape == (len(problem.variables),)
        assert not vector.flags.writeable
    left_shared = _coordinate_index(joint.problems[0], SHARED_NAME)
    right_shared = _coordinate_index(joint.problems[1], SHARED_NAME)
    assert local[0][left_shared] == local[1][right_shared]
    assert not np.shares_memory(local[0], global_unit)
    assert not np.shares_memory(local[1], global_unit)


def test_joint_layout_fingerprint_binds_dataset_and_sharing_order() -> None:
    api = import_module("xrr_fitter.fit.joint_problem")
    left = _problem(seed=821)
    right = _problem(seed=821, size=48)
    baseline = api.compile_joint_problem(("left", "right"), (left, right), _rules())
    repeated = api.compile_joint_problem(("left", "right"), (left, right), _rules())
    reversed_layout = api.compile_joint_problem(
        ("right", "left"),
        (right, left),
        (
            SharingRule(
                "film-thickness",
                (
                    ParameterReference("right", SHARED_NAME),
                    ParameterReference("left", SHARED_NAME),
                ),
            ),
        ),
    )

    assert baseline.layout_fingerprint == repeated.layout_fingerprint
    assert baseline.layout_fingerprint != reversed_layout.layout_fingerprint


def test_joint_sharing_rejects_two_members_from_one_dataset() -> None:
    api = import_module("xrr_fitter.fit.joint_problem")
    rule = SharingRule(
        "ambiguous-left",
        (
            ParameterReference("left", SHARED_NAME),
            ParameterReference("left", LOCAL_NAME),
            ParameterReference("right", SHARED_NAME),
        ),
    )

    with pytest.raises(ValueError, match="sharing|dataset|member"):
        api.compile_joint_problem(
            ("left", "right"),
            (_problem(seed=827), _problem(seed=827)),
            (rule,),
        )


def test_joint_sharing_rejects_missing_and_incompatible_coordinates() -> None:
    api = import_module("xrr_fitter.fit.joint_problem")
    left = _problem(seed=839)
    right = _problem(seed=839)
    coordinate_index = _coordinate_index(right, SHARED_NAME)
    definition_index = right.variables[coordinate_index].parameter_index
    incompatible = replace(
        right,
        variables=(
            *right.variables[:coordinate_index],
            replace(right.variables[coordinate_index], transform="log"),
            *right.variables[coordinate_index + 1 :],
        ),
        parameter_definitions=(
            *right.parameter_definitions[:definition_index],
            replace(right.parameter_definitions[definition_index], transform="log"),
            *right.parameter_definitions[definition_index + 1 :],
        ),
    )

    with pytest.raises(ValueError, match="sharing|coordinate|transform|compatible"):
        api.compile_joint_problem(("left", "right"), (left, incompatible), _rules())

    missing = SharingRule(
        "missing",
        (
            ParameterReference("left", SHARED_NAME),
            ParameterReference("right", "component.9.missing"),
        ),
    )
    with pytest.raises(ValueError, match="sharing|parameter|missing"):
        api.compile_joint_problem(("left", "right"), (left, right), (missing,))


def test_joint_sharing_rejects_incompatible_parameter_families() -> None:
    api = import_module("xrr_fitter.fit.joint_problem")
    left = _problem(seed=841)
    right = _problem(seed=841)
    density = next(value for value in left.parameter_definitions if value.name == SHARED_NAME)
    background_index = next(
        index
        for index, value in enumerate(right.parameter_definitions)
        if value.name == "instrument.background"
    )
    background = right.parameter_definitions[background_index]
    compatible_shape = replace(
        background,
        initial=density.initial,
        lower=density.lower,
        upper=density.upper,
        transform=density.transform,
        unit=density.unit,
    )
    right = replace(
        right,
        parameter_definitions=(
            *right.parameter_definitions[:background_index],
            compatible_shape,
            *right.parameter_definitions[background_index + 1 :],
        ),
    )
    rule = SharingRule(
        "cross-family",
        (
            ParameterReference("left", SHARED_NAME),
            ParameterReference("right", "instrument.background"),
        ),
    )

    with pytest.raises(ValueError, match="sharing|family|compatible|instrument"):
        api.compile_joint_problem(("left", "right"), (left, right), (rule,))


def test_joint_sharing_rejects_instrument_parameters_from_different_instruments() -> None:
    api = import_module("xrr_fitter.fit.joint_problem")
    left = _problem(seed=843)
    right = replace(
        _problem(seed=843),
        instrument=replace(_problem(seed=843).instrument, instrument_id="other-lab"),
    )
    rule = SharingRule(
        "shared-scale",
        (
            ParameterReference("left", "instrument.scale"),
            ParameterReference("right", "instrument.scale"),
        ),
    )

    with pytest.raises(ValueError, match="sharing|instrument|identity|semantic"):
        api.compile_joint_problem(("left", "right"), (left, right), (rule,))


@pytest.mark.parametrize(
    "global_unit",
    [
        pytest.param(np.asarray(0.5), id="scalar"),
        pytest.param(np.asarray([np.nan]), id="nonfinite"),
        pytest.param(np.asarray([-0.1]), id="below-unit-bounds"),
        pytest.param(np.asarray([1.1]), id="above-unit-bounds"),
    ],
)
def test_joint_scatter_rejects_invalid_global_vectors(global_unit: np.ndarray) -> None:
    api = import_module("xrr_fitter.fit.joint_sharing")
    joint = _compile()

    if global_unit.ndim == 1 and global_unit.size == 1:
        global_unit = np.resize(global_unit, len(joint.global_variables))
    with pytest.raises(ValueError, match="global|unit|shape|finite|bounds"):
        api.scatter_joint_vector(joint, global_unit)
