from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support.model_cases import dataset_project, final_fit_result, project, simple_structure

from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterPrior,
    ParameterReference,
    ParameterSetting,
    PriorSpec,
    SharingRule,
)
from xrr_fitter.model.project import ProjectUiState
from xrr_fitter.model.structure import LayerSpec, MaterialSpec, PeriodicBlock, StructureSpec
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.parameters import (
    _reconciled_parameter_sidecars,
    describe_parameters,
    set_constraint_rules,
    set_parameter_priors,
    set_parameter_settings,
    set_sharing_rules,
    validate_constraint_rules,
    validate_parameter_priors,
    validate_parameter_settings,
    validate_sharing_rules,
)
from xrr_fitter.services.projects import new_project, set_batch_mode
from xrr_fitter.services.structures import set_structure


def _source(path: Path) -> Path:
    angles = np.linspace(0.1, 3.2, 64)
    path.write_bytes(xy_bytes(angles, np.geomspace(1.0, 1e-5, angles.size)))
    return path


def _structured_project(tmp_path: Path):
    value = add_dataset(
        new_project(),
        _source(tmp_path / "curve.xy"),
        InstrumentSpec(instrument_id="parameter-service"),
    )
    return set_structure(value, "curve", simple_structure())


def test_parameter_settings_validate_without_reordering_and_reject_unknown(
    tmp_path: Path,
) -> None:
    value = _structured_project(tmp_path)
    definitions = describe_parameters(value, "curve")
    thickness = next(item for item in definitions if item.name == "component.0.thickness_a")
    setting = ParameterSetting(
        thickness.name,
        thickness.initial,
        thickness.lower,
        thickness.upper,
    )

    assert validate_parameter_settings(definitions, (setting,)) == (setting,)
    with pytest.raises(ValueError, match="unknown parameter setting"):
        validate_parameter_settings(
            definitions,
            (ParameterSetting("unknown", 1.0, 0.0, 2.0),),
        )


def test_set_parameter_settings_persists_and_invalidates_only_fit_state(
    tmp_path: Path,
) -> None:
    value = _structured_project(tmp_path)
    result = final_fit_result()
    dataset = replace(value.datasets[0], last_valid_result=result)
    value = replace(
        value,
        datasets=(dataset,),
        ui_state=ProjectUiState(
            active_dataset_id="curve",
            selected_candidate_ids=(("curve", "candidate-0"),),
        ),
    )
    definition = next(item for item in describe_parameters(value, "curve") if item.name == "component.0.thickness_a")
    setting = ParameterSetting(definition.name, 90.0, 20.0, 180.0)

    updated = set_parameter_settings(value, "curve", (setting,))

    assert updated.datasets[0].parameter_settings == (setting,)
    assert updated.datasets[0].structure is dataset.structure
    assert updated.datasets[0].structure_evidence is dataset.structure_evidence
    assert updated.datasets[0].last_valid_result is None
    assert updated.ui_state.selected_candidate_ids == ()


def test_set_parameter_settings_reconciles_priors_against_effective_bounds(
    tmp_path: Path,
) -> None:
    value = _structured_project(tmp_path)
    thickness = _thickness(value)
    prior = ParameterPrior(thickness.name, PriorSpec("normal", (50.0, 5.0)))
    value = set_parameter_priors(value, "curve", (prior,))
    setting = ParameterSetting(thickness.name, 20.0, 10.0, 30.0)

    updated = set_parameter_settings(value, "curve", (setting,))

    assert updated.datasets[0].parameter_settings == (setting,)
    assert updated.datasets[0].parameter_priors == ()


def test_sharing_validation_is_pure_and_does_not_read_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = project(dataset_project("first"), dataset_project("second"))
    rule = SharingRule(
        "shared-thickness",
        (
            ParameterReference("first", "component.0.thickness_a"),
            ParameterReference("second", "component.0.thickness_a"),
        ),
    )

    def fail_read(_path: Path) -> bytes:
        raise AssertionError("sharing declaration validation read source")

    monkeypatch.setattr(Path, "read_bytes", fail_read)

    assert validate_sharing_rules(value, (rule,)) == (rule,)


def test_sharing_validation_rejects_duplicate_ownership_and_allows_same_dataset() -> None:
    value = project(dataset_project("first"), dataset_project("second"))
    shared = ParameterReference("first", "component.0.thickness_a")
    first = SharingRule(
        "first-rule",
        (shared, ParameterReference("second", "component.0.thickness_a")),
    )
    second = SharingRule(
        "second-rule",
        (shared, ParameterReference("second", "component.0.density_scale")),
    )
    same_dataset = SharingRule(
        "same-dataset",
        (
            ParameterReference("first", "component.0.thickness_a"),
            ParameterReference("first", "component.0.density_scale"),
        ),
    )

    with pytest.raises(ValueError, match="multiple|ownership"):
        validate_sharing_rules(value, (first, second))
    assert validate_sharing_rules(value, (same_dataset,)) == (same_dataset,)


def test_set_sharing_rules_invalidates_affected_fit_state() -> None:
    result = final_fit_result()
    value = project(
        dataset_project("first", result=result),
        dataset_project("second", result=result),
    )
    value = replace(
        value,
        ui_state=ProjectUiState(
            selected_candidate_ids=(
                ("first", "candidate-0"),
                ("second", "candidate-0"),
            )
        ),
    )
    rule = SharingRule(
        "shared-thickness",
        (
            ParameterReference("first", "component.0.thickness_a"),
            ParameterReference("second", "component.0.thickness_a"),
        ),
    )

    updated = set_sharing_rules(value, (rule,))

    assert updated.sharing_rules == (rule,)
    assert all(dataset.last_valid_result is None for dataset in updated.datasets)
    assert updated.ui_state.selected_candidate_ids == ()


def _thickness(value, name: str = "component.0.thickness_a"):
    return next(item for item in describe_parameters(value, "curve") if item.name == name)


def test_set_parameter_priors_returns_new_project(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    thickness = _thickness(value)
    prior = ParameterPrior(thickness.name, PriorSpec("normal", (thickness.initial, 5.0)))

    updated = set_parameter_priors(value, "curve", (prior,))

    assert updated is not value
    assert updated.datasets[0].parameter_priors == (prior,)


def test_set_parameter_priors_returns_same_object_when_unchanged(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    thickness = _thickness(value)
    prior = ParameterPrior(thickness.name, PriorSpec("normal", (thickness.initial, 5.0)))
    updated = set_parameter_priors(value, "curve", (prior,))

    assert set_parameter_priors(updated, "curve", (prior,)) is updated


def test_set_parameter_priors_invalidates_fit_state(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    result = final_fit_result()
    dataset = replace(value.datasets[0], last_valid_result=result)
    value = replace(
        value,
        datasets=(dataset,),
        ui_state=ProjectUiState(
            active_dataset_id="curve",
            selected_candidate_ids=(("curve", "candidate-0"),),
        ),
    )
    thickness = _thickness(value)
    prior = ParameterPrior(thickness.name, PriorSpec("normal", (thickness.initial, 5.0)))

    updated = set_parameter_priors(value, "curve", (prior,))

    assert updated.datasets[0].parameter_priors == (prior,)
    assert updated.datasets[0].structure is dataset.structure
    assert updated.datasets[0].last_valid_result is None
    assert updated.ui_state.selected_candidate_ids == ()


def test_validate_parameter_priors_rejects_unknown_parameter(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    definitions = describe_parameters(value, "curve")

    with pytest.raises(ValueError, match="unknown parameter name"):
        validate_parameter_priors(definitions, (ParameterPrior("unknown", PriorSpec("uniform")),))


def test_validate_parameter_priors_rejects_duplicate_names(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    definitions = describe_parameters(value, "curve")
    thickness = _thickness(value)
    prior = ParameterPrior(thickness.name, PriorSpec("uniform"))

    with pytest.raises(ValueError, match="prior names must be unique"):
        validate_parameter_priors(definitions, (prior, prior))


def test_validate_parameter_priors_rejects_center_outside_bounds(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    definitions = describe_parameters(value, "curve")
    thickness = _thickness(value)
    prior = ParameterPrior(thickness.name, PriorSpec("normal", (thickness.upper + 100.0, 5.0)))

    with pytest.raises(ValueError, match="within bounds"):
        validate_parameter_priors(definitions, (prior,))


def test_validate_parameter_priors_rejects_constraint_targets(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    rule = _scaled_rule(
        "curve",
        "component.0.density_scale",
        "component.0.thickness_a",
        0.01,
    )
    value = set_constraint_rules(value, (rule,))
    definitions = describe_parameters(value, "curve")

    with pytest.raises(ValueError, match="constrained parameter"):
        validate_parameter_priors(
            definitions,
            (ParameterPrior(rule.target.parameter_name, PriorSpec("uniform")),),
        )


def test_validate_parameter_priors_scores_roughness_prior_in_unit_fraction(tmp_path: Path) -> None:
    # roughness_fraction 先验落在单位分数 [0,1];一个落在 Å 物理界内但 >1 的中心(在
    # 分数坐标里非法)必须被拒,以证明服务层继承了定义层的坐标系修正,而不是拿 Å 物理界
    # 去校验分数先验。
    value = _structured_project(tmp_path)
    definitions = describe_parameters(value, "curve")
    roughness = next(item for item in definitions if item.transform == "roughness_fraction")
    assert roughness.upper > 1.0
    center = 0.5 * (roughness.lower + roughness.upper)
    prior = ParameterPrior(roughness.name, PriorSpec("normal", (center, 1.0)))

    with pytest.raises(ValueError, match="within bounds"):
        validate_parameter_priors(definitions, (prior,))


def test_compiled_definitions_carry_stored_priors(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    thickness = _thickness(value)
    spec = PriorSpec("normal", (thickness.initial, 5.0))

    updated = set_parameter_priors(value, "curve", (ParameterPrior(thickness.name, spec),))

    carried = _thickness(updated)
    assert carried.prior == spec


def test_reconcile_parameter_sidecars_keeps_only_valid_unique_entries(
    tmp_path: Path,
) -> None:
    value = _structured_project(tmp_path)
    definitions = describe_parameters(value, "curve")
    thickness = _thickness(value)
    setting = ParameterSetting(
        thickness.name,
        thickness.initial,
        thickness.lower,
        thickness.upper,
        thickness.locked,
    )
    prior = ParameterPrior(thickness.name, PriorSpec("normal", (thickness.initial, 5.0)))
    unknown_setting = ParameterSetting("component.99.thickness_a", 10.0, 2.0, 20.0)
    unknown_prior = ParameterPrior("component.99.thickness_a", PriorSpec("uniform"))

    settings, priors = _reconciled_parameter_sidecars(
        definitions,
        (setting, setting, unknown_setting),
        (prior, prior, unknown_prior),
    )

    assert settings == (setting,)
    assert priors == (prior,)


# --- Task 7: expression constraint declarations (services layer) -----------


def _second_source(path: Path) -> Path:
    angles = np.linspace(0.1, 3.2, 64)
    path.write_bytes(xy_bytes(angles, np.geomspace(1.0, 1e-4, angles.size)))
    return path


def _two_dataset_project(tmp_path: Path):
    value = add_dataset(
        new_project(),
        _source(tmp_path / "first.xy"),
        InstrumentSpec(instrument_id="first-instrument"),
    )
    value = set_structure(value, "first", simple_structure())
    value = add_dataset(
        value,
        _second_source(tmp_path / "second.xy"),
        InstrumentSpec(instrument_id="second-instrument"),
    )
    return set_structure(value, "second", simple_structure())


def _periodic_structure() -> StructureSpec:
    air = MaterialSpec("Air", None, None, 0.0j)
    silicon = MaterialSpec("Si", "Si", 2.329)
    silica = MaterialSpec("SiO2", "SiO2", 2.2)
    block = PeriodicBlock("stack", (LayerSpec("film", silica, 20.0, roughness_a=2.0),), repeats=3)
    return StructureSpec(
        fronting=air,
        components=(block,),
        backing=silicon,
        backing_roughness_a=3.0,
    )


def _ref(dataset_id: str, name: str) -> ConstraintNode:
    return ConstraintNode("ref", reference=ParameterReference(dataset_id, name))


def _scaled_rule(dataset_id: str, target: str, source: str, factor: float) -> ConstraintRule:
    return ConstraintRule(
        target=ParameterReference(dataset_id, target),
        expression=ConstraintNode(
            "mul",
            operands=(_ref(dataset_id, source), ConstraintNode("const", value=factor)),
        ),
    )


def test_validate_constraint_rules_accepts_a_well_formed_rule(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    rule = _scaled_rule("curve", "component.0.density_scale", "component.0.thickness_a", 0.01)

    assert validate_constraint_rules(value, (rule,)) == (rule,)


def test_describe_parameters_marks_persisted_constraint_target_as_constrained(
    tmp_path: Path,
) -> None:
    value = _structured_project(tmp_path)
    rule = _scaled_rule(
        "curve",
        "component.0.density_scale",
        "component.0.thickness_a",
        0.01,
    )
    value = set_constraint_rules(value, (rule,))

    definitions = {definition.name: definition for definition in describe_parameters(value, "curve")}

    assert definitions[rule.target.parameter_name].constrained is True


def test_set_constraint_rules_removes_prior_from_the_new_target(
    tmp_path: Path,
) -> None:
    value = _structured_project(tmp_path)
    target = next(
        definition
        for definition in describe_parameters(value, "curve")
        if definition.name == "component.0.density_scale"
    )
    value = set_parameter_priors(
        value,
        "curve",
        (ParameterPrior(target.name, PriorSpec("uniform")),),
    )
    rule = _scaled_rule(
        "curve",
        target.name,
        "component.0.thickness_a",
        0.01,
    )

    updated = set_constraint_rules(value, (rule,))

    assert updated.datasets[0].parameter_priors == ()


def test_describe_parameters_marks_cross_dataset_target_without_local_compile(
    tmp_path: Path,
) -> None:
    value = _two_dataset_project(tmp_path)
    rule = ConstraintRule(
        target=ParameterReference("second", "component.0.density_scale"),
        expression=_ref("first", "component.0.density_scale"),
    )
    value = set_constraint_rules(value, (rule,))

    definitions = {definition.name: definition for definition in describe_parameters(value, "second")}

    assert definitions[rule.target.parameter_name].constrained is True


def test_validate_constraint_rules_rejects_non_constraint_values(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)

    with pytest.raises(TypeError, match="ConstraintRule"):
        validate_constraint_rules(value, ("not-a-rule",))


def test_validate_constraint_rules_reports_cycle_path(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    first = ConstraintRule(
        target=ParameterReference("curve", "component.0.thickness_a"),
        expression=_ref("curve", "component.0.density_scale"),
    )
    second = ConstraintRule(
        target=ParameterReference("curve", "component.0.density_scale"),
        expression=_ref("curve", "component.0.thickness_a"),
    )

    with pytest.raises(ValueError, match="cycle.*->"):
        validate_constraint_rules(value, (first, second))


def test_validate_constraint_rules_rejects_duplicate_targets(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    first = ConstraintRule(
        target=ParameterReference("curve", "component.0.density_scale"),
        expression=ConstraintNode("const", value=0.8),
    )
    second = replace(first, expression=ConstraintNode("const", value=0.9))

    with pytest.raises(ValueError, match="constraint targets.*unique"):
        validate_constraint_rules(value, (first, second))


def test_validate_constraint_rules_rejects_unknown_parameter(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    rule = ConstraintRule(
        target=ParameterReference("curve", "component.0.thickness_a"),
        expression=_ref("curve", "component.99.missing"),
    )

    with pytest.raises(ValueError, match="unknown parameter"):
        validate_constraint_rules(value, (rule,))


def test_validate_constraint_rules_rejects_non_roughness_target_reading_roughness(
    tmp_path: Path,
) -> None:
    value = _structured_project(tmp_path)
    roughness = next(item for item in describe_parameters(value, "curve") if item.transform == "roughness_fraction")
    rule = ConstraintRule(
        target=ParameterReference("curve", "component.0.thickness_a"),
        expression=_ref("curve", roughness.name),
    )

    with pytest.raises(ValueError, match="roughness"):
        validate_constraint_rules(value, (rule,))


def test_validate_constraint_rules_rejects_integer_independent_variable(tmp_path: Path) -> None:
    value = add_dataset(
        new_project(),
        _source(tmp_path / "curve.xy"),
        InstrumentSpec(instrument_id="periodic-instrument"),
    )
    value = set_structure(value, "curve", _periodic_structure())
    names = {item.name for item in describe_parameters(value, "curve")}
    assert "component.0.repeats" in names
    rule = ConstraintRule(
        target=ParameterReference("curve", "component.0.layer.0.thickness_a"),
        expression=_ref("curve", "component.0.repeats"),
    )

    with pytest.raises(ValueError, match="integer"):
        validate_constraint_rules(value, (rule,))


def test_validate_constraint_rules_rejects_conflict_with_sharing(tmp_path: Path) -> None:
    value = _two_dataset_project(tmp_path)
    shared = SharingRule(
        "shared-density",
        (
            ParameterReference("first", "component.0.density_scale"),
            ParameterReference("second", "component.0.density_scale"),
        ),
    )
    value = set_sharing_rules(value, (shared,))
    rule = _scaled_rule("first", "component.0.density_scale", "component.0.thickness_a", 0.01)

    with pytest.raises(ValueError, match="sharing|both"):
        validate_constraint_rules(value, (rule,))


def test_set_constraint_rules_returns_same_object_when_unchanged(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    rule = _scaled_rule("curve", "component.0.density_scale", "component.0.thickness_a", 0.01)
    once = set_constraint_rules(value, (rule,))

    assert once.constraint_rules == (rule,)
    assert set_constraint_rules(once, (rule,)) is once


def test_set_constraint_rules_invalidates_target_dataset_fit_state(tmp_path: Path) -> None:
    value = _structured_project(tmp_path)
    result = final_fit_result()
    dataset = replace(value.datasets[0], last_valid_result=result)
    value = replace(
        value,
        datasets=(dataset,),
        ui_state=ProjectUiState(
            active_dataset_id="curve",
            selected_candidate_ids=(("curve", "candidate-0"),),
        ),
    )
    rule = _scaled_rule("curve", "component.0.density_scale", "component.0.thickness_a", 0.01)

    updated = set_constraint_rules(value, (rule,))

    assert updated.constraint_rules == (rule,)
    assert updated.datasets[0].last_valid_result is None
    assert updated.ui_state.selected_candidate_ids == ()


def test_set_constraint_rules_invalidates_every_referenced_dataset(tmp_path: Path) -> None:
    value = _two_dataset_project(tmp_path)
    # 目标在 second、自变量在 first 的跨数据集规则:两侧数据集都必须进入失效面(修正9)。
    rule = ConstraintRule(
        target=ParameterReference("second", "component.0.density_scale"),
        expression=_ref("first", "component.0.density_scale"),
    )

    updated = set_constraint_rules(value, (rule,))

    assert updated.constraint_rules == (rule,)
    assert updated.datasets[0] is not value.datasets[0]
    assert updated.datasets[1] is not value.datasets[1]


def test_set_constraint_rules_spreads_to_all_datasets_in_joint_mode(tmp_path: Path) -> None:
    value = set_batch_mode(_two_dataset_project(tmp_path), "joint")
    # 规则只触及 first;joint 模式仍把失效面扩散到全部数据集,未被引用的 second 也被重建。
    rule = _scaled_rule("first", "component.0.density_scale", "component.0.thickness_a", 0.01)

    updated = set_constraint_rules(value, (rule,))

    assert updated.constraint_rules == (rule,)
    assert updated.datasets[1].dataset_id == "second"
    assert updated.datasets[1] is not value.datasets[1]
