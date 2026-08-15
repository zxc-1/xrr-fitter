import pytest
from tests.support.drift_cases import drift_case, plain_periodic_structure
from tests.support.model_cases import prepared_data

from xrr_fitter.fit.drift import DRIFT_DATASET, rebind_drift_dataset, rebind_drift_rules
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ConstraintNode, ConstraintRule, ParameterReference, _iter_references


def _plain_member():
    return compile_fit_problem(
        prepared_data(),
        plain_periodic_structure(),
        InstrumentSpec(instrument_id="lab"),
        FitConfig.standard(11),
    )


def test_rebind_drift_dataset_remaps_sentinel_to_member():
    problem = compile_fit_problem(*drift_case())
    assert any(rule.target.dataset_id == DRIFT_DATASET for rule in problem.constraint_rules)
    rebound = rebind_drift_dataset(problem, "sample")
    assert rebound.constraint_rules
    for rule in rebound.constraint_rules:
        assert rule.target.dataset_id == "sample"
        assert all(ref.dataset_id == "sample" for ref in _iter_references(rule.expression))
    assert {rule.target.parameter_name for rule in rebound.constraint_rules} == {
        rule.target.parameter_name for rule in problem.constraint_rules
    }


def test_rebind_drift_dataset_is_noop_without_drift():
    problem = _plain_member()
    assert rebind_drift_dataset(problem, "sample") is problem


def test_rebind_drift_dataset_rejects_reserved_dataset_id():
    problem = compile_fit_problem(*drift_case())

    with pytest.raises(ValueError, match="reserved|dataset"):
        rebind_drift_dataset(problem, DRIFT_DATASET)


def test_rebind_drift_rules_rejects_reserved_dataset_id():
    problem = compile_fit_problem(*drift_case())

    with pytest.raises(ValueError, match="reserved|dataset"):
        rebind_drift_rules(problem.constraint_rules, DRIFT_DATASET)


def test_joint_compile_accepts_drifted_member():
    members = (compile_fit_problem(*drift_case()), compile_fit_problem(*drift_case()))
    joint = compile_joint_problem(("A", "B"), members, (), ())
    drift_rules = [rule for rule in joint.constraint_rules if ".repeat." in rule.target.parameter_name]
    assert drift_rules
    assert all(rule.target.dataset_id != DRIFT_DATASET for rule in joint.constraint_rules)
    assert {rule.target.dataset_id for rule in drift_rules} == {"A", "B"}
    for rule in drift_rules:
        assert all(ref.dataset_id in {"A", "B"} for ref in _iter_references(rule.expression))


def test_joint_compile_rejects_reserved_dataset_id():
    members = (_plain_member(), _plain_member())

    with pytest.raises(ValueError, match="reserved|dataset"):
        compile_joint_problem((DRIFT_DATASET, "B"), members, (), ())


def test_single_compile_rejects_reserved_dataset_id_in_user_constraints():
    data, structure, instrument, config = drift_case()
    rule = ConstraintRule(
        target=ParameterReference(DRIFT_DATASET, "component.0.density_scale"),
        expression=ConstraintNode(
            "ref",
            reference=ParameterReference(DRIFT_DATASET, "instrument.scale"),
        ),
    )

    with pytest.raises(ValueError, match="reserved|dataset"):
        compile_fit_problem(data, structure, instrument, config, (), (rule,))
