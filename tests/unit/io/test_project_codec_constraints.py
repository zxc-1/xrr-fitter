"""Project-codec contracts for expression constraints."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module

import pytest
from tests.support.model_cases import project

from xrr_fitter.io.project_codec import (
    ProjectSchemaError,
    project_from_bytes,
    project_from_dict,
    project_to_bytes,
    project_to_dict,
)
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterDefinition,
    ParameterReference,
)


def _constraint_definition(*, constrained: bool = False) -> ParameterDefinition:
    return ParameterDefinition(
        name="component.0.thickness_a",
        display_name="thickness",
        unit="A",
        category="structure",
        initial=10.0,
        lower=2.0,
        upper=50.0,
        transform="linear",
        locked=False,
        constrained=constrained,
    )


def test_parameter_definition_dict_omits_default_constrained_flag() -> None:
    candidates = import_module("xrr_fitter.io.codec_candidates")

    payload = candidates._parameter_definition_to_dict(_constraint_definition())

    assert "constrained" not in payload


def test_legacy_parameter_definition_without_constrained_decodes_false() -> None:
    candidates = import_module("xrr_fitter.io.codec_candidates")
    payload = candidates._parameter_definition_to_dict(_constraint_definition())
    payload.pop("constrained", None)

    restored = candidates._parameter_definition_from_dict(payload)

    assert restored.constrained is False


def test_parameter_definition_dict_round_trips_constrained_flag() -> None:
    candidates = import_module("xrr_fitter.io.codec_candidates")
    definition = _constraint_definition(constrained=True)

    payload = candidates._parameter_definition_to_dict(definition)
    restored = candidates._parameter_definition_from_dict(payload)

    assert payload["constrained"] is True
    assert restored.constrained is True


def _constraint_reference(parameter_name: str, dataset_id: str = "curve") -> ParameterReference:
    return ParameterReference(dataset_id, parameter_name)


def _multi_level_constraint_rule() -> ConstraintRule:
    expression = ConstraintNode(
        "add",
        operands=(
            ConstraintNode(
                "mul",
                operands=(
                    ConstraintNode("const", value=2.0),
                    ConstraintNode("ref", reference=_constraint_reference("t1")),
                ),
            ),
            ConstraintNode(
                "pow",
                operands=(
                    ConstraintNode("ref", reference=_constraint_reference("t2")),
                    ConstraintNode("const", value=2.0),
                ),
            ),
        ),
    )
    return ConstraintRule(target=_constraint_reference("t0"), expression=expression)


def test_project_round_trips_multi_level_constraint_tree() -> None:
    original = replace(project(), constraint_rules=(_multi_level_constraint_rule(),))

    restored = project_from_bytes(project_to_bytes(original))

    assert restored.constraint_rules == original.constraint_rules


def test_project_without_constraints_omits_the_key() -> None:
    payload = project_to_dict(project())

    assert "constraint_rules" not in payload
    assert b"constraint_rules" not in project_to_bytes(project())


def test_legacy_project_without_constraint_rules_decodes_to_empty() -> None:
    payload = project_to_dict(replace(project(), constraint_rules=(_multi_level_constraint_rule(),)))
    payload.pop("constraint_rules", None)

    restored = project_from_dict(payload)

    assert restored.constraint_rules == ()


def _deep_constraint_node(depth: int) -> dict[str, object]:
    node: dict[str, object] = {
        "op": "ref",
        "reference": {"dataset_id": "curve", "parameter_name": "t1"},
    }
    for _ in range(depth):
        node = {"op": "mul", "operands": [{"op": "const", "value": 2.0}, node]}
    return node


def _deep_constraint_json(depth: int) -> str:
    node = '{"op":"ref","reference":{"dataset_id":"curve","parameter_name":"t1"}}'
    for _ in range(depth):
        node = '{"op":"mul","operands":[{"op":"const","value":2.0},' + node + "]}"
    return node


def test_constraint_tree_exceeding_python_recursion_limit_is_rejected() -> None:
    payload = project_to_dict(replace(project(), constraint_rules=(_multi_level_constraint_rule(),)))
    payload["constraint_rules"][0]["expression"] = _deep_constraint_node(1_500)

    with pytest.raises(ProjectSchemaError, match="constraint expression nesting"):
        project_from_dict(payload)


def test_deep_constraint_bytes_are_rejected_as_a_schema_error() -> None:
    payload = project_to_dict(replace(project(), constraint_rules=(_multi_level_constraint_rule(),)))
    marker = "__DEEP_CONSTRAINT_EXPRESSION__"
    payload["constraint_rules"][0]["expression"] = marker
    document = json.dumps(payload)
    node = _deep_constraint_json(1_500)
    content = document.replace(json.dumps(marker), node, 1).encode("utf-8")

    with pytest.raises(ProjectSchemaError, match="constraint expression nesting|invalid project JSON"):
        project_from_bytes(content)


def test_constraint_reference_fields_must_be_strings() -> None:
    payload = project_to_dict(replace(project(), constraint_rules=(_multi_level_constraint_rule(),)))
    payload["constraint_rules"][0]["target"]["dataset_id"] = 7

    with pytest.raises(ProjectSchemaError, match="parameter reference"):
        project_from_dict(payload)
