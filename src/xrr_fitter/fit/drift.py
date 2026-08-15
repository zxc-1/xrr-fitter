"""Compile-time drift desugaring: per-copy coefficients and constraint rules."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from xrr_fitter.model.parameters import (
    RESERVED_DATASET_ID,
    ConstraintNode,
    ConstraintRule,
    ParameterReference,
)
from xrr_fitter.model.structure import DriftSpec, PeriodicBlock, StructureSpec


def drift_coefficients(drift: DriftSpec, repeats: int) -> tuple[float, ...]:
    """Per-copy modulation constants c_k (c_0=0; copy 0 is the free base cell)."""
    coeffs: list[float] = [0.0]
    if drift.kind == "linear":
        coeffs.extend(float(k) for k in range(1, repeats))
    elif drift.kind == "sine":
        coeffs.extend(math.sin(2.0 * math.pi * k / drift.period + drift.phase) for k in range(1, repeats))
    else:  # random — deterministic, self-contained in drift.seed
        rng = np.random.default_rng(drift.seed)
        coeffs.extend(float(v) for v in rng.uniform(-1.0, 1.0, size=max(0, repeats - 1)))
    return tuple(coeffs)


DRIFT_DATASET = RESERVED_DATASET_ID  # sentinel; evaluation resolves by parameter_name only


def _validate_rebind_dataset_id(dataset_id: object) -> str:
    if not isinstance(dataset_id, str):
        raise TypeError("drift dataset ID must be a string")
    if not dataset_id.strip():
        raise ValueError("drift dataset ID must be nonempty")
    if dataset_id == DRIFT_DATASET:
        raise ValueError(f"drift dataset ID is reserved: {DRIFT_DATASET}")
    return dataset_id


def _ref(name: str) -> ConstraintNode:
    return ConstraintNode("ref", reference=ParameterReference(DRIFT_DATASET, name))


def drift_constraint_rules(structure: StructureSpec) -> tuple[ConstraintRule, ...]:
    """Desugar every drifted block into per-copy ``target = base·(1+scale·c_k)`` rules."""
    rules: list[ConstraintRule] = []
    for index, component in enumerate(structure.components):
        if not isinstance(component, PeriodicBlock) or component.drift is None:
            continue
        prefix = f"component.{index}"
        drift = component.drift
        coeffs = drift_coefficients(drift, component.repeats)
        family = "thickness_a" if drift.target == "thickness" else "roughness_a"
        scale = _ref(f"{prefix}.drift_scale")
        for layer_index in range(len(component.layers)):
            base = _ref(f"{prefix}.layer.{layer_index}.{family}")
            for k in range(1, component.repeats):
                factor = ConstraintNode(
                    "add",
                    operands=(
                        ConstraintNode("const", value=1.0),
                        ConstraintNode("mul", operands=(scale, ConstraintNode("const", value=coeffs[k]))),
                    ),
                )
                target = ParameterReference(DRIFT_DATASET, f"{prefix}.repeat.{k}.layer.{layer_index}.{family}")
                rules.append(ConstraintRule(target=target, expression=ConstraintNode("mul", operands=(base, factor))))
    return tuple(rules)


def _rebind_node(node: ConstraintNode, dataset_id: str) -> ConstraintNode:
    """Rewrite every ``DRIFT_DATASET`` reference in an expression to ``dataset_id``."""
    if node.op == "ref":
        if node.reference.dataset_id == DRIFT_DATASET:
            return ConstraintNode("ref", reference=ParameterReference(dataset_id, node.reference.parameter_name))
        return node
    if node.op == "const":
        return node
    return ConstraintNode(node.op, operands=tuple(_rebind_node(child, dataset_id) for child in node.operands))


def rebind_drift_rules(
    rules: tuple[ConstraintRule, ...],
    dataset_id: str,
) -> tuple[ConstraintRule, ...]:
    """Bind sentinel-generated rules to one local dataset namespace."""
    target_dataset = _validate_rebind_dataset_id(dataset_id)
    rebound: list[ConstraintRule] = []
    for rule in rules:
        target = (
            ParameterReference(target_dataset, rule.target.parameter_name)
            if rule.target.dataset_id == DRIFT_DATASET
            else rule.target
        )
        expression = _rebind_node(rule.expression, target_dataset)
        rebound.append(
            rule
            if target == rule.target and expression == rule.expression
            else ConstraintRule(target=target, expression=expression)
        )
    return tuple(rebound)


def rebind_drift_dataset(problem: object, dataset_id: str) -> object:
    """Rebind a compiled member's ``__drift__`` sentinel rules to its real ``dataset_id``.

    Drift rules are generated dataset-agnostically (target and refs carry the
    ``DRIFT_DATASET`` sentinel) so a single structure desugars identically for any
    dataset. Joint compilation validates each member's local rules against the
    member's own dataset identity, so the sentinel must be rebound before the
    problem enters ``compile_joint_problem``. Returns the problem unchanged (same
    object) when it carries no drift rules, keeping non-drift joint fingerprints
    byte-identical.
    """
    target_dataset = _validate_rebind_dataset_id(dataset_id)
    rules = problem.constraint_rules
    if not any(rule.target.dataset_id == DRIFT_DATASET for rule in rules):
        return problem
    rebound = rebind_drift_rules(rules, target_dataset)
    return replace(problem, constraint_rules=rebound)
