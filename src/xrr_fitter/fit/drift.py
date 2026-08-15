"""Compile-time drift desugaring: per-copy coefficients and constraint rules."""

from __future__ import annotations

import math

import numpy as np

from xrr_fitter.model.parameters import ConstraintNode, ConstraintRule, ParameterReference
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


DRIFT_DATASET = "__drift__"  # sentinel; evaluation resolves by parameter_name only


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
