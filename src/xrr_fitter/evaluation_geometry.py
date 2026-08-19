"""Geometry-specific helpers for the shared evaluation facade.

The public evaluation module owns coordinate decoding and constraint handling.
This implementation module owns only the material-independent geometry needed
to derive roughness caps and their thickness tangents.  It deliberately does
not import :mod:`xrr_fitter.evaluation`, so the facade remains the sole
production consumer of this implementation boundary.
"""

from __future__ import annotations

from math import ceil, isfinite

import numpy as np

from xrr_fitter.model.parameters import ParameterDefinition
from xrr_fitter.model.structure import MAX_EXPANDED_SLABS, GradientLayerSpec, PeriodicBlock
from xrr_fitter.physics.geometry import (
    GRADIENT_INTERNAL_INTERFACE,
    DifferentiableStack,
)
from xrr_fitter.physics.geometry import (
    GeometryExpansion as _GeometryExpansion,
)
from xrr_fitter.physics.geometry import (
    expand_geometry as _expand_geometry,
)
from xrr_fitter.physics.geometry import (
    expand_structure_with_jacobian as _expand_physical_structure_with_jacobian,
)
from xrr_fitter.physics.stack import rebuild_structure


def _interface_neighbor_indices(
    interface_index: int,
    final_medium: int,
) -> tuple[int, ...]:
    """Return finite slab indices adjacent to one expanded interface.

    Incident and backing media are semi-infinite and do not constrain roughness.
    An interior interface may therefore expose one or two finite neighbors.
    """
    return tuple(
        range(
            max(1, interface_index),
            min(final_medium, interface_index + 2),
        )
    )


def _interface_upper(
    thickness_a: np.ndarray,
    neighbor_indices: tuple[int, ...],
) -> float:
    """Apply the strict 49-percent cap or the bare-interface 50 A limit.

    ``nextafter`` keeps the decoded upper strictly legal under stack expansion.
    The bare two-medium case has no finite geometry and uses its explicit cap.
    """
    if not neighbor_indices:
        return 50.0
    minimum = float(np.min(thickness_a[np.asarray(neighbor_indices, dtype=int)]))
    return float(np.nextafter(0.49 * minimum, 0.0))


def _is_roughness_definition(definition: ParameterDefinition) -> bool:
    return definition.transform == "roughness_fraction"


def _definition_name(definition: ParameterDefinition) -> str:
    return definition.name


def _gradient_slab_counts(problem: object) -> dict[str, int]:
    """Resolve the fixed gradient topology compiled from candidate bounds."""
    definitions = {definition.name: definition for definition in problem.parameter_definitions}
    counts: dict[str, int] = {}
    for index, component in enumerate(problem.structure.components):
        if not isinstance(component, GradientLayerSpec):
            continue
        prefix = f"component.{index}"
        maximum = component.microslab_max_a
        upper = definitions[f"{prefix}.thickness_a"].upper
        ratio = upper / maximum
        if not isfinite(ratio) or ratio > MAX_EXPANDED_SLABS:
            raise ValueError(f"gradient slab topology exceeds the expanded slab budget: {prefix}")
        counts[prefix] = max(1, ceil(ratio))
    return counts


def _zero_roughness_values(problem: object) -> dict[str, float]:
    """Neutralize every declared roughness while retaining expanded topology."""
    definitions = filter(_is_roughness_definition, problem.parameter_definitions)
    return dict.fromkeys(map(_definition_name, definitions), 0.0)


def _roughness_definitions(problem: object) -> dict[str, ParameterDefinition]:
    return {
        definition.name: definition
        for definition in problem.parameter_definitions
        if _is_roughness_definition(definition)
    }


def _latent_periodic_roughness_names(problem: object) -> tuple[str, ...]:
    """Return declared periodic roughness coordinates absent from topology.

    Periodic declarations always carry both the cell layer roughnesses and a
    top-termination roughness. Exactly one of those declarations can be latent:
    an inherited top sentinel never materializes ``top_roughness_a``, while a
    single explicit-top repeat consumes the only layer-zero interface.
    """
    names: list[str] = []
    for component_index, component in enumerate(problem.structure.components):
        if not isinstance(component, PeriodicBlock):
            continue
        prefix = f"component.{component_index}"
        if component.top_roughness_a is None:
            names.append(f"{prefix}.top_roughness_a")
        elif component.repeats == 1:
            names.append(f"{prefix}.layer.0.roughness_a")
    return tuple(names)


def _add_latent_roughness_caps(problem: object, dynamic: dict[str, float]) -> None:
    definitions = _roughness_definitions(problem)
    for name in _latent_periodic_roughness_names(problem):
        definition = definitions.get(name)
        if definition is not None and name not in dynamic:
            dynamic[name] = definition.upper


def _is_public_interface(item: tuple[int, str]) -> bool:
    return item[1] != GRADIENT_INTERNAL_INTERFACE


def _roughness_definition_map(problem: object) -> dict[str, ParameterDefinition]:
    return {
        definition.name: definition
        for definition in problem.parameter_definitions
        if _is_roughness_definition(definition)
    }


def _allowed_missing_roughness_names(problem: object) -> frozenset[str]:
    """Return declaration coordinates intentionally absent from expanded interfaces."""
    allowed: set[str] = set()
    for index, component in enumerate(problem.structure.components):
        if not isinstance(component, PeriodicBlock):
            continue
        prefix = f"component.{index}"
        if component.top_roughness_a is None:
            allowed.add(f"{prefix}.top_roughness_a")
        if component.top_roughness_a is not None and component.repeats == 1:
            allowed.add(f"{prefix}.layer.0.roughness_a")
        if (
            component.top_roughness_a is not None
            and component.drift is not None
            and component.drift.target == "roughness"
        ):
            allowed.add(f"{prefix}.layer.0.roughness_a")
    return frozenset(allowed)


def _fill_missing_roughness_caps(
    problem: object,
    dynamic: dict[str, float],
) -> dict[str, float]:
    definitions = _roughness_definition_map(problem)
    allowed_missing = _allowed_missing_roughness_names(problem)
    unknown = set(dynamic) - set(definitions)
    if unknown:
        raise ValueError(f"unknown roughness coordinate mapping: {sorted(unknown)}")
    missing = set(definitions) - set(dynamic) - allowed_missing
    if missing:
        raise ValueError(f"roughness coordinate mapping missing: {sorted(missing)}")
    for name in allowed_missing:
        definition = definitions.get(name)
        if definition is not None:
            dynamic.setdefault(name, definition.upper)
    for name, definition in definitions.items():
        dynamic.setdefault(name, definition.upper)
    return dynamic


def _fill_missing_roughness_jacobians(
    problem: object,
    dynamic: dict[str, tuple[float, np.ndarray]],
) -> dict[str, tuple[float, np.ndarray]]:
    definitions = _roughness_definition_map(problem)
    allowed_missing = _allowed_missing_roughness_names(problem)
    unknown = set(dynamic) - set(definitions)
    if unknown:
        raise ValueError(f"unknown roughness coordinate mapping: {sorted(unknown)}")
    missing = set(definitions) - set(dynamic) - allowed_missing
    if missing:
        raise ValueError(f"roughness coordinate mapping missing: {sorted(missing)}")
    zero = np.zeros(len(problem.variables), dtype=float)
    for name, definition in definitions.items():
        dynamic.setdefault(name, (definition.upper, zero.copy()))
    return dynamic


def _roughness_dynamic_uppers(
    problem: object,
    nonrough_values: dict[str, float],
) -> dict[str, float]:
    """Tighten each named roughness against all repeated interface instances.

    Roughness is zeroed only for this provisional topology expansion. Candidate
    thickness remains active, and repeated periodic names reduce by the strictest
    local neighborhood rather than whichever occurrence is visited last.
    """
    zero_roughness = _zero_roughness_values(problem)
    # Right-hand union deliberately overrides any supplied roughness coordinate.
    # Geometry remains candidate-specific because all thickness values are kept.
    provisional_values = nonrough_values | zero_roughness
    provisional = rebuild_structure(
        problem.structure,
        provisional_values,
    )
    geometry = _expand_geometry(
        provisional,
        gradient_slab_counts=_gradient_slab_counts(problem),
    )
    dynamic: dict[str, float] = {}
    final_medium = geometry.thickness_a.size - 1
    # Repeated source names reduce by minimum, independent of occurrence order.
    # Internal gradient interfaces never receive a public parameter definition.
    for interface, name in filter(_is_public_interface, enumerate(geometry.interface_names)):
        neighbors = _interface_neighbor_indices(interface, final_medium)
        upper = _interface_upper(geometry.limit_thickness_a, neighbors)
        dynamic[name] = min(dynamic.get(name, np.inf), upper)
    return _fill_missing_roughness_caps(problem, dynamic)


def _expand_structure_with_jacobian(
    problem: object,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
    wavelength_a: float,
) -> DifferentiableStack:
    """Adapt the typed evaluation context to the physical stack expander."""
    return _expand_physical_structure_with_jacobian(
        problem.structure,
        values,
        value_jacobians,
        wavelength_a,
        len(problem.variables),
        _gradient_slab_counts(problem),
    )


def _roughness_geometry_context(
    problem: object,
    nonrough_values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
) -> _GeometryExpansion:
    """Build one zero-roughness geometry expansion with thickness tangents."""
    zero_roughness = _zero_roughness_values(problem)
    provisional_values = nonrough_values | zero_roughness
    provisional = rebuild_structure(
        problem.structure,
        provisional_values,
    )
    return _expand_geometry(
        provisional,
        parameter_count=len(problem.variables),
        value_jacobians=value_jacobians,
        gradient_slab_counts=_gradient_slab_counts(problem),
    )


def _active_upper_tangent(
    provisional: _GeometryExpansion,
    neighbors: tuple[int, ...],
    parameter_count: int,
) -> tuple[float, np.ndarray]:
    """Return the strict upper and symmetric tangent at an exact minimum tie.

    The cap follows the thinnest finite adjacent slab. At an exact equality,
    symmetric finite differences select the arithmetic mean subgradient across
    all active neighbors; near ties retain the single active derivative.
    """
    if not neighbors:
        return 50.0, np.zeros(parameter_count, dtype=float)
    indices = np.asarray(neighbors, dtype=int)
    assert provisional.limit_thickness_jacobian is not None
    thickness = provisional.limit_thickness_a[indices]
    minimum = float(np.min(thickness))
    active = indices[thickness == minimum]
    # Exact equality intentionally selects the symmetric minimum subgradient.
    # No tolerance widens this nonsmooth branch to merely nearby thicknesses.
    tangent = 0.49 * np.mean(
        provisional.limit_thickness_jacobian[active],
        axis=0,
    )
    return float(np.nextafter(0.49 * minimum, 0.0)), tangent


def _record_active_upper(
    dynamic: dict[str, tuple[float, np.ndarray]],
    tie_counts: dict[str, int],
    name: str,
    upper: float,
    tangent: np.ndarray,
) -> None:
    """Reduce repeated parameter names by minimum, averaging exact ties.

    A lower occurrence atomically replaces value and tangent. Equal active
    occurrences update an incremental arithmetic mean, making periodic sharing
    independent of repetition count while preserving the frozen traversal.
    """
    previous = dynamic.get(name, (np.inf, tangent))
    # Replacement publishes value and tangent atomically at a stricter cap.
    # Equal caps retain all active occurrences through incremental averaging.
    if upper < previous[0]:
        dynamic[name] = (upper, tangent.copy())
        tie_counts[name] = 1
        return
    if upper == previous[0]:
        count = tie_counts[name]
        dynamic[name] = (
            upper,
            (count * previous[1] + tangent) / (count + 1),
        )
        tie_counts[name] = count + 1


def _add_latent_roughness_cap_jacobians(
    problem: object,
    dynamic: dict[str, tuple[float, np.ndarray]],
    parameter_count: int,
) -> None:
    definitions = _roughness_definitions(problem)
    for name in _latent_periodic_roughness_names(problem):
        definition = definitions.get(name)
        if definition is not None and name not in dynamic:
            dynamic[name] = (definition.upper, np.zeros(parameter_count, dtype=float))


def _roughness_dynamic_upper_jacobians(
    problem: object,
    nonrough_values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
) -> dict[str, tuple[float, np.ndarray]]:
    """Return candidate-specific roughness bounds and thickness tangents.

    Internal gradient interfaces are skipped because they have no public
    coordinate. Locked definitions remain represented so encoding and analytic
    decoding observe the same candidate geometry.
    """
    provisional = _roughness_geometry_context(
        problem,
        nonrough_values,
        value_jacobians,
    )
    dynamic: dict[str, tuple[float, np.ndarray]] = {}
    tie_counts: dict[str, int] = {}
    parameter_count = len(problem.variables)
    final_medium = provisional.thickness_a.size - 1
    for interface, name in filter(
        _is_public_interface,
        enumerate(provisional.interface_names),
    ):
        upper, tangent = _active_upper_tangent(
            provisional,
            _interface_neighbor_indices(interface, final_medium),
            parameter_count,
        )
        _record_active_upper(dynamic, tie_counts, name, upper, tangent)
    return _fill_missing_roughness_jacobians(problem, dynamic)
