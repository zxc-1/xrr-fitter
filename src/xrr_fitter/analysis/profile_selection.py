"""Evidence-driven selection of parameter profiles for uncertainty analysis."""

from __future__ import annotations

from xrr_fitter.model.analysis import UncertaintyReport
from xrr_fitter.model.fitting import FitEvaluationContext

EXHAUSTIVE_PROFILE_LIMIT = 11


def _structural_profile_names(names: tuple[str, ...]) -> set[str]:
    fragments = ("thickness", "period", "density", "roughness")
    return {
        name
        for name in names
        if name == "instrument.angle_offset_deg" or any(fragment in name for fragment in fragments)
    }


def _reported_profile_names(preliminary_report: UncertaintyReport | None) -> set[str]:
    if preliminary_report is None:
        return set()
    selected = set(preliminary_report.boundary_hits)
    for first, second, _value in preliminary_report.strong_correlations:
        selected.update((first, second))
    return selected


def _degeneracy_profile_names(warnings: tuple[str, ...]) -> set[str]:
    selected: set[str] = set()
    for warning in warnings:
        if warning.startswith("\u539a\u5ea6-\u5bc6\u5ea6\u7b80\u5e76:"):
            selected.update(warning.split(":", 2)[1:])
    return selected


def evidence_focused_layout(problem: FitEvaluationContext) -> bool:
    """Return whether a layout needs evidence-focused profile selection."""

    return len(problem.variables) > EXHAUSTIVE_PROFILE_LIMIT


def select_profile_names(
    problem: FitEvaluationContext,
    preliminary_report: UncertaintyReport | None = None,
    *,
    degeneracy_warnings: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Select exhaustive small-problem or evidence-focused large-problem names.

    Large layouts retain structural coordinates, angle offset, boundary hits,
    correlated pairs, and thickness-density degeneracy participants. Derived
    binary profiles are appended in their declaration order.
    """

    from xrr_fitter.analysis.binary_profiles import binary_derived_profiles

    names = tuple(variable.name for variable in problem.variables)
    derived = tuple(item.name for item in binary_derived_profiles(problem))
    if not evidence_focused_layout(problem):
        return names + derived
    required = _structural_profile_names(names)
    required.update(_reported_profile_names(preliminary_report))
    required.update(_degeneracy_profile_names(degeneracy_warnings))
    return tuple(name for name in names if name in required) + derived
