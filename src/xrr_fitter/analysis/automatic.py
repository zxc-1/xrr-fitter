"""Fast automatic quality decisions over persisted analysis evidence."""

from __future__ import annotations

from dataclasses import dataclass

SEARCH_UPGRADE_EVIDENCE = frozenset(
    {
        "distinct_equivalent_clusters",
        "profile_path_merge_failed",
        "insufficient_cluster_support",
    }
)
PROFILE_REVIEW_EVIDENCE = frozenset({"primary_profile_open"})


@dataclass(frozen=True, slots=True)
class AutomaticQualityDecision:
    passed: bool
    search_upgrade: bool
    absorption_names: tuple[str, ...]
    profile_names: tuple[str, ...]
    reasons: tuple[str, ...]


def assess_automatic_quality(
    problem: object,
    result: object,
    profile_limit: int = 4,
) -> AutomaticQualityDecision:
    """Translate fast report evidence into bounded automatic follow-up work."""
    if profile_limit < 0:
        raise ValueError("profile_limit must be nonnegative")
    best = result.best_candidate
    if best is None or not best.valid:
        return AutomaticQualityDecision(False, True, (), (), ("no valid candidate",))
    report = result.uncertainty
    if report is None:
        return AutomaticQualityDecision(False, True, (), (), ("missing quality report",))

    reasons: list[str] = []
    implicated: set[str] = set()
    if report.boundary_hits:
        reasons.append("parameter boundary hit")
        implicated.update(report.boundary_hits)
    if report.strong_correlations:
        reasons.append("strong parameter correlation")
        for first, second, _value in report.strong_correlations:
            implicated.update((first, second))
    if report.systematic_residual or report.residual_autocorrelation:
        reasons.append("systematic residual")
    if report.diagnostics:
        reasons.append("physical diagnostic")

    evidence = tuple(result.classification_evidence)
    reasons.extend(
        code
        for code in evidence
        if code in SEARCH_UPGRADE_EVIDENCE or code in PROFILE_REVIEW_EVIDENCE
    )
    names = tuple(variable.name for variable in problem.variables)
    definition_names = tuple(
        definition.name for definition in problem.parameter_definitions
    )
    structural = tuple(
        name
        for name in names
        if any(fragment in name for fragment in ("thickness", "sld_", "roughness"))
        or name.startswith("instrument.")
    )
    if reasons and not implicated:
        implicated.update(structural)
    profiles = tuple(name for name in names if name in implicated)[:profile_limit]
    absorption = tuple(
        name
        for name in definition_names
        if name.endswith(".sld_imag_a2")
        and (
            report.systematic_residual
            or any(code in SEARCH_UPGRADE_EVIDENCE for code in evidence)
        )
    )
    search_upgrade = (
        any(code in SEARCH_UPGRADE_EVIDENCE for code in evidence)
        or best.stop_reason == "max_nfev"
    )
    return AutomaticQualityDecision(
        not reasons,
        search_upgrade,
        absorption,
        profiles,
        tuple(reasons),
    )
