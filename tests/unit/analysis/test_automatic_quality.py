from types import SimpleNamespace

import pytest

from xrr_fitter.analysis.automatic import assess_automatic_quality


def _problem(
    names: tuple[str, ...],
    definition_names: tuple[str, ...] | None = None,
):
    variables = tuple(SimpleNamespace(name=name) for name in names)
    definitions = tuple(
        SimpleNamespace(name=name)
        for name in (names if definition_names is None else definition_names)
    )
    thresholds = SimpleNamespace(
        equivalent_cost_fraction=0.02,
        equivalent_cost_floor=1e-5,
    )
    return SimpleNamespace(
        variables=variables,
        parameter_definitions=definitions,
        config=SimpleNamespace(confidence=thresholds),
    )


def _result(
    *,
    boundaries=(),
    correlations=(),
    systematic=False,
    autocorrelation=False,
    diagnostics=(),
    evidence=(),
):
    uncertainty = SimpleNamespace(
        boundary_hits=boundaries,
        strong_correlations=correlations,
        systematic_residual=systematic,
        residual_autocorrelation=autocorrelation,
        diagnostics=diagnostics,
    )
    candidate = SimpleNamespace(valid=True, objective=0.01, stop_reason="converged")
    return SimpleNamespace(
        best_candidate=candidate,
        uncertainty=uncertainty,
        classification_evidence=evidence,
    )


def test_clean_fast_evidence_passes_without_profiles() -> None:
    decision = assess_automatic_quality(_problem(("component.0.thickness_a",)), _result())

    assert decision.passed is True
    assert decision.profile_names == ()
    assert decision.search_upgrade is False


def test_evidence_selects_at_most_four_relevant_profiles_in_parameter_order() -> None:
    names = (
        "component.0.thickness_a",
        "component.0.sld_real_a2",
        "component.0.roughness_a",
        "instrument.angle_offset_deg",
        "instrument.scale",
    )
    result = _result(
        boundaries=(names[0], names[1], names[2]),
        correlations=((names[0], names[3], 0.98),),
        systematic=True,
    )

    decision = assess_automatic_quality(_problem(names), result)

    assert decision.passed is False
    assert decision.profile_names == names[:4]


def test_systematic_residual_can_request_only_direct_sld_absorption() -> None:
    variables = ("component.1.thickness_a",)
    definitions = ("component.0.sld_imag_a2", *variables)

    decision = assess_automatic_quality(
        _problem(variables, definitions),
        _result(systematic=True),
    )

    assert decision.absorption_names == ("component.0.sld_imag_a2",)


@pytest.mark.parametrize(
    "code",
    (
        "distinct_equivalent_clusters",
        "profile_path_merge_failed",
        "insufficient_cluster_support",
    ),
)
def test_existing_candidate_evidence_codes_request_one_search_upgrade(code: str) -> None:
    decision = assess_automatic_quality(
        _problem(("component.0.thickness_a",)),
        _result(evidence=(code,)),
    )

    assert decision.search_upgrade is True
    assert code in decision.reasons


def test_existing_open_primary_profile_code_requests_review_not_search_replay() -> None:
    decision = assess_automatic_quality(
        _problem(("component.0.thickness_a",)),
        _result(evidence=("primary_profile_open",)),
    )

    assert decision.passed is False
    assert decision.search_upgrade is False
    assert decision.profile_names == ("component.0.thickness_a",)
