from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from xrr_fitter.model.analysis import ConfidenceClass
from xrr_fitter.model.instrument import PhysicsDiagnostic


def _api():
    return import_module("xrr_fitter.analysis.classification")


def classify_candidate_evidence(*args, **kwargs):
    return _api().classify_candidate_evidence(*args, **kwargs)


def classify_candidate_evidence_with_reasons(*args, **kwargs):
    return _api().classify_candidate_evidence_with_reasons(*args, **kwargs)


def classify_result_with_evidence(*args, **kwargs):
    return _api().classify_result_with_evidence(*args, **kwargs)


def cluster_candidates(*args, **kwargs):
    return _api().cluster_candidates(*args, **kwargs)


def cluster_unit_vectors(*args, **kwargs):
    return _api().cluster_unit_vectors(*args, **kwargs)


def _supported_vectors() -> np.ndarray:
    return np.asarray([[0.10], [0.11], [0.12], [0.13]])


def _classify(vectors: np.ndarray, costs: np.ndarray, **kwargs):
    return classify_candidate_evidence(
        vectors,
        costs,
        cluster_unit_vectors(vectors),
        **kwargs,
    )


def test_equivalent_distant_solutions_are_multiple() -> None:
    vectors = np.asarray([[0.10, 0.10], [0.11, 0.09], [0.80, 0.80], [0.81, 0.79]])

    assert _classify(vectors, np.asarray([0.0100, 0.0101, 0.01005, 0.0101])) is ConfidenceClass.MULTIPLE


def test_equivalent_distant_cluster_precedes_singleton_best_gate() -> None:
    vectors = np.asarray([[0.10], [0.70], [0.71], [0.72]])
    costs = np.asarray([0.0100, 0.01005, 0.4, 0.5])
    clusters = ((0,), (1, 2, 3))

    confidence, reasons = classify_candidate_evidence_with_reasons(vectors, costs, clusters)

    assert confidence is ConfidenceClass.MULTIPLE
    assert reasons == ("distinct_equivalent_clusters",)


def test_equivalent_supported_cluster_precedes_rounding_better_singleton() -> None:
    vectors = np.asarray([[0.10], [0.16], [0.17], [0.18]])
    costs = np.asarray([1e-12, 2e-12, 3e-12, 4e-12])
    clusters = ((0,), (1, 2, 3))

    confidence, reasons = classify_candidate_evidence_with_reasons(vectors, costs, clusters)

    assert confidence is ConfidenceClass.TRUSTED
    assert reasons == ()


def test_materially_better_singleton_remains_untrusted() -> None:
    vectors = np.asarray([[0.10], [0.16], [0.17], [0.18]])
    costs = np.asarray([1e-12, 1e-3, 1.001e-3, 1.002e-3])
    clusters = ((0,), (1, 2, 3))

    confidence, reasons = classify_candidate_evidence_with_reasons(vectors, costs, clusters)

    assert confidence is ConfidenceClass.UNTRUSTED
    assert reasons == ("insufficient_cluster_support",)


def test_cluster_candidates_uses_candidate_unit_vectors_in_input_order() -> None:
    candidates = tuple(
        SimpleNamespace(unit_vector=np.asarray(vector, dtype=float))
        for vector in ((0.10, 0.10), (0.11, 0.09), (0.80, 0.80), (0.81, 0.79))
    )

    assert cluster_candidates(candidates) == ((0, 1), (2, 3))


def test_one_of_four_seeds_is_untrusted() -> None:
    vectors = np.asarray([[0.1], [0.7], [0.8], [0.9]])

    assert _classify(vectors, np.asarray([0.01, 0.2, 0.3, 0.4])) is ConfidenceClass.UNTRUSTED


def test_two_of_four_in_best_cluster_is_correlated_but_three_is_trusted() -> None:
    two = np.asarray([[0.10], [0.11], [0.70], [0.90]])
    three = np.asarray([[0.10], [0.11], [0.12], [0.90]])

    assert _classify(two, np.asarray([0.0100, 0.0101, 0.2, 0.3])) is ConfidenceClass.CORRELATED
    assert _classify(three, np.asarray([0.0100, 0.0101, 0.01005, 0.3])) is ConfidenceClass.TRUSTED


def test_strong_correlation_caps_otherwise_trusted_evidence() -> None:
    vectors = _supported_vectors()

    assert (
        _classify(
            vectors,
            np.asarray([0.0100, 0.0101, 0.01005, 0.01008]),
            strong_correlations=(("a", "b", 0.96),),
        )
        is ConfidenceClass.CORRELATED
    )


@pytest.mark.parametrize(
    ("clusters", "valid", "reason"),
    (
        pytest.param((), None, "missing_candidate_clusters", id="missing-candidate-clusters"),
        pytest.param(
            ((0, 1, 2, 3),),
            np.asarray([True, False, True, True]),
            "invalid_candidate_evidence",
            id="invalid-candidate-evidence",
        ),
        pytest.param(
            ((0,), (1,), (2,), (3,)),
            None,
            "insufficient_cluster_support",
            id="insufficient-cluster-support",
        ),
    ),
)
def test_untrusted_classification_reports_the_deciding_evidence(clusters, valid, reason: str) -> None:
    observed = classify_candidate_evidence_with_reasons(
        np.asarray([[0.10], [0.40], [0.70], [0.90]]),
        np.asarray([0.01, 0.20, 0.30, 0.40]),
        clusters,
        valid=valid,
    )

    assert observed == (ConfidenceClass.UNTRUSTED, (reason,))


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        pytest.param({"boundary_hits": ("film.thickness_a",)}, "boundary_hit", id="boundary-hit"),
        pytest.param(
            {"strong_correlations": (("a", "b", 0.97),)},
            "strong_correlation",
            id="strong-correlation",
        ),
        pytest.param({"profiles_closed": False}, "profile_interval_open", id="profile-interval-open"),
        pytest.param({"systematic_residual": True}, "systematic_residual", id="systematic-residual"),
        pytest.param(
            {
                "diagnostics": (
                    PhysicsDiagnostic(
                        "nevot_croce_applicability_exceeded",
                        "roughness approximation exceeded",
                        (2,),
                    ),
                )
            },
            "nevot_croce_applicability_exceeded",
            id="nevot-croce-applicability-exceeded",
        ),
    ),
)
def test_correlated_classification_reports_the_deciding_evidence(kwargs: dict[str, object], reason: str) -> None:
    vectors = _supported_vectors()

    confidence, reasons = classify_candidate_evidence_with_reasons(
        vectors,
        np.asarray([0.0100, 0.0101, 0.01005, 0.01008]),
        cluster_unit_vectors(vectors),
        **kwargs,
    )

    assert confidence is ConfidenceClass.CORRELATED
    assert reasons == (reason,)


def test_two_seed_cluster_reports_limited_support_reason() -> None:
    vectors = np.asarray([[0.10], [0.11], [0.70], [0.90]])

    assert classify_candidate_evidence_with_reasons(
        vectors,
        np.asarray([0.0100, 0.0101, 0.20, 0.30]),
        cluster_unit_vectors(vectors),
    ) == (ConfidenceClass.CORRELATED, ("two_seed_cluster_support",))


def test_multiple_classification_reports_distant_cluster_and_open_primary_reasons() -> None:
    distant = np.asarray([[0.10, 0.10], [0.11, 0.09], [0.80, 0.80], [0.81, 0.79]])
    close = _supported_vectors()

    assert classify_candidate_evidence_with_reasons(
        distant,
        np.asarray([0.0100, 0.0101, 0.01005, 0.0101]),
        cluster_unit_vectors(distant),
    ) == (ConfidenceClass.MULTIPLE, ("distinct_equivalent_clusters",))
    assert classify_candidate_evidence_with_reasons(
        close,
        np.asarray([0.0100, 0.0101, 0.01005, 0.01008]),
        cluster_unit_vectors(close),
        fully_open_primary_profile=True,
    ) == (ConfidenceClass.MULTIPLE, ("primary_profile_open",))


def test_profile_merge_failure_returns_stable_classification_evidence() -> None:
    vectors = np.asarray([[0.40, 0.40], [0.41, 0.41], [0.47, 0.47], [0.48, 0.48]])

    assert classify_candidate_evidence_with_reasons(
        vectors,
        np.asarray([0.0100, 0.01005, 0.0101, 0.01008]),
        cluster_unit_vectors(vectors),
        profile_path_merge=lambda *_args: False,
    ) == (ConfidenceClass.MULTIPLE, ("profile_path_merge_failed",))


def test_classify_result_with_evidence_forwards_explicit_profile_path_merge(
    monkeypatch,
) -> None:
    profiles = import_module("xrr_fitter.analysis.profiles")

    def fail_default_merge(*_args) -> bool:
        raise AssertionError("default merge was used")

    monkeypatch.setattr(
        profiles,
        "default_profile_path_merge",
        fail_default_merge,
    )
    candidates = tuple(
        SimpleNamespace(
            unit_vector=np.asarray([value]),
            objective=0.0100 + index * 1e-5,
            valid=True,
            stop_reason="converged",
        )
        for index, value in enumerate((0.40, 0.41, 0.47, 0.48))
    )
    report = SimpleNamespace(
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        profiles=(),
        systematic_residual=False,
        diagnostics=(),
    )
    calls: list[tuple[np.ndarray, np.ndarray, float]] = []

    def merge(first: np.ndarray, second: np.ndarray, limit: float) -> bool:
        calls.append((first, second, limit))
        return False

    observed = classify_result_with_evidence(
        SimpleNamespace(config=None, variables=(object(),)),
        candidates,
        report,
        profile_path_merge=merge,
    )

    assert observed == (ConfidenceClass.MULTIPLE, ("profile_path_merge_failed",))
    assert len(calls) == 1


def test_classify_result_builds_default_profile_path_for_close_clusters(
    monkeypatch,
) -> None:
    profiles = import_module("xrr_fitter.analysis.profiles")
    calls: list[tuple[np.ndarray, np.ndarray, float]] = []

    def merge(_problem, first: np.ndarray, second: np.ndarray, limit: float) -> bool:
        calls.append((first, second, limit))
        return False

    monkeypatch.setattr(profiles, "default_profile_path_merge", merge)
    candidates = tuple(
        SimpleNamespace(
            unit_vector=np.asarray([value]),
            objective=0.0100 + index * 1e-5,
            valid=True,
            stop_reason="converged",
        )
        for index, value in enumerate((0.40, 0.41, 0.47, 0.48))
    )
    report = SimpleNamespace(
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        profiles=(),
        systematic_residual=False,
        diagnostics=(),
    )

    observed = classify_result_with_evidence(
        SimpleNamespace(config=None, variables=(object(),)),
        candidates,
        report,
    )

    assert observed == (ConfidenceClass.MULTIPLE, ("profile_path_merge_failed",))
    assert len(calls) == 1


def test_singleton_best_does_not_probe_close_equivalent_profile_path() -> None:
    vectors = np.asarray([[0.10], [0.16]])
    costs = np.asarray([0.010000, 0.010001])
    clusters = cluster_unit_vectors(vectors)
    calls: list[tuple[object, ...]] = []

    observed = classify_candidate_evidence(
        vectors,
        costs,
        clusters,
        profile_path_merge=lambda *args: calls.append(args) is None,
    )

    assert clusters == ((0,), (1,))
    assert observed is ConfidenceClass.UNTRUSTED
    assert calls == []


def test_classify_result_with_evidence_uses_candidate_and_report_state() -> None:
    vectors = (0.10, 0.11, 0.12, 0.13)
    candidates = tuple(
        SimpleNamespace(unit_vector=np.asarray([value]), objective=0.01, valid=True) for value in vectors
    )
    report = SimpleNamespace(
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        profiles=(),
        systematic_residual=False,
        diagnostics=(),
    )

    assert classify_result_with_evidence(SimpleNamespace(config=None), candidates, report) == (
        ConfidenceClass.TRUSTED,
        (),
    )


def test_classify_result_uses_persisted_global_ranking_costs(monkeypatch) -> None:
    module = _api()
    candidates = tuple(
        SimpleNamespace(
            unit_vector=np.asarray([0.1 + 0.01 * index]),
            objective=0.1 + 0.1 * index,
            ranking_objective=ranking,
            valid=True,
            stop_reason="converged",
        )
        for index, ranking in enumerate((10.0, 1.0, 20.0, 30.0))
    )
    report = SimpleNamespace(
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        profiles=(),
        systematic_residual=False,
        diagnostics=(),
    )
    observed: dict[str, np.ndarray] = {}

    def classify(_vectors, costs, _clusters, **_kwargs):
        observed["costs"] = np.asarray(costs)
        return ConfidenceClass.TRUSTED, ()

    monkeypatch.setattr(module, "classify_candidate_evidence_with_reasons", classify)

    module.classify_result_with_evidence(
        SimpleNamespace(config=None),
        candidates,
        report,
    )

    np.testing.assert_array_equal(observed["costs"], np.asarray([10.0, 1.0, 20.0, 30.0]))


def test_model_error_gate_does_not_count_non_acf_systematic_residual() -> None:
    vectors = _supported_vectors()

    confidence, reasons = classify_candidate_evidence_with_reasons(
        vectors,
        np.full(4, 0.01),
        cluster_unit_vectors(vectors),
        systematic_residual=False,
    )

    assert confidence is ConfidenceClass.TRUSTED
    assert "systematic_residual" not in reasons


def test_truncated_low_q_curve_is_not_trusted() -> None:
    vectors = _supported_vectors()

    confidence = _classify(
        vectors,
        np.full(4, 0.01),
        boundary_hits=("component.0.thickness_a",),
    )

    assert confidence is not ConfidenceClass.TRUSTED
