from __future__ import annotations

from collections import Counter

from tests.support.synthetic_recovery import build_corpus, run_corpus


EXPECTED_CATEGORIES = {
    "ambiguous": 20,
    "double_layer": 20,
    "footprint_locked": 20,
    "footprint_released": 20,
    "instrument_effects": 20,
    "mixed_kalpha_dual": 20,
    "mixed_kalpha_mono": 20,
    "model_error": 20,
    "oxide_cap": 20,
    "periodic_mosi": 20,
    "single_layer": 20,
}


def test_corpus_definitions_are_complete_and_deterministic() -> None:
    first = build_corpus()
    second = build_corpus()

    observed = (
        len(first),
        tuple(case.case_id for case in first),
        tuple(case.seed for case in first),
        len({case.case_id for case in first}),
        Counter(case.category for case in first),
    )
    expected = (
        220,
        tuple(case.case_id for case in second),
        tuple(case.seed for case in second),
        220,
        EXPECTED_CATEGORIES,
    )
    assert observed == expected


def test_synthetic_recovery_corpus_meets_approved_thresholds() -> None:
    report = run_corpus(build_corpus())

    observed = (
        report.schema,
        report.status,
        report.case_count,
        report.fit_count,
        report.category_counts,
        report.failed_case_ids,
    )
    expected = (
        "xrr-r23-synthetic-recovery-v1",
        "PASS",
        220,
        220,
        tuple(sorted(EXPECTED_CATEGORIES.items())),
        (),
    )
    assert observed == expected
