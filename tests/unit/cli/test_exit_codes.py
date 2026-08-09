"""Prove the CLI exit-code contract maps published results, not labels."""

from __future__ import annotations

import pytest

import xrr_fitter.api as api
from xrr_fitter.cli import exit_codes


def test_codes_are_the_documented_four() -> None:
    assert (
        exit_codes.SUCCESS,
        exit_codes.NOT_CONVERGED,
        exit_codes.INVALID_INPUT,
        exit_codes.STALE_SOURCE,
    ) == (0, 1, 2, 3)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (api.ConfidenceClass.TRUSTED, 0),
        (api.ConfidenceClass.CORRELATED, 0),
        (api.ConfidenceClass.MULTIPLE, 1),
        (api.ConfidenceClass.UNTRUSTED, 1),
    ],
)
def test_confidence_decides_the_fit_exit_code(confidence, expected) -> None:
    assert exit_codes.confidence_exit_code(confidence) == expected


def test_untrusted_dataset_dominates_a_mixed_result() -> None:
    codes = (
        api.ConfidenceClass.TRUSTED,
        api.ConfidenceClass.UNTRUSTED,
        api.ConfidenceClass.TRUSTED,
    )

    assert exit_codes.worst_exit_code(codes) == exit_codes.NOT_CONVERGED


def test_cancelled_result_is_not_reported_as_success() -> None:
    assert exit_codes.cancelled_exit_code() == exit_codes.NOT_CONVERGED


def test_every_confidence_member_has_a_mapping() -> None:
    for member in api.ConfidenceClass:
        assert exit_codes.confidence_exit_code(member) in {0, 1}


def test_an_empty_dataset_tuple_is_success_not_a_crash() -> None:
    assert exit_codes.worst_exit_code(()) == exit_codes.SUCCESS
