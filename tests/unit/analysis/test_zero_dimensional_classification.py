"""A locked model has one parameter-space point, not an empty candidate set."""

import warnings
from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import fit_candidate

from xrr_fitter.analysis.classification import _rms_distance, cluster_candidates, cluster_unit_vectors


@pytest.mark.parametrize("count", (1, 2, 4))
def test_zero_dimensional_candidates_form_one_cluster_without_empty_means(count) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            clusters = cluster_unit_vectors(np.empty((count, 0)))

    assert clusters == (tuple(range(count)),)


def test_distance_in_zero_dimensional_parameter_space_is_zero() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _rms_distance(np.empty(0), np.empty(0)) == 0.0


def test_locked_candidate_values_keep_input_order_in_the_single_cluster() -> None:
    candidates = tuple(replace(fit_candidate(f"locked-{index}"), unit_vector=np.empty(0)) for index in range(4))

    assert cluster_candidates(candidates) == ((0, 1, 2, 3),)


@pytest.mark.parametrize("distance", (0.0, -0.1, float("inf"), float("nan")))
def test_zero_dimensional_clustering_still_validates_join_distance(distance) -> None:
    with pytest.raises(ValueError, match="join_distance"):
        cluster_unit_vectors(np.empty((4, 0)), distance)


@pytest.mark.parametrize(
    "vectors",
    (np.empty((0, 0)), np.empty((0, 2)), np.empty(0), np.empty((1, 1, 0)), [[float("nan")]], [[float("inf")]]),
)
def test_clustering_still_rejects_missing_candidates_wrong_dimensions_and_nonfinite_values(vectors) -> None:
    with pytest.raises(ValueError, match="nonempty finite matrix"):
        cluster_unit_vectors(vectors)
