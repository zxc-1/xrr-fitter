"""Versioned bootstrap interval policy and separately labelled probability levels.

The Monte Carlo content assurance is not a parameter-coverage guarantee. Its
policy and one-based ranks are persisted on BootstrapResult and sealed together
with the samples; the public nominal confidence remains a separate quantity.
"""

JOINT_POISSON_INTERVAL_METHOD = "poisson_joint_content_tolerance_v1"
LINEAR_INTERVAL_METHOD = "percentile_linear_v1"


def bootstrap_interval_method(method: str) -> str:
    return JOINT_POISSON_INTERVAL_METHOD if method == "joint_poisson_parametric" else LINEAR_INTERVAL_METHOD


def _rank_pair(values, count):
    ranks = tuple(values) if values is not None else ()
    valid = len(ranks) == 2 and all(isinstance(rank, int) and not isinstance(rank, bool) for rank in ranks)
    if not valid or not 1 <= ranks[0] < ranks[1] <= count:
        raise ValueError("bootstrap interval ranks must be ordered one-based sample indices")
    if sum(ranks) != count + 1:
        raise ValueError("bootstrap interval ranks must be symmetric")
    return ranks


def validated_interval_ranks(result):
    if result.interval_method != bootstrap_interval_method(result.method):
        raise ValueError("bootstrap interval policy must match its resampling method")
    ranked = result.interval_method == JOINT_POISSON_INTERVAL_METHOD and bool(result.intervals)
    if ranked:
        return _rank_pair(result.interval_ranks, result.successful_samples)
    if result.interval_ranks is not None:
        raise ValueError("bootstrap interval ranks require an available finite-MC interval")
    return None


class BootstrapIntervalMetadata:
    """Derived labels are fixed by the persisted method, never caller-chosen."""

    __slots__ = ()
    interval_method: str

    @property
    def bootstrap_content_target(self) -> float:
        return 0.96 if self.interval_method == JOINT_POISSON_INTERVAL_METHOD else 0.95

    @property
    def monte_carlo_assurance(self) -> float | None:
        return 0.95 if self.interval_method == JOINT_POISSON_INTERVAL_METHOD else None

    @property
    def diagnostic_error_budget(self) -> float:
        return 0.01 if self.interval_method == JOINT_POISSON_INTERVAL_METHOD else 0.0
