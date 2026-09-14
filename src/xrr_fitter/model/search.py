"""Immutable numerical review and optimizer-budget evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real


def _count(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _members[T](values: object, member_type: type[T], name: str) -> tuple[T, ...]:
    result = tuple(values)
    if any(not isinstance(value, member_type) for value in result):
        raise TypeError(f"{name} contains an invalid member")
    return result


def _review_axes(value: GridReview) -> tuple[tuple[int, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    axes = tuple(value.grid_points)
    identifiers = tuple(value.candidate_ids)
    coarse = tuple(value.coarse_objectives)
    full = tuple(value.full_objectives)
    if not axes:
        raise ValueError("grid_points must contain a dataset axis")
    for count in axes:
        _count(count, "grid_points", 1)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("review candidate IDs must be unique")
    for identifier in identifiers:
        _identity(identifier, "candidate_id")
    return axes, identifiers, coarse, full


def _review_costs(costs: tuple[float, ...], identifiers: tuple[str, ...]) -> tuple[float, ...]:
    if len(costs) != len(identifiers):
        raise ValueError("review objectives must align with candidate IDs")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in costs):
        raise ValueError("review objectives must contain real numeric scalars")
    values = tuple(float(value) for value in costs)
    if any(not (value >= 0 and (isfinite(value) or value == float("inf"))) for value in values):
        raise ValueError("review objectives must be nonnegative or positive infinity")
    return values


@dataclass(frozen=True, slots=True)
class GridReview:
    """One grid's reviewed candidates; grid_points has one axis per dataset.

    grid_evaluations counts prior or rescreening calls on this numerical grid,
    which may itself be full resolution. full_evaluations counts only additional
    full-review cache misses, not every full-resolution physical evaluation.
    """

    grid_points: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    coarse_objectives: tuple[float, ...]
    full_objectives: tuple[float, ...]
    promoted: bool
    grid_evaluations: int
    full_evaluations: int

    def __post_init__(self) -> None:
        axes, identifiers, coarse, full = _review_axes(self)
        object.__setattr__(self, "grid_points", axes)
        object.__setattr__(self, "candidate_ids", identifiers)
        object.__setattr__(self, "coarse_objectives", _review_costs(coarse, identifiers))
        object.__setattr__(self, "full_objectives", _review_costs(full, identifiers))
        if not isinstance(self.promoted, bool):
            raise TypeError("promoted must be bool")
        _count(self.grid_evaluations, "grid_evaluations")
        _count(self.full_evaluations, "full_evaluations")
        if self.full_evaluations > len(self.candidate_ids):
            raise ValueError("full_evaluations exceeds the reviewed candidate count")


@dataclass(frozen=True, slots=True)
class SearchAllocation:
    """A bounded optimizer call assigned to one lineage in a deterministic round."""

    lineage_id: str
    candidate_id: str
    round_index: int
    max_nfev: int
    nfev: int

    def __post_init__(self) -> None:
        _identity(self.lineage_id, "lineage_id")
        _identity(self.candidate_id, "candidate_id")
        _count(self.round_index, "round_index")
        _count(self.max_nfev, "max_nfev", 1)
        _count(self.nfev, "nfev")
        if self.nfev > self.max_nfev:
            raise ValueError("optimizer nfev exceeds allocated budget")


@dataclass(frozen=True, slots=True)
class SearchEvidence:
    candidate_origin: str
    seed: int
    reviews: tuple[GridReview, ...]
    budget_allocations: tuple[SearchAllocation, ...]
    stop_reason: str

    def __post_init__(self) -> None:
        _identity(self.candidate_origin, "candidate_origin")
        _identity(self.stop_reason, "stop_reason")
        _count(self.seed, "seed")
        object.__setattr__(self, "reviews", _members(self.reviews, GridReview, "reviews"))
        object.__setattr__(
            self, "budget_allocations", _members(self.budget_allocations, SearchAllocation, "budget_allocations")
        )

    @property
    def grid_points(self) -> tuple[tuple[int, ...], ...]:
        return tuple(review.grid_points for review in self.reviews)

    @property
    def full_review_evaluations(self) -> int:
        """Additional full-review cache misses, excluding other grid work."""
        return sum(review.full_evaluations for review in self.reviews)

    @property
    def grid_review_evaluations(self) -> int:
        """Prior or rescreening calls on each review's grid_points resolution."""
        return sum(review.grid_evaluations for review in self.reviews)

    @property
    def optimizer_nfev(self) -> int:
        return sum(allocation.nfev for allocation in self.budget_allocations)

    @property
    def optimizer_nfev_limit(self) -> int:
        return sum(allocation.max_nfev for allocation in self.budget_allocations)


def search_evidence_tuple(values: object) -> tuple[SearchEvidence, ...]:
    return _members(values, SearchEvidence, "search_evidence")
