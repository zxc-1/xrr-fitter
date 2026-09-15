"""Strict JSON declarations for search review and budget evidence."""

from __future__ import annotations

from dataclasses import asdict, fields
from math import isfinite

from xrr_fitter.io.codec_common import ProjectSchemaError, _finite_number, _mapping, _sequence
from xrr_fitter.model.fitting import GridReview, SearchAllocation, SearchEvidence


def search_evidence_to_list(values: tuple[SearchEvidence, ...]) -> list[dict[str, object]]:
    return [_evidence_to_dict(value) for value in values]


def _evidence_to_dict(value: SearchEvidence) -> dict[str, object]:
    payload = asdict(value)
    payload["reviews"] = [_review_to_dict(review) for review in payload["reviews"]]
    payload["budget_allocations"] = list(payload["budget_allocations"])
    return payload


def _review_to_dict(review: dict[str, object]) -> dict[str, object]:
    for name in ("candidate_ids", "grid_points"):
        review[name] = list(review[name])
    for name in ("coarse_objectives", "full_objectives"):
        review[name] = [cost if isfinite(cost) else None for cost in review[name]]
    return review


def _review_costs(values: object) -> tuple[float, ...]:
    result = []
    for value in _sequence(values, "review costs"):
        if value is not None and not _finite_number(value):
            raise ProjectSchemaError("review cost must be a finite JSON number or null")
        result.append(float("inf") if value is None else float(value))
    return tuple(result)


def _review_from_dict(value: object) -> GridReview:
    """Require the current GridReview fields, including grid_evaluations."""
    payload = dict(_mapping(value, {item.name for item in fields(GridReview)}, "grid review"))
    for name in ("candidate_ids", "grid_points"):
        payload[name] = tuple(_sequence(payload[name], name))
    for name in ("coarse_objectives", "full_objectives"):
        payload[name] = _review_costs(payload[name])
    return GridReview(**payload)


def search_evidence_from_list(values: object) -> tuple[SearchEvidence, ...]:
    return tuple(_evidence_from_dict(value) for value in _sequence(values, "search evidence"))


def _evidence_from_dict(value: object) -> SearchEvidence:
    payload = dict(_mapping(value, {item.name for item in fields(SearchEvidence)}, "search evidence"))
    payload["reviews"] = tuple(_review_from_dict(item) for item in _sequence(payload["reviews"], "grid reviews"))
    payload["budget_allocations"] = tuple(
        _allocation_from_dict(item) for item in _sequence(payload["budget_allocations"], "budget allocations")
    )
    return SearchEvidence(**payload)


def _allocation_from_dict(value: object) -> SearchAllocation:
    allowed = {field.name for field in fields(SearchAllocation)}
    return SearchAllocation(**_mapping(value, allowed, "search allocation"))
