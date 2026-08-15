"""Reconcile project graph references after one dataset is removed."""

from __future__ import annotations

from dataclasses import replace

from xrr_fitter.model.parameters import ConstraintRule, SharingRule, _iter_references
from xrr_fitter.model.project import DatasetProject, ProjectUiState, XrrProject


def removal_ui_state(
    project: XrrProject,
    datasets: tuple[DatasetProject, ...],
    remaining_ids: set[str],
    invalidated_ids: set[str],
    index: int,
    dataset_id: str,
) -> ProjectUiState:
    active = project.ui_state.active_dataset_id
    if active == dataset_id:
        active = datasets[min(index, len(datasets) - 1)].dataset_id if datasets else None
    selected = tuple(
        item
        for item in project.ui_state.selected_candidate_ids
        if item[0] in remaining_ids and item[0] not in invalidated_ids
    )
    return replace(
        project.ui_state,
        active_dataset_id=active,
        selected_candidate_ids=selected,
    )


def removal_sharing_rules(
    project: XrrProject,
    remaining_ids: set[str],
) -> tuple[SharingRule, ...]:
    return tuple(
        rule for rule in project.sharing_rules if all(member.dataset_id in remaining_ids for member in rule.members)
    )


def removal_constraint_rules(
    project: XrrProject,
    remaining_ids: set[str],
) -> tuple[tuple[ConstraintRule, ...], set[str]]:
    """Drop invalid rules and identify surviving targets needing invalidation."""
    retained: list[ConstraintRule] = []
    affected: set[str] = set()
    for rule in project.constraint_rules:
        references = (rule.target, *_iter_references(rule.expression))
        if rule.target.dataset_id not in remaining_ids:
            continue
        if any(reference.dataset_id not in remaining_ids for reference in references):
            affected.add(rule.target.dataset_id)
            continue
        retained.append(rule)
    return tuple(retained), affected


__all__ = [
    "removal_constraint_rules",
    "removal_sharing_rules",
    "removal_ui_state",
]
