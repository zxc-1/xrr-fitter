"""Canonical ordered skip decisions shared by immutable search value objects."""

from __future__ import annotations


def normalize_skipped_stages(value: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise TypeError("skipped_stages must be an ordered sequence")
    stages = tuple(value)
    if any(stage not in ("A", "B", "C", "D", "E") for stage in stages):
        raise ValueError("skipped_stages must contain only A-E stages")
    if stages != tuple(stage for stage in ("A", "B", "C", "D", "E") if stage in stages):
        raise ValueError("skipped_stages must be unique and in stage order")
    return stages


def search_terminated_early(skipped_stages: tuple[str, ...]) -> bool:
    """C/D may be omitted; skipping A/B/E prevents a complete final ensemble."""
    return bool(set(skipped_stages) & {"A", "B", "E"})
