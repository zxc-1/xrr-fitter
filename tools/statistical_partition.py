"""Deterministic exact-cover partitions of the approved synthetic corpus."""

from __future__ import annotations

from tests.support.synthetic_recovery import _validated_cases
from tests.support.synthetic_recovery_model import SyntheticCase
from tests.support.synthetic_recovery_runs import _case_work_estimate

SHARD_COUNT = 8


def validate_shard_index(index: int) -> int:
    if type(index) is not int or not 0 <= index < SHARD_COUNT:
        raise ValueError(f"shard index must be an integer in [0, {SHARD_COUNT})")
    return index


def shard_cases(cases: tuple[SyntheticCase, ...], index: int) -> tuple[SyntheticCase, ...]:
    index = validate_shard_index(index)
    values = _validated_cases(cases)
    scheduled = sorted(enumerate(values), key=lambda item: (-_case_work_estimate(item[1]), item[0]))
    selected = {position for position, _case in scheduled[index::SHARD_COUNT]}
    return tuple(case for position, case in enumerate(values) if position in selected)
