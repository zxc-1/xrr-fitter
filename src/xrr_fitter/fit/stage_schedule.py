"""Deterministic stage order and child-seed scheduling contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STAGE_ORDER = ("A", "B", "C", "D", "E")


@dataclass(frozen=True, slots=True)
class ChildSeed:
    """One named deterministic stream and its generated integer seed."""

    stream_id: str
    seed: int


def _stream_order(stream_id: str) -> tuple[int, int, str]:
    prefix, separator, suffix = stream_id.partition("-")
    if not separator or not suffix.isdigit():
        return 2, 0, stream_id
    priority = {"E": 0, "B": 1}.get(prefix, 2)
    return priority, int(suffix), stream_id


def reserve_child_seeds(
    master_seed: int,
    stream_ids: tuple[str, ...],
) -> tuple[ChildSeed, ...]:
    """Map named streams to SeedSequence children independent of request order.

    Canonical E streams precede B streams, followed by lexical fallbacks. Results
    are restored to caller order only after every named child has been generated,
    keeping incremental and full reservations identical.
    """

    requested = tuple(stream_ids)
    if len(requested) != len(set(requested)) or any(not value for value in requested):
        raise ValueError("child stream IDs must be nonempty and unique")
    canonical = tuple(sorted(requested, key=_stream_order))
    spawned = np.random.SeedSequence(master_seed).spawn(len(canonical))
    by_stream = {
        stream_id: int(child.generate_state(1, dtype=np.uint64)[0])
        for stream_id, child in zip(canonical, spawned, strict=True)
    }
    return tuple(ChildSeed(stream_id, by_stream[stream_id]) for stream_id in requested)


def remaining_stages(completed_stage: str | None) -> tuple[str, ...]:
    """Return the strict suffix after a completed checkpoint stage."""

    if completed_stage is None:
        return STAGE_ORDER
    if completed_stage not in STAGE_ORDER:
        raise ValueError(f"unsupported fit stage: {completed_stage}")
    return STAGE_ORDER[STAGE_ORDER.index(completed_stage) + 1 :]
