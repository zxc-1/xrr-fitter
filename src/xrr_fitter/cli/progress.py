"""Human-readable and JSON Lines renderings of fit progress."""

from __future__ import annotations

import json
from math import isfinite

import xrr_fitter.api as api


def _objective(value: float) -> float | None:
    return value if isfinite(value) else None


def render_text(progress: api.FitProgress) -> str:
    """Render one progress event as a single human-readable line."""
    dataset = progress.dataset_id or "-"
    objective = _objective(progress.best_objective)
    best = "-" if objective is None else f"{objective:.6g}"
    return (
        f"[{dataset}] 阶段 {progress.stage} "
        f"{progress.completed}/{progress.total} "
        f"best={best} {progress.message}"
    )


def render_json(progress: api.FitProgress) -> str:
    """Render one progress event as a single JSON Lines record."""
    return json.dumps(
        {
            "dataset_id": progress.dataset_id,
            "stage": progress.stage,
            "completed": progress.completed,
            "total": progress.total,
            "best_objective": _objective(progress.best_objective),
            "message": progress.message,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
