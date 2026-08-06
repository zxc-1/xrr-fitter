"""Progress publication helpers for fitting stages."""

from __future__ import annotations

from collections.abc import Callable

from xrr_fitter.fit.candidates import best_candidate_index
from xrr_fitter.model.fitting import FitCandidate, FitProgress
from xrr_fitter.model.progress import downsampled_preview


def emit_progress(
    callback: Callable[[FitProgress], None] | None,
    dataset_id: str | None,
    stage: str,
    completed: int,
    total: int,
    best: float,
    message: str,
    preview: FitCandidate | None = None,
) -> None:
    """Publish progress with a bounded curve when an incumbent changes."""
    if callback is None:
        return
    axes = (
        (None, None)
        if preview is None
        else downsampled_preview(preview.qz_a_inv, preview.model_normalized)
    )
    callback(
        FitProgress(
            dataset_id,
            stage,
            completed,
            total,
            best,
            message,
            preview_qz_a_inv=axes[0],
            preview_model_normalized=axes[1],
        )
    )


def best_preview_candidate(
    candidates: tuple[FitCandidate, ...],
) -> FitCandidate | None:
    """Return the current incumbent used by live progress only."""
    winner = best_candidate_index(candidates)
    return None if winner is None else candidates[winner]
