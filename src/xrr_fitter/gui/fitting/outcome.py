"""Project published fit evidence into feedback without reclassifying the fit."""

from dataclasses import dataclass

import xrr_fitter.api as api


@dataclass(frozen=True, slots=True)
class FitFeedback:
    text: str
    kind: str
    completed: bool


def _valid_result(result: api.FitResult | None) -> bool:
    return result is not None and result.best_candidate is not None and result.best_candidate.valid


def _member_kind(
    result: api.FitResult,
    dataset: api.DatasetProject | None,
    automatic_status: api.AutomaticStatus | None,
) -> str:
    # A retained historical result must not turn this run's failed outcome green.
    if not _valid_result(result) or dataset is None or not _valid_result(dataset.last_valid_result):
        return "error"
    if automatic_status is api.AutomaticStatus.FAILED:
        return "error"
    if automatic_status is api.AutomaticStatus.REVIEW or result.confidence is not api.ConfidenceClass.TRUSTED:
        return "warn"
    return "ok"


def _message(title: str, details: tuple[str | None, ...]) -> str:
    notes = "；".join(dict.fromkeys(note for note in details if note))
    return f"{title}：{notes}" if notes else title


def _member_feedback(
    dataset_id: str,
    result: api.FitResult,
    dataset: api.DatasetProject | None,
    automatic: bool,
) -> FitFeedback:
    # Manual reruns retain automation metadata; it is not this run's verdict.
    automation = dataset.automation if automatic and dataset is not None else None
    status = None if automation is None else automation.status
    kind = _member_kind(result, dataset, status)
    if kind == "ok":
        return FitFeedback("", kind, True)
    reason = None if automation is None else automation.reason
    details = tuple(filter(None, (reason, *result.classification_evidence, *result.warnings)))
    default = result.confidence.value if kind == "warn" else "未发布有效拟合结果"
    return FitFeedback(_message(dataset_id, details or (default,)), kind, kind != "error")


def _combined_feedback(members: tuple[FitFeedback, ...], warnings: tuple[str, ...]) -> FitFeedback:
    kinds = tuple(member.kind for member in members)
    if not members:
        return FitFeedback(_message("未发布拟合结果", warnings), "warn", False)
    failed = kinds.count("error")
    if failed:
        total_failure = failed == len(members)
        title = "拟合失败" if total_failure else f"部分拟合完成（{len(members) - failed}/{len(members)}）"
        kind = "error" if total_failure else "warn"
    elif "warn" in kinds:
        title, kind = "拟合完成，需复核", "warn"
    else:
        title, kind = "拟合完成", "warn" if warnings else "ok"
    details = tuple(member.text for member in members)
    return FitFeedback(_message(title, (*details, *warnings)), kind, failed == 0)


def fit_result_feedback(result: api.ProjectFitResult) -> FitFeedback:
    """Use only this operation's members, not unrelated historical failures."""
    datasets = {dataset.dataset_id: dataset for dataset in result.updated_project.datasets}
    members = tuple(
        _member_feedback(row.dataset_id, row.fit_result, datasets.get(row.dataset_id), result.mode == "automatic")
        for row in result.datasets
    )
    return _combined_feedback(members, result.warnings)
