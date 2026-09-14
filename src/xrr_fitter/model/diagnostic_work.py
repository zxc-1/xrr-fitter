"""Compact joint histograms of diagnostic path exits and real charged work.

Each bin retains all three phase exits together, so a reader can recompute the
per-path cap, maximum and total without trusting a separately stored maximum.
Bins have no replicate coordinates or optimization trajectories. Observed and
null histograms are kept separate by the calibration owner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from xrr_fitter.model.fitting import _pickle_values, _positive_integer

PHASES = ("localize", "handoff", "refine")
EXIT_KINDS = frozenset({"native", "budget", "validation", "numerical_failure", "not_entered", "zero_dimensional"})


def _native_exit(value):
    if not isinstance(value.status, int) or isinstance(value.status, bool) or value.phase == "handoff":
        raise ValueError("native diagnostic exits require an integer optimizer status")
    statuses = (
        {0: True, 1: False, 2: False}
        if value.phase == "localize"
        else {
            -2: False,
            -1: False,
            0: False,
            1: True,
            2: True,
            3: True,
            4: True,
        }
    )
    if value.status not in statuses or value.success is not statuses[value.status]:
        raise ValueError("native diagnostic status and success must match the optimizer semantics")
    if value.nfev == 0:
        raise ValueError("native diagnostic exits require charged optimizer work")


def _exit_fields(value):
    if value.phase not in PHASES or value.kind not in EXIT_KINDS:
        raise ValueError("invalid diagnostic phase or exit kind")
    if not isinstance(value.success, bool):
        raise ValueError("diagnostic exit success must be a boolean")
    if not isinstance(value.message, str) or not value.message.strip():
        raise ValueError("diagnostic exit message must be nonempty")
    _positive_integer(value.nfev, "diagnostic exit nfev", allow_zero=True)
    if value.kind == "native":
        _native_exit(value)
    elif value.status is not None:
        raise ValueError("non-native diagnostic exits cannot invent a native status")


def _exit_kind(value):
    if value.kind in {"not_entered", "zero_dimensional"} and (value.nfev or value.success):
        raise ValueError("unentered diagnostic phases require zero work and no success claim")
    if value.kind in {"budget", "numerical_failure"} and value.success:
        raise ValueError("failed diagnostic phases cannot claim success")
    _entered_exit_kind(value)


def _entered_exit_kind(value):
    if value.kind == "validation" and (value.phase != "handoff" or value.nfev != 1):
        raise ValueError("diagnostic handoff validation requires exactly one charged request")
    if value.kind == "budget" and value.phase == "handoff":
        raise ValueError("handoff has a preallocated request, not an optimizer budget exit")


@dataclass(frozen=True, slots=True)
class DiagnosticSolverExit:
    """An unmodified native exit or an explicitly distinguished internal exit."""

    phase: str
    status: int | None
    success: bool
    message: str
    nfev: int
    kind: str

    def __post_init__(self):
        _exit_fields(self)
        _exit_kind(self)

    def __reduce__(self):
        return type(self), _pickle_values(self)


def _path_transitions(stages, budget):
    localize, handoff, refine = stages
    locked = tuple(stage.kind == "zero_dimensional" for stage in stages)
    if all(locked):
        return
    if any(locked):
        raise ValueError("zero-dimensional diagnostic paths cannot mix with entered stages")
    failed_early = localize.kind == "numerical_failure" or (localize.kind == "budget" and budget < 3)
    if failed_early:
        if handoff.kind != "not_entered" or refine.kind != "not_entered":
            raise ValueError("failed localization cannot enter later diagnostic stages")
        return
    _entered_transitions(localize, handoff, refine)


def _entered_transitions(localize, handoff, refine):
    if localize.kind not in {"native", "budget"} or handoff.kind not in {"validation", "numerical_failure"}:
        raise ValueError("diagnostic path must localize and validate its handoff")
    if handoff.nfev != 1:
        raise ValueError("attempted diagnostic handoff must charge one evaluation")
    allowed = {"native", "budget", "numerical_failure"} if handoff.success else {"not_entered"}
    if refine.kind not in allowed:
        raise ValueError("diagnostic refinement must follow the actual handoff outcome")


def _path_budget(stages, budget):
    localize, handoff, _refine = stages
    cap = (budget - 1) // 2
    if localize.nfev > cap:
        raise ValueError("diagnostic localization exceeds its fixed cap")
    if handoff.nfev > 1 or sum(stage.nfev for stage in stages) > budget:
        raise ValueError("diagnostic path exceeds its original budget")
    _exhausted_budget(stages, budget, cap)


def _exhausted_budget(stages, budget, cap):
    localize, _handoff, refine = stages
    if localize.kind == "budget" and localize.nfev != cap:
        raise ValueError("diagnostic localization budget exit must match its actual cap")
    exhausted = refine.kind == "budget" or (refine.kind == "native" and refine.status == 0)
    if exhausted and sum(stage.nfev for stage in stages) != budget:
        raise ValueError("diagnostic refinement budget exit must exhaust the shared budget")


@dataclass(frozen=True, slots=True)
class DiagnosticPathSummary:
    """One distinct path exit/work combination, repeated count times."""

    budget: int
    stages: tuple[DiagnosticSolverExit, ...]
    count: int = 1

    def __post_init__(self):
        _positive_integer(self.budget, "diagnostic path budget")
        _positive_integer(self.count, "diagnostic path count")
        stages = tuple(self.stages)
        if any(not isinstance(stage, DiagnosticSolverExit) for stage in stages):
            raise TypeError("diagnostic stages must contain DiagnosticSolverExit values")
        stages = tuple(replace(stage) for stage in stages)
        if tuple(stage.phase for stage in stages) != PHASES:
            raise ValueError("diagnostic path requires the ordered three-phase axis")
        _path_budget(stages, self.budget)
        _path_transitions(stages, self.budget)
        object.__setattr__(self, "stages", stages)

    @property
    def nfev(self):
        return sum(stage.nfev for stage in self.stages)

    @property
    def completed(self):
        return self.stages[-1].success or all(stage.kind == "zero_dimensional" for stage in self.stages)

    def __reduce__(self):
        return type(self), _pickle_values(self)


def path_key(path):
    return path.budget, tuple(
        (
            stage.phase,
            stage.kind,
            -3 if stage.status is None else stage.status,
            stage.success,
            stage.message,
            stage.nfev,
        )
        for stage in path.stages
    )


def _declared_axis(paths, declared_paths):
    if not paths:
        return
    dimensions = {path.stages[0].kind == "zero_dimensional" for path in paths}
    if len(dimensions) != 1:
        raise ValueError("diagnostic work cannot mix zero- and nonzero-dimensional paths")
    allowed = {1} if True in dimensions else {3, 4}
    if declared_paths not in allowed:
        raise ValueError("declared diagnostic paths must match the dimensional Sobol start axis")


def _canonical_paths(values):
    paths = tuple(values)
    if any(not isinstance(path, DiagnosticPathSummary) for path in paths):
        raise TypeError("diagnostic work must contain DiagnosticPathSummary values")
    paths = tuple(replace(path) for path in paths)
    keys = tuple(path_key(path) for path in paths)
    if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
        raise ValueError("diagnostic work bins must be unique and canonically ordered")
    if len({path.budget for path in paths}) > 1:
        raise ValueError("diagnostic paths must share the same declared budget")
    return paths


@dataclass(frozen=True, slots=True)
class DiagnosticRefitWork:
    """Canonical joint histogram; all summary counts are derived, not asserted."""

    declared_paths: int = 0
    paths: tuple[DiagnosticPathSummary, ...] = ()

    def __post_init__(self):
        _positive_integer(self.declared_paths, "declared diagnostic paths", allow_zero=True)
        paths = _canonical_paths(self.paths)
        if self.declared_paths > 4 or bool(paths) != bool(self.declared_paths):
            raise ValueError("diagnostic work must bind its nonempty declared path axis")
        _declared_axis(paths, self.declared_paths)
        object.__setattr__(self, "paths", paths)

    @property
    def path_count(self):
        return sum(path.count for path in self.paths)

    @property
    def nfev(self):
        return sum(path.nfev * path.count for path in self.paths)

    @property
    def path_budget(self):
        return self.paths[0].budget if self.paths else 0

    @property
    def max_path_nfev(self):
        return max((path.nfev for path in self.paths), default=0)

    @property
    def completed_paths(self):
        return sum(path.count for path in self.paths if path.completed)

    def __reduce__(self):
        return type(self), _pickle_values(self)


def combine_refit_work(*values: DiagnosticRefitWork) -> DiagnosticRefitWork:
    """Merge all work, including failed paths, without selecting a winner subset."""
    rows = {}
    declaration = None
    for value in values:
        if not isinstance(value, DiagnosticRefitWork):
            raise TypeError("diagnostic work aggregation requires DiagnosticRefitWork")
        value = replace(value)
        if not value.paths:
            continue
        identity = value.declared_paths, value.path_budget
        if declaration is not None and declaration != identity:
            raise ValueError("diagnostic refits changed declared starts or path budget")
        declaration = identity
        for path in value.paths:
            key = path_key(path)
            previous = rows.get(key)
            rows[key] = path if previous is None else replace(previous, count=previous.count + path.count)
    return DiagnosticRefitWork(0 if declaration is None else declaration[0], tuple(rows[key] for key in sorted(rows)))


def _work_value(value):
    if not isinstance(value, DiagnosticRefitWork):
        raise TypeError("diagnostic work must be DiagnosticRefitWork")
    return replace(value)


def _single_refit_work(work):
    if work.path_count > work.declared_paths:
        raise ValueError("diagnostic refit work exceeds declared paths")
    if work.path_count - work.completed_paths > 1:
        raise ValueError("diagnostic refit work cannot continue after its first failed path")


def refit_work(evidence):
    """Validate one runtime result or path prefix against its charged work."""
    work = _work_value(evidence.work)
    _single_refit_work(work)
    if work.nfev != evidence.nfev or work.path_count != evidence.attempted_paths:
        raise ValueError("diagnostic refit work must account for nfev and attempted_paths")
    if evidence.failure_reason is None and work.completed_paths != evidence.attempted_paths:
        raise ValueError("successful diagnostic refit requires complete numerical work")
    return work


def _null_work_counts(evidence, observed, null):
    declared = observed.declared_paths
    if evidence.attempted_count and (not declared or observed.completed_paths != declared):
        raise ValueError("null diagnostic attempts require a complete observed refit")
    if not evidence.successful_count * declared <= null.path_count <= evidence.attempted_count * declared:
        raise ValueError("null diagnostic work must account for successful and attempted refits")
    # Publication/statistic failures can have fully completed numerical work.
    # Only the upper bound is valid: a failed refit need not have a failed path.
    if null.path_count - null.completed_paths > evidence.attempted_count - evidence.successful_count:
        raise ValueError("incomplete null paths cannot exceed the failed replicate count")
    if evidence.successful_count and (not declared or null.completed_paths < evidence.successful_count * declared):
        raise ValueError("successful null refits require their complete numerical work")


def _complete_work(evidence, observed, null):
    declared = observed.declared_paths
    expected = evidence.sample_count * declared
    if not declared or observed.path_count != declared or observed.completed_paths != declared:
        raise ValueError("available diagnostic calibration requires complete observed work")
    if null.path_count != expected or null.completed_paths != expected:
        raise ValueError("available diagnostic calibration requires every declared null path")


def calibration_work(evidence):
    """Own and validate the separate observed/null coupled work histograms.

    Null generation may fail before its optimizer starts, but it cannot erase
    the complete observed refit that preceded every attempted null batch.
    """
    observed, null = _work_value(evidence.observed_work), _work_value(evidence.null_work)
    if evidence.refit_nfev != observed.nfev + null.nfev:
        raise ValueError("diagnostic work must account for every refit_nfev")
    _single_refit_work(observed)
    if null.paths:
        if (observed.declared_paths, observed.path_budget) != (null.declared_paths, null.path_budget):
            raise ValueError("observed and null diagnostic work must use the same estimator")
    _null_work_counts(evidence, observed, null)
    if evidence.status == "available":
        _complete_work(evidence, observed, null)
    return observed, null
