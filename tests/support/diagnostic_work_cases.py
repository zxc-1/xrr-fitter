"""Small explicit work fixtures; they do not claim to run an optimizer."""

from xrr_fitter.model.diagnostic_work import (
    PHASES,
    DiagnosticPathSummary,
    DiagnosticRefitWork,
    DiagnosticSolverExit,
    combine_refit_work,
)


def _native(phase, nfev, *, success=True):
    status = (0 if phase == "localize" else 3) if success else (2 if phase == "localize" else -2)
    return DiagnosticSolverExit(phase, status, success, "fixture native exit", nfev, "native")


def _handoff(success=True):
    return DiagnosticSolverExit("handoff", None, success, "fixture validation", 1, "validation")


def zero_work(paths=1, *, budget=80):
    stages = tuple(
        DiagnosticSolverExit(phase, None, False, "zero_dimensional", 0, "zero_dimensional") for phase in PHASES
    )
    return DiagnosticRefitWork(1, (DiagnosticPathSummary(budget, stages, paths),))


def completed_work(paths=4, *, nfev=None, declared_paths=4, budget=80):
    nfev = 3 * paths if nfev is None else nfev
    quotient, remainder = divmod(nfev, paths)
    assert quotient >= 3
    values = []
    for spent, count in ((quotient, paths - remainder), (quotient + 1, remainder)):
        if count:
            stages = (_native("localize", 1), _handoff(), _native("refine", spent - 2))
            values.append(DiagnosticRefitWork(declared_paths, (DiagnosticPathSummary(budget, stages, count),)))
    return combine_refit_work(*values)


def failed_work(nfev=1, *, declared_paths=4, budget=80):
    skipped = {phase: DiagnosticSolverExit(phase, None, False, "not_entered", 0, "not_entered") for phase in PHASES}
    if nfev == 1:
        stages = (
            DiagnosticSolverExit("localize", None, False, "fixture numeric failure", 1, "numerical_failure"),
            skipped["handoff"],
            skipped["refine"],
        )
    elif nfev == 2:
        stages = (_native("localize", 1), _handoff(False), skipped["refine"])
    else:
        stages = (_native("localize", 1), _handoff(), _native("refine", nfev - 2, success=False))
    return DiagnosticRefitWork(declared_paths, (DiagnosticPathSummary(budget, stages),))
