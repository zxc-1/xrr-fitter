"""Member-wise draws followed by one shared refit per bootstrap replicate."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

from xrr_fitter.analysis.bootstrap_generation import (
    BOOTSTRAP_METHODS,
    Recompile,
    bootstrap_source,
    draw_replicates,
    poll_cancelled,
)
from xrr_fitter.analysis.bootstrap_samples import (
    BootstrapProgress,
    TaskRunner,
    bootstrap_result_from_fits,
    run_tasks,
    validated_bootstrap_names,
    validated_sample_count,
)
from xrr_fitter.evaluation import EvaluationConstraintError
from xrr_fitter.model.analysis import BootstrapResult
from xrr_fitter.model.evaluation import ModelEvaluation
from xrr_fitter.model.fitting import FitEvaluationContext


def _refit(contexts, refit, cancelled):
    poll_cancelled(cancelled)
    if isinstance(contexts, str):
        return contexts
    try:
        fitted = refit(contexts)
    except (EvaluationConstraintError, FloatingPointError) as error:
        fitted = f"{type(error).__name__}:{error}"
    poll_cancelled(cancelled)
    return fitted


def _joint_sources(problems, evaluations):
    if len(problems) < 2 or len(problems) != len(evaluations) or any(not item.valid for item in evaluations):
        raise ValueError("joint bootstrap requires valid aligned member evaluations")
    if any(
        problem.config.noise_model != value.noise_model for problem, value in zip(problems, evaluations, strict=True)
    ):
        raise ValueError("joint bootstrap evaluations must match each member's noise model")
    return tuple(
        bootstrap_source(problem, value.model_normalized, value.fit_residuals)
        for problem, value in zip(problems, evaluations, strict=True)
    )


def _joint_method(problems) -> str:
    methods = tuple(BOOTSTRAP_METHODS[problem.config.noise_model] for problem in problems)
    return "joint_" + (methods[0] if len(set(methods)) == 1 else "mixed:" + ",".join(methods))


def bootstrap_joint_local(
    problems: tuple[FitEvaluationContext, ...],
    evaluations: tuple[ModelEvaluation, ...],
    parameter_names: tuple[str, ...],
    *,
    sample_count: int,
    child_seed: int,
    recompile: Recompile,
    refit: Callable[[tuple[FitEvaluationContext, ...]], np.ndarray | str | None],
    cancelled: Callable[[], bool] | None = None,
    progress: BootstrapProgress | None = None,
    task_runner: TaskRunner | None = None,
) -> BootstrapResult:
    names = validated_bootstrap_names(parameter_names)
    count = validated_sample_count(sample_count)
    sources = _joint_sources(problems, evaluations)
    rng = np.random.default_rng(child_seed)
    draws = draw_replicates(sources, count, rng, recompile, cancelled)
    tasks = tuple(partial(_refit, contexts, refit, cancelled) for contexts in draws)
    fits = run_tasks(tasks, task_runner)
    poll_cancelled(cancelled)
    return bootstrap_result_from_fits(names, fits, count, progress, method=_joint_method(problems))
