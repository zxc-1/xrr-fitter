"""One handler per CLI subcommand, reaching domain behavior only through api."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import xrr_fitter.api as api
from xrr_fitter.cli import exit_codes
from xrr_fitter.cli import progress as progress_module


class CommandError(Exception):
    """A user-facing failure carrying its own exit code."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def _load(path: str) -> api.XrrProject:
    target = Path(path)
    if not target.is_file():
        raise CommandError(f"工程文件不存在：{target}", exit_codes.INVALID_INPUT)
    try:
        return api.load_project(target)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise CommandError(f"工程文件不合法：{error}", exit_codes.INVALID_INPUT) from error


def _require_fresh_sources(project: api.XrrProject) -> None:
    validation = api.inspect_sources(project)
    if not validation.valid:
        detail = "；".join(item.message for item in validation.issues) or "源文件校验失败"
        raise CommandError(detail, exit_codes.STALE_SOURCE)


def _progress_sink(as_json: bool) -> Callable[[api.FitProgress], None]:
    if as_json:
        return lambda event: print(progress_module.render_json(event), flush=True)
    return lambda event: print(progress_module.render_text(event), file=sys.stderr, flush=True)


def run_validate(arguments) -> int:
    """Check sources and fit readiness without running any optimizer."""
    project = _load(arguments.project)
    _require_fresh_sources(project)
    readiness = api.preflight_fit(project)
    print(readiness.message)
    return exit_codes.SUCCESS if readiness.ready else exit_codes.INVALID_INPUT


def run_fit(arguments) -> int:
    """Run the fit pipeline, resuming automatically from any stored checkpoint."""
    project = _load(arguments.project)
    _require_fresh_sources(project)
    sink = _progress_sink(arguments.json_progress)
    result = _fit_result(project, arguments, sink)
    if arguments.output is not None:
        api.save_project(result.updated_project, arguments.output)
    for warning in result.warnings:
        print(warning, file=sys.stderr)
    return exit_codes.fit_exit_code(result)


def _fit_result(project, arguments, sink) -> api.ProjectFitResult:
    if arguments.auto:
        return api.fit_automatically(project, progress_callback=sink)
    readiness = api.preflight_fit(project)
    if not readiness.ready:
        raise CommandError(readiness.message, exit_codes.INVALID_INPUT)
    return api.fit_project(project, progress_callback=sink)


def run_mcmc(arguments) -> int:
    """Sample the selected candidate and persist the augmented project."""
    project = _load(arguments.project)
    _require_fresh_sources(project)
    config = api.McmcConfig(
        walkers=arguments.walkers,
        burn_in=arguments.burn_in,
        production_steps=arguments.steps,
    )
    updated = api.run_mcmc(
        project,
        arguments.dataset,
        arguments.candidate,
        config,
        progress_callback=_progress_sink(arguments.json_progress),
    )
    api.save_project(updated, arguments.output or arguments.project)
    return exit_codes.SUCCESS


def run_export(arguments) -> int:
    """Publish an existing project's results atomically."""
    project = _load(arguments.project)
    manifest = api.export_result(project, arguments.output_dir, include_ort=arguments.ort)
    print(manifest.run_directory)
    return exit_codes.SUCCESS
