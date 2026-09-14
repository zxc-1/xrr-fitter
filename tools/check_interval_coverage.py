#!/usr/bin/env python3
"""Explicit, non-CI fixed-seed interval experiment through the public fit API.

Every selected group runs all seeds 0..199; failures are observations, not reasons
to replace seeds. ``--pilot`` runs only seed 0 and never reports a complete
experiment. A separate correlated robust-log bootstrap example is not a coverage
calibration. Report consumers must distinguish interval yield, conditional
coverage, and the all-seed rate that counts missing intervals as noncoverage.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from os import fstat
from pathlib import Path

from scipy.stats import binomtest

import xrr_fitter.api as api

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

import interval_coverage_cases as cases  # noqa: E402
from interval_coverage_evidence import NOMINAL_COVERAGE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = cases.SEEDS
GROUPS = cases.GROUPS
ROBUST_GROUP = cases.ROBUST_GROUP
TARGET = cases.TARGET
TRUTH = cases.TRUTH
MODES = cases.MODES


def _case_record(group: str, seed: int, runner: Callable) -> dict:
    try:
        payload = runner(group, seed)
    except Exception as error:
        payload = cases.failed_outcome(error)
    fitted = bool(payload["fit_available"])
    available = fitted and bool(payload["interval"]["available"])
    bounds = payload["interval"]["bounds"]
    reason = None if available else payload.get("failure_reason", payload["interval"]["unavailable_reason"])
    return payload | {
        "group": group,
        "seed": seed,
        "parameter_name": TARGET,
        "true_value": TRUTH,
        "unit": "angstrom",
        "fit_available": fitted,
        "interval_available": available,
        "success": available,
        "failure_reason": reason,
        "covered": bool(bounds[0] <= TRUTH <= bounds[1]) if available else None,
    }


def _process_case(task: tuple[str, int]) -> dict:
    group, seed = task
    return _case_record(group, seed, cases.run_case)


def _case_rows(group: str, seeds: tuple[int, ...], runner: Callable | None, workers: int) -> list[dict]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return [_case_record(group, seed, cases.run_case if runner is None else runner) for seed in seeds]
    if runner is not None:
        raise ValueError("injected case_runner requires one worker")
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_process_case, ((group, seed) for seed in seeds)))


def _binomial(covered: int, trials: int) -> dict | None:
    if trials == 0:
        return None
    test = binomtest(covered, trials, NOMINAL_COVERAGE, alternative="two-sided")
    return {
        "successes": covered,
        "trials": trials,
        "null_probability": NOMINAL_COVERAGE,
        "alternative": "two-sided",
        "pvalue": float(test.pvalue),
        "reject_null_at_0_05": bool(test.pvalue < 0.05),
    }


def _coverage(rows: list[dict]) -> dict:
    available = sum(row["interval_available"] for row in rows)
    covered = sum(row["covered"] is True for row in rows)
    return {
        "failed_count": sum(not row["success"] for row in rows),
        "fit_failed_count": sum(not row["fit_available"] for row in rows),
        "interval_available_count": available,
        "interval_unavailable_count": len(rows) - available,
        "covered_count": covered,
        "interval_yield": available / len(rows),
        "conditional_coverage": None if available == 0 else covered / available,
        "all_seed_coverage": covered / len(rows),
        "conditional_binomial_test": _binomial(covered, available),
        "conditional_test_unavailable_reason": "no_available_intervals" if available == 0 else None,
        "all_seed_binomial_test": _binomial(covered, len(rows)),
    }


def run_group(group: str, *, pilot: bool = False, case_runner: Callable | None = None, workers: int = 1) -> dict:
    if group not in GROUPS:
        raise ValueError(f"unknown coverage group: {group}")
    seeds = (0,) if pilot else SEEDS
    rows = _case_rows(group, seeds, case_runner, workers)
    return {
        "group": group,
        "noise_model": MODES[group],
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "required_seed_count": len(SEEDS),
        "protocol_complete": not pilot,
        "nominal_coverage": NOMINAL_COVERAGE,
        "success_definition": "valid public fit and an available formal 95% interval; covering truth is not required",
        "cases": rows,
    } | _coverage(rows)


def run_robust_representative() -> dict:
    return _case_record(ROBUST_GROUP, 0, cases.run_case)


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, action="append")
    parser.add_argument("--pilot", action="store_true", help="only fixed seed 0; never full acceptance")
    parser.add_argument("--workers", type=_positive_integer, default=1)
    return parser.parse_args(argv)


def _protocol_reason(args, groups: tuple[str, ...], reports: list, robust: dict) -> str | None:
    if args.pilot:
        return "pilot_only"
    if set(groups) != set(GROUPS):
        return "not_all_declared_groups"
    if not all(report["protocol_complete"] for report in reports):
        return "incomplete_seed_group"
    interval = robust["interval"]
    attempts = interval.get("details", {}).get("attempted_count", 0)
    if attempts < 200 or interval["method"] != "robust_log_moving_block":
        return "robust_bootstrap_not_completed"
    return None


def run_experiment(args: argparse.Namespace) -> dict:
    groups = tuple(GROUPS if args.group is None else args.group)
    if len(set(groups)) != len(groups):
        raise ValueError("coverage groups must be unique")
    reports = []
    for group in groups:
        print(f"coverage: {group}; seeds={'0 (pilot)' if args.pilot else '0..199'}", file=sys.stderr)
        reports.append(run_group(group, pilot=args.pilot, workers=args.workers))
    robust = run_robust_representative() | {"coverage_calibration_claim": False, "excluded_from_formal_groups": True}
    reason = _protocol_reason(args, groups, reports, robust)
    return {
        "schema": "xrr-interval-coverage-v1",
        "algorithm_version": api.new_project().algorithm_version,
        "objective_version": api.FitConfig.fast(0).objective_version,
        "nominal_coverage": NOMINAL_COVERAGE,
        "seeds": [0] if args.pilot else list(SEEDS),
        "seed_count": 1 if args.pilot else len(SEEDS),
        "required_seed_count": len(SEEDS),
        "declared_groups": list(GROUPS),
        "selected_groups": list(groups),
        "seed_protocol_complete": not args.pilot and set(groups) == set(GROUPS),
        "protocol_complete": reason is None,
        "protocol_incomplete_reason": reason,
        "experiment_definition": cases.experiment_definition(),
        "workers": args.workers,
        "groups": reports,
        "robust_representative": robust,
        "interpretation": {
            "conditional_coverage": "covered / available; eligibility is data-dependent, so selection can affect this rate",
            "all_seed_coverage": "covered / all fixed seeds; missing intervals count as noncoverage, not discarded cases",
            "binomial_tests": "exact two-sided evidence against p=0.95; no automatic pass or threshold adjustment",
            "robust": "one correlated moving-block bootstrap example; not 95% coverage calibration",
            "scope": "known non-target parameters; synthetic single-layer thickness only; not general or real-data calibration",
        },
    }


def canonical_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _discard_reserved_report(target: Path, identity: tuple[int, int], error: BaseException) -> None:
    """Remove only our incomplete inode; a replacement belongs to its writer."""
    try:
        current = target.lstat()
        if (current.st_dev, current.st_ino) == identity:
            target.unlink()
    except FileNotFoundError:
        return
    except OSError as cleanup_error:
        error.add_note(f"could not clean incomplete coverage report: {cleanup_error}")


def _write_report(args: argparse.Namespace, target: Path) -> None:
    stream = target.open("x", encoding="utf-8")
    created = fstat(stream.fileno())
    try:
        with stream:
            serialized = canonical_json(run_experiment(args))
            stream.write(serialized)
        current = target.lstat()
        if (current.st_dev, current.st_ino) != (created.st_dev, created.st_ino):
            raise RuntimeError(f"coverage report output was replaced: {target}")
    except BaseException as error:
        _discard_reserved_report(target, (created.st_dev, created.st_ino), error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target = args.output.resolve()
    if target.is_relative_to(ROOT):
        raise ValueError("coverage report must be outside the repository")
    if target.exists():
        raise FileExistsError(f"coverage report already exists: {target}")
    _write_report(args, target)
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
