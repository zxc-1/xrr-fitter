#!/usr/bin/env python3
"""Non-CI wall-clock and deterministic-work benchmark for automatic fitting."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import monotonic

import xrr_fitter.api as api

if __name__ == "__main__":
    sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
MASTER_SEED = 1701


def _validate_run_identity(
    case_id: str,
    repeat_index: int,
    elapsed_seconds: float,
) -> None:
    if not case_id or repeat_index < 0:
        raise ValueError("case_id and repeat_index must identify one run")
    if not isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise ValueError("elapsed_seconds must be finite and nonnegative")


def _validated_stages(
    stage_nfev: tuple[tuple[str, int], ...],
    total_nfev: int,
) -> tuple[tuple[str, int], ...]:
    stages = tuple(stage_nfev)
    if len({name for name, _count in stages}) != len(stages):
        raise ValueError("stage_nfev names must be unique")
    if any(not name or count < 0 for name, count in stages):
        raise ValueError("stage_nfev values must be named and nonnegative")
    if total_nfev != sum(count for _name, count in stages):
        raise ValueError("total_nfev must equal the stage_nfev sum")
    return stages


def _validate_analysis_counts(bootstrap_count: int, profile_count: int) -> None:
    if bootstrap_count < 0 or profile_count < 0:
        raise ValueError("analysis work counts must be nonnegative")


def _validated_dataset_metrics(
    statuses: tuple[tuple[str, str], ...],
    recovery_errors: tuple[tuple[str, float | None], ...],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, float | None], ...]]:
    status_values = tuple(statuses)
    error_values = tuple(recovery_errors)
    status_ids = tuple(dataset_id for dataset_id, _status in status_values)
    error_ids = tuple(dataset_id for dataset_id, _error in error_values)
    if not status_values or len(set(status_ids)) != len(status_ids):
        raise ValueError("statuses must identify unique datasets")
    if error_ids != status_ids:
        raise ValueError("recovery_errors must align with statuses")
    return status_values, error_values


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", action="store_true")
    mode.add_argument("--batch-size", type=int, choices=(2, 3, 4))
    mode.add_argument("--adaptive", action="store_true")
    parser.add_argument("--repeat", type=_positive_integer, default=1)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(argv)


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    case_id: str
    repeat_index: int
    elapsed_seconds: float
    stage_nfev: tuple[tuple[str, int], ...]
    total_nfev: int
    bootstrap_count: int
    profile_count: int
    statuses: tuple[tuple[str, str], ...]
    recovery_errors: tuple[tuple[str, float | None], ...]

    def __post_init__(self) -> None:
        _validate_run_identity(
            self.case_id,
            self.repeat_index,
            self.elapsed_seconds,
        )
        stages = _validated_stages(self.stage_nfev, self.total_nfev)
        _validate_analysis_counts(self.bootstrap_count, self.profile_count)
        statuses, errors = _validated_dataset_metrics(
            self.statuses,
            self.recovery_errors,
        )
        object.__setattr__(self, "stage_nfev", stages)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "recovery_errors", errors)


def _run_dict(run: BenchmarkRun) -> dict[str, object]:
    value = asdict(run)
    value["stage_nfev"] = dict(run.stage_nfev)
    value["status"] = dict(run.statuses)
    value["recovery_error"] = dict(run.recovery_errors)
    value.pop("statuses")
    value.pop("recovery_errors")
    return value


def build_report(
    mode: str,
    batch_size: int | None,
    repeat: int,
    runs: tuple[BenchmarkRun, ...],
) -> dict[str, object]:
    if not runs:
        raise ValueError("benchmark report requires at least one run")
    elapsed = tuple(run.elapsed_seconds for run in runs)
    return {
        "schema": "xrr-automatic-benchmark-v1",
        "mode": mode,
        "batch_size": batch_size,
        "repeat": repeat,
        "median_seconds": float(median(elapsed)),
        "maximum_seconds": float(max(elapsed)),
        "runs": [_run_dict(run) for run in runs],
    }


def canonical_json(report: dict[str, object]) -> str:
    return json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _recovery_support():
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tests.support import automatic_recovery

    return automatic_recovery


def _benchmark_fixture(root: Path, case_id: str, count: int):
    recovery = _recovery_support()
    fixture = (
        recovery.build_direct_sld_fixture(root)
        if count == 1
        else recovery.build_shared_local_fixture(root)
    )
    project = replace(fixture.project, datasets=fixture.project.datasets[:count])
    return replace(
        fixture,
        case_id=case_id,
        project=project,
        targets=fixture.targets[:count],
    )


def _adaptive_cases():
    recovery = _recovery_support()
    return (
        ("direct-sld", recovery.build_direct_sld_fixture),
        ("shared-local", recovery.build_shared_local_fixture),
        ("isolated-outlier", recovery.build_isolated_outlier_fixture),
        ("roughness-release", recovery.build_roughness_release_fixture),
    )


def _recovery_error(dataset: api.DatasetProject, target) -> float | None:
    result = dataset.last_valid_result
    candidate = None if result is None else result.best_candidate
    if candidate is None:
        return None
    fitted = {parameter.name: parameter.value for parameter in candidate.parameters}
    if any(name not in fitted for name, _truth in target.parameters):
        return None
    errors = tuple(
        abs(fitted[name] - truth) / abs(truth)
        if truth != 0.0
        else abs(fitted[name] - truth)
        for name, truth in target.parameters
    )
    return max(errors, default=0.0)


def _benchmark_recovery_case(
    fixture,
    repeat_index: int,
) -> BenchmarkRun:
    ledger = _recovery_support().AutomaticWorkLedger()
    started = monotonic()
    result = api.fit_automatically(
        fixture.project,
        fixture.import_batch_id,
        checkpoint_callback=ledger.observe,
    )
    elapsed = monotonic() - started
    updated = result.updated_project
    ledger.observe(updated)
    statuses = tuple(
        (dataset.dataset_id, dataset.automation.status.value)
        for dataset in updated.datasets
    )
    errors = tuple(
        (dataset.dataset_id, _recovery_error(dataset, target))
        for dataset, target in zip(
            updated.datasets,
            fixture.targets,
            strict=True,
        )
    )
    stages, bootstrap, profiles = ledger.metrics()
    return BenchmarkRun(
        fixture.case_id,
        repeat_index,
        elapsed,
        stages,
        sum(count for _stage, count in stages),
        bootstrap,
        profiles,
        statuses,
        errors,
    )


def _mode(args: argparse.Namespace) -> tuple[str, int | None]:
    if args.single:
        return "single", None
    if args.batch_size is not None:
        return "batch", args.batch_size
    return "adaptive", None


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    mode, batch_size = _mode(args)
    runs = []
    with TemporaryDirectory(prefix="xrr-automatic-benchmark-") as directory:
        root = Path(directory)
        for repeat_index in range(args.repeat):
            if mode == "adaptive":
                for case_id, builder in _adaptive_cases():
                    case_root = root / f"{repeat_index}-{case_id}"
                    case_root.mkdir()
                    fixture = builder(case_root)
                    if fixture.case_id != case_id:
                        raise RuntimeError("adaptive fixture registry identity mismatch")
                    runs.append(
                        _benchmark_recovery_case(
                            fixture,
                            repeat_index,
                        )
                    )
                continue
            count = 1 if mode == "single" else batch_size
            assert count is not None
            case_id = "single" if count == 1 else f"batch-{count}"
            case_root = root / f"{repeat_index}-{case_id}"
            case_root.mkdir()
            fixture = _benchmark_fixture(case_root, case_id, count)
            runs.append(
                _benchmark_recovery_case(
                    fixture,
                    repeat_index,
                )
            )
    return build_report(mode, batch_size, args.repeat, tuple(runs))


def _print_human(report: dict[str, object]) -> None:
    print(f"mode: {report['mode']}")
    print(f"median_seconds: {report['median_seconds']:.6f}")
    print(f"maximum_seconds: {report['maximum_seconds']:.6f}")
    for run in report["runs"]:
        print(
            f"{run['case_id']}[{run['repeat_index']}]: "
            f"{run['elapsed_seconds']:.6f}s, nfev={run['total_nfev']}, "
            f"profiles={run['profile_count']}, status={run['status']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    if args.as_json:
        print(canonical_json(report))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
