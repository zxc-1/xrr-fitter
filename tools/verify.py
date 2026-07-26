#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence


PYTHON = "{python}"
REPORT = "{report}"
Runner = Callable[..., object]


@dataclass(frozen=True)
class Mode:
    commands: tuple[tuple[str, ...], ...]


PYTEST_PREFIX = (
    PYTHON,
    "-m",
    "pytest",
    "-o",
    "addopts=",
    "--strict-config",
    "--strict-markers",
    "-p",
    "no:cacheprovider",
    "-p",
    "tests.outcome_gate",
    "--basetemp",
    f"{REPORT}/pytest-tmp",
)

MODE_REGISTRY: Mapping[str, Mode] = {
    "quality": Mode(
        commands=(
            (PYTHON, "tools/check_radon.py", "--output", f"{REPORT}/radon.json"),
            PYTEST_PREFIX
            + (
                "tests/architecture/test_dependency_rules.py",
                "tests/architecture/test_naming_rules.py",
                "tests/architecture/test_quality_gate.py",
                "tests/architecture/test_removed_legacy_modules.py",
                "-q",
            ),
        )
    ),
    "tools": Mode(commands=(PYTEST_PREFIX + ("tests/unit/tools", "-q"),)),
    "unit": Mode(
        commands=(PYTEST_PREFIX + ("tests/unit/model", "tests/unit/io", "-q"),)
    ),
}


def build_environment(repo_root: str | Path, report_dir: str | Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    report = Path(report_dir).resolve()
    mpl = report / "mpl-cache"
    xdg = report / "xdg-cache"
    mpl.mkdir(parents=True, exist_ok=True)
    xdg.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PYTEST_"):
            environment.pop(name)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(mpl)
    environment["XDG_CACHE_HOME"] = str(xdg)
    return environment


def _materialize(command: Sequence[str], report_dir: Path) -> tuple[str, ...]:
    replacements = {PYTHON: sys.executable, REPORT: str(report_dir)}
    return tuple(
        replacements.get(token, token.replace(REPORT, str(report_dir))) for token in command
    )


def _invoke(
    runner: Runner,
    command: Sequence[str],
    *,
    root: Path,
    environment: Mapping[str, str],
) -> None:
    runner(tuple(command), cwd=root, env=dict(environment), check=True)


def run_mode(
    name: str,
    mode: Mode,
    *,
    repo_root: str | Path,
    report_dir: str | Path,
    runner: Runner = subprocess.run,
) -> None:
    root = Path(repo_root).resolve()
    report = Path(report_dir).resolve()
    report.mkdir(parents=True, exist_ok=True)
    environment = build_environment(root, report)
    hygiene = (sys.executable, "tools/check_hygiene.py")
    _invoke(runner, hygiene, root=root, environment=environment)
    failure: BaseException | None = None
    try:
        for command in mode.commands:
            args = _materialize(command, report)
            _invoke(runner, args, root=root, environment=environment)
    except BaseException as error:
        failure = error
    try:
        _invoke(runner, hygiene, root=root, environment=environment)
    except BaseException as error:
        if failure is None:
            failure = error
    if failure is not None:
        raise failure


def _run_with_report(name: str, report_dir: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    run_mode(name, MODE_REGISTRY[name], repo_root=root, report_dir=report_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=tuple(MODE_REGISTRY))
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    if args.report_dir is not None:
        _run_with_report(args.mode, args.report_dir)
        return 0
    with tempfile.TemporaryDirectory(prefix=f"xrr-r23-{args.mode}-") as directory:
        _run_with_report(args.mode, Path(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
