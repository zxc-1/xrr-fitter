"""Statistical input routing: explicit fitting or provenance-bound replay, never fallback."""

from __future__ import annotations

from pathlib import Path

from verify_registry import REPORT, Mode
from verify_report import _resolve_output_path


def _require_option_scope(
    active: bool,
    name: str,
    allowed_modes: set[str],
    message: str,
) -> None:
    if active and name not in allowed_modes:
        raise ValueError(message)


def _validate_statistical_permission(
    name: str, results: object | None, compute: bool, producer: object | None = None
) -> None:
    _require_option_scope(
        producer is not None,
        name,
        {"statistical", "release"},
        "statistical producer is only valid for statistical or release",
    )
    if producer is not None and results is None:
        raise ValueError("statistical producer requires explicit result inputs")
    _require_option_scope(
        compute,
        name,
        {"statistical", "release"},
        "statistical compute permission is only valid for statistical or release",
    )
    if compute and results is not None:
        raise ValueError("statistical compute and result replay are mutually exclusive")
    if name in {"statistical", "release"} and results is None and compute is not True:
        raise ValueError("statistical fitting requires explicit --compute-statistical or --statistical-results")


def _statistical_results_path(name: str, root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if name not in {"statistical", "release"}:
        raise ValueError("statistical results are only valid for statistical or release")
    path = _resolve_output_path(value, "statistical results")
    if path.is_relative_to(root) or not path.is_dir():
        raise ValueError("statistical results must be an existing external directory")
    return path


def _statistical_producer_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = _resolve_output_path(value, "statistical producer")
    if path.is_relative_to(root) or not path.is_file():
        raise ValueError("statistical producer must be an existing external descriptor")
    return path


def _statistical_input_mode(mode: Mode, results: Path | None, producer: Path | None) -> Mode:
    if results is None:
        options = ("-p", "tests.statistical_gate", "--compute-statistical")
        return Mode(tuple((*command, *options) for command in mode.commands))
    options = (
        "-p",
        "tests.statistical_gate",
        "--statistical-results",
        str(results),
        "--statistical-report",
        f"{REPORT}/statistical-evidence.json",
    )
    if producer is not None:
        options = (*options, "--statistical-producer", str(producer))
    return Mode(tuple((*command, *options) for command in mode.commands))
