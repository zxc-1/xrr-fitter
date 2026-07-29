"""Immutable values and deterministic identities for owner-data workflows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SCHEMA = "xrr-r23-approved-data-candidate-v1"
NOT_RUN = "NOT_RUN: owner post-delivery acceptance"
OPERATIONS = (
    "load_project",
    "preview_source_update",
    "accept_source_update",
    "preflight_fit",
    "fit_project",
    "select_candidate",
    "save_project",
    "load_project_round_trip",
    "export_result",
)


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RunRecord:
    ordinal: int
    seed: int
    project: FileRecord
    exports: tuple[FileRecord, ...]
    plots: tuple[FileRecord, ...]
    normalized_result: dict[str, object]


@dataclass(frozen=True, slots=True)
class CaseRecord:
    case_id: str
    source: FileRecord
    configuration_sha256: str
    operations: tuple[str, ...]
    runs: tuple[RunRecord, ...]
    normalized_result: dict[str, object]
    conclusion: str


@dataclass(frozen=True, slots=True)
class GuiResult:
    case_id: str
    project_reopened: bool
    exports_verified: bool
    plots_verified: bool


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    source_name: str
    template_name: str
    repeated_seed: int
    fresh_seed: int


CASE_SPECS = (
    CaseSpec("known_single_layer", "known_single_layer.dat", "single-layer.xrrproj.json", 1101, 1102),
    CaseSpec("unstable_multilayer", "unstable_multilayer.dat", "mo-si-periodic.xrrproj.json", 3301, 3302),
    CaseSpec(
        "workable_mo_si_multilayer",
        "workable_mo_si_multilayer.dat",
        "mo-si-periodic.xrrproj.json",
        2201,
        2202,
    ),
)


def canonical_json_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("candidate evidence is not canonical JSON data") from error
    return (payload + "\n").encode("utf-8")


def file_record(path: Path, relative: str) -> FileRecord:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or pure.as_posix() != relative or ".." in pure.parts:
        raise ValueError("evidence path must be a normalized relative POSIX path")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence must be a regular file: {relative}")
    content = path.read_bytes()
    if not content:
        raise ValueError(f"evidence must not be empty: {relative}")
    return FileRecord(relative, len(content), hashlib.sha256(content).hexdigest())


def file_value(record: FileRecord) -> dict[str, object]:
    return {"path": record.path, "size": record.size, "sha256": record.sha256}


def owner_source(root: Path, spec: CaseSpec) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("owner-data root must be a regular directory")
    source = root / spec.source_name
    if source.is_symlink():
        raise ValueError(f"owner source must be a regular file: {spec.source_name}")
    if not source.is_file():
        raise FileNotFoundError(f"missing owner source: {spec.source_name}")
    if not source.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"owner source escapes its root: {spec.source_name}")
    return source


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _tree_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"identity input must be a regular file: {path}")
        content = path.read_bytes()
        values = (
            path.relative_to(ROOT).as_posix().encode("utf-8"),
            str(len(content)).encode("ascii"),
            hashlib.sha256(content).hexdigest().encode("ascii"),
        )
        for value in values:
            digest.update(_frame(value))
    return digest.hexdigest()


def candidate_environment() -> dict[str, object]:
    machine = platform.machine().lower()
    if sys.version_info[:2] != (3, 12) or sys.platform != "darwin" or machine != "arm64":
        raise RuntimeError("approved-data capture requires Python 3.12 on macOS arm64")
    from PySide6 import __version__ as pyside_version
    from PySide6.QtCore import qVersion

    lock = ROOT / "requirements-macos-arm64-py312.lock"
    acceptance_paths = (
        ROOT / "tests/acceptance/test_real_data_workflows.py",
        ROOT / "tests/acceptance/test_gui_real_data_workflows.py",
        ROOT / "tests/support/approved_workflows.py",
        ROOT / "tests/support/approved_workflow_capture.py",
        ROOT / "tests/support/approved_workflow_gui.py",
        Path(__file__).resolve(),
    )
    return {
        "python_version": platform.python_version(),
        "platform": "macos-arm64",
        "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "production_tree_sha256": _tree_sha256((ROOT / "src/xrr_fitter").rglob("*.py")),
        "acceptance_test_tree_sha256": _tree_sha256(acceptance_paths),
        "qt_runtime_identity": f"PySide6 {pyside_version} / Qt {qVersion()}",
    }


def workflow_contract_sha256() -> str:
    value = {
        "schema": "xrr-r23-approved-workflow-v1",
        "operations": list(OPERATIONS),
        "cases": [
            {
                "case_id": spec.case_id,
                "source_name": spec.source_name,
                "template_name": spec.template_name,
                "repeated_seed": spec.repeated_seed,
                "fresh_seed": spec.fresh_seed,
            }
            for spec in CASE_SPECS
        ],
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def configuration_sha256(spec: CaseSpec, source: FileRecord) -> str:
    template = ROOT / "examples" / spec.template_name
    document = json.loads(template.read_text(encoding="utf-8"))
    document["fit_config"]["master_seed"] = 0
    document["datasets"][0]["source_path"] = spec.source_name
    document["datasets"][0]["source_sha256"] = source.sha256
    value = {
        "schema": "xrr-r23-approved-configuration-v1",
        "case_id": spec.case_id,
        "template": document,
        "operations": list(OPERATIONS),
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalize_source_paths(value: object, source_name: str) -> object:
    if isinstance(value, dict):
        return {
            key: source_name if key == "source_path" else normalize_source_paths(item, source_name)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_source_paths(item, source_name) for item in value]
    return value
