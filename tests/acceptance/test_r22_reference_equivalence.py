from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
COMPARATOR = ROOT / "tools/compare_r22_reference.py"
MANIFEST = ROOT / "verification/r22/reference/manifest.json"
COLLECTIONS = ROOT / "verification/r22/collections"
RELEASE_SPEC = ROOT / "verification/release-spec.json"
GROUPS = (
    "model_project",
    "io",
    "physics",
    "fit_compile",
    "fit_search",
    "analysis",
    "services",
    "gui",
)


def _oracle_snapshot() -> tuple[tuple[str, int, str], ...]:
    oracle = ROOT / "verification/r22"
    return tuple(
        (
            path.relative_to(oracle).as_posix(),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(oracle.rglob("*"))
        if path.is_file()
    )


def _run(*args: str) -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        (sys.executable, str(COMPARATOR), *args),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_frozen_r22_reference_matches_all_registered_r23_groups(tmp_path: Path) -> None:
    before = _oracle_snapshot()
    checked = _run(
        "--self-check",
        str(MANIFEST),
        "--collections-root",
        str(COLLECTIONS),
        "--release-spec",
        str(RELEASE_SPEC),
    )
    report_dir = tmp_path / "comparison"
    compared = _run(
        "--all-groups",
        "--manifest",
        str(MANIFEST),
        "--report-dir",
        str(report_dir),
    )

    release_spec = json.loads(RELEASE_SPEC.read_bytes())
    assert checked["r22_oracle_tree_sha256"] == release_spec["r22_oracle_tree_sha256"]
    assert checked["r22_oracle_file_count"] == release_spec["r22_oracle_file_count"]
    assert compared["status"] == "PASS"
    assert compared["group_count"] == 8
    assert tuple(result["group"] for result in compared["groups"]) == GROUPS
    report = report_dir / "r22-reference-report.json"
    assert json.loads(report.read_bytes()) == compared
    assert _oracle_snapshot() == before
