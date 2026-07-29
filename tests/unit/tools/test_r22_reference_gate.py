from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.unit.tools.test_compare_r22_reference import GROUPS, _canonical, _reference


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _tree_hash(root: Path) -> tuple[str, int]:
    records = []
    for path in root.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            records.append(
                (
                    path.relative_to(root).as_posix(),
                    len(content),
                    hashlib.sha256(content).hexdigest(),
                )
            )
    digest = hashlib.sha256()
    for path, size, sha256 in sorted(records):
        for value in (path.encode(), str(size).encode(), sha256.encode()):
            digest.update(_frame(value))
    return digest.hexdigest(), len(records)


def _oracle_binding(tmp_path: Path) -> tuple[Path, Path, Path, str, int]:
    verification = tmp_path / "verification"
    r22_root = verification / "r22"
    manifest = _reference(r22_root)
    collections = r22_root / "collections"
    collections.mkdir()
    (collections / "tests-active.json").write_bytes(_canonical({"suite": "tests"}))
    (r22_root / "reference-sidecar-lock.json").write_bytes(
        _canonical({"schema": "fixture-lock-v1"})
    )
    tree_sha256, file_count = _tree_hash(r22_root)
    release_spec = verification / "release-spec.json"
    release_spec.write_bytes(
        _canonical(
            {
                "schema": "xrr-r23-release-spec-v1",
                "r22_oracle_tree_sha256": tree_sha256,
                "r22_oracle_file_count": file_count,
            }
        )
    )
    return manifest, collections, release_spec, tree_sha256, file_count


def _fixture_registry() -> dict[str, object]:
    return {
        group: (
            lambda _context, selected=group: {
                f"golden/{selected}.json": {"group": selected}
            }
        )
        for group in GROUPS
    }


def test_self_check_binds_complete_r22_tree_to_release_spec(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest, collections, release_spec, tree_sha256, file_count = _oracle_binding(tmp_path)

    observed = module.self_check(
        manifest,
        collections_root=collections,
        release_spec=release_spec,
    )

    assert observed["r22_oracle_tree_sha256"] == tree_sha256
    assert observed["r22_oracle_file_count"] == file_count
    (collections / "tests-active.json").write_bytes(_canonical({"suite": "drift"}))
    with pytest.raises(ValueError, match="oracle.*(hash|tree|digest)|digest.*drift"):
        module.self_check(
            manifest,
            collections_root=collections,
            release_spec=release_spec,
        )


def test_self_check_rejects_wrong_collections_root_and_noncanonical_release_spec(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest, collections, release_spec, _tree_sha256, _file_count = _oracle_binding(tmp_path)
    wrong = tmp_path / "collections"
    wrong.mkdir()

    with pytest.raises(ValueError, match="collections"):
        module.self_check(
            manifest,
            collections_root=wrong,
            release_spec=release_spec,
        )

    release_spec.write_text('{"schema": "xrr-r23-release-spec-v1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        module.self_check(
            manifest,
            collections_root=collections,
            release_spec=release_spec,
        )


def test_all_groups_requires_exact_order_and_writes_canonical_external_report(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path)
    repository = tmp_path / "repository"
    repository.mkdir()
    report_dir = tmp_path / "report"

    observed = module.compare_all_groups(
        manifest,
        report_dir,
        registry=_fixture_registry(),
        repo_root=repository,
    )

    report = report_dir / "r22-reference-report.json"
    assert observed == json.loads(report.read_bytes())
    assert report.read_bytes() == _canonical(observed)
    assert observed["schema"] == "xrr-r22-reference-comparison-v1"
    assert observed["status"] == "PASS"
    assert observed["group_count"] == 8
    assert tuple(result["group"] for result in observed["groups"]) == GROUPS
    assert all(result["status"] == "PASS" for result in observed["groups"])


@pytest.mark.parametrize(
    "registry",
    [
        {group: adapter for group, adapter in list(_fixture_registry().items())[:-1]},
        {**_fixture_registry(), "extra": lambda _context: {}},
        dict(reversed(tuple(_fixture_registry().items()))),
    ],
    ids=("missing", "extra", "order"),
)
def test_all_groups_rejects_registry_coverage_or_order_drift_before_writing(
    tmp_path: Path,
    load_tool_module,
    registry: dict[str, object],
) -> None:
    module = load_tool_module("compare_r22_reference")
    report_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="registry|eight|group|order"):
        module.compare_all_groups(
            _reference(tmp_path),
            report_dir,
            registry=registry,
            repo_root=tmp_path / "repository",
        )

    assert not report_dir.exists()


def test_all_groups_rejects_report_directory_inside_repository(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    repository = tmp_path / "repository"
    repository.mkdir()

    with pytest.raises(ValueError, match="outside the repository"):
        module.compare_all_groups(
            _reference(tmp_path),
            repository / "report",
            registry=_fixture_registry(),
            repo_root=repository,
        )
