from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import tomllib
from pathlib import Path

import pytest

BUILD = ("setuptools==75.8.2", "wheel==0.45.1")
GENERATED_METADATA = (
    "PKG-INFO",
    "setup.cfg",
    "src/xrr_fitter.egg-info/PKG-INFO",
    "src/xrr_fitter.egg-info/SOURCES.txt",
    "src/xrr_fitter.egg-info/dependency_links.txt",
    "src/xrr_fitter.egg-info/requires.txt",
    "src/xrr_fitter.egg-info/top_level.txt",
)
ENTRY_POINT_METADATA = "src/xrr_fitter.egg-info/entry_points.txt"
INPUT_DIRECTORIES = ("docs", "examples", "src", "tests", "tools", "verification")
INPUT_FILES = (
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
    "requirements-macos-arm64-py312.lock",
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _pyproject_text() -> str:
    return """[build-system]
requires = ["setuptools==75.8.2", "wheel==0.45.1"]
build-backend = "setuptools.build_meta"

[project]
name = "xrr-fitter"
version = "0.2.0"
requires-python = ">=3.12,<3.13"
dependencies = ["numpy>=2.0,<3"]

[project.optional-dependencies]
test = ["pytest>=8.3,<9"]
"""


def _lock_text(*lines: str) -> str:
    values = lines or ("numpy==2.1.3", "pytest==8.3.5", *BUILD)
    return "\n".join(sorted(values, key=str.casefold)) + "\n"


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_pyproject_text(), encoding="utf-8")
    lock = tmp_path / "requirements.lock"
    lock.write_text(_lock_text(), encoding="utf-8")
    return pyproject, lock


def _stub_metadata(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    monkeypatch.setattr(module, "build_generated_metadata", lambda _payload: GENERATED_METADATA)


def _metadata_archive(tmp_path: Path, members: tuple[str, ...]) -> Path:
    archive = tmp_path / f"metadata-{len(tuple(tmp_path.iterdir()))}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name in members:
            content = name.encode()
            info = tarfile.TarInfo(f"xrr_fitter-0.2.0/{name}")
            info.size = len(content)
            handle.addfile(info, io.BytesIO(content))
    return archive


def test_release_spec_binds_exact_metadata_lock_and_content_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_tool_module
) -> None:
    module = load_tool_module("build_release_spec")
    _stub_metadata(monkeypatch, module)
    pyproject, lock = _fixture(tmp_path)

    spec = module.calculate_release_spec(pyproject, lock)

    assert spec["schema"] == "xrr-r23-release-spec-v1"
    assert spec["build_system"] == {
        "requires": list(BUILD),
        "build_backend": "setuptools.build_meta",
    }
    assert spec["runtime_dependencies"] == ["numpy>=2.0,<3"]
    assert spec["test_dependencies"] == ["pytest>=8.3,<9"]
    assert spec["lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert spec["sdist_content_policy"] == {
        "input_directories": list(INPUT_DIRECTORIES),
        "input_files": list(INPUT_FILES),
        "generated_metadata": list(GENERATED_METADATA),
    }


def test_manifest_matches_the_audited_sdist_input_allowlist() -> None:
    root = Path(__file__).resolve().parents[3]
    expected = (
        "include MANIFEST.in",
        "include README.md",
        "include pyproject.toml",
        "include requirements-macos-arm64-py312.lock",
        "include LICENSE",
        "graft docs",
        "graft examples",
        "graft src",
        "graft tests",
        "graft tools",
        "graft verification",
        "global-exclude *.py[cod]",
        "global-exclude .DS_Store",
        "prune **/__pycache__",
        "prune **/.pytest_cache",
    )

    assert tuple((root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()) == expected


def test_committed_pyproject_declares_the_single_gui_entrypoint() -> None:
    root = Path(__file__).resolve().parents[3]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert payload["project"]["gui-scripts"] == {"xrr-fitter": "xrr_fitter.__main__:main"}


def test_calculation_binds_build_requirements_to_the_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_tool_module,
) -> None:
    module = load_tool_module("build_release_spec")
    _stub_metadata(monkeypatch, module)
    pyproject, lock = _fixture(tmp_path)
    lock.write_text(_lock_text("numpy==2.1.3", "pytest==8.3.5"), encoding="utf-8")

    with pytest.raises(ValueError, match="missing setuptools"):
        module.calculate_release_spec(pyproject, lock)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    (
        ("setuptools==75.8.2", "setuptools==75.8.1", "pinned"),
        ("setuptools.build_meta", "hatchling.build", "backend"),
        ('dependencies = ["numpy>=2.0,<3"]', 'dependencies = "numpy>=2.0,<3"', "arrays"),
        ('test = ["pytest>=8.3,<9"]', 'test = "pytest>=8.3,<9"', "arrays"),
        ('dependencies = ["numpy>=2.0,<3"]', 'dependencies = ["not ??? valid"]', "invalid"),
    ),
)
def test_pyproject_dependency_and_build_system_drift_is_rejected(
    tmp_path: Path,
    old: str,
    new: str,
    match: str,
    load_tool_module,
) -> None:
    module = load_tool_module("build_release_spec")
    pyproject, _lock = _fixture(tmp_path)
    pyproject.write_text(_pyproject_text().replace(old, new), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        module._pyproject(pyproject)


@pytest.mark.parametrize(
    ("content", "match"),
    (
        (b"numpy==2.1.3\npytest==8.3.5\n", "missing setuptools"),
        (b"numpy==2.1.3\npytest==8.3.5\nsetuptools==75.8.2\nwheel==0.45.0\n", "wheel"),
        (b"numpy==2.1.3\npytest==8.3.5\nsetuptools==75.8.2\nwheel>=0.45.1\n", "non-exact"),
        (b"numpy ==2.1.3\npytest==8.3.5\nsetuptools==75.8.2\nwheel==0.45.1\n", "canonical"),
        (b"numpy==2.1.3\r\npytest==8.3.5\r\nsetuptools==75.8.2\r\nwheel==0.45.1\r\n", "canonical"),
        (b"numpy==2.1.3\npytest==8.3.5\nsetuptools==75.8.2\nwheel==0.45.1", "canonical"),
        (b"not a requirement ???\n", "invalid"),
        (
            _lock_text("numpy[extra]==2.1.3", "pytest==8.3.5", *BUILD).encode(),
            "extras or markers",
        ),
        (
            _lock_text(
                'numpy==2.1.3; python_version >= "3.12"',
                "pytest==8.3.5",
                *BUILD,
            ).encode(),
            "extras or markers",
        ),
        (
            _lock_text(
                "evil @ git+https://example.invalid/evil.git@1111111111111111111111111111111111111111",
                "numpy==2.1.3",
                "pytest==8.3.5",
                *BUILD,
            ).encode(),
            "undeclared direct",
        ),
        (
            _lock_text(
                "evil @ git+https://example.invalid/evil.git@main",
                "numpy==2.1.3",
                "pytest==8.3.5",
                *BUILD,
            ).encode(),
            "full lowercase commit",
        ),
    ),
)
def test_lock_drift_and_noncanonical_content_is_rejected(
    tmp_path: Path,
    content: bytes,
    match: str,
    load_tool_module,
) -> None:
    module = load_tool_module("build_release_spec")
    pyproject, lock = _fixture(tmp_path)
    lock.write_bytes(content)
    payload = module._pyproject(pyproject)
    declared = module._parse_requirements(
        (
            *payload["build-system"]["requires"],
            *payload["project"]["dependencies"],
            *payload["project"]["optional-dependencies"]["test"],
        )
    )

    with pytest.raises(ValueError, match=match):
        module._lock(lock, declared)


def test_pyproject_and_lock_read_failures_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_tool_module,
) -> None:
    module = load_tool_module("build_release_spec")
    pyproject, lock = _fixture(tmp_path)
    original_text = Path.read_text
    original_bytes = Path.read_bytes

    def denied_text(path: Path, *args, **kwargs) -> str:
        if path == pyproject:
            raise PermissionError("denied")
        return original_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied_text)
    with pytest.raises(ValueError, match="read pyproject"):
        module._pyproject(pyproject)
    monkeypatch.setattr(Path, "read_text", original_text)

    def denied_bytes(path: Path) -> bytes:
        if path == lock:
            raise PermissionError("denied")
        return original_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", denied_bytes)
    with pytest.raises(ValueError, match="read dependency lock"):
        module._lock(lock, ())


def test_pinned_fixture_builds_twice_with_exact_metadata_set(load_tool_module) -> None:
    module = load_tool_module("build_release_spec")
    payload = module._pyproject(Path(__file__).resolve().parents[3] / "pyproject.toml")

    assert module.build_generated_metadata(payload) == tuple(sorted((*GENERATED_METADATA, ENTRY_POINT_METADATA)))


def test_dynamic_version_project_can_build_release_fixture(load_tool_module) -> None:
    module = load_tool_module("build_release_spec")
    root = Path(__file__).resolve().parents[3]
    payload = module._pyproject(root / "pyproject.toml")

    fixture = module._fixture_toml(payload)

    assert 'version = "0.2.2"' in fixture


def test_pyproject_rejects_static_and_dynamic_version_sources(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("build_release_spec")
    pyproject, _lock = _fixture(tmp_path)
    pyproject.write_text(
        _pyproject_text().replace(
            'version = "0.2.0"',
            'version = "0.2.0"\ndynamic = ["version"]',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both statically and dynamically"):
        module._pyproject(pyproject)


def test_generated_metadata_rejects_unknown_egg_info_member(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("build_release_spec")
    archive = _metadata_archive(
        tmp_path,
        (*GENERATED_METADATA, "src/xrr_fitter.egg-info/unreviewed.txt"),
    )

    with pytest.raises(ValueError, match="metadata"):
        module._generated_metadata(archive)


def test_generated_metadata_dual_run_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_tool_module,
) -> None:
    module = load_tool_module("build_release_spec")
    payload = {"project": {}}
    observed = iter((GENERATED_METADATA, (*GENERATED_METADATA, "entry_points.txt")))
    monkeypatch.setattr(module, "_assert_build_environment", lambda: None)
    monkeypatch.setattr(module, "_build_fixture", lambda _root, _payload: tmp_path / "fixture.tar.gz")
    monkeypatch.setattr(module, "_generated_metadata", lambda _archive: next(observed))

    with pytest.raises(ValueError, match="deterministic"):
        module.build_generated_metadata(payload)


def test_release_spec_output_is_atomic_canonical_and_byte_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_tool_module,
) -> None:
    module = load_tool_module("build_release_spec")
    _stub_metadata(monkeypatch, module)
    pyproject, lock = _fixture(tmp_path)
    output = tmp_path / "release-spec.json"
    output.write_text('{"stale": true}\n', encoding="utf-8")

    module.write_release_spec(pyproject, lock, output)
    first = output.read_bytes()

    assert first == _canonical(json.loads(first))
    module.write_release_spec(pyproject, lock, output)
    assert output.read_bytes() == first


def test_committed_release_spec_matches_canonical_recalculation(load_tool_module) -> None:
    module = load_tool_module("build_release_spec")
    root = Path(__file__).resolve().parents[3]
    output = root / "verification" / "release-spec.json"

    committed = output.read_bytes()
    expected = module.calculate_release_spec(
        root / "pyproject.toml",
        root / "requirements-macos-arm64-py312.lock",
    )

    assert committed == _canonical(expected)


def test_atomic_write_uses_same_directory_file_and_directory_fsync_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_tool_module,
) -> None:
    module = load_tool_module("build_release_spec")
    output = tmp_path / "release-spec.json"
    events: list[str] = []
    real_mkstemp = module.tempfile.mkstemp
    real_fsync = module.os.fsync
    real_replace = module.os.replace

    def tracked_mkstemp(*args, **kwargs):
        assert Path(kwargs["dir"]) == tmp_path
        events.append("mkstemp")
        return real_mkstemp(*args, **kwargs)

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(f"fsync-{kind}")
        real_fsync(descriptor)

    def tracked_replace(source: Path, destination: Path) -> None:
        assert Path(source).parent == tmp_path
        assert Path(destination) == output
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(module.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(module.os, "replace", tracked_replace)

    module._atomic_write(output, b"{}\n")

    assert events == ["mkstemp", "fsync-file", "replace", "fsync-directory"]


def test_calculation_failure_preserves_existing_output_without_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_tool_module,
) -> None:
    module = load_tool_module("build_release_spec")
    _stub_metadata(monkeypatch, module)
    pyproject, lock = _fixture(tmp_path)
    output = tmp_path / "release-spec.json"
    original = b"previous release spec\n"
    output.write_bytes(original)
    lock.write_text("numpy>=2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        module.write_release_spec(pyproject, lock, output)

    assert output.read_bytes() == original
    assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))


def test_atomic_replace_failure_preserves_existing_output_without_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_tool_module,
) -> None:
    module = load_tool_module("build_release_spec")
    output = tmp_path / "release-spec.json"
    original = b"previous release spec\n"
    output.write_bytes(original)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        module._atomic_write(output, b"new release spec\n")

    assert output.read_bytes() == original
    assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))
