from __future__ import annotations

from pathlib import Path

import pytest

LOCK = "requirements-windows-x64-py312.lock"

PYPROJECT = """\
[build-system]
requires = ["setuptools==75.8.2", "wheel==0.45.1"]
build-backend = "setuptools.build_meta"

[project]
name = "xrr-fitter"
version = "0.2.2"
requires-python = ">=3.12,<3.13"
dependencies = ["numpy>=2.0,<3"]

[project.optional-dependencies]
test = ["pytest>=8.3,<9"]

[tool.xrr.windows-packaging]
requires = ["pyinstaller==6.21.0"]
"""


def _write(root: Path, body: str) -> Path:
    path = root / "pyproject.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_windows_environment_describes_win_amd64_cpython312(load_tool_module) -> None:
    module = load_tool_module("lock_windows_environment")
    assert module.WINDOWS_ENVIRONMENT["sys_platform"] == "win32"
    assert module.WINDOWS_ENVIRONMENT["os_name"] == "nt"
    assert module.WINDOWS_ENVIRONMENT["platform_system"] == "Windows"
    assert module.WINDOWS_ENVIRONMENT["platform_machine"] == "AMD64"
    assert module.WINDOWS_ENVIRONMENT["python_version"] == "3.12"
    assert module.WINDOWS_ENVIRONMENT["platform_python_implementation"] == "CPython"


def test_target_arguments_pin_the_windows_wheel_platform(load_tool_module) -> None:
    module = load_tool_module("lock_windows_environment")
    assert module.TARGET_ARGUMENTS == (
        "--only-binary=:all:",
        "--platform",
        "win_amd64",
        "--python-version",
        "3.12",
        "--implementation",
        "cp",
        "--abi",
        "cp312",
    )


def test_reads_build_runtime_and_windows_packaging_requirements(
    load_tool_module, tmp_path: Path
) -> None:
    module = load_tool_module("lock_windows_environment")
    build, runtime, packaging = module.read_windows_dependencies(_write(tmp_path, PYPROJECT))
    assert build == ("setuptools==75.8.2", "wheel==0.45.1")
    assert runtime == ("numpy>=2.0,<3",)
    assert packaging == ("pyinstaller==6.21.0",)


def test_rejects_pyproject_without_windows_packaging_table(
    load_tool_module, tmp_path: Path
) -> None:
    module = load_tool_module("lock_windows_environment")
    body = PYPROJECT.replace('[tool.xrr.windows-packaging]\nrequires = ["pyinstaller==6.21.0"]\n', "")
    with pytest.raises(ValueError, match="invalid Windows packaging metadata"):
        module.read_windows_dependencies(_write(tmp_path, body))


def test_rejects_direct_reference_in_windows_closure(load_tool_module, tmp_path: Path) -> None:
    module = load_tool_module("lock_windows_environment")
    body = PYPROJECT.replace(
        '["pyinstaller==6.21.0"]',
        '["pyinstaller @ git+https://example.invalid/p.git@' + "0" * 40 + '"]',
    )
    with pytest.raises(ValueError, match="cannot contain direct references"):
        module.read_windows_dependencies(_write(tmp_path, body))


def test_keeps_win32_dependencies_and_drops_other_platforms(load_tool_module) -> None:
    module = load_tool_module("lock_windows_environment")
    metadata = {
        "requires_dist": [
            "altgraph",
            "macholib>=1.8; sys_platform == 'darwin'",
            "pefile>=2022.5.30; sys_platform == 'win32'",
            "pywin32-ctypes>=0.2.1; sys_platform == 'win32'",
            "importlib-metadata>=4.6; python_version < '3.10'",
            "pytest>=2.7.3; extra == 'hook-testing'",
        ]
    }
    assert module._dependencies_for_windows(metadata) == (
        "altgraph",
        "pefile>=2022.5.30",
        "pywin32-ctypes>=0.2.1",
    )


def test_strips_markers_before_handing_requirements_back_to_pip(load_tool_module) -> None:
    """pip re-evaluates markers against the host and silently omits win32-only pins."""
    module = load_tool_module("lock_windows_environment")
    metadata = {"requires_dist": ["pefile>=2022.5.30; sys_platform == 'win32'"]}
    assert module._dependencies_for_windows(metadata) == ("pefile>=2022.5.30",)


def test_committed_lock_satisfies_the_declared_windows_closure(load_tool_module) -> None:
    module = load_tool_module("lock_windows_environment")
    root = Path(__file__).resolve().parents[3]
    build, runtime, packaging = module.read_windows_dependencies(root / "pyproject.toml")
    report = module.validate_lock_text(
        (root / LOCK).read_bytes().decode("utf-8"),
        build_dependencies=(*build, *packaging),
        runtime_dependencies=runtime,
    )
    assert report["refnx_commit"] is None
    assert report["requirement_count"] > 0


def test_committed_lock_excludes_test_only_distributions() -> None:
    root = Path(__file__).resolve().parents[3]
    names = {
        line.partition("==")[0].casefold().replace("_", "-")
        for line in (root / LOCK).read_text(encoding="utf-8").splitlines()
        if line
    }
    assert not names & {"pytest", "pytest-qt", "refnx", "radon", "openpyxl", "build"}
    assert {"pyinstaller", "pefile", "pywin32-ctypes", "altgraph"} <= names
    assert "macholib" not in names


def test_committed_locks_agree_on_shared_distributions() -> None:
    root = Path(__file__).resolve().parents[3]

    def pins(name: str) -> dict[str, str]:
        return {
            line.partition("==")[0].casefold().replace("_", "-"): line.partition("==")[2]
            for line in (root / name).read_text(encoding="utf-8").splitlines()
            if line and "==" in line
        }

    windows = pins(LOCK)
    macos = pins("requirements-macos-arm64-py312.lock")
    divergent = {
        name: (macos[name], windows[name])
        for name in set(macos) & set(windows)
        if macos[name] != windows[name]
    }
    assert divergent == {}
