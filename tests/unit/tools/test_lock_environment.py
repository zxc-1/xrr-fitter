from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools" / "lock_environment.py"
PYPROJECT = ROOT / "pyproject.toml"
BUILD = ("setuptools==75.8.2", "wheel==0.45.1")
RUNTIME = (
    "numpy>=2.0,<3",
    "scipy>=1.14,<2",
    "periodictable>=2.0,<3",
    "pandas>=2.2,<3",
    "xlsxwriter>=3.2,<4",
    "matplotlib>=3.9,<4",
    "PySide6>=6.8,<7",
)
REFNX = "refnx @ git+https://github.com/refnx/refnx.git@3d3808f66a14a8200eba020f8dff53f4d1e059bc"
TEST = (
    "pytest>=8.3,<9",
    "pytest-qt>=4.4,<5",
    "openpyxl>=3.1,<4",
    "radon==6.0.1",
    "build>=1.2,<2",
    REFNX,
)
LOCK_LINES = (
    "build==1.2.2.post1",
    "matplotlib==3.9.4",
    "numpy==2.1.3",
    "openpyxl==3.1.5",
    "pandas==2.2.3",
    "periodictable==2.0.2",
    "PySide6==6.8.2",
    "pytest==8.3.5",
    "pytest-qt==4.4.0",
    "radon==6.0.1",
    REFNX,
    "scipy==1.14.1",
    "setuptools==75.8.2",
    "wheel==0.45.1",
    "XlsxWriter==3.2.0",
)


def _lock_text(lines: tuple[str, ...] = LOCK_LINES) -> str:
    return "\n".join(sorted(lines, key=str.casefold)) + "\n"


def _replace(old: str, new: str) -> str:
    return _lock_text(tuple(new if line == old else line for line in LOCK_LINES))


def _add(line: str) -> str:
    return _lock_text((*LOCK_LINES, line))


def _pollute_resolver_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    pollution = {
        "PYTHONPATH": "/caller/pythonpath",
        "PYTHONHOME": "/caller/pythonhome",
        "VIRTUAL_ENV": "/caller/venv",
        "PYTHONOPTIMIZE": "2",
        "PIP_INDEX_URL": "https://caller.invalid/simple",
        "PIP_CONFIG_FILE": "/caller/pip.conf",
        "PIP_REQUIRE_VIRTUALENV": "1",
    }
    for name, value in pollution.items():
        monkeypatch.setenv(name, value)
    return pollution


def _shared_resolver_environment(
    calls: list[tuple[tuple[str, ...], dict[str, object]]],
) -> dict[str, str]:
    assert len(calls) == 4
    first, second, third, fourth = (call[1]["env"] for call in calls)
    assert first == second == third == fourth
    controlled = first
    assert isinstance(controlled, dict)
    return controlled


def _assert_python_environment(controlled: dict[str, str], pollution: dict[str, str]) -> None:
    assert controlled["PYTHONDONTWRITEBYTECODE"] == "1"
    python_names = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONOPTIMIZE"}
    assert python_names.isdisjoint(controlled)
    assert python_names <= set(pollution)


def _assert_pip_environment(controlled: dict[str, str], pollution: dict[str, str]) -> None:
    assert controlled["PIP_CONFIG_FILE"] == os.devnull
    assert controlled["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert controlled["PIP_NO_INPUT"] == "1"
    assert {"PIP_INDEX_URL", "PIP_REQUIRE_VIRTUALENV"}.isdisjoint(controlled)
    assert controlled["PIP_CONFIG_FILE"] != pollution["PIP_CONFIG_FILE"]
    pip_names = {name for name in controlled if name.startswith("PIP_")}
    assert pip_names == {"PIP_CONFIG_FILE", "PIP_DISABLE_PIP_VERSION_CHECK", "PIP_NO_INPUT"}


def _assert_resolver_call_kwargs(
    calls: list[tuple[tuple[str, ...], dict[str, object]]],
) -> None:
    assert set(calls[-1][1]) == {"capture_output", "env"}
    assert tuple(set(kwargs) for _command, kwargs in calls[:-1]) == ({"env"},) * 3


BAD_LOCKS = (
    pytest.param(_add("-e ."), id="editable"),
    pytest.param(_add("local-pkg @ file:///tmp/pkg"), id="local-file-url"),
    pytest.param(_add("/tmp/pkg"), id="absolute-path"),
    pytest.param(
        _replace(REFNX, "refnx @ git+https://github.com/refnx/refnx.git@main"),
        id="unfixed-vcs",
    ),
    pytest.param(
        _replace(
            REFNX,
            "refnx @ git+https://github.com/refnx/refnx.git@3d3808f66a14a8200eba",
        ),
        id="short-vcs-commit",
    ),
    pytest.param(
        _replace(
            REFNX,
            "refnx @ git+https://github.com/refnx/refnx.git@3D3808F66A14A8200EBA020F8DFF53F4D1E059BC",
        ),
        id="uppercase-vcs-commit",
    ),
    pytest.param(
        _replace(REFNX, REFNX.replace("refnx @", "refnx[dev] @")),
        id="vcs-extra",
    ),
    pytest.param(
        _replace(REFNX, f'{REFNX} ; python_version >= "3.12"'),
        id="vcs-marker",
    ),
    pytest.param(
        _add("other @ git+https://github.com/example/other.git@1111111111111111111111111111111111111111"),
        id="undeclared-vcs",
    ),
    pytest.param(
        _add("other @ https://example.invalid/other-1.0-py3-none-any.whl"),
        id="undeclared-non-vcs-url",
    ),
    pytest.param(
        _replace(
            REFNX,
            "refnx @ git+https://example.invalid/refnx.git@3d3808f66a14a8200eba020f8dff53f4d1e059bc",
        ),
        id="wrong-vcs-url",
    ),
    pytest.param(
        _replace(
            REFNX,
            "refnx @ git+https://github.com/refnx/refnx.git@1111111111111111111111111111111111111111",
        ),
        id="wrong-vcs-commit",
    ),
    pytest.param(_replace("numpy==2.1.3", "numpy>=2.1.3"), id="non-exact"),
    pytest.param(_add("numpy==2.1.3"), id="duplicate"),
    pytest.param("\n".join(reversed(LOCK_LINES)) + "\n", id="unsorted"),
    pytest.param(_replace("numpy==2.1.3", "numpy ==2.1.3"), id="non-canonical-spacing"),
)


def test_lock_accepts_canonical_exact_closure_and_declared_refnx(load_tool_module) -> None:
    module = load_tool_module("lock_environment")
    text = _lock_text()

    observed = module.validate_lock_text(
        text,
        build_dependencies=BUILD,
        runtime_dependencies=RUNTIME,
        test_dependencies=TEST,
    )

    assert observed == {
        "requirement_count": len(LOCK_LINES),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "refnx_commit": "3d3808f66a14a8200eba020f8dff53f4d1e059bc",
    }


@pytest.mark.parametrize("bad", BAD_LOCKS)
def test_lock_rejects_each_noncanonical_or_unbound_mutation(load_tool_module, bad: str) -> None:
    module = load_tool_module("lock_environment")

    with pytest.raises(ValueError):
        module.validate_lock_text(
            bad,
            build_dependencies=BUILD,
            runtime_dependencies=RUNTIME,
            test_dependencies=TEST,
        )


@pytest.mark.parametrize(
    ("kind", "error_text"),
    (
        ("missing", "lock path must be a regular file"),
        ("directory", "lock path must be a regular file"),
        ("invalid-utf8", "lock must be readable UTF-8"),
        ("parse-error", "invalid lock requirement"),
    ),
)
def test_check_cli_returns_nonzero_for_path_decode_or_parse_error(tmp_path: Path, kind: str, error_text: str) -> None:
    lock = tmp_path / "requirements.lock"
    if kind == "directory":
        lock.mkdir()
    elif kind == "invalid-utf8":
        lock.write_bytes(b"numpy==2.1.3\n\xff")
    elif kind == "parse-error":
        lock.write_text(_add("not a valid requirement ???"), encoding="utf-8")

    result = subprocess.run(
        (
            sys.executable,
            str(TOOL),
            "--check",
            str(lock),
            "--pyproject",
            str(PYPROJECT),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert error_text in result.stderr


def test_check_permission_error_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_tool_module
) -> None:
    module = load_tool_module("lock_environment")
    lock = tmp_path / "requirements.lock"
    lock.write_text(_lock_text(), encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def denied(path: Path) -> bytes:
        if path == lock:
            raise PermissionError("denied by test")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", denied)

    with pytest.raises(ValueError, match="readable UTF-8"):
        module.main(("--check", str(lock), "--pyproject", str(PYPROJECT)))


def test_resolver_failure_preserves_existing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_tool_module
) -> None:
    module = load_tool_module("lock_environment")
    output = tmp_path / "requirements.lock"
    original = b"previous-lock-bytes\n"
    output.write_bytes(original)

    def fail(_pyproject: Path):
        raise subprocess.CalledProcessError(1, ("pip", "install"))

    monkeypatch.setattr(module, "resolve_lock", fail)

    with pytest.raises(subprocess.CalledProcessError):
        module.main(("--output", str(output), "--pyproject", str(PYPROJECT)))
    assert output.read_bytes() == original


def test_success_atomically_replaces_and_two_runs_are_byte_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_tool_module
) -> None:
    module = load_tool_module("lock_environment")
    output = tmp_path / "requirements.lock"
    output.write_bytes(b"old lock\n")
    replacements: list[tuple[Path, Path]] = []
    original_replace = module.os.replace

    def record_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", record_replace)
    module.write_lock(output, ("radon==6.0.1", "numpy==2.1.3"))
    first = output.read_bytes()
    module.write_lock(output, ("numpy==2.1.3", "radon==6.0.1"))

    assert output.read_bytes() == first == b"numpy==2.1.3\nradon==6.0.1\n"
    assert len(replacements) == 2
    assert all(source.parent == output.parent for source, _ in replacements)
    assert all(destination == output for _, destination in replacements)
    assert not tuple(tmp_path.glob(".requirements.lock.*.tmp"))


def test_resolver_uses_pinned_pip_declared_specs_and_freeze_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_tool_module
) -> None:
    module = load_tool_module("lock_environment")
    project = tmp_path / "project"
    project.mkdir()
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        "[build-system]\n"
        'requires = ["setuptools==75.8.2", "wheel==0.45.1"]\n'
        "[project]\n"
        'dependencies = ["numpy>=2,<3"]\n'
        "[project.optional-dependencies]\n"
        f'test = ["{REFNX}"]\n',
        encoding="utf-8",
    )
    pollution = _pollute_resolver_environment(monkeypatch)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        normalized = tuple(command)
        calls.append((normalized, kwargs))
        resolved = ("numpy==2.1.3", REFNX, *BUILD)
        stdout = _lock_text(resolved) if "freeze" in normalized else ""
        return subprocess.CompletedProcess(normalized, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(module, "_run", fake_run)

    pins, build, runtime, tests = module.resolve_lock(pyproject)

    commands = [command for command, _kwargs in calls]
    resolver = Path(commands[0][-1])
    python = resolver / "bin" / "python"
    assert commands == [
        (sys.executable, "-m", "venv", str(resolver)),
        (str(python), "-m", "pip", "install", "pip==26.1.2"),
        (
            str(python),
            "-m",
            "pip",
            "install",
            "setuptools==75.8.2",
            "wheel==0.45.1",
            "numpy>=2,<3",
            REFNX,
        ),
        (str(python), "-m", "pip", "freeze", "--exclude-editable"),
    ]
    assert not resolver.is_relative_to(project)
    assert not resolver.exists()
    assert pins == tuple(_lock_text(("numpy==2.1.3", REFNX, *BUILD)).splitlines())
    assert (build, runtime, tests) == (BUILD, ("numpy>=2,<3",), (REFNX,))
    controlled = _shared_resolver_environment(calls)
    _assert_python_environment(controlled, pollution)
    _assert_pip_environment(controlled, pollution)
    _assert_resolver_call_kwargs(calls)


def test_resolver_refuses_temporary_directory_inside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_tool_module
) -> None:
    module = load_tool_module("lock_environment")
    project = tmp_path / "project"
    project.mkdir()
    pyproject = project / "pyproject.toml"
    pyproject.write_text(PYPROJECT.read_text(encoding="utf-8"), encoding="utf-8")
    inside = project / "temp"
    inside.mkdir()
    monkeypatch.setattr(module.tempfile, "tempdir", str(inside))

    with pytest.raises(ValueError, match="outside the repository"):
        module.resolve_lock(pyproject)
