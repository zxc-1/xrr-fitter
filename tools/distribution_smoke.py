"""Fresh-environment installation and application-entrypoint smoke tests."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
from typing import Callable
import venv

from distribution_archive import verify_wheel_dependencies
from distribution_source import build_environment


Runner = Callable[..., object]


def _smoke_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PATH"] = ""
    locations = {
        "HOME": root / "smoke-home",
        "XDG_CACHE_HOME": root / "xdg-cache",
        "MPLCONFIGDIR": root / "mpl-cache",
    }
    for name, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def smoke_installed(
    environment_root: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> None:
    root = Path(environment_root).resolve()
    python = root / "bin" / "python"
    entrypoint = root / "bin" / "xrr-fitter"
    if not python.is_file() or not entrypoint.is_file():
        raise ValueError("installed smoke environment is missing absolute executables")
    commands = (
        (python, "-c", "import xrr_fitter.api"),
        (python, "-m", "xrr_fitter", "--help"),
        (entrypoint, "--help"),
    )
    environment = _smoke_environment(root)
    for command in commands:
        runner(
            tuple(str(value) for value in command),
            cwd=root,
            env=environment,
            check=True,
        )


def _install_environment(root: Path, lock_file: Path, runner: Runner) -> Path:
    environment = root.resolve()
    venv.EnvBuilder(with_pip=True, clear=False).create(environment)
    python = environment / "bin" / "python"
    child = os.environ.copy()
    child.pop("PYTHONPATH", None)
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    child["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    runner(
        (str(python), "-m", "pip", "install", "pip==26.1.2"),
        cwd=environment,
        env=child,
        check=True,
    )
    runner(
        (str(python), "-m", "pip", "install", "-r", str(lock_file.resolve())),
        cwd=environment,
        env=child,
        check=True,
    )
    return environment


def _install_wheel(environment: Path, wheel: Path, runner: Runner) -> None:
    python = environment / "bin" / "python"
    child = os.environ.copy()
    child.pop("PYTHONPATH", None)
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    runner(
        (str(python), "-m", "pip", "install", "--no-deps", str(wheel.resolve())),
        cwd=environment,
        env=child,
        check=True,
    )


def smoke_wheel(repository: Path, wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="xrr-r23-wheel-smoke-") as directory:
        environment = _install_environment(
            Path(directory) / "venv",
            repository / "requirements-macos-arm64-py312.lock",
            subprocess.run,
        )
        _install_wheel(environment, wheel, subprocess.run)
        smoke_installed(environment)


def _safe_sdist_path(name: str, path: PurePosixPath) -> bool:
    return bool(name) and not (
        path.is_absolute()
        or path.as_posix() != name
        or ".." in path.parts
        or "\\" in name
    )


def _sdist_member_path(member: tarfile.TarInfo) -> PurePosixPath:
    if not (member.isfile() or member.isdir()):
        raise ValueError("sdist archive contains an unsupported member type")
    name = member.name.rstrip("/")
    path = PurePosixPath(name)
    if not _safe_sdist_path(name, path):
        raise ValueError("sdist archive member path is unsafe")
    return path


def _sdist_members(archive: tarfile.TarFile) -> tuple[tuple[tarfile.TarInfo, ...], str]:
    members = tuple(archive.getmembers())
    if not members:
        raise ValueError("sdist archive is empty")
    roots: set[str] = set()
    names: set[str] = set()
    for member in members:
        path = _sdist_member_path(member)
        name = path.as_posix()
        if name in names:
            raise ValueError("sdist archive contains a duplicate member")
        names.add(name)
        roots.add(path.parts[0])
    if len(roots) != 1:
        raise ValueError("sdist archive must contain exactly one source root")
    return members, roots.pop()


def extract_sdist(sdist: str | Path, destination: str | Path) -> Path:
    archive_path = Path(sdist).resolve()
    output = Path(destination).resolve()
    if Path(sdist).is_symlink() or not archive_path.is_file():
        raise ValueError("sdist archive must be a regular file")
    if Path(destination).is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise ValueError("sdist extraction directory must be empty and regular")
    with tarfile.open(archive_path, "r:gz") as archive:
        members, root_name = _sdist_members(archive)
        archive.extractall(output, members=members, filter="data")
    source = output / root_name
    if source.is_symlink() or not source.is_dir():
        raise ValueError("sdist archive source root is not a regular directory")
    return source


def smoke_sdist(repository: Path, sdist: Path, epoch: int) -> None:
    with tempfile.TemporaryDirectory(prefix="xrr-r23-sdist-smoke-") as directory:
        root = Path(directory)
        environment = _install_environment(
            root / "venv",
            repository / "requirements-macos-arm64-py312.lock",
            subprocess.run,
        )
        python = environment / "bin" / "python"
        source_root = root / "source"
        source_root.mkdir()
        source = extract_sdist(sdist, source_root)
        output = root / "rebuilt"
        output.mkdir()
        subprocess.run(
            (
                str(python),
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output),
                str(source),
            ),
            cwd=environment,
            env=build_environment(root, epoch),
            check=True,
        )
        wheels = tuple(output.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("sdist rebuild must create exactly one wheel")
        verify_wheel_dependencies(wheels[0], repository / "pyproject.toml")
        _install_wheel(environment, wheels[0], subprocess.run)
        smoke_installed(environment)
