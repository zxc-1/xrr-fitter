from __future__ import annotations

from email import policy
from email.parser import BytesParser
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile

from packaging.markers import Marker
from packaging.requirements import Requirement
import pytest
from tools.verify_distribution import canonicalize_sdist


ROOT = Path(__file__).resolve().parents[2]
RELEASE_SPEC = ROOT / "verification" / "release-spec.json"
BUILD_VERSIONS = {"setuptools": "75.8.2", "wheel": "0.45.1"}


def _release_spec() -> dict[str, object]:
    return json.loads(RELEASE_SPEC.read_text(encoding="utf-8"))


def test_gui_entrypoint_and_shell_modules_are_explicit() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["gui-scripts"] == {
        "xrr-fitter": "xrr_fitter.__main__:main"
    }
    required = {
        ROOT / "src/xrr_fitter/__main__.py",
        ROOT / "src/xrr_fitter/gui/__init__.py",
        ROOT / "src/xrr_fitter/gui/application.py",
        ROOT / "src/xrr_fitter/gui/document.py",
        ROOT / "src/xrr_fitter/gui/main_window.py",
    }
    assert all(path.is_file() and not path.is_symlink() for path in required)


def _repository_inputs(policy: dict[str, object]) -> tuple[PurePosixPath, ...]:
    result = subprocess.run(
        ("git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    directories = set(policy["input_directories"])
    files = set(policy["input_files"])
    paths = tuple(
        PurePosixPath(value.decode("utf-8"))
        for value in result.stdout.split(b"\0")
        if value
    )
    selected = tuple(
        path for path in paths if path.as_posix() in files or path.parts[0] in directories
    )
    assert {path.as_posix() for path in selected if len(path.parts) == 1} == files
    assert {path.parts[0] for path in selected if len(path.parts) > 1} == directories
    return selected


def _copy_inputs(
    staging: Path,
    paths: tuple[PurePosixPath, ...],
    epoch: int,
) -> None:
    for relative in paths:
        source = ROOT.joinpath(*relative.parts)
        assert source.is_file() and not source.is_symlink(), relative
        target = staging.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.utime(target, (epoch, epoch))


def _assert_pinned_builder(spec: dict[str, object]) -> None:
    build_system = spec["build_system"]
    assert build_system == {
        "build_backend": "setuptools.build_meta",
        "requires": ["setuptools==75.8.2", "wheel==0.45.1"],
    }
    observed = {name: importlib.metadata.version(name) for name in BUILD_VERSIONS}
    assert observed == BUILD_VERSIONS


def _build_distributions(
    root: Path,
    inputs: tuple[PurePosixPath, ...],
    epoch: int,
) -> tuple[Path, Path]:
    staging = root / "staging"
    output = root / "artifacts"
    root.mkdir()
    staging.mkdir()
    output.mkdir()
    _copy_inputs(staging, inputs, epoch)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SOURCE_DATE_EPOCH"] = str(epoch)
    subprocess.run(
        (
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
            str(staging),
        ),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    wheels = tuple(output.glob("*.whl"))
    sdists = tuple(output.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1
    canonicalize_sdist(sdists[0], epoch)
    return wheels[0], sdists[0]


@pytest.fixture(scope="session")
def built_distributions(tmp_path_factory: pytest.TempPathFactory):
    spec = _release_spec()
    _assert_pinned_builder(spec)
    policy = spec["sdist_content_policy"]
    inputs = _repository_inputs(policy)
    epoch = int(
        subprocess.check_output(
            ("git", "show", "-s", "--format=%ct", "HEAD"),
            cwd=ROOT,
            text=True,
        ).strip()
    )
    root = tmp_path_factory.mktemp("distribution")
    first_wheel, first_sdist = _build_distributions(root / "first", inputs, epoch)
    time.sleep(1.1)
    second_wheel, second_sdist = _build_distributions(root / "second", inputs, epoch)
    return spec, inputs, first_wheel, first_sdist, second_wheel, second_sdist


def _project_identity() -> tuple[str, str]:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    normalized = project["name"].replace("-", "_").replace(".", "_")
    return normalized, project["version"]


def _wheel_metadata() -> set[str]:
    name, version = _project_identity()
    root = f"{name}-{version}.dist-info"
    members = {f"{root}/{name}" for name in ("METADATA", "RECORD", "WHEEL", "top_level.txt", "LICENSE")}
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if payload["project"].get("gui-scripts"):
        members.add(f"{root}/entry_points.txt")
    return members


def _wheel_files(archive: Path) -> set[str]:
    with zipfile.ZipFile(archive) as handle:
        members = tuple(handle.infolist())
    assert all(not member.is_dir() for member in members)
    assert all(stat.S_IFMT(member.external_attr >> 16) != stat.S_IFLNK for member in members)
    return {member.filename for member in members}


def _project_requirements(project: dict[str, object]) -> tuple[Requirement, ...]:
    expected = [Requirement(value) for value in project["dependencies"]]
    for extra, requirements in project["optional-dependencies"].items():
        for value in requirements:
            requirement = Requirement(value)
            extra_marker = Marker(f'extra == "{extra}"')
            requirement.marker = (
                extra_marker
                if requirement.marker is None
                else Marker(f"({requirement.marker}) and ({extra_marker})")
            )
            expected.append(requirement)
    return tuple(expected)


def test_wheel_contains_only_package_and_exact_distribution_metadata(
    built_distributions,
) -> None:
    spec, inputs, wheel, _sdist, _second_wheel, _second_sdist = built_distributions
    expected_package = {
        path.as_posix().removeprefix("src/")
        for path in inputs
        if path.parts[:2] == ("src", "xrr_fitter")
    }
    observed = _wheel_files(wheel)
    assert observed == expected_package | _wheel_metadata()
    policy = spec["wheel_content_policy"]
    assert policy["package_root"] == "xrr_fitter"
    assert policy["include_distribution_metadata"] is True
    assert {PurePosixPath(path).parts[0] for path in observed}.isdisjoint(
        policy["forbidden_roots"]
    )


def test_wheel_requires_dist_exactly_matches_pyproject(
    built_distributions,
) -> None:
    spec, _inputs, wheel, _sdist, _second_wheel, _second_sdist = built_distributions
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    with zipfile.ZipFile(wheel) as archive:
        metadata = tuple(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        assert len(metadata) == 1
        message = BytesParser(policy=policy.default).parsebytes(archive.read(metadata[0]))
    observed = tuple(Requirement(value) for value in message.get_all("Requires-Dist", ()))
    assert observed == _project_requirements(project)
    assert tuple(spec["runtime_dependencies"]) == tuple(project["dependencies"])


def _sdist_files(archive: Path) -> tuple[str, set[str]]:
    with tarfile.open(archive, "r:gz") as handle:
        members = tuple(handle.getmembers())
    assert all(member.isfile() or member.isdir() for member in members)
    files = tuple(PurePosixPath(member.name) for member in members if member.isfile())
    roots = {path.parts[0] for path in files}
    assert len(roots) == 1
    root = roots.pop()
    return root, {PurePosixPath(*path.parts[1:]).as_posix() for path in files}


def test_sdist_matches_release_spec_inputs_and_generated_metadata(
    built_distributions,
) -> None:
    spec, inputs, _wheel, sdist, _second_wheel, _second_sdist = built_distributions
    root, observed = _sdist_files(sdist)
    name, version = _project_identity()
    assert root == f"{name}-{version}"
    generated = set(spec["sdist_content_policy"]["generated_metadata"])
    assert observed == {path.as_posix() for path in inputs} | generated


def test_repeated_builds_are_byte_for_byte_reproducible(
    built_distributions,
) -> None:
    _spec, _inputs, wheel, sdist, second_wheel, second_sdist = built_distributions
    assert second_wheel.name == wheel.name
    assert second_sdist.name == sdist.name
    assert second_wheel.read_bytes() == wheel.read_bytes()
    assert second_sdist.read_bytes() == sdist.read_bytes()


def test_distribution_build_leaves_repository_without_build_outputs(
    built_distributions,
) -> None:
    assert built_distributions
    assert not (ROOT / "build").exists()
    assert not (ROOT / "dist").exists()
    assert not tuple((ROOT / "src").glob("*.egg-info"))
