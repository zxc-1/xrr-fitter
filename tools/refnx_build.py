#!/usr/bin/env python3
"""Build refnx offline from verified inputs in a disposable, external builder."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.markers import Marker  # noqa: E402
from packaging.requirements import Requirement  # noqa: E402
from packaging.tags import sys_tags  # noqa: E402
from packaging.utils import canonicalize_name, parse_wheel_filename  # noqa: E402

from audit_reports import _write  # noqa: E402
from lock_audit_tools import _require_version  # noqa: E402
from package_cache import copy_verified  # noqa: E402
from package_downloads import _summary, install_command, prepare_report, require_install_target  # noqa: E402
from package_manifest import manifest_bytes  # noqa: E402
from refnx_build_manifest import (  # noqa: E402
    TARGET,
    input_records,
    read_inputs,
    source_manifest,
    verify_inputs,
)
from refnx_build_runtime import (  # noqa: E402
    build_command,
    build_environment,
    builder_install_command,
    file_record,
    run_checked,
    toolchain_evidence,
    verify_builder_command,
)
from vcs_source import _read_archive, archive_files, verify_source  # noqa: E402
from wheel_inventory import inspect_wheel  # noqa: E402


def extract_source(root: Path, source: dict, archive: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise ValueError("refnx source destination already exists")
    verify_source(root, source, archive)
    content = _read_archive(archive)
    prefix = f"refnx-{source['vcs']['commit']}"
    files = archive_files(content, prefix)
    if any(".git" in PurePosixPath(name).parts for name in files):
        raise ValueError("verified refnx archive must not supply fabricated Git metadata")
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as handle:
        for name, record in files.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with handle.extractfile(f"{prefix}/{name}") as stream, path.open("xb") as output:
                output.write(stream.read())
            path.chmod(0o755 if record["executable"] else 0o644)
            if file_record(path)["sha256"] != record["sha256"]:
                raise ValueError("refnx source changed while extracting")
    verify_source(root, source, archive)


def source_project(source: Path, manifest: dict) -> dict:
    value = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    if value["build-system"]["build-backend"] != "mesonpy" or value["project"]["name"] != "refnx":
        raise ValueError("refnx source build backend or identity changed")
    entries = {item["name"]: (item["version"], item["sha256"]) for item in manifest["wheels"]}
    for requirement in value["build-system"]["requires"]:
        _require_version(Requirement(requirement), entries)
    return value["project"]


def _requirement(value: str, extra: str | None = None) -> str:
    requirement = Requirement(value)
    requirement.name = canonicalize_name(requirement.name)
    if extra is not None:
        marker = f"({requirement.marker}) and " if requirement.marker else ""
        requirement.marker = Marker(f'{marker}extra == "{canonicalize_name(extra)}"')
    return str(requirement)


def _project_requirements(project: dict) -> list[str]:
    values = [_requirement(value) for value in project["dependencies"]]
    for extra, requirements in project.get("optional-dependencies", {}).items():
        values.extend(_requirement(value, extra) for value in requirements)
    return sorted(values)


def _require_native_extensions(inventory: dict) -> None:
    native = {
        item["path"].split(".")[0]
        for item in inventory["files"]
        if item.get("format") in {"mach-o", "mach-o-universal"}
    }
    if not {"refnx/_lib/_cutil", "refnx/reflect/_creflect", "refnx/reduce/_cevent"}.issubset(native):
        raise ValueError("refnx wheel is missing required native extensions")


def wheel_result(directory: Path, project: dict) -> tuple[dict, dict]:
    paths = list(directory.iterdir())
    if len(paths) != 1 or paths[0].suffix != ".whl":
        raise ValueError("refnx build must produce exactly one wheel")
    path = paths[0]
    name, version, _build, tags = parse_wheel_filename(path.name)
    if name != "refnx" or str(version) != project["version"] or not tags.intersection(sys_tags()):
        raise ValueError("refnx wheel identity or target differs from verified source")
    record = {"name": name, "version": str(version), **file_record(path)}
    inventory = inspect_wheel(path, record)
    if sorted(_requirement(value) for value in inventory["requirements"]) != _project_requirements(project):
        raise ValueError("refnx wheel dependencies differ from verified source")
    _require_native_extensions(inventory)
    return record, inventory


def _local_requirements(directory: Path, records: list[dict]) -> str:
    return "".join(
        f"{item['name']} @ {(directory / item['filename']).resolve().as_uri()} --hash=sha256:{item['sha256']}\n"
        for item in records
    )


def _build_stage(root: Path, inputs: Path, report: Path, stage: Path, manifest: dict, environment: dict) -> dict:
    source = source_manifest(root)
    copy_verified(inputs, stage / "inputs", input_records(root, manifest))
    extract_source(root, source, stage / "inputs/refnx-source.tar.gz", stage / "source")
    project = source_project(stage / "source", manifest)
    toolchain = toolchain_evidence(root, report, environment)
    environment = build_environment(environment, stage, toolchain)
    for name in ("home", "tmp", "cache"):
        (stage / name).mkdir()
    python = stage / "venv/bin/python"
    commands = (
        ("builder-venv", (sys.executable, "-m", "venv", "--without-pip", str(stage / "venv"))),
        ("builder-install", builder_install_command(python, report / "builder-install.requirements")),
        ("builder-versions", verify_builder_command(python, manifest)),
        ("builder-check", (str(python), "-m", "pip", "check")),
        ("build", build_command(python, stage / "source", stage / "wheels")),
    )
    _write(report, "builder-install.requirements", _local_requirements(stage / "inputs", manifest["wheels"]))
    for label, command in commands:
        run_checked(command, root=root, report=report, label=label, environment=environment)
    outputs = [file_record(path) for path in (stage / "wheels").iterdir()]
    copy_verified(stage / "wheels", report / "wheels", outputs)
    record, inventory = wheel_result(report / "wheels", project)
    verify_inputs(root, manifest, stage / "inputs")
    return {
        "schema": "xrr-refnx-build-v1",
        "state": "PASS",
        "source": source,
        "build_inputs": manifest,
        "builder_versions": json.loads((report / "builder-versions.stdout").read_text()),
        "toolchain": toolchain,
        "wheel": record,
        "inventory": inventory,
        "offline": True,
        "derived_wheel_reused": False,
        "source_git_metadata": False,
    }


def native_smoke_command(result: dict) -> tuple[str, ...]:
    files = {item["path"]: item["sha256"] for item in result["inventory"]["files"] if item["kind"] == "native"}
    script = (
        "import hashlib,importlib,importlib.metadata as m,json,sys; from pathlib import Path; "
        "d=m.distribution('refnx'); assert d.version==sys.argv[1]; expected=json.loads(sys.argv[2]); "
        "actual={n:hashlib.sha256(Path(d.locate_file(n)).read_bytes()).hexdigest() for n in expected}; "
        "assert actual==expected,actual; names=('refnx._lib._cutil','refnx.reflect._creflect'); "
        "imports={n:hashlib.sha256(Path(importlib.import_module(n).__file__).read_bytes()).hexdigest() for n in names}; "
        "print(json.dumps({'files':actual,'imports':imports,'import_scope':'reflect'},sort_keys=True))"
    )
    return sys.executable, "-c", script, result["wheel"]["version"], json.dumps(files, sort_keys=True)


def _install(root: Path, report: Path, result: dict, environment: dict) -> None:
    _write(report, "install.requirements", _local_requirements(report / "wheels", [result["wheel"]]))
    run_checked(install_command(report), root=root, report=report, label="install", environment=environment)
    run_checked(
        (sys.executable, "-m", "pip", "check"),
        root=root,
        report=report,
        label="installed-check",
        environment=environment,
    )
    observed = json.loads(
        run_checked(
            native_smoke_command(result), root=root, report=report, label="native-import", environment=environment
        )
    )
    expected = {
        item["path"].split(".")[0].replace("/", "."): item["sha256"]
        for item in result["inventory"]["files"]
        if item["kind"] == "native"
    }
    if any(expected[name] != digest for name, digest in observed["imports"].items()):
        raise ValueError("imported refnx native bytes differ from the just-built wheel")
    actual = file_record(report / "wheels" / result["wheel"]["filename"])
    if any(result["wheel"][name] != value for name, value in actual.items()):
        raise ValueError("refnx wheel bytes changed during installation")
    result["installed_native"] = observed


def build_refnx(root: Path, inputs: Path, report: Path, *, install: bool = False) -> int:
    require_install_target(TARGET)
    if Path(sys.prefix).resolve().is_relative_to(root.resolve()):
        raise ValueError("refnx build requires an external virtual environment")
    manifest = read_inputs(root)
    verify_inputs(root, manifest, inputs)
    report, guard, environment = prepare_report(root, report)
    try:
        with tempfile.TemporaryDirectory(prefix="refnx-build-stage-", dir=report) as temporary:
            result = _build_stage(root, inputs, report, Path(temporary), manifest, environment)
        guard()
        verify_inputs(root, manifest, inputs)
        if install:
            _install(root, report, result, environment)
        guard()
        _write(report, "build.json", manifest_bytes(result).decode("ascii"))
    except subprocess.CalledProcessError as error:
        _summary(report, "refnx-build", error.returncode)
        return error.returncode
    _summary(report, "refnx-build", 0, installed=install)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    try:
        return build_refnx(args.repo_root, args.input_dir, args.report_dir, install=args.install)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
