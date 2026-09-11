"""Bind ordinary, bootstrap, test-only and rebuilt VCS installation inputs."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.tags import sys_tags  # noqa: E402
from packaging.utils import parse_wheel_filename  # noqa: E402
from packaging.version import Version  # noqa: E402

from distribution_manifest import _json_object  # noqa: E402
from installed_files import InstalledSnapshot  # noqa: E402
from installed_inventory import WheelInput  # noqa: E402
from package_downloads import read_manifest  # noqa: E402
from package_manifest import (  # noqa: E402
    _verified_file,
    target_tags,
    verify_wheels,
)
from wheel_inventory import inspect_wheel  # noqa: E402

PIN = re.compile(r"([a-z0-9][a-z0-9-]*)==([^\s]+) --hash=sha256:([0-9a-f]{64})")


def read_bound(path: Path) -> bytes:
    snapshot = InstalledSnapshot(path.absolute().parent)
    content = snapshot.read(path, limit=32 * 1024**2)
    snapshot.finish()
    return content


class InputBindings:
    def __init__(self, paths: dict[str, Path], *, limit: int = 32 * 1024**2) -> None:
        self.limit = limit
        self.paths = {}
        self.contents = {}
        self.snapshots = {}
        self.add(paths)

    def add(self, paths: dict[str, Path]) -> None:
        for name, path in paths.items():
            if name in self.paths:
                raise ValueError("duplicate installation input binding")
            snapshot = InstalledSnapshot(path.absolute().parent)
            self.paths[name] = path
            self.contents[name] = snapshot.read(path, limit=self.limit)
            self.snapshots[name] = snapshot

    def guard(self) -> None:
        for name, path in self.paths.items():
            snapshot = self.snapshots[name]
            if snapshot.read(path, limit=self.limit) != self.contents[name]:
                raise ValueError("installation input binding changed")
            snapshot.finish()

    def hashes(self) -> dict[str, str]:
        return {name: hashlib.sha256(data).hexdigest() for name, data in self.contents.items()}


def json_object(content: bytes) -> dict:
    value = json.loads(content.decode("utf-8-sig"), object_pairs_hook=_json_object)
    if not isinstance(value, dict):
        raise ValueError("bound JSON evidence must be an object")
    return value


def read_hash_lock(path: Path) -> dict[str, dict[str, str]]:
    result = {}
    for line in read_bound(path).decode("ascii").splitlines():
        match = PIN.fullmatch(line)
        if match is None or match[1] in result:
            raise ValueError("auxiliary lock requires unique exact hash-pinned wheels")
        name, version, digest = match.groups()
        if str(Version(version)) != version:
            raise ValueError("auxiliary lock version is not canonical")
        result[name] = {"version": version, "sha256": digest}
    if not result:
        raise ValueError("auxiliary lock cannot be empty")
    return result


def locked_wheel(path: Path, pins: dict, target: str) -> dict:
    name, version, _build, tags = parse_wheel_filename(path.name)
    if name not in pins or str(version) != pins[name]["version"] or not tags.intersection(target_tags(target)):
        raise ValueError("auxiliary wheel identity differs from the pinned target")
    _verified_file(path, pins[name]["sha256"])
    record = {"name": str(name), "version": str(version), "filename": path.name, "sha256": pins[name]["sha256"]}
    return {"path": path, "record": record}


def _auxiliary_records(paths: list[Path], pins: dict, target: str) -> dict:
    if len(paths) != len(pins) or any(path.suffix != ".whl" for path in paths):
        raise ValueError("auxiliary directory must contain exactly its locked wheels")
    result = {}
    for path in paths:
        item = locked_wheel(path, pins, target)
        name = item["record"]["name"]
        if name in result:
            raise ValueError("duplicate auxiliary wheel identity")
        result[name] = item
    if set(result) != set(pins):
        raise ValueError("auxiliary wheels do not cover the exact pins")
    return result


def auxiliary_wheels(directory: Path, pins: dict, target: str) -> dict:
    if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
        raise ValueError("auxiliary wheel directory must be regular")
    snapshot = InstalledSnapshot(directory)
    result = _auxiliary_records(list(directory.iterdir()), pins, target)
    verify_wheels(directory, [item["record"] for item in result.values()])
    snapshot.finish()
    return result


def _merge_pins(*groups: dict) -> dict:
    result = {}
    for group in groups:
        for name, pin in group.items():
            if name in result and result[name] != pin:
                raise ValueError("conflicting auxiliary lock pins")
            result[name] = pin
    return result


def _bootstrap_inputs(root, target, pip_wheel, ordinary):
    pins = read_hash_lock(root / "tools/bootstrap-requirements.lock")
    pip = locked_wheel(pip_wheel, pins, target)
    if pip["record"]["name"] != "pip":
        raise ValueError("installer input must be the locked pip wheel")
    for name, pin in pins.items():
        if name == "pip":
            continue
        if name not in ordinary or any(ordinary[name][key] != value for key, value in pin.items()):
            raise ValueError("bootstrap overlap differs from ordinary wheel bytes")
    return WheelInput(**pip, provenance="bootstrap-lock", direct_url=False), pins


def _refnx_build_identity(root: Path, build: dict) -> None:
    expected_source = json_object(read_bound(root / "tools/package-manifests/refnx-source.json"))
    expected_builder = json_object(read_bound(root / "tools/package-manifests/refnx-build-macos-arm64-py312.json"))
    if build["schema"] != "xrr-refnx-build-v1":
        raise ValueError("refnx build report schema differs")
    if build["state"] != "PASS" or build["offline"] is not True or build["derived_wheel_reused"] is not False:
        raise ValueError("refnx installation requires a fresh successful offline build")
    if build["source"] != expected_source or build["build_inputs"] != expected_builder:
        raise ValueError("refnx build source or builder identities differ")
    versions = {wheel["name"]: wheel["version"] for wheel in expected_builder["wheels"]}
    if build["source_git_metadata"] is not False or build["builder_versions"] != versions:
        raise ValueError("refnx build metadata or builder versions differ")


def _refnx_input(root: Path, report: Path) -> WheelInput:
    build = json_object(read_bound(report))
    _refnx_build_identity(root, build)
    record = build["wheel"]
    name, version, _build, tags = parse_wheel_filename(record["filename"])
    if (name, str(version)) != ("refnx", record["version"]) or not tags.intersection(sys_tags()):
        raise ValueError("refnx wheel identity does not support the running interpreter")
    directory = report.parent / "wheels"
    snapshot = InstalledSnapshot(directory)
    verified = verify_wheels(directory, [record])
    if verified[0]["size"] != record["size"]:
        raise ValueError("refnx build wheel size differs")
    path = directory / record["filename"]
    snapshot.verify(path, record["sha256"])
    if inspect_wheel(path, record) != build["inventory"]:
        raise ValueError("refnx build inventory differs from its wheel bytes")
    verify_wheels(directory, [record])
    snapshot.finish()
    return WheelInput(path, record, "refnx-source-build", True)


def _windows_inputs(root, directory, bootstrap, ordinary):
    test_pins = read_hash_lock(root / "tools/windows-test-requirements.lock")
    items = auxiliary_wheels(directory, _merge_pins(bootstrap, test_pins), "windows-x64-py312")
    if set(test_pins).intersection(ordinary):
        raise ValueError("Windows test-only inputs overlap the ordinary manifest")
    return [WheelInput(**items[name], provenance="windows-test-lock", direct_url=False) for name in sorted(test_pins)]


def _wheel_snapshots(inputs: list[WheelInput]) -> list[InstalledSnapshot]:
    result = []
    for item in inputs:
        snapshot = InstalledSnapshot(item.path.absolute().parent)
        snapshot.verify(item.path, item.record["sha256"])
        result.append(snapshot)
    return result


def load_installation_inputs(
    root: Path,
    manifest_path: Path,
    wheel_directory: Path,
    pip_wheel: Path,
    *,
    refnx_build: Path | None = None,
    auxiliary_directory: Path | None = None,
):
    paths = {"ordinary-manifest": manifest_path, "bootstrap-lock": root / "tools/bootstrap-requirements.lock"}
    bound = InputBindings(paths)
    manifest = read_manifest(root, manifest_path)
    bound.add(_platform_paths(root, refnx_build, auxiliary_directory))
    verify_wheels(wheel_directory, manifest["wheels"])
    ordinary = {item["name"]: item for item in manifest["wheels"]}
    inputs = [
        WheelInput(wheel_directory / item["filename"], item, "ordinary-manifest", True) for item in manifest["wheels"]
    ]
    pip, bootstrap = _bootstrap_inputs(root, manifest["target"], pip_wheel, ordinary)
    inputs.append(pip)
    _platform_inputs(root, manifest, refnx_build, auxiliary_directory, bootstrap, ordinary, inputs, paths)
    snapshots = _wheel_snapshots(inputs)

    def guard():
        bound.guard()
        for snapshot in snapshots:
            snapshot.finish()
        if read_manifest(root, manifest_path) != manifest:
            raise ValueError("ordinary installation manifest changed")
        verify_wheels(wheel_directory, manifest["wheels"])
        if auxiliary_directory is not None:
            auxiliary_wheels(
                auxiliary_directory,
                _merge_pins(bootstrap, read_hash_lock(paths["windows-test-lock"])),
                manifest["target"],
            )

    guard()
    return manifest, inputs, pip, bound.hashes(), guard


def _platform_paths(root, refnx, auxiliary) -> dict:
    result = {}
    if refnx is not None:
        result.update(
            {
                "refnx-build": refnx,
                "refnx-source": root / "tools/package-manifests/refnx-source.json",
                "refnx-builder": root / "tools/package-manifests/refnx-build-macos-arm64-py312.json",
            }
        )
    if auxiliary is not None:
        result["windows-test-lock"] = root / "tools/windows-test-requirements.lock"
    return result


def _platform_inputs(root, manifest, refnx, auxiliary, bootstrap, ordinary, inputs, paths):
    if manifest["target"] == "macos-arm64-py312":
        if refnx is None or auxiliary is not None or [item["name"] for item in manifest["vcs"]] != ["refnx"]:
            raise ValueError("macOS installed closure requires its separate refnx build evidence")
        inputs.append(_refnx_input(root, refnx))
        paths.update(
            {
                "refnx-build": refnx,
                "refnx-source": root / "tools/package-manifests/refnx-source.json",
                "refnx-builder": root / "tools/package-manifests/refnx-build-macos-arm64-py312.json",
            }
        )
    else:
        if auxiliary is None or refnx is not None or manifest["vcs"]:
            raise ValueError("Windows installed closure requires the separate auxiliary wheels")
        inputs.extend(_windows_inputs(root, auxiliary, bootstrap, ordinary))
        paths["windows-test-lock"] = root / "tools/windows-test-requirements.lock"
