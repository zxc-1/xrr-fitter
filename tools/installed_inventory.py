"""Verify installed distributions against archive bytes and exact installer transforms."""

from __future__ import annotations

import hashlib
import importlib.util
import ntpath
import os
import platform
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from installed_diagnostics import InstalledMismatch, mismatch_evidence  # noqa: E402
from installed_files import InstalledSnapshot, VerifiedWheel  # noqa: E402
from installed_transforms import (  # noqa: E402
    bytecode,
    compile_source,
    direct_url,
    entry_scripts,
    pip_template,
    record_bytes,
    record_row,
    wheel_record,
)


@dataclass(frozen=True)
class InstallLayout:
    root: Path
    library: Path
    scripts: Path
    interpreter: str
    target: str

    def destination(self, path: str, info: str) -> tuple[Path, str]:
        parts = PurePosixPath(path).parts
        if not parts[0].endswith(".data"):
            return self.library / path, "copy"
        if len(parts) < 3 or parts[0] != info.removesuffix(".dist-info") + ".data":
            raise ValueError("wheel relocation directory differs from its distribution")
        roots = {"data": self.root, "scripts": self.scripts, "purelib": self.library, "platlib": self.library}
        if parts[1] not in roots:
            raise ValueError("unsupported wheel installation relocation scheme")
        return roots[parts[1]].joinpath(*parts[2:]), parts[1]

    def record_path(self, path: Path) -> str:
        return os.path.relpath(path, self.library).replace(os.sep, "/")


@dataclass(frozen=True)
class WheelInput:
    path: Path
    record: dict
    provenance: str
    direct_url: bool
    requested: bool = True


def current_layout(target: str) -> InstallLayout:
    expected = {"macos-arm64-py312": ("darwin", "arm64"), "windows-x64-py312": ("win32", "amd64")}
    actual = sys.platform, platform.machine().lower()
    if expected.get(target) != actual or sys.version_info[:2] != (3, 12):
        raise ValueError("installed target differs from the running CPython 3.12 platform")
    if platform.python_implementation() != "CPython" or sys.flags.optimize != 0 or sys.prefix == sys.base_prefix:
        raise ValueError("installation inspection requires an unoptimized CPython virtual environment")
    paths = sysconfig.get_paths()
    if paths["purelib"] != paths["platlib"]:
        raise ValueError("split purelib/platlib installation is not supported")
    return InstallLayout(Path(sys.prefix), Path(paths["purelib"]), Path(paths["scripts"]), sys.executable, target)


def _info_directory(wheel: VerifiedWheel) -> str:
    names = [
        name.split("/")[0] for name in wheel.files if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
    ]
    if len(names) != 1:
        raise ValueError("installed wheel has ambiguous distribution metadata")
    return names[0]


class InstallationInspection:
    def __init__(self, layout: InstallLayout) -> None:
        self.layout = layout
        self.snapshot = InstalledSnapshot(layout.root)
        self.snapshot.anchor(layout.library / "probe", missing=True)
        self.snapshot.anchor(layout.scripts / "probe", missing=True)
        self.members = self.snapshot.members()
        self.canonical_paths = {os.path.normcase(name): name for name in self.members}
        self.files: dict[str, dict] = {}
        self.packages: list[dict] = []
        self.compile_failures: list[dict] = []
        self.compile_warnings: list[dict] = []

    def claim(self, path, digest, item, source, transform, *, expected=None) -> dict:
        try:
            file = self.snapshot.verify(path, digest)
        except (OSError, ValueError) as error:
            evidence = mismatch_evidence(self.snapshot, path, digest, item, source, transform, expected)
            message = f"installed {transform} bytes differ for {item.record['name']} {evidence['path']}: {error}"
            raise InstalledMismatch(message, evidence) from error
        canonical = self.canonical_paths.get(os.path.normcase(file["path"]))
        if canonical is None:
            raise ValueError("installed file appeared during inspection")
        file["path"] = canonical
        claim = {
            "package": item.record["name"],
            "wheel_sha256": item.record["sha256"],
            "source": source,
            "transform": transform,
        }
        previous = self.files.setdefault(canonical, {**file, "claims": []})
        if (previous["sha256"], previous["size"]) != (digest, file["size"]) or claim in previous["claims"]:
            raise ValueError("conflicting or duplicate installed file claim")
        previous["claims"].append(claim)
        return file

    def generated(self, path, content, item, source, transform, rows, *, unhashed=False) -> None:
        self.claim(path, hashlib.sha256(content).hexdigest(), item, source, transform, expected=content)
        relative = self.layout.record_path(path)
        rows.append((relative, "", "") if unhashed else record_row(relative, content))

    def copy_member(self, wheel, item, info, row, rows) -> None:
        name, digest, size = row
        destination, scheme = self.layout.destination(name, info)
        content = wheel.read(name) if destination.suffix == ".py" or scheme == "scripts" else None
        if scheme == "scripts" and content.startswith(b"#!python"):
            content = b"#!" + os.fsencode(self.layout.interpreter) + os.linesep.encode() + content.partition(b"\n")[2]
            self.generated(destination, content, item, name, "wheel-script-shebang", rows)
        else:
            self.claim(destination, wheel.files[name]["sha256"], item, name, "wheel-copy", expected=content)
            rows.append((self.layout.record_path(destination), digest, size))
        if destination.suffix == ".py":
            self.compiled(destination, content, item, name, rows)

    def compiled(self, destination, content, item, name, rows) -> None:
        # pip joins the library with its slash-separated RECORD path before compileall.
        join = ntpath.join if self.layout.target == "windows-x64-py312" else os.path.join
        filename = join(str(self.layout.library), self.layout.record_path(destination))
        code, notices, failure = compile_source(content, filename)
        self.compile_warnings.extend({"package": item.record["name"], "source": name, **notice} for notice in notices)
        if failure is not None:
            self.compile_failures.append({"package": item.record["name"], "source": name, "error": failure})
            cache = self.snapshot.anchor(Path(importlib.util.cache_from_source(filename)), missing=True)
            if cache.exists():
                raise ValueError("noncompilable source has unexpected installed bytecode")
            return
        path, data = bytecode(content, filename, code, self.snapshot)
        self.generated(path, data, item, name, "cpython-bytecode", rows, unhashed=True)

    def metadata(self, item, info, rows) -> None:
        values = {"INSTALLER": b"pip\n"}
        if item.requested:
            values["REQUESTED"] = b""
        if item.direct_url:
            values["direct_url.json"] = direct_url(item.path, item.record["sha256"])
        for name, content in values.items():
            self.generated(self.layout.library / info / name, content, item, info, f"pip-{name}", rows)

    def package(self, item, pip, template, native_loaders) -> None:
        with VerifiedWheel(item.path, item.record, native_loaders=native_loaders) as wheel:
            info = _info_directory(wheel)
            record_path = f"{info}/RECORD"
            rows: list[tuple] = []
            for row in wheel_record(wheel, info):
                if row[0] != record_path:
                    self.copy_member(wheel, item, info, row, rows)
            for path, data in entry_scripts(wheel, info, self.layout, pip, template, self.snapshot):
                self.generated(path, data, item, f"{info}/entry_points.txt", "pip-entry-point", rows)
            self.metadata(item, info, rows)
            rows.append((record_path, "", ""))
            self.claim(
                self.layout.library / record_path,
                hashlib.sha256(record_bytes(rows)).hexdigest(),
                item,
                record_path,
                "pip-RECORD",
                expected=record_bytes(rows),
            )
            self.packages.append(
                {"wheel": dict(item.record), "provenance": item.provenance, "inventory": wheel.inventory}
            )

    def finish(self, pip) -> dict:
        current = self.snapshot.members()
        if set(current) != set(self.members):
            raise ValueError("installed environment membership changed during inspection")
        unknown = [path for name, path in current.items() if name not in self.files]
        if any(path.is_relative_to(self.layout.library) for path in unknown):
            raise ValueError("unowned installed site-packages files")
        unowned = [self.snapshot.unowned(path) for path in sorted(unknown)]
        self.snapshot.finish()
        for file in self.files.values():
            file["claims"].sort(key=lambda claim: (claim["package"], claim["source"], claim["transform"]))
        return {
            "schema": "xrr-installed-inventory-v1",
            "target": self.layout.target,
            "library_path": self.layout.library.relative_to(self.layout.root).as_posix(),
            "python_version": platform.python_version(),
            "installer_wheel_sha256": pip.record["sha256"],
            "interpreter_path_sha256": hashlib.sha256(self.layout.interpreter.encode()).hexdigest(),
            "packages": self.packages,
            "files": [self.files[name] for name in sorted(self.files)],
            "compile_failures": self.compile_failures,
            "compile_warnings": self.compile_warnings,
            "unowned": unowned,
            "aggregate": "incomplete",
            "boundary": "verified wheel installation; host interpreter, OS and runtime loader closure are not established",
        }


def inspect_installation(
    layout: InstallLayout, wheels: list[WheelInput], pip: WheelInput, *, native_loaders: bool = False
) -> dict:
    names = [item.record["name"] for item in wheels]
    if len(names) != len(set(names)) or not wheels:
        raise ValueError("installed wheel identities must be unique and nonempty")
    if pip not in wheels:
        raise ValueError("pinned installer wheel must be included in the installed closure")
    inspection = InstallationInspection(layout)
    with VerifiedWheel(pip.path, pip.record) as installer:
        template = pip_template(installer)
        for item in sorted(wheels, key=lambda value: value.record["name"]):
            inspection.package(item, installer, template, native_loaders)
        return inspection.finish(pip)
