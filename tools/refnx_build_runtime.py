"""Commands and observed platform toolchain for the job-private refnx builder."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from audit_reports import run_logged  # noqa: E402
from package_manifest import _verified_file  # noqa: E402


def build_command(python: Path, source: Path, wheels: Path) -> tuple[str, ...]:
    return (
        str(python),
        "-m",
        "pip",
        "--isolated",
        "--verbose",
        "wheel",
        "--no-index",
        "--no-deps",
        "--no-build-isolation",
        "--no-cache-dir",
        "--wheel-dir",
        str(wheels),
        str(source),
    )


def builder_install_command(python: Path, requirements: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pip",
        "--isolated",
        "--python",
        str(python),
        "install",
        "--no-index",
        "--no-deps",
        "--no-cache-dir",
        "--require-hashes",
        "--only-binary=:all:",
        "--requirement",
        str(requirements),
    )


def build_environment(inherited: dict, stage: Path, toolchain: dict) -> dict[str, str]:
    return {
        **{name: inherited[name] for name in ("LANG", "LC_ALL") if name in inherited},
        "PATH": f"{stage}/venv/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(stage / "home"),
        "TMPDIR": str(stage / "tmp"),
        "XDG_CACHE_HOME": str(stage / "cache"),
        "CC": toolchain["cc"],
        "CXX": toolchain["cxx"],
        "SDKROOT": toolchain["sdk_path"],
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_INPUT": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def run_checked(command, *, root: Path, report: Path, label: str, environment: dict) -> str:
    code = run_logged(command, root=root, report=report, label=label, environment=environment)
    if code != 0:
        raise subprocess.CalledProcessError(code, command)
    return (report / f"{label}.stdout").read_text(encoding="utf-8").strip()


def file_record(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("refnx build evidence requires a regular file")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return _verified_file(path, digest)


def toolchain_evidence(root: Path, report: Path, environment: dict) -> dict:
    def observe(label, command):
        return run_checked(command, root=root, report=report, label=label, environment=environment)

    cc = Path(observe("cc-path", ("/usr/bin/xcrun", "--find", "clang")))
    cxx = Path(observe("cxx-path", ("/usr/bin/xcrun", "--find", "clang++")))
    sdk = Path(observe("sdk-path", ("/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"))).resolve(strict=True)
    return {
        "cc": str(cc),
        "cxx": str(cxx),
        "sdk_path": str(sdk),
        "cc_file": file_record(cc.resolve(strict=True)),
        "cxx_file": file_record(cxx.resolve(strict=True)),
        "compiler_version": observe("cc-version", (str(cc), "--version")),
        "sdk_version": observe("sdk-version", ("/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-version")),
        "sdk_settings": file_record((sdk / "SDKSettings.json").resolve(strict=True)),
        "os_build": observe("os-build", ("/usr/bin/sw_vers", "-buildVersion")),
        "cpu": observe("cpu", ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string")),
        "machine": platform.machine(),
        "python": sys.version,
        "external_boundary": "host CPython, Apple compiler and macOS SDK; not a cross-host reproducible wheel claim",
    }


def verify_builder_command(python: Path, manifest: dict) -> tuple[str, ...]:
    expected = {item["name"]: item["version"] for item in manifest["wheels"]}
    script = (
        "import importlib.metadata as m,json,sys; from packaging.utils import canonicalize_name; "
        "items=list(m.distributions()); actual={canonicalize_name(d.metadata['Name']):d.version for d in items}; "
        "assert len(items)==len(actual) and actual==json.loads(sys.argv[1]),actual; "
        "print(json.dumps(actual,sort_keys=True))"
    )
    return str(python), "-c", script, json.dumps(expected, sort_keys=True)
