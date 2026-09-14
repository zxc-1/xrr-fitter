"""Small Qt source-build wheel fixtures, never used as native acceptance evidence."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import shutil
import struct
import zipfile
from pathlib import Path

from tests.support.native_fixtures import macho, macho_command

INFO = "pyside6_essentials-6.11.2.dist-info"
PLUGIN = "PySide6/Qt/plugins/platforms/libqcocoa.dylib"


def _records(files):
    stream = io.StringIO(newline="")
    rows = []
    for name, content in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
        rows.append((name, f"sha256={digest}", str(len(content))))
    csv.writer(stream).writerows([*rows, (f"{INFO}/RECORD", "", "")])
    return stream.getvalue().encode()


def _plugin(*, cpu=0x100000C, minos=13 << 16, rpath="@loader_path/../../lib"):
    commands = [macho_command(0x8000001C, rpath, dylib=False), struct.pack("<6I", 0x32, 24, 1, minos, 26 << 16, 0)]
    return macho(commands, cpu=cpu)


def _upstream(tmp_path):
    files = {
        PLUGIN: b"original vendor plugin",
        "PySide6/module.py": b"value = 42\n",
        f"{INFO}/METADATA": b"Name: PySide6-Essentials\nVersion: 6.11.2\nRequires-Dist: shiboken6==6.11.2\n",
        f"{INFO}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp310-abi3-macosx_13_0_universal2\n",
        f"{INFO}/licenses/LICENSE": b"original license notice\n",
    }
    files[f"{INFO}/RECORD"] = _records(files)
    path = tmp_path / "pyside6_essentials-6.11.2-cp310-abi3-macosx_13_0_universal2.whl"
    _write_zip(path, files)
    record = {
        "name": "pyside6-essentials",
        "version": "6.11.2",
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return path, record, files


def _write_zip(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in sorted(files.items()):
            archive.writestr(name, value)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "tools/package-manifests/qt-cocoa-macos-arm64-py312.json"


def qt_build_case(load_tool_module, tmp_path):
    module = load_tool_module("qt_cocoa_evidence")
    inputs = load_tool_module("qt_cocoa_inputs")
    wheel = load_tool_module("qt_cocoa_wheel")
    root = tmp_path / "repo"
    manifest = inputs.read_inputs(ROOT)
    for name in [MANIFEST, *module.BUILD_CODE, *[r["path"] for r in manifest["recipes"]]]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (tmp_path / "upstream").mkdir()
    source, upstream, _ = _upstream(tmp_path / "upstream")
    manifest["upstream_wheel"] = upstream
    manifest["sources"] = manifest["sources"][:2]
    for record in manifest["sources"]:
        content = record["path"].encode()
        record.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    (root / MANIFEST).write_text(json.dumps(manifest))
    (root / "tools/package-manifests/macos-arm64-py312.json").write_text(
        json.dumps({"target": "macos-arm64-py312", "wheels": [upstream]})
    )
    additions = module.source_additions(root, manifest, lambda record: record["path"].encode())
    report = tmp_path / "report"
    (report / "wheels").mkdir(parents=True)
    plugin = tmp_path / "libqcocoa.dylib"
    plugin.write_bytes(_plugin())
    derived = wheel.derive_wheel(source, upstream, plugin, report / "wheels", additions)
    derivation = wheel.verify_derivation(source, upstream, report / "wheels" / derived["filename"], derived, additions)
    result = module.build_receipt(
        root, manifest, derived, derivation, {"machine": "arm64"}, {name: 0 for name in module.CASES}
    )
    (report / "build.json").write_text(json.dumps(result))
    return module, root, source.parent, report, result
