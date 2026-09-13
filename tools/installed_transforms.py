"""Reconstruct CPython and pinned pip installation outputs without running them."""

from __future__ import annotations

import ast
import base64
import configparser
import csv
import hashlib
import importlib.util
import io
import json
import marshal
import ntpath
import os
import re
import struct
import textwrap
import warnings
import zipfile
from pathlib import Path, PurePosixPath


def record_row(path: str, data: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
    return path, f"sha256={digest}", str(len(data))


def record_bytes(rows: list[tuple]) -> bytes:
    result = io.StringIO(newline="")
    csv.writer(result).writerows(sorted(rows))
    return result.getvalue().encode("utf-8")


def wheel_record(wheel, info: str) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(wheel.read(f"{info}/RECORD").decode("utf-8"))))
    if any(len(row) != 3 for row in rows):
        raise ValueError("wheel RECORD must have three fields")
    if len({row[0] for row in rows}) != len(rows) or {row[0] for row in rows} != set(wheel.files):
        raise ValueError("wheel RECORD must cover every unique member")
    for path, digest, size in rows:
        expected = ("", "") if path == f"{info}/RECORD" else _archive_digest(wheel.files[path])
        if (digest, size) != expected:
            raise ValueError("wheel RECORD bytes disagree with the verified archive")
    return rows


def _archive_digest(file: dict) -> tuple[str, str]:
    digest = base64.urlsafe_b64encode(bytes.fromhex(file["sha256"])).decode().rstrip("=")
    return f"sha256={digest}", str(file["size"])


def _template_assignment(node) -> bool:
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "script_template" for target in node.targets
    )


def _pinned_script_maker(wheel) -> ast.ClassDef:
    if (wheel.record["name"], wheel.record["version"]) != ("pip", "26.1.2"):
        raise ValueError("installed transformations require the pinned pip 26.1.2 wheel")
    source = ast.parse(wheel.read("pip/_internal/operations/install/wheel.py"))
    classes = [node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "PipScriptMaker"]
    if len(classes) != 1:
        raise ValueError("pinned pip script template class differs")
    return classes[0]


def _dedented_template(expression) -> str:
    if not isinstance(expression, ast.Call) or len(expression.args) != 1:
        raise ValueError("pinned pip script template differs")
    if ast.unparse(expression.func) != "textwrap.dedent" or expression.keywords:
        raise ValueError("pinned pip script template transform differs")
    value = ast.literal_eval(expression.args[0])
    if not isinstance(value, str):
        raise ValueError("pip script template must be text")
    return textwrap.dedent(value)


def pip_template(wheel) -> str:
    maker = _pinned_script_maker(wheel)
    values = [node.value for node in maker.body if _template_assignment(node)]
    if len(values) != 1:
        raise ValueError("pinned pip script template differs")
    return _dedented_template(values[0])


def _entry_points(content: bytes) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read_string(content.decode("utf-8"))
    return {
        section: dict(parser.items(section)) if parser.has_section(section) else {}
        for section in ("console_scripts", "gui_scripts")
    }


def _versioned_scripts(values: dict[str, str], package: str) -> dict[str, str]:
    result = dict(values)
    if package == "pip" and "pip" in result:
        entry = result["pip"]
        result = {name: spec for name, spec in result.items() if re.fullmatch(r"pip\d+(\.\d+)?", name) is None}
        result.update({name: entry for name in ("pip", "pip3", "pip3.12")})
    return result


def _entry_body(specification: str, template: str) -> bytes:
    match = re.fullmatch(r"\s*([\w.]+)\s*:\s*([\w.]+)(?:\s*\[[^\]]*\])?\s*", specification)
    if match is None:
        raise ValueError("invalid installed entry point callable")
    module, function = match.groups()
    if not all(part.isidentifier() for part in (module + "." + function).split(".")):
        raise ValueError("invalid installed entry point identifier")
    return (template % {"module": module, "import_name": function.split(".")[0], "func": function}).encode()


def shebang(interpreter: str, windows: bool) -> bytes:
    executable = f'"{interpreter}"' if " " in interpreter else interpreter
    raw = executable.encode("utf-8")
    if windows or (b" " not in raw and len(raw) + 3 <= 512):
        return b"#!" + raw + b"\n"
    return b"#!/bin/sh\n'''exec' " + raw + b' "$0" "$@"\n' + b"' '''\n"


def windows_launcher(actual: bytes, launcher: bytes, header: bytes, body: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(actual)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != "__main__.py":
                raise ValueError("Windows entry point launcher has unexpected members")
            date_time = members[0].date_time
    except zipfile.BadZipFile as error:
        raise ValueError("invalid Windows entry point launcher") from error
    buffer = io.BytesIO()
    entry = zipfile.ZipInfo("__main__.py", date_time)
    entry.create_system = 0
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(entry, body)
    return launcher + header + buffer.getvalue()


def entry_scripts(wheel, info, layout, pip, template, snapshot):
    entry = f"{info}/entry_points.txt"
    if entry not in wheel.files:
        return []
    groups = _entry_points(wheel.read(entry))
    groups["console_scripts"] = _versioned_scripts(groups["console_scripts"], wheel.record["name"])
    result = []
    for group, scripts in groups.items():
        for name, spec in scripts.items():
            result.append(_entry_script(name, spec, group, layout, pip, template, snapshot))
    if len({str(path) for path, _ in result}) != len(result):
        raise ValueError("duplicate installed entry point destination")
    return result


def _entry_script(name, spec, group, layout, pip, template, snapshot):
    if PurePosixPath(name).name != name or name in {".", ".."} or "\\" in name or ":" in name:
        raise ValueError("entry point escapes the scripts directory")
    windows = layout.target == "windows-x64-py312"
    gui = group == "gui_scripts"
    interpreter = layout.interpreter
    if windows and gui:
        directory, filename = ntpath.split(interpreter)
        interpreter = ntpath.join(directory, filename.replace("python", "pythonw"))
    header, body = shebang(interpreter, windows), _entry_body(spec, template)
    if not windows:
        return layout.scripts / name, header + body
    stem, extension = os.path.splitext(name)
    path = layout.scripts / ((stem if extension.startswith(".py") else name) + ".exe")
    launcher = pip.read(f"pip/_vendor/distlib/{'w' if gui else 't'}64.exe")
    return path, windows_launcher(snapshot.read(path), launcher, header, body)


def direct_url(path: Path, checksum: str) -> bytes:
    value = {
        "archive_info": {"hash": f"sha256={checksum}", "hashes": {"sha256": checksum}},
        "url": path.resolve().as_uri(),
    }
    return json.dumps(value, sort_keys=True).encode("utf-8")


def compile_source(content: bytes, filename: str) -> tuple[object | None, list[dict], str | None]:
    try:
        with warnings.catch_warnings(record=True) as observed:
            code = compile(content, filename, "exec", dont_inherit=True, optimize=0)
    except SyntaxError as error:
        return None, [], f"{error.msg}; line {error.lineno}"
    notices = [{"category": item.category.__name__, "message": str(item.message)} for item in observed]
    return code, notices, None


def bytecode(content: bytes, filename: str, code, snapshot) -> tuple[Path, bytes]:
    cache = Path(importlib.util.cache_from_source(filename))
    actual = snapshot.read(cache)
    if len(actual) < 16 or actual[:4] != importlib.util.MAGIC_NUMBER:
        raise ValueError("installed bytecode magic differs from the current CPython")
    flags = struct.unpack("<I", actual[4:8])[0]
    if flags not in (0, 1, 3):
        raise ValueError("invalid installed bytecode flags")
    header = importlib.util.MAGIC_NUMBER + struct.pack("<I", flags)
    if flags == 0:
        source = snapshot.anchor(Path(filename))
        header += struct.pack("<II", int(source.stat().st_mtime) & 0xFFFFFFFF, len(content) & 0xFFFFFFFF)
    else:
        header += importlib.util.source_hash(content)
    return cache, header + marshal.dumps(code)
