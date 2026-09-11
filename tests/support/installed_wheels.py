from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import py_compile
import textwrap
import zipfile
from pathlib import Path

SCRIPT_TEMPLATE = (
    "import sys\nfrom %(module)s import %(import_name)s\n"
    "if __name__ == '__main__':\n"
    "    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
    "    sys.exit(%(func)s())\n"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_row(name: str, data: bytes) -> tuple[str, str, str]:
    value = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    return name, f"sha256={value}", str(len(data))


def record_bytes(rows) -> bytes:
    result = io.StringIO(newline="")
    csv.writer(result).writerows(sorted(tuple(row) for row in rows))
    return result.getvalue().encode()


def make_wheel(root: Path, *, name="sample", version="1.0", entries=None) -> dict:
    root.mkdir(exist_ok=True, parents=True)
    info = f"{name}-{version}.dist-info"
    files = {
        f"{info}/METADATA": f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n".encode(),
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        **(entries or {}),
    }
    rows = [record_row(path, data) for path, data in files.items()]
    files[f"{info}/RECORD"] = record_bytes([*rows, (f"{info}/RECORD", "", "")])
    path = root / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, data in files.items():
            archive.writestr(member, data)
    return {
        "path": path,
        "record": {"name": name, "version": version, "filename": path.name, "sha256": digest(path.read_bytes())},
        "provenance": "test-wheel",
        "direct_url": True,
    }


def make_pip(root: Path) -> dict:
    source = f"import textwrap\nclass PipScriptMaker:\n    script_template = textwrap.dedent({SCRIPT_TEMPLATE!r})\n"
    return make_wheel(
        root,
        name="pip",
        version="26.1.2",
        entries={
            "pip/_internal/operations/install/wheel.py": source.encode(),
        },
    )


def layout_values(root: Path) -> dict:
    library = root / "lib/python3.12/site-packages"
    scripts = root / "bin"
    library.mkdir(parents=True)
    scripts.mkdir()
    return {
        "root": root,
        "library": library,
        "scripts": scripts,
        "interpreter": str(scripts / "python"),
        "target": "macos-arm64-py312",
    }


def _destination(name: str, layout: dict) -> Path:
    parts = Path(name).parts
    if parts[0].endswith(".data"):
        roots = {"data": layout["root"], "scripts": layout["scripts"], "purelib": layout["library"]}
        return roots[parts[1]].joinpath(*parts[2:])
    return layout["library"] / name


def _record_path(path: Path, layout: dict) -> str:
    return os.path.relpath(path, layout["library"]).replace(os.sep, "/")


def _install_member(name, data, layout, rows, *, hash_pyc=False):
    path = _destination(name, layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    if ".data/scripts/" in name and data.startswith(b"#!python"):
        data = b"#!" + os.fsencode(layout["interpreter"]) + os.linesep.encode() + data.partition(b"\n")[2]
    path.write_bytes(data)
    rows.append(record_row(_record_path(path, layout), data))
    if path.suffix != ".py":
        return
    try:
        mode = py_compile.PycInvalidationMode.CHECKED_HASH if hash_pyc else py_compile.PycInvalidationMode.TIMESTAMP
        raw_path = str(layout["library"] / _record_path(path, layout))
        cache = py_compile.compile(raw_path, doraise=True, invalidation_mode=mode)
    except py_compile.PyCompileError:
        return
    rows.append((_record_path(Path(cache), layout), "", ""))


def install_fixture(wheel: dict, layout: dict, *, entry_points=(), hash_pyc=False) -> None:
    info = f"{wheel['record']['name']}-{wheel['record']['version']}.dist-info"
    rows = []
    with zipfile.ZipFile(wheel["path"]) as archive:
        for name in archive.namelist():
            if name == f"{info}/RECORD":
                continue
            _install_member(name, archive.read(name), layout, rows, hash_pyc=hash_pyc)
    generated = {f"{info}/INSTALLER": b"pip\n", f"{info}/REQUESTED": b""}
    if wheel["direct_url"]:
        checksum = wheel["record"]["sha256"]
        url = {
            "archive_info": {"hash": f"sha256={checksum}", "hashes": {"sha256": checksum}},
            "url": wheel["path"].resolve().as_uri(),
        }
        generated[f"{info}/direct_url.json"] = json.dumps(url, sort_keys=True).encode()
    for name, data in generated.items():
        path = layout["library"] / name
        path.write_bytes(data)
        rows.append(record_row(name, data))
    for name, module, function in entry_points:
        script = textwrap.dedent(SCRIPT_TEMPLATE) % {
            "module": module,
            "import_name": function.split(".")[0],
            "func": function,
        }
        data = b"#!" + os.fsencode(layout["interpreter"]) + b"\n" + script.encode()
        path = layout["scripts"] / name
        path.write_bytes(data)
        rows.append(record_row(_record_path(path, layout), data))
    (layout["library"] / info / "RECORD").write_bytes(record_bytes([*rows, (f"{info}/RECORD", "", "")]))


def sample_installation(tmp_path: Path, *, entries=None, hash_pyc=False):
    layout = layout_values(tmp_path / "venv")
    pip = make_pip(tmp_path / "wheels")
    pip["direct_url"] = False
    install_fixture(pip, layout, hash_pyc=hash_pyc)
    sample = make_wheel(
        tmp_path / "wheels",
        entries=entries or {"sample/__init__.py": b"raise RuntimeError('must not import audited code')\n"},
    )
    install_fixture(sample, layout, hash_pyc=hash_pyc)
    return layout, [pip, sample]


def rewrite_installed_record(path: Path, library: Path) -> None:
    rows = list(csv.reader(io.StringIO(path.read_text(), newline="")))
    result = []
    for name, checksum, size in rows:
        if checksum:
            result.append(record_row(name, (library / name).read_bytes()))
        else:
            result.append((name, checksum, size))
    path.write_bytes(record_bytes(result))
