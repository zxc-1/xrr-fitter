"""Byte-level provenance candidates for frozen files and recompiled Python code."""

from __future__ import annotations

import hashlib
import json
import marshal
import types
from collections import defaultdict
from contextlib import ExitStack
from pathlib import PurePosixPath
from urllib.parse import quote

from frozen_code import code_fields_digest
from installed_files import VerifiedWheel
from installed_transforms import compile_source
from native_inventory import wheel_file_reference


def _file_component(ref: str, name: str, digest: str, **properties) -> dict:
    return {
        "type": "file",
        "bom-ref": ref,
        "name": name,
        "hashes": [{"alg": "SHA-256", "content": digest}],
        "properties": [{"name": key, "value": str(value)} for key, value in sorted(properties.items())],
    }


def _renamed(code, filename: str):
    constants = tuple(
        _renamed(value, filename) if isinstance(value, types.CodeType) else value for value in code.co_consts
    )
    return code.replace(co_consts=constants, co_filename=filename)


def serialized_source(content: bytes, filename: str) -> bytes:
    code, _notices, failure = compile_source(content, "<verified-source>")
    if failure is not None:
        raise ValueError("executable source cannot compile with the current CPython")
    transformed = _renamed(code, filename)
    return marshal.dumps(transformed)


def _code_match(content: bytes, filename: str, data: bytes) -> dict | None:
    expected = serialized_source(content, filename)
    if expected == data:
        return {"transform": "pyinstaller-code"}
    actual_fields = code_fields_digest(data)
    if actual_fields != code_fields_digest(expected):
        return None
    return {
        "transform": "pyinstaller-code-marshal-layout",
        "code_fields_sha256": actual_fields,
        "expected_wire_sha256": hashlib.sha256(expected).hexdigest(),
        "layout_boundary": "valid marshal object-sharing/interning only; every code field compared",
    }


class ExecutableSources:
    def __init__(
        self, inputs, inventory: dict, source_files: dict[str, bytes], source: dict, *, entry_point: str
    ) -> None:
        self.inputs, self.inventory = inputs, inventory
        self.source_files, self.source, self.entry_point = source_files, source, entry_point
        self.stack = ExitStack()
        self.archives, self.paths, self.raw = {}, defaultdict(list), defaultdict(list)
        self.components, self.dependencies = {}, {}
        self.known = {}
        prefix = inventory["library_path"] + "/"
        for file in inventory["files"]:
            self.raw[(file["sha256"], file["size"])].append(file)
            if file["path"].startswith(prefix):
                self.known[file["path"].removeprefix(prefix)] = file

    def __enter__(self):
        try:
            for item in self.inputs:
                wheel = self.stack.enter_context(VerifiedWheel(item.path, item.record))
                self.archives[item.record["sha256"]] = wheel
                for path in wheel.files:
                    self.paths[path].append(wheel)
            builders = [item.record for item in self.inputs if item.record["name"] == "pyinstaller"]
            if len(builders) != 1 or builders[0]["version"] != "6.21.0":
                raise ValueError("executable provenance requires the pinned PyInstaller 6.21.0 wheel")
            self.builder = builders[0]
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def _package(self, wheel: dict) -> str:
        ref = f"urn:xrr:archive:{wheel['sha256']}"
        self.components[ref] = {
            "type": "library",
            "bom-ref": ref,
            "name": wheel["name"],
            "version": wheel["version"],
            "hashes": [{"alg": "SHA-256", "content": wheel["sha256"]}],
            "properties": [{"name": "xrr:role", "value": "input-provenance-not-complete-bundled-package"}],
        }
        return ref

    def _wheel_file(self, wheel, path: str) -> str:
        ref = wheel_file_reference(wheel.record, path)
        self.components[ref] = _file_component(ref, path, wheel.files[path]["sha256"])
        self.dependencies[ref] = [self._package(wheel.record)]
        return ref

    def _git_file(self, path: str) -> str:
        ref = f"urn:xrr:source:{self.source['source_tree']}:{quote(path, safe='')}"
        self.components[ref] = _file_component(ref, path, hashlib.sha256(self.source_files[path]).hexdigest())
        return ref

    def _installed_file(self, file: dict) -> str:
        ref = f"urn:xrr:installed:{quote(file['path'], safe='')}"
        self.components[ref] = _file_component(
            ref,
            file["path"],
            file["sha256"],
            **{"xrr:installed:claims": json.dumps(file["claims"], sort_keys=True), "xrr:role": "verified-input-bytes"},
        )
        parents = set()
        for claim in file["claims"]:
            wheel = self.archives[claim["wheel_sha256"]]
            parents.add(self._package(wheel.record))
            if claim["source"] in wheel.files:
                parents.add(self._wheel_file(wheel, claim["source"]))
        self.dependencies[ref] = sorted(parents)
        return ref

    def _code_paths(self, entry: dict) -> tuple[list[str], str]:
        path = entry["path"]
        if entry["container"] == "pyz":
            relative = path.replace(".", "/") + ("/__init__.py" if entry["kind"] == 1 else ".py")
            return [relative, "src/" + relative], relative.replace("/", "\\")
        filename = path.removesuffix(".py") + ".py"
        candidates = [
            "PyInstaller/loader/" + filename,
            "PyInstaller/hooks/rthooks/" + filename,
            "_pyinstaller_hooks_contrib/rthooks/" + filename,
        ]
        if path == PurePosixPath(self.entry_point).stem:
            candidates.append(self.entry_point)
        return candidates, filename.replace("/", "\\")

    def _code_candidates(self, paths: list[str]):
        for path in paths:
            if path in self.source_files:
                yield self.source_files[path], lambda path=path: self._git_file(path)
            for wheel in self.paths.get(path, ()):
                yield wheel.read(path), lambda wheel=wheel, path=path: self._wheel_file(wheel, path)

    def _code(self, entry: dict, data: bytes) -> list[dict]:
        paths, filename = self._code_paths(entry)
        candidates = list(self._code_candidates(paths))
        matches = [
            (reference(), proof)
            for content, reference in candidates
            if (proof := _code_match(content, filename, data)) is not None
        ]
        if candidates and not matches:
            raise ValueError(f"executable bytecode differs from the verified source: {entry['path']}")
        if not candidates and any(path.startswith("src/xrr_fitter/") for path in paths):
            raise ValueError(f"executable application source is missing: {entry['path']}")
        if not matches:
            return []
        builder = self._package(self.builder)
        return [{**proof, "filename": filename, "inputs": sorted([ref, builder])} for ref, proof in matches]

    def match(self, entry: dict, data: bytes) -> list[dict]:
        if entry["container"] == "pyz" and entry["kind"] == 3:
            return []
        if entry["container"] == "pyz" or entry["kind"] in {"s", "m", "M"}:
            return self._code(entry, data)
        checksum = hashlib.sha256(data).hexdigest()
        known = self.known.get(entry["path"])
        if known is not None and (known["sha256"], known["size"]) != (checksum, len(data)):
            raise ValueError(f"executable bytes differ from the verified source path: {entry['path']}")
        return [
            {"transform": "installed-byte-copy", "inputs": [self._installed_file(file)]}
            for file in self.raw[(checksum, len(data))]
        ]
