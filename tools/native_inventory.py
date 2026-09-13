"""Bind native declarations to wheel-local paths, not presumed runtime loads."""

from __future__ import annotations

import json
import posixpath
from urllib.parse import quote


def wheel_file_reference(wheel: dict, path: str) -> str:
    return f"urn:xrr:wheel:{wheel['sha256']}:{quote(path, safe='')}"


def _properties(values: dict) -> list[dict]:
    return [{"name": name, "value": str(value)} for name, value in sorted(values.items())]


def _architecture(image: dict) -> tuple:
    if image["format"] == "pe":
        return "pe", image["machine"]
    return "mach-o", image["cpu_type"], image["cpu_subtype"]


def _target_family(image: dict, target: str) -> str:
    expected = {"macos-arm64-py312": ("mach-o", 0x100000C), "windows-x64-py312": ("pe", 0x8664)}
    return "target" if _architecture(image)[:2] == expected[target] else "foreign"


def _native_files(wheels: list[dict], inventories: list[dict]) -> dict[str, dict]:
    by_name = {wheel["name"]: wheel for wheel in wheels}
    return {
        wheel_file_reference(by_name[inventory["name"]], file["path"]): file
        for inventory in inventories
        for file in inventory["files"]
        if file["kind"] == "native"
    }


def _image_records(files: dict[str, dict]) -> list[dict]:
    records = []
    for ref, file in sorted(files.items()):
        if "native" not in file:
            continue
        for index, image in enumerate(file["native"]["images"]):
            records.append(
                {
                    "ref": f"{ref}:native:{index}",
                    "file_ref": ref,
                    "path": file["path"],
                    "image": image,
                    "unparsed": bool(file["native"]["unparsed"]),
                }
            )
    return records


def _loadable(record: dict) -> bool:
    image = record["image"]
    if "member_path" in image or record["path"].split("/", 1)[0].endswith(".data"):
        return False
    return image["format"] == "pe" or image["file_type"] in {6, 7}


def _path_index(records: list[dict]) -> dict[tuple, list[str]]:
    result: dict[tuple, list[str]] = {}
    for record in records:
        if _loadable(record):
            result.setdefault((record["path"], _architecture(record["image"])), []).append(record["ref"])
    return result


def _safe_path(path: str) -> str | None:
    normalized = posixpath.normpath(path)
    if normalized == ".." or normalized.startswith(("../", "/")):
        return None
    return normalized


def _loader_path(value: str, source: str) -> str | None:
    prefix = "@loader_path"
    if value != prefix and not value.startswith(prefix + "/"):
        return None
    return _safe_path(posixpath.join(posixpath.dirname(source), value[len(prefix) :].lstrip("/")))


def _rpath_paths(record: dict, name: str) -> tuple[list[str], str | None]:
    paths = []
    rpaths = record["image"]["rpaths"]
    incomplete = not rpaths
    for rpath in rpaths:
        base = _loader_path(rpath, record["path"])
        path = None if base is None else _safe_path(posixpath.join(base, name[len("@rpath/") :]))
        if path is None:
            incomplete = True
        else:
            paths.append(path)
    return paths, "runtime-rpath-context" if incomplete else None


def _request_paths(record: dict, name: str) -> tuple[list[str], str | None]:
    image = record["image"]
    if image["format"] == "pe":
        return [], "windows-loader-context-not-inspected"
    if not _loadable(record) and ("member_path" in image or image["file_type"] == 1):
        return [], "static-archive-link-context"
    if record["path"].split("/", 1)[0].endswith(".data"):
        return [], "wheel-relocation-context"
    if name.startswith("@rpath/"):
        return _rpath_paths(record, name)
    path = _loader_path(name, record["path"])
    if path is not None:
        return [path], None
    return [], "external-or-system-path" if name.startswith("/") else "runtime-loader-context"


def _resolve_request(record: dict, request: dict, index: dict) -> dict:
    paths, reason = _request_paths(record, request["name"])
    candidates = sorted({ref for path in paths for ref in index.get((path, _architecture(record["image"])), [])})
    if record["unparsed"]:
        reason = "unsupported-loader-declarations"
    result = {**request, "candidates": candidates}
    if reason is None and len(candidates) == 1:
        return {**result, "state": "archive-path", "target": candidates[0]}
    if reason is None:
        reason = "ambiguous-archive-path" if candidates else "no-matching-archive-path"
    return {**result, "state": "unresolved", "reason": reason}


def _image_component(record: dict, requests: list[dict], target: str) -> dict:
    image = record["image"]
    return {
        "type": "file",
        "bom-ref": record["ref"],
        "name": record["path"],
        "hashes": [{"alg": "SHA-256", "content": image["sha256"]}],
        "properties": _properties(
            {
                "xrr:native:architecture": json.dumps(_architecture(image)),
                "xrr:native:target-family": _target_family(image, target),
                "xrr:native:declarations": json.dumps(image, sort_keys=True),
                "xrr:native:requests": json.dumps(requests, sort_keys=True),
                "xrr:native:resolution-scope": "declared-wheel-local-paths-not-runtime-loads",
            }
        ),
    }


def _file_evidence(file: dict) -> dict:
    if "native" not in file:
        return {
            "state": "not-inspected",
            "unparsed": [{"reason": "native loader declarations not inspected"}],
            "containers": {},
        }
    containers = {key: file["native"][key] for key in ("members", "slices") if key in file["native"]}
    return {"state": "inspected", "unparsed": file["native"]["unparsed"], "containers": containers}


def _file_components(files: dict[str, dict]) -> dict:
    return {
        ref: {
            "components": [],
            "properties": _properties(
                {
                    "xrr:native:inspection": _file_evidence(file)["state"],
                    "xrr:native:unparsed": json.dumps(_file_evidence(file)["unparsed"], sort_keys=True),
                    "xrr:native:container-evidence": json.dumps(_file_evidence(file)["containers"], sort_keys=True),
                }
            ),
        }
        for ref, file in sorted(files.items())
    }


def build_native_inventory(wheels: list[dict], inventories: list[dict], target: str) -> dict:
    files = _native_files(wheels, inventories)
    records = _image_records(files)
    index = _path_index(records)
    result = {
        "files": _file_components(files),
        "dependencies": [],
        "images": len(records),
        "unparsed": sum(len(_file_evidence(file)["unparsed"]) for file in files.values()),
        "resolved": 0,
        "unresolved": 0,
    }
    for record in records:
        requests = [_resolve_request(record, request, index) for request in record["image"]["imports"]]
        result["files"][record["file_ref"]]["components"].append(_image_component(record, requests, target))
        result["dependencies"].append(
            {
                "ref": record["ref"],
                "dependsOn": sorted({item["target"] for item in requests if item["state"] == "archive-path"}),
            }
        )
        resolved = sum(item["state"] == "archive-path" for item in requests)
        result["resolved"] += resolved
        result["unresolved"] += len(requests) - resolved
    return result
