"""Retain bounded byte evidence for failed installation checks, never a fallback."""

from __future__ import annotations

import base64
import hashlib
import os
import sys
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from frozen_code import code_fields_digest  # noqa: E402

CAPTURE_LIMIT = 64 * 1024


class InstalledMismatch(ValueError):
    def __init__(self, message: str, evidence: dict) -> None:
        super().__init__(message)
        self.evidence = evidence


def _byte_evidence(content: bytes, transform: str) -> dict:
    result = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        "prefix_hex": content[:64].hex(),
    }
    if len(content) <= CAPTURE_LIMIT:
        result["bytes_base64"] = base64.b64encode(content).decode("ascii")
    if transform == "cpython-bytecode":
        result["header_hex"] = content[:16].hex()
        try:
            result["code_fields_sha256"] = code_fields_digest(content[16:])
        except ValueError as error:
            result["code_fields_error"] = str(error)
    return result


def mismatch_evidence(snapshot, path, digest, item, source, transform, expected) -> dict:
    result = {
        "package": item.record["name"],
        "wheel_sha256": item.record["sha256"],
        "path": os.path.relpath(path, snapshot.root).replace(os.sep, "/"),
        "source": source,
        "transform": transform,
        "expected": {"sha256": digest} if expected is None else _byte_evidence(expected, transform),
    }
    try:
        result["actual"] = _byte_evidence(snapshot.read(path), transform)
    except (OSError, ValueError) as error:
        result["actual"] = {"read_error": f"{type(error).__name__}: {error}"}
    return result
