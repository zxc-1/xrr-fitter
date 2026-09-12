#!/usr/bin/env python3
"""Choose explicit replay or a manually authorized first-attempt computation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.dont_write_bytecode = True

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from installed_inputs import json_object  # noqa: E402
from statistical_github import validate_selection  # noqa: E402


def select_execution(*, compute: bool, producer_json: str, event: str, attempt: str) -> str:
    if compute and producer_json:
        raise ValueError("statistical compute and producer replay are mutually exclusive")
    if compute is True:
        if event != "workflow_dispatch" or attempt != "1":
            raise ValueError("NOT_READY: only explicit first-attempt workflow_dispatch can authorize new fits")
        return "compute"
    if not producer_json:
        raise ValueError("NOT_READY: provide explicit producer evidence; statistical fitting is disabled")
    validate_selection(json_object(producer_json.encode("utf-8")))
    return "replay"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute", choices=("true", "false"), default="false")
    parser.add_argument("--producer-json", default="")
    args = parser.parse_args(argv)
    try:
        mode = select_execution(
            compute=args.compute == "true",
            producer_json=args.producer_json,
            event=os.environ.get("GITHUB_EVENT_NAME", ""),
            attempt=os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        )
    except (ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    print(f"mode={mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
