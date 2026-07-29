"""Command-line dispatch for R22 reference checks and comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence


SelfCheck = Callable[..., dict[str, object]]
CompareGroup = Callable[[Path, str], dict[str, object]]
CompareAll = Callable[[Path, Path], dict[str, object]]


def _reject(
    parser: argparse.ArgumentParser,
    message: str,
    *conditions: bool,
) -> None:
    if any(conditions):
        parser.error(message)


def _run_self_check(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    checker: SelfCheck,
) -> int:
    _reject(
        parser,
        "--self-check cannot be combined with comparison options",
        args.manifest is not None,
        args.group is not None,
        args.all_groups,
        args.report_dir is not None,
    )
    _reject(
        parser,
        "--collections-root and --release-spec are required together",
        (args.collections_root is None) != (args.release_spec is None),
    )
    result = checker(
        args.self_check,
        collections_root=args.collections_root,
        release_spec=args.release_spec,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_all_groups(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    compare: CompareAll,
) -> int:
    _reject(
        parser,
        "--all-groups requires --manifest and --report-dir",
        args.manifest is None,
        args.report_dir is None,
    )
    _reject(
        parser,
        "--all-groups cannot be combined with single-group or self-check options",
        args.group is not None,
        args.collections_root is not None,
        args.release_spec is not None,
    )
    print(json.dumps(compare(args.manifest, args.report_dir), sort_keys=True))
    return 0


def _run_group(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    compare: CompareGroup,
) -> int:
    _reject(
        parser,
        "--manifest and --group are required together",
        args.manifest is None,
        args.group is None,
    )
    _reject(
        parser,
        "single-group comparison does not accept aggregate options",
        args.report_dir is not None,
        args.collections_root is not None,
        args.release_spec is not None,
    )
    print(json.dumps(compare(args.manifest, args.group), sort_keys=True))
    return 0


def run_cli(
    argv: Sequence[str] | None,
    *,
    groups: Sequence[str],
    self_check: SelfCheck,
    compare_group: CompareGroup,
    compare_all: CompareAll,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--group", choices=groups)
    parser.add_argument("--all-groups", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--collections-root", type=Path)
    parser.add_argument("--release-spec", type=Path)
    args = parser.parse_args(argv)
    if args.self_check is not None:
        return _run_self_check(args, parser, self_check)
    if args.all_groups:
        return _run_all_groups(args, parser, compare_all)
    return _run_group(args, parser, compare_group)
