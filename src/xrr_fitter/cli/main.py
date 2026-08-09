"""Console entry point for unattended XRR workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from multiprocessing import freeze_support
import sys

from xrr_fitter.cli import commands, exit_codes


HANDLERS = {
    "fit": commands.run_fit,
    "mcmc": commands.run_mcmc,
    "export": commands.run_export,
    "validate": commands.run_validate,
}


def _add_progress_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json-progress",
        action="store_true",
        help="把进度写成 stdout 的 JSON Lines，而不是 stderr 的文本行",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the complete subcommand parser."""
    parser = argparse.ArgumentParser(
        prog="xrr-fitter-cli",
        description="X 射线反射率拟合的无头命令行入口",
    )
    subparsers = parser.add_subparsers(dest="command")

    fit = subparsers.add_parser("fit", help="运行拟合流水线")
    fit.add_argument("project")
    fit.add_argument("--auto", action="store_true", help="走自动批次拟合路径")
    fit.add_argument("--output", help="把更新后的工程写到该路径")
    _add_progress_flag(fit)

    mcmc = subparsers.add_parser("mcmc", help="对选定候选运行 MCMC 采样")
    mcmc.add_argument("project")
    mcmc.add_argument("--dataset", required=True)
    mcmc.add_argument("--candidate", required=True)
    mcmc.add_argument("--walkers", type=int, required=True)
    mcmc.add_argument("--burn-in", type=int, required=True, dest="burn_in")
    mcmc.add_argument("--steps", type=int, required=True)
    mcmc.add_argument("--output")
    _add_progress_flag(mcmc)

    export = subparsers.add_parser("export", help="发布已有结果")
    export.add_argument("project")
    export.add_argument("output_dir")

    validate = subparsers.add_parser("validate", help="只读校验工程与源文件")
    validate.add_argument("project")
    return parser


def _dispatch(arguments: argparse.Namespace) -> int:
    return HANDLERS[arguments.command](arguments)


def main(argv: Sequence[str] | None = None) -> int:
    """Enable frozen workers, parse arguments, and run one subcommand."""
    freeze_support()
    parser = build_parser()
    arguments = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if arguments.command is None:
        parser.print_usage(sys.stderr)
        return exit_codes.INVALID_INPUT
    try:
        return _dispatch(arguments)
    except commands.CommandError as error:
        print(str(error), file=sys.stderr)
        return error.code


if __name__ == "__main__":
    raise SystemExit(main())
