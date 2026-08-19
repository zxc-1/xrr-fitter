#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

MANAGED_ROOTS = ("examples", "src", "tests", "tools")
PRUNED_DIRS = {".git", ".pytest_cache", "__pycache__", "build", "dist"}
EXPECTED_RADON_VERSION = "6.0.1"


@dataclass(frozen=True)
class DiscoveryIssue:
    kind: str
    path: str
    detail: str


def _discovery_issues(root: Path, relative: Path) -> list[DiscoveryIssue]:
    if relative.parts[0] not in MANAGED_ROOTS:
        detail = "Python source is outside a managed root"
        return [DiscoveryIssue("ownership", relative.as_posix(), detail)]
    if _ignored(root, relative):
        detail = "managed Python source is ignored"
        return [DiscoveryIssue("ignore-policy", relative.as_posix(), detail)]
    return []


def _prunable(root: Path, relative: Path) -> bool:
    if relative.name in PRUNED_DIRS or relative.name.endswith(".egg-info"):
        return True
    # A directory git ignores outside the managed roots is a local artifact such as
    # a virtual environment, so it carries no source the gate governs.
    return relative.parts[0] not in MANAGED_ROOTS and _ignored(root, relative)


def _ignored(root: Path, relative: Path) -> bool:
    result = subprocess.run(
        ("git", "check-ignore", "-q", "--", relative.as_posix()),
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def discover_python_files(root: str | Path) -> tuple[tuple[Path, ...], tuple[DiscoveryIssue, ...]]:
    repository = Path(root).resolve()
    paths: list[Path] = []
    issues: list[DiscoveryIssue] = []
    for current, directories, names in os.walk(repository):
        base = Path(current).relative_to(repository)
        directories[:] = [name for name in directories if not _prunable(repository, base / name)]
        for name in names:
            if not name.endswith(".py") or not (Path(current) / name).is_file():
                continue
            relative = base / name
            paths.append(relative)
            issues.extend(_discovery_issues(repository, relative))
    ordered_paths = tuple(sorted(paths, key=lambda item: item.as_posix()))
    ordered_issues = tuple(sorted(issues, key=lambda item: (item.path, item.kind, item.detail)))
    return ordered_paths, ordered_issues


def _issue(kind: str, path: str, detail: str, **fields: object) -> dict[str, object]:
    return {"kind": kind, "path": path, "detail": detail, **fields}


def _symbol_records(item: dict[str, object], scores: list[float]) -> list[dict[str, object]]:
    supplied = item.get("symbols")
    if isinstance(supplied, list) and len(supplied) == len(scores):
        return [dict(symbol) for symbol in supplied]
    return [{"name": f"<block:{index}>", "line": 0, "complexity": score} for index, score in enumerate(scores, start=1)]


def _block_issues(path: str, symbols: list[dict[str, object]]) -> list[dict[str, object]]:
    from radon.complexity import cc_rank

    issues: list[dict[str, object]] = []
    for symbol in symbols:
        score = float(symbol["complexity"])
        rank = cc_rank(score)
        symbol.update(complexity=score, rank=rank)
        if score > 10:
            name = str(symbol.get("name", ""))
            line = int(symbol.get("line", 0))
            detail = f"{name} line {line}: CC {score:g} rank {rank} exceeds 10"
            issues.append(_issue("block-cc", path, detail, symbol=name, line=line, score=score, rank=rank))
    return issues


def _file_metrics(item: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]], list[float]]:
    from radon.complexity import cc_rank
    from radon.metrics import mi_rank

    path = str(item["path"])
    if "analysis_error" in item:
        detail = str(item["analysis_error"])
        return dict(item), [_issue("analysis", path, detail)], []
    scores = [float(value) for value in item.get("blocks", [])]
    symbols = _symbol_records(item, scores)
    issues = _block_issues(path, symbols)
    average = sum(scores) / len(scores) if scores else 0.0
    cc_label = cc_rank(average)
    if average > 5.0:
        detail = f"file average CC {average:g} rank {cc_label} exceeds 5.0"
        issues.append(_issue("file-average", path, detail, score=average, rank=cc_label))
    mi_value = float(item["mi"])
    mi_label = mi_rank(mi_value)
    if mi_label != "A":
        detail = f"MI {mi_value:g} rank {mi_label} is not A"
        issues.append(_issue("mi-rank", path, detail, score=mi_value, rank=mi_label))
    normalized = {
        **item,
        "blocks": scores,
        "symbols": symbols,
        "cc_average": average,
        "cc_rank": cc_label,
        "mi": mi_value,
        "mi_rank": mi_label,
    }
    return normalized, issues, scores


def evaluate_metrics(files: Sequence[dict[str, object]], *, radon_version: str) -> dict[str, object]:
    from radon.complexity import cc_rank

    issues: list[dict[str, object]] = []
    normalized: list[dict[str, object]] = []
    all_scores: list[float] = []
    if radon_version != EXPECTED_RADON_VERSION:
        detail = f"Radon {EXPECTED_RADON_VERSION} is required; found {radon_version}"
        issues.append(_issue("radon-version", ".", detail))
    for item in sorted(files, key=lambda row: str(row["path"])):
        metrics, file_issues, scores = _file_metrics(item)
        normalized.append(metrics)
        issues.extend(file_issues)
        all_scores.extend(scores)
    average = sum(all_scores) / len(all_scores) if all_scores else 0.0
    rank = cc_rank(average)
    if average > 5.0:
        detail = f"repository average CC {average:g} rank {rank} exceeds 5.0"
        issues.append(_issue("repository-average", ".", detail, score=average, rank=rank))
    issues.sort(key=lambda item: (str(item["path"]), str(item["kind"]), str(item["detail"])))
    return {
        "schema": "xrr-r23-radon-report-v1",
        "radon_version": radon_version,
        "status": "PASS" if not issues else "FAIL",
        "repository_cc_average": average,
        "repository_cc_rank": rank,
        "files": normalized,
        "issues": issues,
    }


def _block_key(block: object) -> tuple[object, ...]:
    return (
        type(block).__name__,
        getattr(block, "name", ""),
        getattr(block, "lineno", 0),
        getattr(block, "col_offset", 0),
        getattr(block, "endline", 0),
    )


def _children(block: object) -> Iterable[object]:
    for attribute in ("methods", "closures", "inner_classes"):
        yield from getattr(block, attribute, ()) or ()


def _flatten_blocks(blocks: Iterable[object]) -> list[object]:
    observed: list[object] = []
    seen: set[tuple[object, ...]] = set()
    pending = list(blocks)
    while pending:
        block = pending.pop(0)
        key = _block_key(block)
        if key not in seen:
            seen.add(key)
            observed.append(block)
        pending.extend(_children(block))
    return observed


def _halstead(value: object) -> dict[str, object]:
    total = dict(value.total._asdict())
    functions = [{"name": name, **dict(report._asdict())} for name, report in value.functions]
    return {"total": total, "functions": functions}


def _source_metrics(path: Path, relative: Path) -> dict[str, object]:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
    from radon.raw import analyze

    source = path.read_text(encoding="utf-8")
    blocks = _flatten_blocks(cc_visit(source))
    raw = analyze(source)
    symbols = [
        {
            "name": str(getattr(block, "name", "")),
            "line": int(getattr(block, "lineno", 0)),
            "complexity": float(block.complexity),
        }
        for block in blocks
    ]
    return {
        "path": relative.as_posix(),
        "mi": float(mi_visit(source, multi=True)),
        "blocks": [float(block.complexity) for block in blocks],
        "symbols": symbols,
        "raw": dict(raw._asdict()),
        "halstead": _halstead(h_visit(source)),
    }


def _analyze_files(repository: Path, paths: Sequence[Path]) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    for relative in paths:
        try:
            metrics.append(_source_metrics(repository / relative, relative))
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            metrics.append({"path": relative.as_posix(), "analysis_error": detail})
    return metrics


def _git_python_paths(repository: Path) -> tuple[set[str] | None, dict[str, object] | None]:
    result = subprocess.run(
        ("git", "ls-files", "-co", "--exclude-standard", "--", "*.py"),
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        return None, _issue("git-set", ".", f"git ls-files failed: {detail}")
    try:
        lines = result.stdout.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        return None, _issue("git-set", ".", f"git ls-files decode failure: {error}")
    return set(lines), None


def _git_set_issues(repository: Path, paths: Sequence[Path]) -> list[dict[str, object]]:
    git_paths, command_issue = _git_python_paths(repository)
    if command_issue is not None or git_paths is None:
        return [command_issue] if command_issue is not None else []
    filesystem_paths = {path.as_posix() for path in paths}
    issues = [
        _issue("git-set", path, "Git lists Python source missing from filesystem discovery")
        for path in sorted(git_paths - filesystem_paths)
    ]
    for path in sorted(filesystem_paths - git_paths):
        relative = Path(path)
        if not _ignored(repository, relative):
            detail = "filesystem Python source is missing from Git tracked/untracked view"
            issues.append(_issue("git-set", path, detail))
    return issues


def build_report(root: str | Path) -> dict[str, object]:
    import radon

    repository = Path(root).resolve()
    paths, discovery_issues = discover_python_files(repository)
    report = evaluate_metrics(_analyze_files(repository, paths), radon_version=radon.__version__)
    issues = [asdict(issue) for issue in discovery_issues]
    issues.extend(_git_set_issues(repository, paths))
    issues.extend(report["issues"])
    if not paths:
        issues.append(_issue("empty-discovery", ".", "no Python source discovered"))
    issues.sort(key=lambda item: (str(item["path"]), str(item["kind"]), str(item["detail"])))
    return {**report, "status": "PASS" if not issues else "FAIL", "issues": issues}


def _canonical(value: object) -> bytes:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (serialized + "\n").encode()


def _print_issues(report: dict[str, object]) -> None:
    for issue in report["issues"]:
        print(f"{issue['kind']}: {issue['path']}: {issue['detail']}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.repo_root)
    if args.output:
        args.output.write_bytes(_canonical(report))
    _print_issues(report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
