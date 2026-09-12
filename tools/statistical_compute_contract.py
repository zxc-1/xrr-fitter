"""Normalize only the reviewed opt-in wrapper, never the shard computation body."""

from __future__ import annotations

import ast

PERMISSION_GUARD = """if allow_compute is not True:
    raise ValueError("statistical shard fitting requires explicit --compute permission")"""
PARSER_PERMISSION = 'parser.add_argument("--compute", action="store_true", help="explicitly allow new shard fits")'
PARSER_GUARD = """if not args.compute:
    parser.error("statistical shard fitting requires explicit --compute permission")"""
GUARDED_CALL = "run_shard(ROOT, args.report_dir, args.shard_index, allow_compute=args.compute)"
ORIGINAL_CALL = "run_shard(ROOT, args.report_dir, args.shard_index)"


def _statement(source: str) -> ast.stmt:
    return ast.parse(source).body[0]


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _normalize_runner(node: ast.FunctionDef) -> None:
    expected = ast.parse("def f(*, allow_compute: bool = False): pass").body[0].args
    if [_dump(item) for item in node.args.kwonlyargs] != [_dump(item) for item in expected.kwonlyargs]:
        return
    if [_dump(item) for item in node.args.kw_defaults] != [_dump(item) for item in expected.kw_defaults]:
        return
    if not node.body or _dump(node.body[0]) != _dump(_statement(PERMISSION_GUARD)):
        return
    node.args.kwonlyargs.clear()
    node.args.kw_defaults.clear()
    node.body.pop(0)


def _normalize_cli(node: ast.FunctionDef) -> None:
    known = {_dump(_statement(source)) for source in (PARSER_PERMISSION, PARSER_GUARD, GUARDED_CALL)}
    observed = [_dump(item) for item in node.body]
    if any(observed.count(item) != 1 for item in known):
        return
    call = _dump(_statement(GUARDED_CALL))
    node.body = [
        _statement(ORIGINAL_CALL) if _dump(item) == call else item
        for item in node.body
        if _dump(item) not in known or _dump(item) == call
    ]


def shard_semantics(content: bytes) -> bytes:
    tree = ast.parse(content)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "run_shard":
            _normalize_runner(node)
        elif node.name == "main":
            _normalize_cli(node)
    return _dump(tree).encode("utf-8")
