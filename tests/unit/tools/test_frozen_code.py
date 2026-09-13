from __future__ import annotations

import marshal

import pytest


def test_code_fields_ignore_only_valid_marshal_interning_and_reference_layout(load_tool_module, monkeypatch):
    module = load_tool_module("frozen_code")
    code = compile("def f():\n    return ('text', 2**80, -0.0, 2j)\n", "module.py", "exec", dont_inherit=True)
    legacy, current = marshal.dumps(code, 3), marshal.dumps(code, 4)
    assert legacy != current
    monkeypatch.setattr(marshal, "loads", lambda *args: pytest.fail("do not deserialize untrusted code objects"))
    assert module.code_fields_digest(legacy) == module.code_fields_digest(current)


@pytest.mark.parametrize(
    "source,filename", [("answer = 43", "module.py"), ("answer = 42", "other.py"), ("answer = -0.0", "module.py")]
)
def test_code_field_changes_are_not_normalized_away(load_tool_module, source, filename):
    module = load_tool_module("frozen_code")
    expected = marshal.dumps(compile("answer = 42", "module.py", "exec", dont_inherit=True))
    changed = marshal.dumps(compile(source, filename, "exec", dont_inherit=True))
    assert module.code_fields_digest(expected) != module.code_fields_digest(changed)


@pytest.mark.parametrize("content", [b"c", b"r\x00\x00\x00\x00", b"[\xff\xff\xff\x7f", b"N", b"?"])
def test_code_parser_rejects_bad_references_types_and_lengths(load_tool_module, content):
    module = load_tool_module("frozen_code")
    with pytest.raises(ValueError):
        module.code_fields_digest(content)


def test_code_parser_requires_the_entire_wire_stream_and_preserves_constant_types(load_tool_module):
    module = load_tool_module("frozen_code")
    values = [
        marshal.dumps(compile(f"value = {value}", "module.py", "exec", dont_inherit=True))
        for value in ["1", "True", "'1'", "b'1'", "0.0", "-0.0"]
    ]
    assert len({module.code_fields_digest(value) for value in values}) == len(values)
    with pytest.raises(ValueError, match="trailing"):
        module.code_fields_digest(values[0] + b"extra")


def test_code_parser_handles_ellipsis_defaults_without_ignoring_them(load_tool_module):
    module = load_tool_module("frozen_code")
    values = [
        marshal.dumps(compile(f"def f(value={default}):\n    return value\n", "module.py", "exec", dont_inherit=True))
        for default in ("...", "None")
    ]
    assert module.code_fields_digest(values[0]) != module.code_fields_digest(values[1])
