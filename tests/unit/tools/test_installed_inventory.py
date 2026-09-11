from __future__ import annotations

import csv
import io
import json

import pytest
from tests.support.installed_wheels import (
    digest,
    install_fixture,
    make_wheel,
    record_bytes,
    rewrite_installed_record,
    sample_installation,
)


def inspect(module, layout, wheels):
    inputs = [module.WheelInput(**wheel) for wheel in wheels]
    return module.inspect_installation(module.InstallLayout(**layout), inputs, inputs[0])


@pytest.mark.parametrize("hash_pyc", [False, True])
def test_verified_installation_replays_every_byte_without_importing(tmp_path, load_tool_module, hash_pyc):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(tmp_path, hash_pyc=hash_pyc)
    result = inspect(module, layout, wheels)
    files = {item["path"]: item for item in result["files"]}
    assert result["aggregate"] == "incomplete"
    assert result["unowned"] == []
    assert len(result["packages"]) == 2
    assert any(claim["transform"] == "cpython-bytecode" for item in files.values() for claim in item["claims"])
    for relative, record in files.items():
        data = (layout["root"] / relative).read_bytes()
        assert (record["sha256"], record["size"]) == (digest(data), len(data))


@pytest.mark.parametrize(
    "relative",
    [
        "sample/__init__.py",
        "sample-1.0.dist-info/INSTALLER",
        "sample-1.0.dist-info/REQUESTED",
        "sample-1.0.dist-info/direct_url.json",
    ],
)
def test_rewritten_record_cannot_authorize_modified_installed_bytes(tmp_path, load_tool_module, relative):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(tmp_path)
    path = layout["library"] / relative
    path.write_bytes(path.read_bytes() + b"changed")
    rewrite_installed_record(layout["library"] / "sample-1.0.dist-info/RECORD", layout["library"])
    with pytest.raises(ValueError, match="bytes|metadata"):
        inspect(module, layout, wheels)


def test_bytecode_is_recompiled_not_trusted_from_its_unhashed_record(tmp_path, load_tool_module):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(tmp_path)
    path = next((layout["library"] / "sample").rglob("*.pyc"))
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="bytes"):
        inspect(module, layout, wheels)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "escape"])
def test_installation_rejects_incomplete_or_forged_record_paths(tmp_path, load_tool_module, change):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(tmp_path)
    path = layout["library"] / "sample-1.0.dist-info/RECORD"
    rows = list(csv.reader(io.StringIO(path.read_text())))
    if change == "missing":
        rows = rows[1:]
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows.append(("../../../../outside" if change == "escape" else "undeclared.py", "", ""))
    path.write_bytes(record_bytes(rows))
    with pytest.raises(ValueError, match="RECORD|bytes"):
        inspect(module, layout, wheels)


def test_unowned_site_file_fails_instead_of_becoming_an_ignored_exception(tmp_path, load_tool_module):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(tmp_path)
    (layout["library"] / "extra.py").write_text("unauthorized = True\n")
    with pytest.raises(ValueError, match="unowned"):
        inspect(module, layout, wheels)


@pytest.mark.parametrize("member", ["sample/__init__.py", "sample"])
def test_installed_symlink_never_reads_outside_the_prefix(tmp_path, load_tool_module, member):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(tmp_path)
    path = layout["library"] / member
    backup = path.with_name(path.name + ".original")
    path.rename(backup)
    path.symlink_to(backup, target_is_directory=backup.is_dir())
    with pytest.raises(ValueError, match="symlink|link"):
        inspect(module, layout, wheels)


def test_shared_identical_wheel_paths_keep_every_claim(tmp_path, load_tool_module):
    module = load_tool_module("installed_inventory")
    contents = {"shared/data.bin": b"same bytes"}
    layout, wheels = sample_installation(tmp_path, entries=contents)
    second = make_wheel(tmp_path / "wheels", name="second", entries=contents)
    install_fixture(second, layout)
    result = inspect(module, layout, [*wheels, second])
    shared = next(item for item in result["files"] if item["path"].endswith("shared/data.bin"))
    assert {claim["package"] for claim in shared["claims"]} == {"sample", "second"}


def test_conflicting_shared_claims_fail_even_when_one_record_matches(tmp_path, load_tool_module):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(tmp_path, entries={"shared/data.bin": b"first"})
    second = make_wheel(tmp_path / "wheels", name="second", entries={"shared/data.bin": b"second"})
    install_fixture(second, layout)
    with pytest.raises(ValueError, match="bytes|conflict"):
        inspect(module, layout, [*wheels, second])


def test_verified_unrendered_template_has_explicit_compile_evidence(tmp_path, load_tool_module):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(tmp_path, entries={"sample/recipe.py": b"value = {{ invalid template }}\n"})
    # A deterministic syntax error, rather than a filename exception.
    with pytest.raises(SyntaxError):
        compile(b"value = {{ invalid template }}\n", "recipe.py", "exec")
    result = inspect(module, layout, wheels)
    assert len(result["compile_failures"]) == 1
    assert result["compile_failures"][0]["source"] == "sample/recipe.py"


def test_data_relocation_and_shebang_are_derived_from_wheel_bytes(tmp_path, load_tool_module):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(
        tmp_path,
        entries={
            "sample-1.0.data/scripts/example.py": b"#!python\nprint('not executed')\n",
            "sample-1.0.data/data/share/example.txt": b"relocated bytes",
        },
    )
    result = inspect(module, layout, wheels)
    script = next(item for item in result["files"] if item["path"] == "bin/example.py")
    assert script["claims"][0]["transform"] == "wheel-script-shebang"
    assert (layout["root"] / "share/example.txt").read_bytes() == b"relocated bytes"


def test_entry_point_body_is_verified_even_after_record_rehash(tmp_path, load_tool_module):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(
        tmp_path,
        entries={
            "sample-1.0.dist-info/entry_points.txt": b"[console_scripts]\nhello = sample:main\n",
        },
    )
    install_fixture(wheels[1], layout, entry_points=[("hello", "sample", "main")])
    assert inspect(module, layout, wheels)["aggregate"] == "incomplete"
    script = layout["scripts"] / "hello"
    script.write_bytes(script.read_bytes().replace(b"sys.exit(main())", b"sys.exit(0)"))
    rewrite_installed_record(layout["library"] / "sample-1.0.dist-info/RECORD", layout["library"])
    with pytest.raises(ValueError, match="bytes"):
        inspect(module, layout, wheels)


def test_direct_url_cannot_point_at_another_equal_named_archive(tmp_path, load_tool_module):
    module = load_tool_module("installed_inventory")
    layout, wheels = sample_installation(tmp_path)
    path = layout["library"] / "sample-1.0.dist-info/direct_url.json"
    data = json.loads(path.read_text())
    data["url"] = "file:///unverified/sample-1.0-py3-none-any.whl"
    path.write_text(json.dumps(data, sort_keys=True))
    rewrite_installed_record(path.with_name("RECORD"), layout["library"])
    with pytest.raises(ValueError, match="bytes|metadata"):
        inspect(module, layout, wheels)
