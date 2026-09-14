from __future__ import annotations

import hashlib
import io
import marshal
import types
import zipfile

import pytest
from tests.support.executable_archives import carchive, pyz
from tests.support.installed_wheels import install_fixture, make_wheel, sample_installation

SOURCE = {"source_commit": "a" * 40, "source_tree": "b" * 40}
CODE = b"def main():\n    return 42\n"


def _serialized(source, filename):
    def renamed(code):
        constants = tuple(renamed(item) if isinstance(item, types.CodeType) else item for item in code.co_consts)
        return code.replace(co_consts=constants, co_filename=filename)

    compiled = compile(source, "original.py", "exec", dont_inherit=True)
    transformed = renamed(compiled)
    return marshal.dumps(transformed)


def _sources(tmp_path, load_tool_module):
    installed = load_tool_module("installed_inventory")
    module = load_tool_module("executable_sources")
    layout, wheels = sample_installation(
        tmp_path, entries={"sample/__init__.py": CODE, "sample/data.bin": b"copied data"}
    )
    builder = make_wheel(tmp_path / "wheels", name="pyinstaller", version="6.21.0")
    install_fixture(builder, layout)
    inputs = [installed.WheelInput(**value) for value in [*wheels, builder]]
    inventory = installed.inspect_installation(installed.InstallLayout(**layout), inputs, inputs[0])
    return module.ExecutableSources(
        inputs, inventory, {"src/xrr_fitter/sample.py": CODE}, SOURCE, entry_point="src/xrr_fitter/sample.py"
    )


def test_executable_matches_exact_installed_bytes_and_transformed_source(tmp_path, load_tool_module):
    module = load_tool_module("executable_inventory")
    data = carchive(
        [
            ("sample/data.bin", b"copied data", "x", False),
            (
                "PYZ.pyz",
                pyz(
                    [
                        ("sample", _serialized(CODE, r"sample\__init__.py"), 1),
                        ("xrr_fitter.sample", _serialized(CODE, r"xrr_fitter\sample.py"), 0),
                    ]
                ),
                "z",
                False,
            ),
        ]
    )
    with _sources(tmp_path, load_tool_module) as sources:
        result = module.inspect_executable(data, sources, native_loaders=False)
    assert result["aggregate"] == "incomplete"
    assert len(result["files"]) == 4
    claims = [claim for row in result["files"] for claim in row["claims"]]
    assert {claim["transform"] for claim in claims} == {"installed-byte-copy", "pyinstaller-code"}
    assert all(row["sha256"] for row in result["files"])
    assert result["executable"]["sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("name,kind", [("sample", 1), ("xrr_fitter.sample", 0)])
def test_executable_rejects_code_drift_from_known_sources(tmp_path, load_tool_module, name, kind):
    module = load_tool_module("executable_inventory")
    data = carchive([("PYZ.pyz", pyz([(name, _serialized(b"tampered=True", "different.py"), kind)]), "z", False)])
    with _sources(tmp_path, load_tool_module) as sources:
        with pytest.raises(ValueError, match="source|bytecode"):
            module.inspect_executable(data, sources, native_loaders=False)


def test_executable_rejects_a_known_data_path_with_different_bytes(tmp_path, load_tool_module):
    module = load_tool_module("executable_inventory")
    with _sources(tmp_path, load_tool_module) as sources:
        with pytest.raises(ValueError, match="source|bytes"):
            module.inspect_executable(
                carchive([("sample/data.bin", b"tampered", "x", False)]), sources, native_loaders=False
            )


def test_frozen_installed_record_uses_installed_bytes_not_the_archive_record(tmp_path, load_tool_module):
    module = load_tool_module("executable_inventory")
    sources = _sources(tmp_path, load_tool_module)
    data = (tmp_path / "venv/lib/python3.12/site-packages/sample-1.0.dist-info/RECORD").read_bytes()
    with sources:
        result = module.inspect_executable(
            carchive([("sample-1.0.dist-info/RECORD", data, "x", False)]), sources, native_loaders=False
        )
    assert result["files"][0]["claims"][0]["transform"] == "installed-byte-copy"


def test_unknown_os_bytes_and_bootloader_remain_explicitly_unresolved(tmp_path, load_tool_module):
    module = load_tool_module("executable_inventory")
    with _sources(tmp_path, load_tool_module) as sources:
        result = module.inspect_executable(
            carchive([("os-runtime.dll", b"unbound", "b", False)]), sources, native_loaders=False
        )
    assert result["files"][0]["claims"] == []
    assert result["files"][0]["unresolved"]
    assert result["framing"]["prefix"]["unresolved"]


def _base_zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in entries:
            archive.writestr(path, content)
    return buffer.getvalue()


def test_executable_inventories_base_library_zip_without_importing_it(tmp_path, load_tool_module):
    module = load_tool_module("executable_inventory")
    zipped = _base_zip([("stdlib.pyc", b"untrusted bytecode")])
    with _sources(tmp_path, load_tool_module) as sources:
        result = module.inspect_executable(
            carchive([("base_library.zip", zipped, "x", True)]), sources, native_loaders=False
        )
    nested = next(row for row in result["files"] if row["container"] == "zip")
    assert nested["path"] == "stdlib.pyc"
    assert nested["sha256"] == hashlib.sha256(b"untrusted bytecode").hexdigest()
    assert nested["claims"] == [] and nested["unresolved"]


def test_executable_sbom_refs_bind_only_observed_provenance_not_the_whole_dev_closure(tmp_path, load_tool_module):
    module = load_tool_module("executable_inventory")
    sbom = load_tool_module("executable_sbom")
    with _sources(tmp_path, load_tool_module) as sources:
        inventory = module.inspect_executable(
            carchive([("sample/data.bin", b"copied data", "x", True)]), sources, native_loaders=False
        )
    bom = sbom.assemble_executable_sbom(inventory, SOURCE, {"installed-inventory": "c" * 64})
    refs = {row["bom-ref"] for row in bom["components"]}
    assert len(refs) == len(bom["components"])
    assert all(row["ref"] in refs and set(row["dependsOn"]) <= refs for row in bom["dependencies"])
    assert not any(row["name"] == "pip" for row in bom["components"])
    assert bom["compositions"] == [{"aggregate": "incomplete"}]
