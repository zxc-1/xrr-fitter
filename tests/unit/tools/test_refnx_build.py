from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/refnx_build.py").is_file(), "missing isolated offline refnx build"
    return load_tool_module("refnx_build")


def _source(module, tmp_path, entries):
    source = json.loads((ROOT / "tools/package-manifests/refnx-source.json").read_text())
    prefix = f"refnx-{source['vcs']['commit']}"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content, mode in entries:
            item = tarfile.TarInfo(f"{prefix}/{name}")
            item.size, item.mode = len(content), mode
            archive.addfile(item, io.BytesIO(content))
    content = stream.getvalue()
    files = module.archive_files(content, prefix)
    source.update(file_count=len(files), files_sha256=hashlib.sha256(module.manifest_bytes(files)).hexdigest())
    source["archive"].update(sha256=hashlib.sha256(content).hexdigest(), size=len(content))
    path = tmp_path / "source.tar.gz"
    path.write_bytes(content)
    return source, path


def test_build_command_cannot_fetch_or_resolve_implicit_dependencies(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    python, source, wheels = (tmp_path / name for name in ("venv/bin/python", "source", "wheels"))
    assert module.build_command(python, source, wheels) == (
        str(python),
        "-m",
        "pip",
        "--isolated",
        "--verbose",
        "wheel",
        "--no-index",
        "--no-deps",
        "--no-build-isolation",
        "--no-cache-dir",
        "--wheel-dir",
        str(wheels),
        str(source),
    )


def test_builder_install_uses_pinned_parent_pip_with_offline_hashes(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    python, requirements = tmp_path / "venv/bin/python", tmp_path / "offline.requirements"
    assert module.builder_install_command(python, requirements) == (
        sys.executable,
        "-m",
        "pip",
        "--isolated",
        "--python",
        str(python),
        "install",
        "--no-index",
        "--no-deps",
        "--no-cache-dir",
        "--require-hashes",
        "--only-binary=:all:",
        "--requirement",
        str(requirements),
    )


def test_build_environment_does_not_inherit_unlocked_compiler_or_cache_flags(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    inherited = {
        "CFLAGS": "-ffast-math",
        "PIP_INDEX_URL": "https://invalid/",
        "PYTHONPATH": "/untrusted",
        "PATH": "/custom/bin",
    }
    toolchain = {"cc": "/usr/bin/clang", "cxx": "/usr/bin/clang++", "sdk_path": "/SDK/MacOSX.sdk"}
    environment = module.build_environment(inherited, tmp_path, toolchain)
    assert not {"CFLAGS", "PIP_INDEX_URL", "PYTHONPATH"} & environment.keys()
    assert environment["PATH"] == f"{tmp_path}/venv/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    assert environment["CC"] == toolchain["cc"]
    assert environment["CXX"] == toolchain["cxx"]
    assert environment["SDKROOT"] == toolchain["sdk_path"]
    assert environment["HOME"] == str(tmp_path / "home")
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_source_extraction_verifies_bytes_and_preserves_executable_bit(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    source, archive = _source(
        module, tmp_path, [("tools/version.py", b"print('1.0')\n", 0o755), ("code.py", b"x = 1\n", 0o644)]
    )
    target = tmp_path / "extracted"
    module.extract_source(ROOT, source, archive, target)
    assert (target / "code.py").read_bytes() == b"x = 1\n"
    assert (target / "tools/version.py").stat().st_mode & 0o111
    assert not (target / ".git").exists()
    assert archive.is_file()


@pytest.mark.parametrize("failure", ["bytes", "git", "existing"])
def test_source_extraction_fails_without_overwriting_existing_data(load_tool_module, tmp_path, failure):
    module = _module(load_tool_module)
    member = ".git/config" if failure == "git" else "code.py"
    source, archive = _source(module, tmp_path, [(member, b"keep\n", 0o644)])
    target = tmp_path / "extracted"
    if failure == "bytes":
        archive.write_bytes(b"modified")
    if failure == "existing":
        target.mkdir()
        (target / "sentinel").write_bytes(b"keep")
    with pytest.raises(ValueError):
        module.extract_source(ROOT, source, archive, target)
    if failure == "existing":
        assert (target / "sentinel").read_bytes() == b"keep"
    else:
        assert not target.exists()


def _wheel(directory, *, version="0.1.65.dev0", native=True, tag="cp311-abi3-macosx_11_0_arm64", extra=""):
    directory.mkdir()
    path = directory / f"refnx-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"refnx-{version}.dist-info/METADATA",
            f"Name: refnx\nVersion: {version}\nRequires-Dist: numpy>=2.0\nLicense-Expression: BSD-3-Clause\n{extra}",
        )
        archive.writestr("refnx/__init__.py", "")
        archive.writestr("refnx/reflect/_creflect.pxd", "# declarations, not the native binary\n")
        if native:
            for package, name in (("_lib", "_cutil"), ("reflect", "_creflect"), ("reduce", "_cevent")):
                archive.writestr(f"refnx/{package}/{name}.abi3.so", b"\xcf\xfa\xed\xfe" + b"\x00" * 28)
    return path


def test_derived_wheel_binds_actual_bytes_and_required_native_extensions(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    wheel = _wheel(tmp_path / "wheels")
    record, inventory = module.wheel_result(wheel.parent, {"version": "0.1.65.dev0", "dependencies": ["numpy>=2.0"]})
    assert record["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert record["filename"] == wheel.name
    assert record["size"] == wheel.stat().st_size
    assert len([item for item in inventory["files"] if item["kind"] == "native"]) == 3


def test_derived_wheel_uses_the_build_host_not_fixed_download_target(load_tool_module, tmp_path, monkeypatch):
    from packaging.tags import Tag

    module = _module(load_tool_module)
    monkeypatch.setattr(module, "sys_tags", lambda: iter((Tag("cp312", "abi3", "macosx_26_0_arm64"),)), raising=False)
    wheel = _wheel(tmp_path / "wheels", tag="cp312-abi3-macosx_26_0_arm64")
    record, _inventory = module.wheel_result(wheel.parent, {"version": "0.1.65.dev0", "dependencies": ["numpy>=2.0"]})
    assert record["filename"] == wheel.name


def test_derived_wheel_accounts_for_declared_source_extras(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    wheel = _wheel(tmp_path / "wheels", extra='Requires-Dist: pytest; extra == "test"\n')
    project = {"version": "0.1.65.dev0", "dependencies": ["numpy>=2.0"], "optional-dependencies": {"test": ["pytest"]}}
    _record, inventory = module.wheel_result(wheel.parent, project)
    assert 'pytest; extra == "test"' in inventory["requirements"]


@pytest.mark.parametrize("failure", ["version", "native", "dependencies", "extra"])
def test_derived_wheel_rejects_incomplete_or_unexpected_output(load_tool_module, tmp_path, failure):
    module = _module(load_tool_module)
    path = _wheel(
        tmp_path / "wheels", version="9.0" if failure == "version" else "0.1.65.dev0", native=failure != "native"
    )
    project = {"version": "0.1.65.dev0", "dependencies": ["numpy>=2.0"]}
    if failure == "dependencies":
        project["dependencies"].append("missing>=1")
    if failure == "extra":
        (path.parent / "unverified.txt").write_bytes(b"extra")
    with pytest.raises(ValueError):
        module.wheel_result(path.parent, project)


def test_invalid_build_inputs_fail_before_creating_environment(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    report = tmp_path / "report"
    with pytest.raises(ValueError, match="exactly"):
        module.build_refnx(ROOT, inputs, report)
    assert not report.exists()


def test_toolchain_preserves_cpp_driver_symlink_name(load_tool_module, tmp_path, monkeypatch):
    _module(load_tool_module)
    runtime = load_tool_module("refnx_build_runtime")
    compiler = tmp_path / "clang"
    compiler.write_bytes(b"compiler")
    cpp = tmp_path / "clang++"
    cpp.symlink_to(compiler)
    sdk = tmp_path / "SDK"
    sdk.mkdir()
    (sdk / "SDKSettings.json").write_bytes(b"{}\n")
    observations = {"cc-path": str(compiler), "cxx-path": str(cpp), "sdk-path": str(sdk)}
    monkeypatch.setattr(runtime, "run_checked", lambda command, **kwargs: observations.get(kwargs["label"], "observed"))
    result = runtime.toolchain_evidence(ROOT, tmp_path, {})
    assert result["cxx"] == str(cpp), "resolving clang++ to clang changes the driver's link semantics"
    assert result["cxx_file"]["sha256"] == hashlib.sha256(compiler.read_bytes()).hexdigest()


def test_compiler_failure_retains_logs_but_never_installs_or_publishes_success(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    calls = []
    monkeypatch.setattr(module, "verify_inputs", lambda *args: [])

    def fail(root, inputs, report, stage, manifest, environment):
        (report / "build.stderr").write_text("compiler failed\n")
        (stage / "temporary").write_text("job owned\n")
        raise module.subprocess.CalledProcessError(23, ["compiler"])

    monkeypatch.setattr(module, "_build_stage", fail)
    monkeypatch.setattr(module, "_install", lambda *args: calls.append(args))
    report = tmp_path / "report"
    assert module.build_refnx(ROOT, tmp_path / "inputs", report, install=True) == 23
    assert not calls
    assert not (report / "build.json").exists()
    assert (report / "build.stderr").read_text() == "compiler failed\n"
    assert not list(report.glob("refnx-build-stage-*"))
    assert json.loads((report / "summary.json").read_text())["state"] == "FAIL"


def test_native_smoke_checks_all_bytes_without_importing_optional_reduce_dependencies(load_tool_module):
    module = _module(load_tool_module)
    assert hasattr(module, "native_smoke_command"), "native smoke must separate byte inventory from optional imports"
    files = [
        {"path": f"refnx/{package}/{name}.abi3.so", "sha256": "a" * 64, "kind": "native"}
        for package, name in (("_lib", "_cutil"), ("reflect", "_creflect"), ("reduce", "_cevent"))
    ]
    command = module.native_smoke_command({"wheel": {"version": "0.1.65.dev0"}, "inventory": {"files": files}})
    assert set(json.loads(command[-1])) == {item["path"] for item in files}
    assert "refnx._lib._cutil" in command[2]
    assert "refnx.reflect._creflect" in command[2]
    assert "refnx.reduce._cevent" not in command[2]


def test_installed_native_hash_is_not_overwritten_by_same_name_cython_declarations(
    load_tool_module, tmp_path, monkeypatch
):
    module = _module(load_tool_module)
    wheel = _wheel(tmp_path / "wheels")
    record, inventory = module.wheel_result(wheel.parent, {"version": "0.1.65.dev0", "dependencies": ["numpy>=2.0"]})
    imports = {
        item["path"].split(".")[0].replace("/", "."): item["sha256"]
        for item in inventory["files"]
        if item["kind"] == "native" and "/reduce/" not in item["path"]
    }
    observed = {
        "imports": imports,
        "files": {item["path"]: item["sha256"] for item in inventory["files"] if item["kind"] == "native"},
    }
    monkeypatch.setattr(
        module,
        "run_checked",
        lambda command, **kwargs: json.dumps(observed) if kwargs["label"] == "native-import" else "",
    )
    result = {"wheel": record, "inventory": inventory}
    module._install(ROOT, tmp_path, result, {})
    assert result["installed_native"] == observed
