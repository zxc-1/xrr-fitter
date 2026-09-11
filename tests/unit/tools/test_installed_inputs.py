from __future__ import annotations

import json

import pytest
from tests.support.installed_wheels import make_wheel


def test_hash_lock_requires_unique_exact_pins(tmp_path, load_tool_module):
    module = load_tool_module("installed_inputs")
    path = tmp_path / "inputs.lock"
    path.write_text(f"sample==1.0 --hash=sha256:{'a' * 64}\n")
    assert module.read_hash_lock(path) == {"sample": {"version": "1.0", "sha256": "a" * 64}}


@pytest.mark.parametrize(
    "line",
    [
        "sample>=1",
        "sample==1.0",
        "sample @ https://example.invalid/a.whl",
        f"sample==1.0 --hash=sha256:{'a' * 64}\nsample==1.0 --hash=sha256:{'a' * 64}",
    ],
)
def test_hash_lock_rejects_unbound_or_duplicate_inputs(tmp_path, load_tool_module, line):
    module = load_tool_module("installed_inputs")
    path = tmp_path / "inputs.lock"
    path.write_text(line + "\n")
    with pytest.raises(ValueError, match="lock|pin"):
        module.read_hash_lock(path)


def test_auxiliary_wheels_bind_the_exact_separate_lock_without_expanding_ordinary_manifest(tmp_path, load_tool_module):
    module = load_tool_module("installed_inputs")
    wheel = make_wheel(tmp_path / "wheels")
    pins = {"sample": {key: wheel["record"][key] for key in ("version", "sha256")}}
    result = module.auxiliary_wheels(tmp_path / "wheels", pins, "windows-x64-py312")
    assert result["sample"]["record"] == wheel["record"]
    (tmp_path / "wheels/extra.txt").write_text("not an input")
    with pytest.raises(ValueError, match="exact|auxiliary"):
        module.auxiliary_wheels(tmp_path / "wheels", pins, "windows-x64-py312")


@pytest.mark.parametrize("change", ["version", "hash", "name"])
def test_auxiliary_wheel_identity_and_bytes_are_independently_bound(tmp_path, load_tool_module, change):
    module = load_tool_module("installed_inputs")
    wheel = make_wheel(tmp_path / "wheels")
    pins = {"sample": {"version": "1.0", "sha256": wheel["record"]["sha256"]}}
    if change == "version":
        pins["sample"]["version"] = "2.0"
    elif change == "hash":
        pins["sample"]["sha256"] = "a" * 64
    else:
        pins["other"] = pins.pop("sample")
    with pytest.raises(ValueError, match="identity|bytes|pin"):
        module.auxiliary_wheels(tmp_path / "wheels", pins, "windows-x64-py312")


def test_hash_lock_symlink_is_not_accepted_as_a_pinned_source(tmp_path, load_tool_module):
    module = load_tool_module("installed_inputs")
    real = tmp_path / "real.lock"
    real.write_text(f"sample==1.0 --hash=sha256:{'a' * 64}\n")
    path = tmp_path / "linked.lock"
    path.symlink_to(real)
    with pytest.raises(ValueError, match="regular|symlink"):
        module.read_hash_lock(path)


def test_auxiliary_directory_is_checked_after_every_input_read(tmp_path, load_tool_module, monkeypatch):
    module = load_tool_module("installed_inputs")
    directory = tmp_path / "wheels"
    wheel = make_wheel(directory)
    pins = {"sample": {key: wheel["record"][key] for key in ("version", "sha256")}}
    original = module.locked_wheel

    def changed(*args):
        result = original(*args)
        (directory / "unexpected.txt").write_bytes(b"not recorded")
        return result

    monkeypatch.setattr(module, "locked_wheel", changed)
    with pytest.raises(ValueError, match="directory|membership|exact"):
        module.auxiliary_wheels(directory, pins, "windows-x64-py312")


def test_auxiliary_directory_rejects_junctions(tmp_path, load_tool_module, monkeypatch):
    module = load_tool_module("installed_inputs")
    directory = tmp_path / "wheels"
    wheel = make_wheel(directory)
    pins = {"sample": {key: wheel["record"][key] for key in ("version", "sha256")}}
    monkeypatch.setattr(type(directory), "is_junction", lambda self: self == directory)
    with pytest.raises(ValueError, match="regular|junction"):
        module.auxiliary_wheels(directory, pins, "windows-x64-py312")


def _loader_fixture(tmp_path, module, monkeypatch):
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    packaging = make_wheel(tmp_path / "ordinary", name="packaging")
    pip = make_wheel(tmp_path / "auxiliary", name="pip", version="26.1.2")
    lock = root / "tools/bootstrap-requirements.lock"
    lock.write_text(
        "".join(
            f"{w['record']['name']}=={w['record']['version']} --hash=sha256:{w['record']['sha256']}\n"
            for w in (packaging, pip)
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"target": "windows-x64-py312", "wheels": [packaging["record"]], "vcs": []}))
    monkeypatch.setattr(module, "read_manifest", lambda root, path: json.loads(path.read_text()))
    return root, manifest_path, pip, lock


def test_inputs_are_bound_before_parsing_not_after_a_lock_change(tmp_path, load_tool_module, monkeypatch):
    module = load_tool_module("installed_inputs")
    root, manifest_path, pip, lock = _loader_fixture(tmp_path, module, monkeypatch)

    def change_lock(*args):
        lock.write_text(lock.read_text().replace("pip==26.1.2", "pip==99.0"))

    monkeypatch.setattr(module, "_platform_inputs", change_lock)
    with pytest.raises(ValueError, match="input.*changed|identity"):
        result = module.load_installation_inputs(root, manifest_path, tmp_path / "ordinary", pip["path"])
        result[-1]()


def test_bootstrap_wheel_is_guarded_through_the_complete_inspection(tmp_path, load_tool_module, monkeypatch):
    module = load_tool_module("installed_inputs")
    root, manifest, pip, _lock = _loader_fixture(tmp_path, module, monkeypatch)
    monkeypatch.setattr(module, "_platform_inputs", lambda *args: None)
    result = module.load_installation_inputs(root, manifest, tmp_path / "ordinary", pip["path"])
    pip["path"].write_bytes(b"changed after input resolution")
    with pytest.raises(ValueError, match="wheel|identity|bytes"):
        result[-1]()


def test_report_input_limits_are_explicit_and_still_enforced(tmp_path, load_tool_module):
    module = load_tool_module("installed_inputs")
    path = tmp_path / "report.json"
    path.write_bytes(b"1234")
    assert module.InputBindings({"report": path}, limit=4).contents["report"] == b"1234"
    with pytest.raises(ValueError, match="limit"):
        module.InputBindings({"report": path}, limit=3)


def test_bound_json_rejects_duplicate_keys_instead_of_overwriting_evidence(load_tool_module):
    module = load_tool_module("installed_inputs")
    with pytest.raises(ValueError, match="duplicate"):
        module.json_object(b'{"state":"FAIL","state":"PASS"}')


def test_refnx_report_schema_is_required_even_when_its_wheel_matches(tmp_path, load_tool_module):
    module = load_tool_module("installed_inputs")
    root = tmp_path / "repo"
    (root / "tools/package-manifests").mkdir(parents=True)
    for name in ("refnx-source.json", "refnx-build-macos-arm64-py312.json"):
        (root / "tools/package-manifests" / name).write_text("{}")
    wheel = make_wheel(tmp_path / "build/wheels", name="refnx")
    report = {
        "schema": "wrong",
        "state": "PASS",
        "offline": True,
        "derived_wheel_reused": False,
        "source_git_metadata": False,
        "source": {},
        "build_inputs": {},
        "builder_versions": {},
        "wheel": wheel["record"],
        "inventory": module.inspect_wheel(wheel["path"], wheel["record"]),
    }
    path = tmp_path / "build/build.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="schema|build"):
        module._refnx_input(root, path)


def _refnx_report(module, tmp_path):
    root = tmp_path / "repo"
    manifests = root / "tools/package-manifests"
    manifests.mkdir(parents=True)
    builder = {"wheels": [{"name": "builder", "version": "1.0"}]}
    (manifests / "refnx-source.json").write_text("{}")
    (manifests / "refnx-build-macos-arm64-py312.json").write_text(json.dumps(builder))
    wheel = make_wheel(tmp_path / "build/wheels", name="refnx")
    record = {**wheel["record"], "size": wheel["path"].stat().st_size}
    value = {
        "schema": "xrr-refnx-build-v1",
        "state": "PASS",
        "offline": True,
        "derived_wheel_reused": False,
        "source_git_metadata": False,
        "source": {},
        "build_inputs": builder,
        "builder_versions": {"builder": "1.0"},
        "wheel": record,
        "inventory": module.inspect_wheel(wheel["path"], record),
    }
    path = tmp_path / "build/build.json"
    path.write_text(json.dumps(value))
    return root, path, value


def test_refnx_input_preserves_independent_build_provenance(tmp_path, load_tool_module):
    module = load_tool_module("installed_inputs")
    root, path, report = _refnx_report(module, tmp_path)
    item = module._refnx_input(root, path)
    assert item.provenance == "refnx-source-build"
    assert item.record == report["wheel"]


@pytest.mark.parametrize("change", ["git-metadata", "builder-versions", "size", "extra-output"])
def test_refnx_input_rejects_unbound_build_contracts(tmp_path, load_tool_module, change):
    module = load_tool_module("installed_inputs")
    root, path, report = _refnx_report(module, tmp_path)
    if change == "git-metadata":
        report["source_git_metadata"] = True
    elif change == "builder-versions":
        report["builder_versions"]["builder"] = "2.0"
    elif change == "size":
        report["wheel"]["size"] += 1
    else:
        (path.parent / "wheels/unrecorded.txt").write_text("unexpected output")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="build|wheel|size|exact"):
        module._refnx_input(root, path)
