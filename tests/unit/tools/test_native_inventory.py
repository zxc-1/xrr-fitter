from __future__ import annotations

import hashlib
import json
import struct

from tests.support.native_fixtures import macho, macho_command, pe_image, static_archive, universal


def _report(load_tool_module, files, *, target="macos-arm64-py312"):
    parser = load_tool_module("native_binary")
    inventories = {}
    for package, path, content in files:
        inventory = inventories.setdefault(
            package, {"name": package, "version": "1", "requirements": [], "licenses": [], "files": []}
        )
        parsed = parser.inspect_native(content)
        inventory["files"].append(
            {
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
                "kind": "native",
                "format": parsed["format"],
                "native": parsed,
            }
        )
    manifest = {
        "target": target,
        "lock_sha256": "a" * 64,
        "vcs": [],
        "wheels": [
            {
                "name": name,
                "version": "1",
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
                "url": f"https://example.invalid/{name}.whl",
            }
            for name in sorted(inventories)
        ],
    }
    return load_tool_module("package_sbom").assemble_sbom(manifest, list(inventories.values()))


def _file(report, path, package="sample"):
    component = next(item for item in report["components"] if item["name"] == package)
    return next(item for item in component["components"] if item["name"] == path)


def _properties(component):
    return {item["name"]: item["value"] for item in component["properties"]}


def _requests(image):
    return json.loads(_properties(image)["xrr:native:requests"])


def _dependencies(report, image):
    return next(item["dependsOn"] for item in report["dependencies"] if item["ref"] == image["bom-ref"])


def test_loader_path_edges_bind_exact_native_image_bytes_without_claiming_runtime_closure(load_tool_module):
    source = macho(
        [macho_command(12, "@loader_path/../lib/dependency.dylib"), macho_command(12, "/usr/lib/libSystem.B.dylib")]
    )
    dependency = macho()
    result = _report(
        load_tool_module,
        [("sample", "sample/bin/source.so", source), ("sample", "sample/lib/dependency.dylib", dependency)],
    )
    image = _file(result, "sample/bin/source.so")["components"][0]
    target = _file(result, "sample/lib/dependency.dylib")["components"][0]
    assert image["hashes"] == [{"alg": "SHA-256", "content": hashlib.sha256(source).hexdigest()}]
    assert _dependencies(result, image) == [target["bom-ref"]]
    requests = _requests(image)
    assert requests[0]["state"] == "archive-path" and requests[0]["target"] == target["bom-ref"]
    assert requests[1]["state"] == "unresolved" and "system" in requests[1]["reason"]
    properties = _properties(result["metadata"])
    assert properties["xrr:inventory:native-resolved-requests"] == "1"
    assert properties["xrr:inventory:native-unresolved-requests"] == "1"
    assert properties["xrr:inventory:installed-distribution"] == "not-inspected"
    assert result["compositions"] == [{"aggregate": "incomplete"}]


def test_declared_rpath_is_resolved_from_the_loader_not_a_basename_search(load_tool_module):
    source = macho(
        [macho_command(12, "@rpath/dependency.dylib"), macho_command(0x8000001C, "@loader_path/../lib", dylib=False)]
    )
    result = _report(
        load_tool_module,
        [("sample", "sample/bin/source.so", source), ("sample", "sample/lib/dependency.dylib", macho())],
    )
    image = _file(result, "sample/bin/source.so")["components"][0]
    target = _file(result, "sample/lib/dependency.dylib")["components"][0]
    assert _dependencies(result, image) == [target["bom-ref"]]


def test_unbound_rpath_context_prevents_a_false_internal_resolution(load_tool_module):
    source = macho(
        [
            macho_command(12, "@rpath/dependency.dylib"),
            macho_command(0x8000001C, "@executable_path", dylib=False),
            macho_command(0x8000001C, "@loader_path", dylib=False),
        ]
    )
    result = _report(
        load_tool_module, [("sample", "sample/source.so", source), ("sample", "sample/dependency.dylib", macho())]
    )
    image = _file(result, "sample/source.so")["components"][0]
    request = _requests(image)[0]
    assert _dependencies(result, image) == []
    assert request["state"] == "unresolved" and request["reason"] == "runtime-rpath-context"
    assert len(request["candidates"]) == 1


def test_missing_rpath_does_not_guess_from_a_matching_basename(load_tool_module):
    source = macho([macho_command(12, "@rpath/dependency.dylib")])
    result = _report(
        load_tool_module, [("sample", "sample/source.so", source), ("sample", "elsewhere/dependency.dylib", macho())]
    )
    image = _file(result, "sample/source.so")["components"][0]
    assert _dependencies(result, image) == [] and _requests(image)[0]["candidates"] == []


def test_duplicate_paths_across_wheels_remain_ambiguous(load_tool_module):
    source = macho([macho_command(12, "@loader_path/dependency.dylib")])
    result = _report(
        load_tool_module,
        [
            ("sample", "sample/source.so", source),
            ("sample", "sample/dependency.dylib", macho()),
            ("another", "sample/dependency.dylib", macho()),
        ],
    )
    image = _file(result, "sample/source.so")["components"][0]
    request = _requests(image)[0]
    assert _dependencies(result, image) == []
    assert request["reason"] == "ambiguous-archive-path" and len(request["candidates"]) == 2


def test_universal_dependencies_link_only_the_matching_architecture(load_tool_module):
    cpus = [0x1000007, 0x100000C]
    source = universal([(cpu, macho([macho_command(12, "@loader_path/dependency.dylib")], cpu=cpu)) for cpu in cpus])
    dependency = universal([(cpu, macho(cpu=cpu)) for cpu in cpus])
    result = _report(
        load_tool_module, [("sample", "sample/source.so", source), ("sample", "sample/dependency.dylib", dependency)]
    )
    images = _file(result, "sample/source.so")["components"]
    targets = _file(result, "sample/dependency.dylib")["components"]
    for image, target in zip(images, targets, strict=True):
        assert _dependencies(result, image) == [target["bom-ref"]]
        assert _properties(image)["xrr:native:architecture"] == _properties(target)["xrr:native:architecture"]
    assert [_properties(image)["xrr:native:target-family"] for image in images] == ["foreign", "target"]


def test_wrong_architecture_at_the_right_path_is_not_a_dependency(load_tool_module):
    source = macho([macho_command(12, "@loader_path/dependency.dylib")])
    result = _report(
        load_tool_module,
        [("sample", "sample/source.so", source), ("sample", "sample/dependency.dylib", macho(cpu=0x1000007))],
    )
    image = _file(result, "sample/source.so")["components"][0]
    assert _dependencies(result, image) == [] and _requests(image)[0]["state"] == "unresolved"


def test_archive_members_are_not_mistaken_for_installed_loadable_libraries(load_tool_module):
    source = macho([macho_command(12, "@loader_path/archive.a")])
    leaf = macho()
    result = _report(
        load_tool_module,
        [("sample", "sample/source.so", source), ("sample", "sample/archive.a", static_archive([("library.o", leaf)]))],
    )
    image = _file(result, "sample/source.so")["components"][0]
    member = _file(result, "sample/archive.a")["components"][0]
    assert _dependencies(result, image) == []
    assert member["hashes"] == [{"alg": "SHA-256", "content": hashlib.sha256(leaf).hexdigest()}]
    assert json.loads(_properties(member)["xrr:native:declarations"])["member_path"] == ["library.o"]


def test_pe_forwarders_are_retained_without_inventing_windows_runtime_resolution(load_tool_module):
    result = _report(load_tool_module, [("sample", "sample/library.dll", pe_image())])
    image = _file(result, "sample/library.dll")["components"][0]
    assert {request["kind"] for request in _requests(image)} == {"load", "delay-load", "forwarder"}
    assert all(request["state"] == "unresolved" for request in _requests(image))
    assert _dependencies(result, image) == []
    properties = _properties(image)
    assert properties["xrr:native:target-family"] == "foreign"
    assert json.loads(properties["xrr:native:declarations"])["forwarded_exports"] == ["python312.PyObject_Call"]


def test_unknown_native_content_is_visible_in_file_and_aggregate_evidence(load_tool_module):
    result = _report(load_tool_module, [("sample", "sample/unknown.so", b"unknown object")])
    properties = _properties(_file(result, "sample/unknown.so"))
    assert json.loads(properties["xrr:native:unparsed"])
    assert _properties(result["metadata"])["xrr:inventory:native-unparsed-records"] == "1"
    assert result["compositions"] == [{"aggregate": "incomplete"}]


def test_archive_member_inventory_is_not_lost_when_emitting_the_sbom(load_tool_module):
    content = static_archive([("object.o", b"unknown member")])
    result = _report(load_tool_module, [("sample", "sample/library.a", content)])
    properties = _properties(_file(result, "sample/library.a"))
    containers = json.loads(properties["xrr:native:container-evidence"])
    assert containers["members"][0]["sha256"] == hashlib.sha256(b"unknown member").hexdigest()


def test_unparsed_loader_context_blocks_internal_resolution(load_tool_module):
    source = macho([macho_command(12, "@loader_path/dependency.dylib"), struct.pack("<2I", 0x8000FFFF, 8)])
    result = _report(
        load_tool_module, [("sample", "sample/source.so", source), ("sample", "sample/dependency.dylib", macho())]
    )
    image = _file(result, "sample/source.so")["components"][0]
    assert _dependencies(result, image) == []
    assert _requests(image)[0]["reason"] == "unsupported-loader-declarations"


def test_wheel_relocations_are_not_mistaken_for_an_installed_layout(load_tool_module):
    source = macho([macho_command(12, "@loader_path/dependency.dylib")])
    prefix = "sample-1.data/platlib/sample/"
    result = _report(
        load_tool_module, [("sample", prefix + "source.so", source), ("sample", prefix + "dependency.dylib", macho())]
    )
    image = _file(result, prefix + "source.so")["components"][0]
    assert _dependencies(result, image) == [] and _requests(image)[0]["reason"] == "wheel-relocation-context"
