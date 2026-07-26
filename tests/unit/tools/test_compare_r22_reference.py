from __future__ import annotations

import hashlib
import json
import warnings
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


GROUPS = (
    "model_project",
    "io",
    "physics",
    "fit_compile",
    "fit_search",
    "analysis",
    "services",
    "gui",
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _embedded(content: str) -> dict[str, object]:
    encoded = content.encode()
    return {"content": content, "size": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _provenance() -> dict[str, object]:
    configurations = {}
    for group in GROUPS:
        value = {"case": group}
        configurations[group] = {"value": value, "sha256": hashlib.sha256(_canonical(value)).hexdigest()}
    return {
        "schema": "xrr-r22-reference-provenance-v1",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "generator": _embedded("print('generator')\n"),
        "collection_lock": _embedded("numpy==2.3.2\n"),
        "python": {"implementation": "CPython", "version": "3.12.13"},
        "platform": {"system": "Darwin", "machine": "arm64"},
        "inputs": [
            {
                "input_id": "fixture",
                "input_class": "bundled-example-data",
                "path": "xrr_fitter/examples/input.xy",
                "size": 4,
                "sha256": hashlib.sha256(b"x\n\n\n").hexdigest(),
            }
        ],
        "seeds": {group: ([17] if group in {"fit_search", "analysis"} else []) for group in GROUPS},
        "configurations": configurations,
        "real_data_acceptance": {"status": "NOT_RUN", "reason": "owner post-delivery acceptance"},
    }


def _record(path: Path, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _reference(tmp_path: Path, *, include_npz: bool = False) -> Path:
    root = tmp_path / "reference"
    golden = root / "golden"
    golden.mkdir(parents=True)
    replay_input = root / "xrr_fitter/examples/input.xy"
    replay_input.parent.mkdir(parents=True)
    replay_input.write_bytes(b"x\n\n\n")
    artifacts = []
    groups = {}
    for group in GROUPS:
        relative = f"golden/{group}.json"
        path = root / relative
        path.write_bytes(_canonical({"group": group}))
        artifacts.append(_record(path, root))
        policies: dict[str, object] = {relative: "exact"}
        paths = [relative]
        if include_npz and group == "physics":
            array_relative = "golden/physics.npz"
            np.savez(root / array_relative, q=np.asarray([0.1, 0.2], dtype="<f8"))
            artifacts.append(_record(root / array_relative, root))
            paths.append(array_relative)
            policies[array_relative] = {"kind": "mapping", "fields": {"q": "array_equal"}}
        groups[group] = {
            "artifacts": paths,
            "comparison_policy": {"kind": "mapping", "fields": policies},
            "input_ids": ["fixture"],
        }
    builder = Path(__file__).resolve().parents[3] / "tools" / "build_r22_reference.py"
    manifest = {
        "schema": "xrr-r22-reference-v1",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "archive_sha256": "1" * 64,
        "freeze_receipt_sha256": "2" * 64,
        "release_identity_sha256": "3" * 64,
        "product_manifest_sha256": "4" * 64,
        "builder_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
        "reference_sidecar_manifest_sha256": "5" * 64,
        "reference_sidecar_tree_sha256": "6" * 64,
        "reference_sidecar_lock_sha256": "7" * 64,
        "real_data_acceptance_status": "NOT_RUN",
        "real_data_acceptance_reason": "owner post-delivery acceptance",
        "provenance": _provenance(),
        "groups": groups,
        "artifacts": artifacts,
    }
    path = root / "manifest.json"
    path.write_bytes(_canonical(manifest))
    return path


def _rewrite_manifest(path: Path, mutate) -> None:
    payload = json.loads(path.read_bytes())
    mutate(payload)
    path.write_bytes(_canonical(payload))


def _replace_record(payload: dict[str, object], record: dict[str, object]) -> None:
    artifacts = payload["artifacts"]
    index = next(index for index, item in enumerate(artifacts) if item["path"] == record["path"])
    artifacts[index] = record


def test_reference_manifest_validator_binds_exact_groups_artifacts_and_inputs(
    load_tool_module,
) -> None:
    module = load_tool_module("reference_manifest")
    groups = {
        group: {
            "artifacts": [f"golden/{group}.json"],
            "comparison_policy": {
                "kind": "mapping",
                "fields": {f"golden/{group}.json": "exact"},
            },
            "input_ids": ["fixture"],
        }
        for group in GROUPS
    }
    artifacts = {f"golden/{group}.json" for group in GROUPS}

    observed = module.validate_groups(
        groups,
        artifact_paths=artifacts,
        known_input_ids={"fixture"},
    )

    assert tuple(observed) == GROUPS
    assert observed == groups


def test_self_check_recomputes_every_file_hash(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path, include_npz=True)
    observed = module.self_check(manifest)
    assert observed == {
        "schema": "xrr-r22-reference-v1",
        "source_commit": "a" * 40,
        "group_count": 8,
        "artifact_count": 9,
        "input_count": 1,
    }
    (manifest.parent / "golden/physics.json").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|size"):
        module.self_check(manifest)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(extra=True), "field|manifest"),
        (lambda value: value.pop("reference_sidecar_lock_sha256"), "field|manifest"),
        (lambda value: value["groups"].pop("gui"), "eight groups"),
        (lambda value: value["artifacts"].append(value["artifacts"][0]), "duplicate"),
        (lambda value: value.update(real_data_acceptance_status="PASS"), "real-data"),
        (lambda value: value["provenance"].update(source_tree="c" * 40), "tree"),
    ],
)
def test_self_check_rejects_manifest_schema_and_provenance_drift(
    tmp_path: Path, load_tool_module, mutation, match: str
) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path)
    _rewrite_manifest(manifest, mutation)
    with pytest.raises(ValueError, match=match):
        module.self_check(manifest)


def test_self_check_rejects_undeclared_file_symlink_and_noncanonical_json(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path)
    extra = manifest.parent / "golden/extra.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared"):
        module.self_check(manifest)
    extra.unlink()
    extra.symlink_to(manifest.parent / "golden/io.json")
    with pytest.raises(ValueError, match="symlink"):
        module.self_check(manifest)
    extra.unlink()
    target = manifest.parent / "golden/io.json"
    target.write_text('{"group": "io"}\n', encoding="utf-8")
    _rewrite_manifest(
        manifest,
        lambda value: _replace_record(value, _record(target, manifest.parent)),
    )
    with pytest.raises(ValueError, match="canonical"):
        module.self_check(manifest)


@pytest.mark.parametrize("mutation", ["object", "order", "timestamp"])
def test_self_check_rejects_unsafe_or_nondeterministic_npz(
    tmp_path: Path, load_tool_module, mutation: str
) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path, include_npz=True)
    path = manifest.parent / "golden/physics.npz"
    if mutation == "object":
        np.savez(path, q=np.asarray([object()], dtype=object))
    elif mutation == "order":
        np.savez(path, z=np.ones(1), q=np.ones(1))
    else:
        import zipfile

        with zipfile.ZipFile(path, "w") as archive:
            info = zipfile.ZipInfo("q.npy", date_time=(2026, 1, 1, 0, 0, 0))
            from io import BytesIO

            payload = BytesIO()
            np.save(payload, np.ones(1), allow_pickle=False)
            archive.writestr(info, payload.getvalue())
    _rewrite_manifest(
        manifest,
        lambda value: _replace_record(value, _record(path, manifest.parent)),
    )
    with pytest.raises(ValueError, match="object|key|order|metadata|NPZ"):
        module.self_check(manifest)


def test_comparison_policy_engine_covers_exact_array_physics_and_mapping(load_tool_module) -> None:
    module = load_tool_module("compare_r22_reference")
    module.compare_value({"id": "x", "values": [1, 2]}, {"id": "x", "values": [1, 2]})
    module.compare_value(np.asarray([1.0, np.nan]), np.asarray([1.0, np.nan]), "array_equal")
    reference = np.asarray([1e-13, 1e-8])
    actual = reference + np.asarray([1e-12, 1e-10 + 5e-7 * 1e-8])
    module.compare_value(reference, actual, "physics_reflectivity")
    policy = {"kind": "mapping", "fields": {"a": "exact", "b": "array_equal"}}
    module.compare_value({"a": 1, "b": np.ones(2)}, {"a": 1, "b": np.ones(2)}, policy)
    tolerance = {"kind": "scalar_tolerance", "absolute": 0.02, "relative": 0.0}
    module.compare_value(1.0, 1.01, tolerance)
    bounds = {"kind": "scalar_bounds", "minimum": 0.9, "maximum": 1.0}
    module.compare_value(0.95, 0.95, bounds)


@pytest.mark.parametrize(
    ("reference", "actual", "policy"),
    [
        ({"id": 1}, {"id": True}, "exact"),
        (np.ones(2, dtype="<f8"), np.ones(2, dtype="<f4"), "array_equal"),
        (np.ones(2, dtype="<f8"), np.ones(2, dtype="<f4"), "physics_reflectivity"),
        (np.asarray([1e-13]), np.asarray([1.2e-12]), "physics_reflectivity"),
        ({"a": 1}, {"a": 1, "b": 2}, {"kind": "mapping", "fields": {"a": "exact"}}),
        (1.0, 1.03, {"kind": "scalar_tolerance", "absolute": 0.02, "relative": 0.0}),
        (0.95, 1.1, {"kind": "scalar_bounds", "minimum": 0.9, "maximum": 1.0}),
    ],
)
def test_comparison_policy_engine_rejects_drift(
    load_tool_module, reference: object, actual: object, policy: object
) -> None:
    module = load_tool_module("compare_r22_reference")
    with pytest.raises(ValueError):
        module.compare_value(reference, actual, policy)


def test_unregistered_group_fails_without_discovery(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path)
    with pytest.raises(ValueError, match="not registered"):
        module.compare_group(manifest, "physics", registry={})


def test_physics_adapter_replays_committed_reference_after_registration(
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    root = Path(__file__).resolve().parents[3]
    assert tuple(module.GROUP_REGISTRY) == ("model_project", "io", "physics")
    source = (root / "tools/reference_groups/physics.py").read_text(encoding="utf-8")
    assert "tests.support" not in source
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert module.compare_group(
            root / "verification/r22/reference/manifest.json",
            "physics",
        ) == {"group": "physics", "status": "PASS", "artifact_count": 2}


def test_registered_group_uses_closed_adapter_and_policy(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path)
    observed = []

    def adapter(context):
        observed.append(context)
        return {"golden/model_project.json": {"group": "model_project"}}

    result = module.compare_group(manifest, "model_project", registry={"model_project": adapter})
    assert result == {"group": "model_project", "status": "PASS", "artifact_count": 1}
    assert observed == [
        module.ReplayContext(
            group="model_project",
            artifacts=("golden/model_project.json",),
            inputs=(
                module.ReplayInput(
                    input_id="fixture",
                    input_class="bundled-example-data",
                    path="xrr_fitter/examples/input.xy",
                    size=4,
                    sha256=hashlib.sha256(b"x\n\n\n").hexdigest(),
                    content=b"x\n\n\n",
                ),
            ),
            configuration={"case": "model_project"},
            seeds=(),
        )
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda entry: entry["comparison_policy"]["fields"].clear(),
        lambda entry: entry.update(input_ids=[]),
        lambda entry: entry.update(input_ids=["missing"]),
        lambda entry: entry.update(configuration={"duplicate": True}),
    ],
)
def test_self_check_rejects_group_replay_metadata_drift(
    tmp_path: Path,
    load_tool_module,
    mutation,
) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path)
    _rewrite_manifest(manifest, lambda value: mutation(value["groups"]["physics"]))

    with pytest.raises(ValueError, match="artifact|input|metadata|policy"):
        module.self_check(manifest)


def test_self_check_binds_replay_input_bytes(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path)

    assert module.self_check(manifest)["input_count"] == 1

    replay_input = manifest.parent / "xrr_fitter/examples/input.xy"
    replay_input.write_bytes(b"drift")
    with pytest.raises(ValueError, match="input.*size|input.*hash"):
        module.self_check(manifest)


@pytest.mark.parametrize("mutation", ["missing", "symlink", "extra"])
def test_self_check_rejects_missing_symlink_or_extra_replay_input(
    tmp_path: Path,
    load_tool_module,
    mutation: str,
) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path)
    replay_input = manifest.parent / "xrr_fitter/examples/input.xy"
    if mutation == "missing":
        replay_input.unlink()
    elif mutation == "symlink":
        replay_input.unlink()
        replay_input.symlink_to(manifest.parent / "golden/io.json")
    else:
        extra = replay_input.with_name("extra.xy")
        extra.write_bytes(b"extra\n")

    with pytest.raises(ValueError, match="input|symlink|undeclared|missing"):
        module.self_check(manifest)


def test_self_check_rejects_replay_input_path_traversal(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    manifest = _reference(tmp_path)
    _rewrite_manifest(
        manifest,
        lambda value: value["provenance"]["inputs"][0].update(path="../input.xy"),
    )

    with pytest.raises(ValueError, match="input.*path"):
        module.self_check(manifest)


def _committed_model_context(module):
    root = Path(__file__).resolve().parents[3]
    reference_root = root / "verification/r22/reference"
    manifest = json.loads(
        (reference_root / "manifest.json").read_text(encoding="utf-8")
    )
    provenance = manifest["provenance"]
    by_id = {record["input_id"]: record for record in provenance["inputs"]}
    entry = manifest["groups"]["model_project"]
    inputs = tuple(
        module.ReplayInput(
            input_id=by_id[input_id]["input_id"],
            input_class=by_id[input_id]["input_class"],
            path=by_id[input_id]["path"],
            size=by_id[input_id]["size"],
            sha256=by_id[input_id]["sha256"],
            content=(reference_root / by_id[input_id]["path"]).read_bytes(),
        )
        for input_id in entry["input_ids"]
    )
    return module.ReplayContext(
        group="model_project",
        artifacts=tuple(entry["artifacts"]),
        inputs=inputs,
        configuration=provenance["configurations"]["model_project"]["value"],
        seeds=tuple(provenance["seeds"]["model_project"]),
    )


def _mutated_model_document(context, mutation):
    replay_input = context.inputs[0]
    document = json.loads(replay_input.content)
    mutation(document)
    content = _canonical(document)
    changed = replace(
        replay_input,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )
    return replace(context, inputs=(changed, *context.inputs[1:]))


def test_model_project_adapter_is_explicit_and_matches_committed_reference(
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    root = Path(__file__).resolve().parents[3]

    assert tuple(module.GROUP_REGISTRY) == ("model_project", "io", "physics")
    source = (root / "tools/reference_groups/model_project.py").read_text(encoding="utf-8")
    assert "tests.support" not in source
    assert module.compare_group(
        root / "verification/r22/reference/manifest.json",
        "model_project",
    ) == {"group": "model_project", "status": "PASS", "artifact_count": 1}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: replace(context, group="io"),
        lambda context: replace(context, artifacts=()),
        lambda context: replace(context, inputs=context.inputs[:-1]),
        lambda context: replace(
            context,
            inputs=context.inputs + (context.inputs[0],),
        ),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], content=b"{}"), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], input_id="wrong"), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], input_class="wrong"), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], path="wrong.json"), *context.inputs[1:]),
        ),
        lambda context: replace(context, configuration={"cases": ["single-layer"]}),
        lambda context: replace(
            context,
            configuration={**context.configuration, "extra": True},
        ),
        lambda context: replace(context, seeds=(1,)),
    ],
)
def test_model_project_adapter_rejects_replay_context_field_drift(
    load_tool_module,
    mutation,
) -> None:
    module = load_tool_module("compare_r22_reference")
    adapter = module.GROUP_REGISTRY["model_project"]
    context = _committed_model_context(module)

    with pytest.raises(ValueError, match="model_project|artifact|input|configuration|seed"):
        adapter(mutation(context))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.pop("algorithm_version"),
        lambda document: document.update(extra=True),
        lambda document: document["datasets"][0].pop("instrument"),
        lambda document: document["datasets"][0].update(extra=True),
    ],
)
def test_model_project_adapter_rejects_missing_or_extra_project_fields(
    load_tool_module,
    mutation,
) -> None:
    module = load_tool_module("compare_r22_reference")
    adapter = module.GROUP_REGISTRY["model_project"]
    context = _mutated_model_document(_committed_model_context(module), mutation)

    with pytest.raises(ValueError, match="project|dataset|field"):
        adapter(context)


def _committed_group_context(module, group: str):
    root = Path(__file__).resolve().parents[3]
    reference_root = root / "verification/r22/reference"
    manifest = json.loads(
        (reference_root / "manifest.json").read_text(encoding="utf-8")
    )
    provenance = manifest["provenance"]
    by_id = {record["input_id"]: record for record in provenance["inputs"]}
    entry = manifest["groups"][group]
    inputs = tuple(
        module.ReplayInput(
            input_id=by_id[input_id]["input_id"],
            input_class=by_id[input_id]["input_class"],
            path=by_id[input_id]["path"],
            size=by_id[input_id]["size"],
            sha256=by_id[input_id]["sha256"],
            content=(reference_root / by_id[input_id]["path"]).read_bytes(),
        )
        for input_id in entry["input_ids"]
    )
    return module.ReplayContext(
        group=group,
        artifacts=tuple(entry["artifacts"]),
        inputs=inputs,
        configuration=provenance["configurations"][group]["value"],
        seeds=tuple(provenance["seeds"][group]),
    )


def test_io_adapter_is_explicit_and_matches_committed_reference(
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    root = Path(__file__).resolve().parents[3]

    assert tuple(module.GROUP_REGISTRY) == ("model_project", "io", "physics")
    source = (root / "tools/reference_groups/io.py").read_text(encoding="utf-8")
    assert "tests.support" not in source
    assert module.compare_group(
        root / "verification/r22/reference/manifest.json",
        "io",
    ) == {"group": "io", "status": "PASS", "artifact_count": 2}


def test_io_adapter_replays_serialized_project_bytes(
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("compare_r22_reference")
    adapter = module.GROUP_REGISTRY["io"]
    context = _committed_group_context(module, "io")
    monkeypatch.setitem(adapter.__globals__, "project_to_bytes", lambda _project: b"{}")

    with pytest.raises(ValueError, match="project|field"):
        adapter(context)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: replace(context, group="model_project"),
        lambda context: replace(context, artifacts=context.artifacts[:-1]),
        lambda context: replace(context, inputs=context.inputs[:-1]),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], content=b"drift"), *context.inputs[1:]),
        ),
        lambda context: replace(context, configuration={"cases": []}),
        lambda context: replace(context, seeds=(1,)),
    ],
)
def test_io_adapter_rejects_replay_context_field_drift(
    load_tool_module,
    mutation,
) -> None:
    module = load_tool_module("compare_r22_reference")
    adapter = module.GROUP_REGISTRY["io"]
    context = _committed_group_context(module, "io")

    with pytest.raises(ValueError, match="io|artifact|input|configuration|seed"):
        adapter(mutation(context))


# Physics replay rejects drift across every hash-bound input and configuration axis.
def _physics_configuration_drift(context, path: tuple[str, ...], value: object):
    configuration = deepcopy(context.configuration)
    target = configuration
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return replace(context, configuration=configuration)


def _physics_project_content_drift(context):
    replay_input = context.inputs[0]
    document = json.loads(replay_input.content)
    document["datasets"][0]["dataset_id"] += "-drift"
    content = _canonical(document)
    changed = replace(
        replay_input,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )
    return replace(context, inputs=(changed, context.inputs[1]))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: replace(context, group="io"),
        lambda context: replace(context, artifacts=context.artifacts[:-1]),
        lambda context: replace(context, artifacts=tuple(reversed(context.artifacts))),
        lambda context: replace(context, inputs=context.inputs[:-1]),
        lambda context: replace(context, inputs=tuple(reversed(context.inputs))),
        lambda context: replace(context, inputs=(replace(context.inputs[0], input_id="single-layer-project"), context.inputs[1])),
        lambda context: replace(context, inputs=(replace(context.inputs[0], path=context.inputs[1].path), context.inputs[1])),
        lambda context: replace(context, inputs=(replace(context.inputs[0], content=b"drift"), context.inputs[1])),
        lambda context: replace(context, inputs=(replace(context.inputs[0], sha256="0" * 64), context.inputs[1])),
        _physics_project_content_drift,
        lambda context: _physics_configuration_drift(context, ("q_grid", "count"), 127),
        lambda context: _physics_configuration_drift(context, ("q_grid", "start"), 0.006),
        lambda context: _physics_configuration_drift(context, ("q_grid", "stop"), 0.36),
        lambda context: _physics_configuration_drift(context, ("theta_grid", "count"), 127),
        lambda context: _physics_configuration_drift(context, ("theta_grid", "start"), 0.06),
        lambda context: _physics_configuration_drift(context, ("theta_grid", "stop"), 2.6),
        lambda context: _physics_configuration_drift(context, ("relative_sigma",), 0.003),
        lambda context: _physics_configuration_drift(context, ("profile_step_a",), 1.0),
        lambda context: replace(context, configuration={**context.configuration, "extra": True}),
        lambda context: replace(context, seeds=(1,)),
    ],
)
def test_physics_adapter_rejects_replay_context_field_drift(
    load_tool_module,
    mutation,
) -> None:
    module = load_tool_module("compare_r22_reference")
    adapter = module.GROUP_REGISTRY["physics"]
    context = _committed_group_context(module, "physics")
    with pytest.raises(ValueError, match="physics|artifact|input|configuration|seed"):
        adapter(mutation(context))
