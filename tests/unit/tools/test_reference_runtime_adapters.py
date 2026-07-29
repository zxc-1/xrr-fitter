"""Runtime replay contracts for fit, stochastic, service, and GUI adapters."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import warnings

import pytest


REGISTERED_GROUPS = (
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
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


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


def _fit_compile_project_content_drift(context):
    replay_input = context.inputs[1]
    document = json.loads(replay_input.content)
    document["datasets"][0]["dataset_id"] += "-drift"
    content = _canonical(document)
    changed = replace(
        replay_input,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )
    return replace(context, inputs=(context.inputs[0], changed, *context.inputs[2:]))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: replace(context, group="physics"),
        lambda context: replace(context, artifacts=context.artifacts[:-1]),
        lambda context: replace(context, artifacts=tuple(reversed(context.artifacts))),
        lambda context: replace(context, inputs=context.inputs[:-1]),
        lambda context: replace(context, inputs=tuple(reversed(context.inputs))),
        lambda context: replace(
            context,
            inputs=(
                replace(context.inputs[0], input_id="single-layer-data"),
                *context.inputs[1:],
            ),
        ),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], content=b"drift"), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], sha256="0" * 64), *context.inputs[1:]),
        ),
        _fit_compile_project_content_drift,
        lambda context: replace(context, configuration={"cases": ["single-layer"]}),
        lambda context: replace(
            context,
            configuration={**context.configuration, "extra": True},
        ),
        lambda context: replace(context, seeds=(1,)),
    ],
)
def test_fit_compile_adapter_rejects_replay_context_field_drift(
    load_tool_module,
    mutation,
) -> None:
    module = load_tool_module("compare_r22_reference")
    adapter = module.GROUP_REGISTRY["fit_compile"]
    context = _committed_group_context(module, "fit_compile")
    with pytest.raises(ValueError, match="fit_compile|artifact|input|configuration|seed"):
        adapter(mutation(context))


@pytest.mark.parametrize("group", ["fit_search", "analysis"])
@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: replace(context, group="physics"),
        lambda context: replace(context, artifacts=context.artifacts[:-1]),
        lambda context: replace(context, artifacts=tuple(reversed(context.artifacts))),
        lambda context: replace(context, inputs=context.inputs[:-1]),
        lambda context: replace(context, inputs=tuple(reversed(context.inputs))),
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
            inputs=(replace(context.inputs[0], path="wrong.xy"), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], content=b"drift"), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], sha256="0" * 64), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            configuration={**context.configuration, "extra": True},
        ),
        lambda context: replace(context, seeds=context.seeds + (0,)),
    ],
    ids=[
        "group",
        "artifact-missing",
        "artifact-order",
        "input-missing",
        "input-order",
        "input-id",
        "input-class",
        "input-path",
        "input-content",
        "input-hash",
        "configuration",
        "seeds",
    ],
)
def test_stochastic_adapters_reject_replay_context_field_drift(
    load_tool_module,
    group: str,
    mutation,
) -> None:
    module = load_tool_module("compare_r22_reference")
    adapter = module.GROUP_REGISTRY[group]
    context = _committed_group_context(module, group)
    with pytest.raises(ValueError, match=f"{group}|artifact|input|configuration|seed"):
        adapter(mutation(context))


def test_services_adapter_is_explicit_and_matches_committed_reference(
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    root = Path(__file__).resolve().parents[3]
    assert tuple(module.GROUP_REGISTRY) == REGISTERED_GROUPS
    source = (root / "tools/reference_groups/services.py").read_text(encoding="utf-8")
    assert "tests.support" not in source
    assert "verification/r22" not in source
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert module.compare_group(
            root / "verification/r22/reference/manifest.json",
            "services",
        ) == {"group": "services", "status": "PASS", "artifact_count": 1}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: replace(context, group="analysis"),
        lambda context: replace(context, artifacts=()),
        lambda context: replace(context, inputs=context.inputs[:-1]),
        lambda context: replace(context, inputs=tuple(reversed(context.inputs))),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], input_id="wrong"), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], content=b"drift"), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            configuration={**context.configuration, "extra": True},
        ),
        lambda context: replace(context, seeds=(1, 2)),
    ],
    ids=(
        "group",
        "artifact",
        "input-missing",
        "input-order",
        "input-id",
        "input-content",
        "configuration",
        "seeds",
    ),
)
def test_services_adapter_rejects_replay_context_field_drift(
    load_tool_module,
    mutation,
) -> None:
    module = load_tool_module("compare_r22_reference")
    adapter = module.GROUP_REGISTRY["services"]
    context = _committed_group_context(module, "services")
    with pytest.raises(ValueError, match="services|artifact|input|configuration|seed"):
        adapter(mutation(context))


def test_gui_adapter_is_explicit_and_matches_committed_reference(
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    root = Path(__file__).resolve().parents[3]
    assert tuple(module.GROUP_REGISTRY) == REGISTERED_GROUPS
    source = (root / "tools/reference_groups/gui.py").read_text(encoding="utf-8")
    for forbidden in ("tests.support", "verification/r22", "read_bytes(", "read_text("):
        assert forbidden not in source
    assert module.compare_group(
        root / "verification/r22/reference/manifest.json",
        "gui",
    ) == {"group": "gui", "status": "PASS", "artifact_count": 1}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: replace(context, group="services"),
        lambda context: replace(context, artifacts=()),
        lambda context: replace(context, inputs=context.inputs[:-1]),
        lambda context: replace(context, inputs=tuple(reversed(context.inputs))),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], input_id="wrong"), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            inputs=(replace(context.inputs[0], content=b"drift"), *context.inputs[1:]),
        ),
        lambda context: replace(
            context,
            configuration={**context.configuration, "extra": True},
        ),
        lambda context: replace(context, seeds=(1,)),
    ],
    ids=(
        "group",
        "artifact",
        "input-missing",
        "input-order",
        "input-id",
        "input-content",
        "configuration",
        "seeds",
    ),
)
def test_gui_adapter_rejects_replay_context_field_drift(
    load_tool_module,
    mutation,
) -> None:
    module = load_tool_module("compare_r22_reference")
    adapter = module.GROUP_REGISTRY["gui"]
    context = _committed_group_context(module, "gui")
    with pytest.raises(ValueError, match="gui|artifact|input|configuration|seed"):
        adapter(mutation(context))
