"""Replay the frozen R22 project-model reference through R23 model values."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields, is_dataclass, replace
from enum import Enum
import hashlib
import json

from xrr_fitter.io.project_codec import project_from_bytes, project_to_dict
from xrr_fitter.model.project import (
    DatasetProject,
    ProjectUiState,
    XrrProject,
    with_active_dataset,
    with_batch_mode,
    with_dataset_fit_mask,
    with_workspace_state,
)


ARTIFACT = "golden/model_project.json"
INPUTS = {
    "mo-si-periodic-project": (
        "bundled-example-project",
        "xrr_fitter/examples/mo-si-periodic.xrrproj.json",
        "mo-si-periodic",
    ),
    "single-layer-project": (
        "bundled-example-project",
        "xrr_fitter/examples/single-layer.xrrproj.json",
        "single-layer",
    ),
}
INPUT_ORDER = tuple(INPUTS)
CONFIGURATION = {
    "cases": ["single-layer", "mo-si-periodic"],
    "operations": [
        "load_project",
        "project_to_dict",
        "with_active_dataset",
        "with_batch_mode",
        "fit_mask_transition",
        "ui_state_transition",
    ],
}


def _json_sequence(value: tuple[object, ...] | list[object]) -> list[object]:
    return [_json_value(item) for item in value]


def _json_dataclass(value: object) -> dict[str, object]:
    return {
        field.name: _json_value(getattr(value, field.name))
        for field in dataclass_fields(value)
    }


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (tuple, list)):
        return _json_sequence(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_dataclass(value)
    return value


def _validate_input_identity(value: object, expected: tuple[str, str, str]) -> str:
    input_class, path, stem = expected
    if value.input_class != input_class or value.path != path:
        raise ValueError("model_project input identity drift")
    return stem


def _validate_input_content(value: object) -> bytes:
    content = value.content
    if not isinstance(content, bytes):
        raise ValueError("model_project input content must be bytes")
    digest = hashlib.sha256(content).hexdigest()
    if value.size != len(content) or value.sha256 != digest:
        raise ValueError("model_project input size or hash drift")
    return content


def _validate_context_identity(context: object) -> None:
    if context.group != "model_project":
        raise ValueError("model_project group drift")
    if tuple(context.artifacts) != (ARTIFACT,):
        raise ValueError("model_project artifact drift")
    if context.configuration != CONFIGURATION:
        raise ValueError("model_project configuration drift")
    if tuple(context.seeds):
        raise ValueError("model_project seed drift")


def _input_contents(inputs: tuple[object, ...]) -> dict[str, bytes]:
    contents: dict[str, bytes] = {}
    for value in inputs:
        try:
            expected = INPUTS[value.input_id]
        except (AttributeError, KeyError) as error:
            raise ValueError("model_project input identity drift") from error
        stem = _validate_input_identity(value, expected)
        contents[stem] = _validate_input_content(value)
    return contents


def _validate_context(context: object) -> dict[str, bytes]:
    _validate_context_identity(context)
    inputs = tuple(context.inputs)
    if tuple(value.input_id for value in inputs) != INPUT_ORDER:
        raise ValueError("model_project input set or order drift")
    return _input_contents(inputs)


def _cases(contents: dict[str, bytes]) -> tuple[dict[str, XrrProject], dict[str, object]]:
    projects: dict[str, XrrProject] = {}
    cases: dict[str, object] = {}
    for stem in CONFIGURATION["cases"]:
        content = contents[stem]
        project = project_from_bytes(content)
        serialized = project_to_dict(project)
        if serialized != json.loads(content.decode("utf-8")):
            raise ValueError(f"model_project roundtrip drift: {stem}")
        projects[stem] = project
        cases[stem] = {
            "project": serialized,
            "project_field_order": [field.name for field in dataclass_fields(XrrProject)],
            "dataset_field_order": [field.name for field in dataclass_fields(DatasetProject)],
            "roundtrip_equal": True,
        }
    return projects, cases


def _transitions(project: XrrProject) -> dict[str, object]:
    dataset = project.datasets[0]
    active = with_active_dataset(project, dataset.dataset_id)
    second = replace(dataset, dataset_id="single-layer-copy")
    batch = with_batch_mode(replace(project, datasets=(dataset, second)), "joint")
    mask = tuple(index % 7 != 0 for index in range(len(dataset.fit_mask)))
    masked = with_dataset_fit_mask(project, dataset.dataset_id, mask)
    state = ProjectUiState(
        active_dataset_id=dataset.dataset_id,
        selected_candidate_ids=project.ui_state.selected_candidate_ids,
        expert_mode=True,
        workspace_splitter_sizes=(300, 700, 400),
        left_splitter_sizes=(260, 500),
        plot_tab_index=1,
    )
    workspace = with_workspace_state(project, state)
    included = sum(masked.datasets[0].fit_mask)
    return {
        "active_dataset": _json_value(active.ui_state),
        "batch_mode": {
            "before": project.batch_mode,
            "after": batch.batch_mode,
            "dataset_ids": [item.dataset_id for item in batch.datasets],
        },
        "fit_mask": {
            "dataset_id": masked.datasets[0].dataset_id,
            "included": included,
            "excluded": len(masked.datasets[0].fit_mask) - included,
        },
        "ui_state": _json_value(workspace.ui_state),
    }


def replay(context: object) -> dict[str, object]:
    """Build the normalized model-project artifact from declared replay inputs."""
    contents = _validate_context(context)
    projects, cases = _cases(contents)
    return {
        ARTIFACT: {
            "schema": "xrr-r22-model-project-reference-v1",
            "cases": cases,
            "transitions": _transitions(projects["single-layer"]),
        }
    }
