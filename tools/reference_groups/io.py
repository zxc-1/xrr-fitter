"""Replay frozen R22 data and project inputs through the R23 I/O boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

import numpy as np

from xrr_fitter.io.project_codec import (
    project_from_bytes,
    project_to_bytes,
    project_to_dict,
)
from xrr_fitter.io.xy import read_xy_bytes
from xrr_fitter.model.data import DataColumnMapping, PreparedData, with_fit_mask


ARTIFACTS = ("golden/io.json", "golden/io.npz")
INPUTS = {
    "mo-si-periodic-data": (
        "bundled-example-data",
        "xrr_fitter/examples/mo-si-periodic.xy",
    ),
    "mo-si-periodic-project": (
        "bundled-example-project",
        "xrr_fitter/examples/mo-si-periodic.xrrproj.json",
    ),
    "single-layer-data": (
        "bundled-example-data",
        "xrr_fitter/examples/single-layer.xy",
    ),
    "single-layer-project": (
        "bundled-example-project",
        "xrr_fitter/examples/single-layer.xrrproj.json",
    ),
}
INPUT_ORDER = tuple(INPUTS)
CONFIGURATION = {
    "cases": ["single-layer", "mo-si-periodic"],
    "operations": [
        "read_xy",
        "duplicate_merge",
        "with_fit_mask",
        "save_load_roundtrip",
    ],
}


def _validate_input(value: object) -> bytes:
    try:
        expected_class, expected_path = INPUTS[value.input_id]
    except (AttributeError, KeyError) as error:
        raise ValueError("io input identity drift") from error
    if value.input_class != expected_class or value.path != expected_path:
        raise ValueError("io input identity drift")
    content = value.content
    if not isinstance(content, bytes):
        raise ValueError("io input content must be bytes")
    digest = hashlib.sha256(content).hexdigest()
    if value.size != len(content) or value.sha256 != digest:
        raise ValueError("io input size or hash drift")
    return content


def _validate_context(context: object) -> dict[str, bytes]:
    if context.group != "io":
        raise ValueError("io group drift")
    if tuple(context.artifacts) != ARTIFACTS:
        raise ValueError("io artifact drift")
    if context.configuration != CONFIGURATION:
        raise ValueError("io configuration drift")
    if tuple(context.seeds):
        raise ValueError("io seed drift")
    inputs = tuple(context.inputs)
    if tuple(value.input_id for value in inputs) != INPUT_ORDER:
        raise ValueError("io input set or order drift")
    return {value.input_id: _validate_input(value) for value in inputs}


def _column_mapping(value: DataColumnMapping) -> dict[str, object]:
    return {
        "two_theta": value.two_theta,
        "intensity": value.intensity,
        "intensity_sigma": value.intensity_sigma,
        "resolution": value.resolution,
        "resolution_kind": value.resolution_kind,
    }


def _summary(data: PreparedData) -> dict[str, object]:
    return {
        "source_sha256": data.source_sha256,
        "raw_row_count": len(data.raw_rows),
        "raw_parse_status": list(data.raw_parse_status),
        "source_row_groups": [list(group) for group in data.source_row_groups],
        "column_mapping": _column_mapping(data.column_mapping),
        "normalization": float(data.normalization),
        "r_floor": float(data.r_floor),
        "fit_ready": bool(data.fit_ready),
        "warnings": list(data.warnings),
    }


def _arrays(data: PreparedData, prefix: str) -> dict[str, np.ndarray]:
    result = {
        f"{prefix}_fit_mask": data.fit_mask,
        f"{prefix}_intensity_normalized": data.intensity_normalized,
        f"{prefix}_intensity_raw": data.intensity_raw,
        f"{prefix}_qz_a_inv": data.qz_a_inv,
        f"{prefix}_two_theta_deg": data.two_theta_deg,
        f"{prefix}_validation_mask": data.validation_mask,
    }
    optional = (
        ("intensity_sigma_raw", data.intensity_sigma_raw),
        ("intensity_sigma_normalized", data.intensity_sigma_normalized),
        ("resolution_raw", data.resolution_raw),
        ("sigma_q_a_inv", data.sigma_q_a_inv),
    )
    for name, value in optional:
        if value is not None:
            result[f"{prefix}_{name}"] = value
    return result


def _case(
    contents: dict[str, bytes],
    stem: str,
) -> tuple[object, PreparedData, bool]:
    project_content = contents[f"{stem}-project"]
    project = project_from_bytes(project_content)
    source = json.loads(project_content.decode("utf-8"))
    encoded = project_to_bytes(project)
    reloaded = project_from_bytes(encoded)
    roundtrip = encoded == project_content and project_to_dict(reloaded) == source
    dataset = reloaded.datasets[0]
    input_id = f"{stem}-data"
    data = read_xy_bytes(
        contents[input_id],
        source_path=PurePosixPath(INPUTS[input_id][1]),
        beam=dataset.beam,
        import_angle_offset_deg=dataset.import_angle_offset_deg,
        column_mapping=dataset.column_mapping,
    )
    return reloaded, data, roundtrip


def _duplicate_case(beam: object) -> tuple[bytes, PreparedData, PreparedData]:
    content = b"0.100 10.0 1.0\n0.100 14.0 2.0\n0.200 5.0 1.0\n"
    mapping = DataColumnMapping(two_theta=0, intensity=1, intensity_sigma=2)
    data = read_xy_bytes(
        content,
        source_path="duplicate-reference.xy",
        beam=beam,
        column_mapping=mapping,
    )
    masked = with_fit_mask(data, np.asarray([False, True], dtype=np.bool_))
    return content, data, masked


def replay(context: object) -> dict[str, object]:
    """Build the exact normalized R22 I/O artifacts from declared bytes."""
    contents = _validate_context(context)
    summaries: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {}
    roundtrip: dict[str, bool] = {}
    cases: dict[str, tuple[object, PreparedData]] = {}
    prefixes = {"single-layer": "single", "mo-si-periodic": "mo_si"}
    for stem in CONFIGURATION["cases"]:
        project, data, equal = _case(contents, stem)
        cases[stem] = (project, data)
        summaries[stem] = _summary(data)
        arrays.update(_arrays(data, prefixes[stem]))
        roundtrip[stem] = equal

    single_project, single_data = cases["single-layer"]
    duplicate_content, duplicate, duplicate_masked = _duplicate_case(
        single_project.datasets[0].beam
    )
    arrays.update(
        {
            "duplicate_fit_mask": duplicate.fit_mask,
            "duplicate_fit_mask_after_transition": duplicate_masked.fit_mask,
            "duplicate_intensity_normalized": duplicate.intensity_normalized,
            "duplicate_intensity_raw": duplicate.intensity_raw,
            "duplicate_intensity_sigma_raw": duplicate.intensity_sigma_raw,
            "duplicate_qz_a_inv": duplicate.qz_a_inv,
            "single_fit_mask_after_transition": with_fit_mask(
                single_data,
                np.asarray(
                    [index % 11 != 0 for index in range(single_data.fit_mask.size)],
                    dtype=np.bool_,
                ),
            ).fit_mask,
        }
    )
    summary = {
        "schema": "xrr-r22-io-reference-v1",
        "cases": summaries,
        "duplicate_merge": {
            "input_sha256": hashlib.sha256(duplicate_content).hexdigest(),
            "column_mapping": _column_mapping(duplicate.column_mapping),
            "raw_parse_status": list(duplicate.raw_parse_status),
            "source_row_groups": [list(group) for group in duplicate.source_row_groups],
            "warnings": list(duplicate.warnings),
        },
        "project_roundtrip": roundtrip,
    }
    return {ARTIFACTS[0]: summary, ARTIFACTS[1]: arrays}
