"""Lossless XRR text import with stable source provenance."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil, isfinite, sqrt
from pathlib import Path

import numpy as np

from xrr_fitter.model.data import (
    BeamSpec,
    DataColumnMapping,
    PreparedData,
    fit_ready,
    qz_from_two_theta,
)
from xrr_fitter.model.instrument import resolution_to_sigma_q


DataRecord = tuple[float, float, float, float, int]


@dataclass(frozen=True, slots=True)
class _MergedRows:
    two_theta: np.ndarray
    intensity: np.ndarray
    sigmas: np.ndarray
    resolutions: np.ndarray
    row_groups: tuple[tuple[int, ...], ...]


def _numeric_columns(row: str) -> list[float] | None:
    values: list[float] = []
    for token in row.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            return None
    return values or None


def _mapped_indices(mapping: DataColumnMapping) -> tuple[int, ...]:
    values = (
        mapping.two_theta,
        mapping.intensity,
        mapping.intensity_sigma,
        mapping.resolution,
    )
    return tuple(value for value in values if value is not None)


def _row_flags(
    parsed: tuple[list[float] | None, ...],
    mapping: DataColumnMapping,
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    mapped = _mapped_indices(mapping)
    required = max(mapped) + 1
    parseable = tuple(values is not None and len(values) >= required for values in parsed)
    numeric = tuple(
        bool(
            values is not None
            and len(values) >= required
            and all(np.isfinite(values[index]) for index in mapped)
        )
        for values in parsed
    )
    return parseable, numeric


def _data_window(
    parseable: tuple[bool, ...],
    numeric: tuple[bool, ...],
    source_path: Path,
) -> tuple[int, int]:
    data_start = next(
        (
            index
            for index in range(max(0, len(numeric) - 1))
            if numeric[index] and numeric[index + 1]
        ),
        None,
    )
    if data_start is None:
        raise ValueError(f"no consecutive numeric data rows: {source_path}")
    data_end = max(index for index in range(data_start, len(parseable)) if parseable[index])
    return data_start, data_end


def _optional_value(
    values: list[float],
    column: int | None,
    raw_index: int,
    *,
    resolution: bool,
) -> float:
    if column is None:
        return float("nan")
    value = values[column]
    valid = np.isfinite(value) and (value >= 0.0 if resolution else value > 0.0)
    if not valid:
        label = "resolution" if resolution else "intensity uncertainty"
        raise ValueError(f"invalid {label} at raw row {raw_index}")
    return value


def _record(
    values: list[float],
    mapping: DataColumnMapping,
    raw_index: int,
) -> DataRecord:
    return (
        values[mapping.two_theta],
        values[mapping.intensity],
        _optional_value(
            values,
            mapping.intensity_sigma,
            raw_index,
            resolution=False,
        ),
        _optional_value(
            values,
            mapping.resolution,
            raw_index,
            resolution=True,
        ),
        raw_index,
    )


def _collect_rows(
    parsed: tuple[list[float] | None, ...],
    parseable: tuple[bool, ...],
    data_start: int,
    data_end: int,
    mapping: DataColumnMapping,
) -> tuple[tuple[str, ...], list[DataRecord]]:
    statuses: list[str] = []
    records: list[DataRecord] = []
    for raw_index, values in enumerate(parsed):
        if raw_index < data_start:
            status = "header"
        elif raw_index > data_end:
            status = "footer"
        elif not parseable[raw_index]:
            status = "malformed"
        else:
            status = "data"
        statuses.append(status)
        if status == "data":
            assert values is not None
            records.append(_record(values, mapping, raw_index))
    return tuple(statuses), records


def _sort_key(record: DataRecord) -> tuple[bool, float, int]:
    finite = bool(np.isfinite(record[0]))
    return not finite, record[0] if finite else 0.0, record[4]


def _group_end(records: list[DataRecord], cursor: int) -> int:
    end = cursor + 1
    while (
        end < len(records)
        and np.isfinite(records[cursor][0])
        and records[end][0] == records[cursor][0]
    ):
        end += 1
    return end


def _merged_intensity(
    group: list[DataRecord],
    has_sigma: bool,
) -> tuple[float, float]:
    intensities = np.asarray([item[1] for item in group])
    sigmas = np.asarray([item[2] for item in group])
    if has_sigma and np.all(np.isfinite(sigmas) & (sigmas > 0.0)):
        inverse_variance = 1.0 / sigmas**2
        weighted = np.sum(inverse_variance * intensities) / np.sum(inverse_variance)
        sigma = np.sqrt(1.0 / np.sum(inverse_variance))
        return float(weighted), float(sigma)
    return float(np.median(intensities)), float("nan")


def _merge_records(
    records: list[DataRecord],
    mapping: DataColumnMapping,
) -> _MergedRows:
    records.sort(key=_sort_key)
    angles: list[float] = []
    intensities: list[float] = []
    sigmas: list[float] = []
    resolutions: list[float] = []
    row_groups: list[tuple[int, ...]] = []
    cursor = 0
    while cursor < len(records):
        end = _group_end(records, cursor)
        group = records[cursor:end]
        intensity, sigma = _merged_intensity(
            group,
            mapping.intensity_sigma is not None,
        )
        resolution = (
            float(np.median([item[3] for item in group]))
            if mapping.resolution is not None
            else float("nan")
        )
        angles.append(float(group[0][0]))
        intensities.append(intensity)
        sigmas.append(sigma)
        resolutions.append(resolution)
        row_groups.append(tuple(item[4] for item in group))
        cursor = end
    return _MergedRows(
        two_theta=np.asarray(angles, dtype=float),
        intensity=np.asarray(intensities, dtype=float),
        sigmas=np.asarray(sigmas, dtype=float),
        resolutions=np.asarray(resolutions, dtype=float),
        row_groups=tuple(row_groups),
    )


def _normalization(
    intensity: np.ndarray,
    base_valid: np.ndarray,
) -> tuple[float, list[str]]:
    positive_indices = np.flatnonzero(base_valid & (intensity > 0.0))
    if positive_indices.size:
        low_count = min(
            positive_indices.size,
            max(20, ceil(0.10 * positive_indices.size)),
        )
        value = float(np.percentile(intensity[positive_indices[:low_count]], 95))
        warnings: list[str] = []
    else:
        value = 1.0
        warnings = ["没有正强度点，归一化因子使用 1"]
    if not isfinite(value) or value <= 0.0:
        return 1.0, [*warnings, "归一化因子无效，使用 1"]
    return value, warnings


def _noise_floor(
    intensity: np.ndarray,
    base_valid: np.ndarray,
    normalization: float,
) -> tuple[float, list[str]]:
    eligible = np.flatnonzero(base_valid)
    high_count = ceil(0.20 * eligible.size)
    if high_count < 20:
        return 1e-8, ["高角点不足，R_floor 使用 1e-8"]
    high_values = intensity[eligible[-high_count:]] / normalization
    differences = np.diff(high_values)
    median = np.median(differences)
    mad = np.median(np.abs(differences - median))
    sigma_noise = mad / (0.67448975 * sqrt(2.0))
    return max(1e-12, 3.0 * float(sigma_noise)), []


def _derived_state(
    merged: _MergedRows,
    wavelength_a: float,
    angle_offset_deg: float,
    statuses: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, float, float, bool, list[str]]:
    qz, theta_positive = qz_from_two_theta(
        merged.two_theta,
        wavelength_a,
        angle_offset_deg,
    )
    base_valid = theta_positive & np.isfinite(merged.intensity)
    normalization, warnings = _normalization(merged.intensity, base_valid)
    r_floor, floor_warnings = _noise_floor(
        merged.intensity,
        base_valid,
        normalization,
    )
    warnings.extend(floor_warnings)
    normalized = merged.intensity / normalization
    validation = (
        np.isfinite(merged.two_theta)
        & np.isfinite(merged.intensity)
        & theta_positive
        & (normalized + r_floor > 0.0)
    )
    ready = fit_ready(merged.two_theta, qz, validation)
    invalid_count = int(np.count_nonzero(~validation))
    if invalid_count:
        warnings.append(f"{invalid_count} 个派生点未通过校验")
    if "malformed" in statuses:
        warnings.append("数据区包含无法解析的行")
    if not ready:
        warnings.append("有效唯一点不足 30 或 qz 跨度为零")
    return qz, normalized, normalization, r_floor, ready, warnings


def _optional_arrays(
    merged: _MergedRows,
    mapping: DataColumnMapping,
    normalization: float,
    wavelength_a: float,
    angle_offset_deg: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    intensity_sigma = None
    normalized_sigma = None
    if mapping.intensity_sigma is not None:
        intensity_sigma = np.asarray(merged.sigmas, dtype=float)
        normalized_sigma = intensity_sigma / normalization
    resolution = None
    sigma_q = None
    if mapping.resolution is not None:
        resolution = np.asarray(merged.resolutions, dtype=float)
        assert mapping.resolution_kind is not None
        sigma_q = resolution_to_sigma_q(
            merged.two_theta,
            resolution,
            mapping.resolution_kind,
            wavelength_a,
            angle_offset_deg,
        )
    return intensity_sigma, normalized_sigma, resolution, sigma_q


def _validated_request(
    beam: BeamSpec,
    import_angle_offset_deg: float,
    column_mapping: DataColumnMapping | None,
) -> DataColumnMapping:
    if not isinstance(beam, BeamSpec):
        raise TypeError("beam must be a BeamSpec")
    if not isfinite(import_angle_offset_deg):
        raise ValueError("import_angle_offset_deg must be finite")
    mapping = column_mapping or DataColumnMapping()
    if not isinstance(mapping, DataColumnMapping):
        raise TypeError("column_mapping must be a DataColumnMapping or None")
    return mapping


def _prepare_xy(
    source_path: Path,
    source_bytes: bytes,
    beam: BeamSpec,
    import_angle_offset_deg: float,
    mapping: DataColumnMapping,
) -> PreparedData:
    raw_rows = tuple(source_bytes.decode("utf-8-sig").splitlines())
    parsed = tuple(_numeric_columns(row) for row in raw_rows)
    parseable, numeric = _row_flags(parsed, mapping)
    start, end = _data_window(parseable, numeric, source_path)
    statuses, records = _collect_rows(parsed, parseable, start, end, mapping)
    merged = _merge_records(records, mapping)
    qz, normalized, normalization, r_floor, ready, warnings = _derived_state(
        merged,
        beam.effective_wavelength_a,
        import_angle_offset_deg,
        statuses,
    )
    intensity_sigma, normalized_sigma, resolution, sigma_q = _optional_arrays(
        merged,
        mapping,
        normalization,
        beam.effective_wavelength_a,
        import_angle_offset_deg,
    )
    validation = (
        np.isfinite(merged.two_theta)
        & np.isfinite(merged.intensity)
        & (qz > 0.0)
        & (normalized + r_floor > 0.0)
    )
    return PreparedData(
        source_path=source_path,
        source_sha256=sha256(source_bytes).hexdigest(),
        raw_rows=raw_rows,
        raw_parse_status=statuses,
        source_row_groups=merged.row_groups,
        beam=beam,
        import_angle_offset_deg=import_angle_offset_deg,
        two_theta_deg=merged.two_theta,
        intensity_raw=merged.intensity,
        intensity_sigma_raw=intensity_sigma,
        resolution_raw=resolution,
        qz_a_inv=qz,
        intensity_normalized=normalized,
        intensity_sigma_normalized=normalized_sigma,
        sigma_q_a_inv=sigma_q,
        validation_mask=validation,
        fit_mask=validation,
        normalization=normalization,
        r_floor=r_floor,
        fit_ready=ready,
        warnings=tuple(warnings),
        column_mapping=mapping,
    )


def read_xy_bytes(
    content: bytes,
    *,
    source_path: str | Path,
    beam: BeamSpec,
    import_angle_offset_deg: float = 0.0,
    column_mapping: DataColumnMapping | None = None,
) -> PreparedData:
    """Parse already-bound source bytes through the authoritative importer."""
    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    mapping = _validated_request(beam, import_angle_offset_deg, column_mapping)
    return _prepare_xy(
        Path(source_path),
        content,
        beam,
        import_angle_offset_deg,
        mapping,
    )


def read_xy(
    path: str | Path,
    beam: BeamSpec,
    import_angle_offset_deg: float = 0.0,
    column_mapping: DataColumnMapping | None = None,
) -> PreparedData:
    """Read one UTF-8 XRR curve and preserve every raw row and source byte hash."""
    source_path = Path(path)
    mapping = _validated_request(beam, import_angle_offset_deg, column_mapping)
    return _prepare_xy(
        source_path,
        source_path.read_bytes(),
        beam,
        import_angle_offset_deg,
        mapping,
    )
