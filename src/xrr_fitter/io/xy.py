"""Canonical XRR text serialization and lossless provenance-aware import.

Generated curves use one stable two-column ASCII representation. Import keeps
the original bytes, raw rows, parse status, and derived row identities intact.
"""

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
    log_domain_mask,
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


def xy_bytes(two_theta_deg: object, intensity: object) -> bytes:
    """Serialize one finite curve using the canonical two-column XY format."""
    angles = np.asarray(two_theta_deg, dtype=float)
    intensities = np.asarray(intensity, dtype=float)
    valid = (
        angles.ndim == 1
        and intensities.ndim == 1
        and angles.size > 0
        and angles.shape == intensities.shape
        and np.all(np.isfinite(angles))
        and np.all(np.isfinite(intensities))
    )
    if not valid:
        raise ValueError("curve axes must be aligned nonempty finite vectors")
    rows = ["# 2theta_deg intensity"]
    rows.extend(f"{angle:.10f} {value:.16e}" for angle, value in zip(angles, intensities, strict=True))
    return ("\n".join(rows) + "\n").encode("ascii")


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
        bool(values is not None and len(values) >= required and all(np.isfinite(values[index]) for index in mapped))
        for values in parsed
    )
    return parseable, numeric


def _data_window(
    parseable: tuple[bool, ...],
    numeric: tuple[bool, ...],
    source_path: Path,
) -> tuple[int, int]:
    data_start = next(
        (index for index in range(max(0, len(numeric) - 1)) if numeric[index] and numeric[index + 1]),
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
    while end < len(records) and np.isfinite(records[cursor][0]) and records[end][0] == records[cursor][0]:
        end += 1
    return end


def _stable_midpoint(left: float, right: float) -> float:
    left, right = float(left), float(right)
    span = right - left
    if isfinite(span):
        return left + span / 2.0
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return 0.0
    return scale * ((left / scale) * 0.5 + (right / scale) * 0.5)


def _stable_median(values: object) -> float:
    array = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        median = float(np.median(array))
    if isfinite(median):
        return median
    ordered = np.sort(array)
    midpoint = ordered.size // 2
    if ordered.size % 2:
        return float(ordered[midpoint])
    return _stable_midpoint(ordered[midpoint - 1], ordered[midpoint])


def _ordinary_inverse_variance(
    intensities: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[float, float] | None:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        inverse_variance = 1.0 / sigmas**2
    if np.any(~np.isfinite(inverse_variance)):
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        total = np.sum(inverse_variance)
        weighted = np.sum(inverse_variance * intensities) / total
        sigma = np.sqrt(1.0 / total)
    if all(np.isfinite(value) for value in (weighted, sigma)):
        return float(weighted), float(sigma)
    return None


def _scaled_inverse_variance(
    intensities: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[float, float]:
    scale = float(np.min(sigmas))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        relative = scale / sigmas
        weights = relative**2
        total = np.sum(weights)
        intensity_scale = float(np.max(np.abs(intensities)))
        if intensity_scale == 0.0:
            weighted = 0.0
        else:
            normalized = intensities / intensity_scale
            normalized_mean = np.sum(weights * normalized) / total
            weighted = intensity_scale * np.clip(normalized_mean, -1.0, 1.0)
        sigma = scale / np.sqrt(total)
    if not all(np.isfinite(value) for value in (weighted, sigma)):
        raise ValueError("inverse-variance merge is not finite")
    return float(weighted), float(sigma)


def _merged_intensity(
    group: list[DataRecord],
    has_sigma: bool,
) -> tuple[float, float]:
    intensities = np.asarray([item[1] for item in group])
    sigmas = np.asarray([item[2] for item in group])
    if has_sigma and np.all(np.isfinite(sigmas) & (sigmas > 0.0)):
        # Keep the ordinary expression bit-stable when it is representable.
        # Extremely small finite sigmas overflow their inverse variance; scale
        # by the smallest sigma in that exceptional path instead of producing
        # inf/inf and a NaN merged intensity.
        ordinary = _ordinary_inverse_variance(intensities, sigmas)
        return ordinary if ordinary is not None else _scaled_inverse_variance(intensities, sigmas)
    return _stable_median(intensities), float("nan")


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
        resolution = _stable_median([item[3] for item in group]) if mapping.resolution is not None else float("nan")
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
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        high_values = intensity[eligible[-high_count:]] / normalization
    if not np.all(np.isfinite(high_values)):
        return 1e-8, ["高角归一化强度不可表示，R_floor 使用 1e-8"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        differences = np.diff(high_values)
        median = np.median(differences)
        mad = np.median(np.abs(differences - median))
        sigma_noise = mad / (0.67448975 * sqrt(2.0))
        floor = 3.0 * float(sigma_noise)
    if not isfinite(floor):
        return 1e-8, ["高角噪声估计不可表示，R_floor 使用 1e-8"]
    return max(1e-12, floor), []


def _derived_state(
    merged: _MergedRows,
    wavelength_a: float,
    angle_offset_deg: float,
    statuses: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, bool, list[str]]:
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
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        normalized = merged.intensity / normalization
    validation = (
        np.isfinite(merged.two_theta)
        & np.isfinite(merged.intensity)
        & theta_positive
        & log_domain_mask(normalized, r_floor)
    )
    ready = fit_ready(merged.two_theta, qz, validation)
    invalid_count = int(np.count_nonzero(~validation))
    if invalid_count:
        warnings.append(f"{invalid_count} 个派生点未通过校验")
    if "malformed" in statuses:
        warnings.append("数据区包含无法解析的行")
    if not ready:
        warnings.append("有效唯一点不足 30 或 qz 跨度为零")
    return qz, normalized, normalization, r_floor, validation, ready, warnings


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
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            normalized_sigma = intensity_sigma / normalization
        if not np.all(np.isfinite(normalized_sigma)):
            raise ValueError("normalized intensity uncertainty is not finite")
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
    qz, normalized, normalization, r_floor, validation, ready, warnings = _derived_state(
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
