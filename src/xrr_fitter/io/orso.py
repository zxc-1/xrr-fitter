"""ORSO ``.ort`` serialization for a single selected dataset.

Imports only orsopy from third parties (registered gate) plus in-layer io/model
code. Confidence and reproducibility payloads travel in two frozen extension
keys, ``xrr_fitter.confidence`` and ``xrr_fitter.reduction``; orsopy's
``Reduction`` dataclass rejects arbitrary keys, so seed/fit-config data cannot
live under ``reduction`` proper. Non-finite / non-positive-Qz rows and rows
lacking aligned finite fit outputs are dropped and counted in the extension.
"""

from __future__ import annotations

import datetime
import io
from numbers import Real

import numpy as np
from jsonschema import ValidationError
from orsopy import fileio
from orsopy.fileio.base import ContentHash, _read_header_data, _validate_header_data

from xrr_fitter.io.codec_declarations import _fit_config_to_dict
from xrr_fitter.io.export_tables import DatasetExportData
from xrr_fitter.version import __version__ as PACKAGE_VERSION

# Measurement start date is not recorded anywhere in the export context; a fixed
# epoch sentinel plus an explicit comment marks it as unavailable rather than
# fabricating a plausible date (剩余风险 L372 -> 修正 11).
START_DATE_SENTINEL = datetime.datetime(1970, 1, 1)
EXPERIMENT_COMMENT = "measurement start date and instrument model unavailable in export context"
INSTRUMENT_SENTINEL = "unknown"
OWNER_AFFILIATION = "XRR-Fitter automated export"
EXCLUDED_ROW_REASON = "non_finite_or_nonpositive_export_row"
COVARIANCE_ABSENT_REASON = "covariance not estimated for this fit result"


def orso_bytes(context: DatasetExportData, *, covariance: np.ndarray | None) -> bytes:
    """Serialize one selected dataset to a validated ORSO ``.ort`` document."""
    if not isinstance(context, DatasetExportData):
        raise TypeError("context must be DatasetExportData")
    data = context.data
    selected = context.selected
    result = context.result  # property raises if no fit result exists
    mask = _export_mask(context)

    columns, matrix = _data_segment(data, mask)
    info = fileio.Orso(
        data_source=_data_source(context, mask),
        reduction=_reduction(context),
        columns=columns,
        **_extensions(context, result, selected, mask, covariance),
    )
    dataset = fileio.OrsoDataset(info=info, data=matrix)
    buffer = io.StringIO()
    fileio.save_orso([dataset], buffer)
    text = buffer.getvalue()
    _validate_serialized_header(text)
    return text.encode("utf-8")


def _validate_serialized_header(text: str) -> None:
    # 唯一 schema 校验点（spec 三层验证之层 2），也是 orsopy 升级时的首个检查点。
    # orsopy 1.2.3 未提供公开 header 校验 API；pyproject 精确 pin 该版本，
    # 此处集中封装私有 ``_read_header_data`` / ``_validate_header_data`` 依赖。
    # 校验对象是「序列化后再由 orsopy 自身 reader 解析回来」的 header：
    # ``Orso.to_dict()`` 直出 ``datetime`` 对象，会被 jsonschema 的字符串日期约束拒绝，
    # 故改走 ``_read_header_data`` 得到字符串化 header 再校验。失败直接抛、不降级，
    # ``publish_export_run`` 会清理由当前调用持有的 private partial 目录。
    header_dicts, _rows, _version = _read_header_data(io.StringIO(text))
    try:
        _validate_header_data(header_dicts)
    except ValidationError as error:
        raise ValueError(f"ORSO schema validation failed: {error.message}") from error


def _export_mask(context: DatasetExportData) -> np.ndarray:
    data = context.data
    selected = context.selected
    mask = np.asarray(data.validation_mask, dtype=bool).copy()
    qz = np.asarray(data.qz_a_inv, dtype=float)
    mask &= np.isfinite(qz) & (qz > 0.0)
    arrays = [
        data.intensity_normalized,
        selected.qz_a_inv,
        selected.model_normalized,
        selected.log_residuals_decades,
    ]
    if data.intensity_sigma_normalized is not None:
        arrays.append(data.intensity_sigma_normalized)
    if data.sigma_q_a_inv is not None:
        arrays.append(data.sigma_q_a_inv)
    for values in arrays:
        mask &= np.isfinite(np.asarray(values, dtype=float))
    if not np.any(mask):
        raise ValueError("ORSO export has no finite aligned rows")
    return mask


def _data_segment(data, mask):
    """Build the ORSO column spec and the row-filtered numeric matrix.

    ORSO 反射率 schema 通过 ``prefixItems`` 把前四列位置固定为 ``Qz, R, sR, sQz``。
    因此只有在实测反射率不确定度 ``sR`` 存在时，才把逐点 ``sQz`` 作为第 4
    个标准误差列写入；否则不臆造 ``sR``，逐点 Q 分辨率由 ``xrr_fitter.reduction``
    扩展段承载。拟合模型曲线与残差不是实测反射率，改由 ``xrr_fitter.model``
    扩展段承载，而非数据列。
    """
    qz = np.asarray(data.qz_a_inv, dtype=float)[mask]
    reflectivity = np.asarray(data.intensity_normalized, dtype=float)[mask]
    columns = [fileio.Column(name="Qz", unit="1/angstrom"), fileio.Column(name="R")]
    arrays = [qz, reflectivity]
    sigma = data.intensity_sigma_normalized
    if sigma is not None:
        columns.append(fileio.ErrorColumn(error_of="R", error_type="uncertainty", value_is="sigma"))
        arrays.append(np.asarray(sigma, dtype=float)[mask])
        if data.sigma_q_a_inv is not None:
            columns.append(fileio.ErrorColumn(error_of="Qz", error_type="resolution", value_is="sigma"))
            arrays.append(np.asarray(data.sigma_q_a_inv, dtype=float)[mask])
    return columns, np.column_stack(arrays)


def _data_source(context, mask):
    """Assemble the ORSO ``data_source`` from honest project fields.

    ``incident_angle`` spans the grazing angle range of the surviving rows
    (``two_theta / 2 + offset``); title/sample reuse the dataset display name.
    """
    data = context.data
    dataset = context.dataset
    theta_deg = np.asarray(data.two_theta_deg, dtype=float)[mask] / 2.0 + data.import_angle_offset_deg
    incident_angle = fileio.ValueRange(min=float(theta_deg.min()), max=float(theta_deg.max()), unit="deg")
    wavelength = fileio.Value(magnitude=float(data.beam.effective_wavelength_a), unit="angstrom")
    settings = fileio.InstrumentSettings(incident_angle=incident_angle, wavelength=wavelength)
    source = fileio.File(
        file=data.source_path.name,
        hash=ContentHash(digest=data.source_sha256, algorithm="sha256"),
    )
    measurement = fileio.Measurement(
        instrument_settings=settings,
        data_files=[source],
    )
    experiment = fileio.Experiment(
        title=dataset.display_name,
        instrument=INSTRUMENT_SENTINEL,
        start_date=START_DATE_SENTINEL,
        probe="x-ray",
        comment=EXPERIMENT_COMMENT,
    )
    owner = fileio.Person(name="XRR-Fitter", affiliation=OWNER_AFFILIATION)
    sample = fileio.Sample(name=dataset.display_name)
    return fileio.DataSource(owner=owner, experiment=experiment, sample=sample, measurement=measurement)


def _reduction(context):
    """Minimal ORSO ``Reduction``; seed/config payload lives in the extension.

    orsopy's ``Reduction`` rejects unknown keys, so ``timestamp`` is omitted
    (no honest export-time value) and reproducibility data is attached under
    ``xrr_fitter.reduction`` instead.
    """
    software = fileio.Software(name="XRR-Fitter", version=PACKAGE_VERSION)
    return fileio.Reduction(software=software)


def _extensions(context, result, selected, mask, covariance):
    """Three frozen extension keys: confidence payload, reproducibility payload,
    and the fit model curve.

    ``xrr_fitter.model`` carries the fitted Qz axis, model reflectivity, and
    log-decade residuals (row-filtered by ``mask`` to align with the exported
    data rows). These are fit outputs, not measured reflectivity, so they cannot
    occupy ORSO data columns (前四列位置被 schema 固定为 ``Qz, R, sR, sQz``) and
    live here instead. The fitted Qz is explicit because an optimized angle
    offset can make it differ from the imported data Qz column.
    """
    uncertainty = context.selected_uncertainty
    covariance_names = _covariance_names(uncertainty)
    selected_covariance = (
        _validated_covariance(covariance, covariance_names)
        if uncertainty is not None and covariance is not None
        else None
    )
    confidence = {
        "class_name": result.confidence.name,
        "display": result.confidence.value,
        "reason_codes": list(result.classification_evidence),
        "parameters": [
            {"name": p.name, "value": p.value, "lower": p.lower, "upper": p.upper} for p in selected.parameters
        ],
        "error_bars": _error_bars(uncertainty),
        "excluded_rows": {"count": int(np.count_nonzero(~mask)), "reason": EXCLUDED_ROW_REASON},
    }
    if selected_covariance is None:
        confidence["covariance_absent_reason"] = context.uncertainty_absent_reason or COVARIANCE_ABSENT_REASON
    else:
        confidence["covariance"] = {
            "names": covariance_names,
            "matrix": selected_covariance.tolist(),
        }
    reduction = {
        "service_seed_tree_version": context.replay_identity.service_seed_tree_version,
        "project_master_seed": context.project.master_seed,
        "algorithm_version": str(context.project.algorithm_version),
        "schema_version": int(context.project.schema_version),
        "fit_config": _fit_config_to_dict(context.project.fit_config),
        "beam": _beam_payload(context.data.beam),
    }
    pointwise_resolution = _pointwise_resolution_payload(context.data, mask)
    if pointwise_resolution is not None:
        reduction["pointwise_resolution"] = pointwise_resolution
    model = {
        "qz_a_inv": np.asarray(selected.qz_a_inv, dtype=float)[mask].tolist(),
        "reflectivity": np.asarray(selected.model_normalized, dtype=float)[mask].tolist(),
        "residual_decades": np.asarray(selected.log_residuals_decades, dtype=float)[mask].tolist(),
    }
    return {
        "xrr_fitter.confidence": confidence,
        "xrr_fitter.reduction": reduction,
        "xrr_fitter.model": model,
    }


def _pointwise_resolution_payload(data, mask):
    sigma_q = data.sigma_q_a_inv
    if sigma_q is None or data.intensity_sigma_normalized is not None:
        return None
    return {
        "error_of": "Qz",
        "error_type": "resolution",
        "value_is": "sigma",
        "unit": "1/angstrom",
        "values": np.asarray(sigma_q, dtype=float)[mask].tolist(),
    }


def _beam_payload(beam):
    payload = {
        "kind": beam.kind,
        "effective_wavelength_a": beam.effective_wavelength_a,
    }
    if beam.kind == "mixed_kalpha":
        payload.update(
            {
                "wavelength_1_a": beam.wavelength_1_a,
                "wavelength_2_a": beam.wavelength_2_a,
                "intensity_ratio_21": beam.intensity_ratio_21,
            }
        )
    return payload


def _error_bars(uncertainty):
    if uncertainty is None:
        return []
    return [_error_bar(interval) for interval in uncertainty.bootstrap_intervals]


def _finite_bound(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and bool(np.isfinite(value))


def _error_bar(interval):
    if not isinstance(interval, (tuple, list)) or len(interval) != 3:
        raise ValueError("bootstrap intervals must contain name, lower, and upper")
    name, lower, upper = interval
    if not isinstance(name, str) or not name.strip():
        raise ValueError("bootstrap intervals must contain finite ordered bounds")
    if not _finite_bound(lower) or not _finite_bound(upper) or lower > upper:
        raise ValueError("bootstrap intervals must contain finite ordered bounds")
    return {"name": name, "lower": float(lower), "upper": float(upper)}


def _covariance_names(uncertainty):
    return [] if uncertainty is None else list(uncertainty.correlation_names)


def _validated_covariance(covariance, names):
    matrix = np.asarray(covariance, dtype=float)
    expected = (len(names), len(names))
    structurally_valid = (
        matrix.ndim == 2
        and matrix.shape == expected
        and np.all(np.isfinite(matrix))
        and np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)
    )
    if not structurally_valid:
        raise ValueError(
            "covariance must be a finite symmetric positive-semidefinite square matrix matching correlation names"
        )
    symmetric = (matrix + matrix.T) / 2.0
    if matrix.size:
        eigenvalues = np.linalg.eigvalsh(symmetric)
        tolerance = 1e-10 * float(np.max(np.abs(symmetric)))
        if float(np.min(eigenvalues)) < -tolerance:
            raise ValueError(
                "covariance must be a finite symmetric positive-semidefinite square matrix matching correlation names"
            )
    return symmetric
