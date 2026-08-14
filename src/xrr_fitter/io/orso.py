"""ORSO ``.ort`` serialization for a single selected dataset.

Imports only orsopy from third parties (registered gate) plus in-layer io/model
code. Confidence and reproducibility payloads travel in two frozen extension
keys, ``xrr_fitter.confidence`` and ``xrr_fitter.reduction``; orsopy's
``Reduction`` dataclass rejects arbitrary keys, so seed/fit-config data cannot
live under ``reduction`` proper. Non-finite / non-positive-Qz rows are dropped
via ``PreparedData.validation_mask`` (修正 8) and counted in the extension.
"""

from __future__ import annotations

import datetime
import io

import numpy as np
from orsopy import fileio
from orsopy.fileio.base import _read_header_data, _validate_header_data

from xrr_fitter.io.export_tables import DatasetExportData
from xrr_fitter.io.project_codec import project_to_dict

# Measurement start date is not recorded anywhere in the export context; a fixed
# epoch sentinel plus an explicit comment marks it as unavailable rather than
# fabricating a plausible date (剩余风险 L372 -> 修正 11).
START_DATE_SENTINEL = datetime.datetime(1970, 1, 1)
EXPERIMENT_COMMENT = "measurement start date and instrument model unavailable in export context"
INSTRUMENT_SENTINEL = "unknown"
OWNER_AFFILIATION = "XRR-Fitter automated export"
EXCLUDED_ROW_REASON = "non_finite_or_nonpositive_qz"
COVARIANCE_ABSENT_REASON = "covariance not estimated for this fit result"


def orso_bytes(context: DatasetExportData, *, covariance: np.ndarray | None) -> bytes:
    """Serialize one selected dataset to a validated ORSO ``.ort`` document."""
    if not isinstance(context, DatasetExportData):
        raise TypeError("context must be DatasetExportData")
    data = context.data
    selected = context.selected
    result = context.result  # property raises if no fit result exists
    mask = np.asarray(data.validation_mask, dtype=bool)

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
    # 唯一 schema 校验点（spec 三层验证之层 2），也是 orsopy 升级时的首个检查点。
    # 校验对象是「序列化后再由 orsopy 自身 reader 解析回来」的 header：
    # ``Orso.to_dict()`` 直出 ``datetime`` 对象，会被 jsonschema 的字符串日期约束拒绝，
    # 故改走 ``_read_header_data`` 得到字符串化 header 再校验。失败直接抛、不降级
    # （``publish_export_run`` 此刻尚未建目录，天然不留半成品）。
    header_dicts, _rows, _version = _read_header_data(io.StringIO(text))
    _validate_header_data(header_dicts)
    return text.encode("utf-8")


def _data_segment(data, mask):
    """Build the ORSO column spec and the row-filtered numeric matrix.

    ORSO 反射率 schema 通过 ``prefixItems`` 把前四列位置固定为 ``Qz, R, sR, sQz``：
    位置 2/3 若出现非误差列会直接校验失败。因此这里只发出诚实的约减反射率列——
    ``Qz, R``，以及导入携带 ``intensity_sigma_normalized`` 时的 ``sR``——绝不臆造
    ``sQz`` 分辨率列。拟合模型曲线与残差不是实测反射率，改由 ``xrr_fitter.model``
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
    measurement = fileio.Measurement(instrument_settings=settings, data_files=[data.source_path.name])
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
    software = fileio.Software(name="XRR-Fitter", version=str(context.project.algorithm_version))
    return fileio.Reduction(software=software)


def _extensions(context, result, selected, mask, covariance):
    """Three frozen extension keys: confidence payload, reproducibility payload,
    and the fit model curve.

    ``xrr_fitter.model`` carries the model reflectivity and log-decade residuals
    (row-filtered by ``mask`` to align with the exported data rows). These are
    fit outputs, not measured reflectivity, so they cannot occupy ORSO data
    columns (前四列位置被 schema 固定为 ``Qz, R, sR, sQz``) and live here instead.
    """
    confidence = {
        "class_name": result.confidence.name,
        "display": result.confidence.value,
        "reason_codes": list(result.classification_evidence),
        "parameters": [
            {"name": p.name, "value": p.value, "lower": p.lower, "upper": p.upper} for p in selected.parameters
        ],
        "error_bars": _error_bars(result.uncertainty),
        "excluded_rows": {"count": int(np.count_nonzero(~mask)), "reason": EXCLUDED_ROW_REASON},
    }
    if covariance is None:
        confidence["covariance_absent_reason"] = COVARIANCE_ABSENT_REASON
    else:
        confidence["covariance"] = {
            "names": _covariance_names(result.uncertainty, selected),
            "matrix": np.asarray(covariance, dtype=float).tolist(),
        }
    reduction = {
        "service_seed_tree_version": context.replay_identity.service_seed_tree_version,
        "project_master_seed": context.project.master_seed,
        "algorithm_version": str(context.project.algorithm_version),
        "schema_version": int(context.project.schema_version),
        "fit_config": project_to_dict(context.project)["fit_config"],
    }
    model = {
        "reflectivity": np.asarray(selected.model_normalized, dtype=float)[mask].tolist(),
        "residual_decades": np.asarray(selected.log_residuals_decades, dtype=float)[mask].tolist(),
    }
    return {
        "xrr_fitter.confidence": confidence,
        "xrr_fitter.reduction": reduction,
        "xrr_fitter.model": model,
    }


def _error_bars(uncertainty):
    if uncertainty is None:
        return []
    return [{"name": name, "lower": lower, "upper": upper} for name, lower, upper in uncertainty.bootstrap_intervals]


def _covariance_names(uncertainty, selected):
    if uncertainty is not None and uncertainty.correlation_names:
        return list(uncertainty.correlation_names)
    return [p.name for p in selected.parameters]
