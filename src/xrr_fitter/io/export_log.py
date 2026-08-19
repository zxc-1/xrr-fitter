"""Deterministic text-log serialization for one selected export candidate."""

from __future__ import annotations

import json

import numpy as np

from xrr_fitter.io.export_tables import DatasetExportData
from xrr_fitter.model.instrument import PhysicsDiagnostic


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _diagnostic_qz_range(
    context: DatasetExportData,
    indices: tuple[int, ...],
) -> str:
    size = context.data.qz_a_inv.size
    valid = tuple(index for index in indices if 0 <= index < size)
    if not valid:
        return "[]"
    qz = context.data.qz_a_inv[list(valid)]
    return f"[{float(np.min(qz)):.12g},{float(np.max(qz)):.12g}]"


def _diagnostic_line(
    context: DatasetExportData,
    diagnostic: PhysicsDiagnostic,
) -> str:
    indices = _compact_json(diagnostic.point_indices)
    qz_range = _diagnostic_qz_range(context, diagnostic.point_indices)
    return f"{diagnostic.code}: {diagnostic.message}; full_data_indices={indices}; qz_a_inv_range={qz_range}"


def _persisted_diagnostics(
    context: DatasetExportData,
) -> tuple[PhysicsDiagnostic, ...]:
    uncertainty = context.selected_uncertainty
    if uncertainty is None:
        return context.selected.diagnostics
    return (*context.selected.diagnostics, *uncertainty.diagnostics)


def _rejected_surface_oxide(
    context: DatasetExportData,
    diagnostics: tuple[PhysicsDiagnostic, ...],
) -> bool:
    residual = any(value.code == "surface_thin_layer_residual" for value in diagnostics)
    return context.matching_surface_oxide_rejection and residual


def run_log_bytes(context: DatasetExportData) -> bytes:
    """Serialize stable warnings, seed lineage, stages, and diagnostics."""
    result = context.result
    identity = context.replay_identity
    report = context.selected_uncertainty
    mcmc = None if report is None else report.mcmc
    lines = [
        f"dataset_id: {context.dataset.dataset_id}",
        f"confidence: {result.confidence.value}",
        f"candidate_count: {len(result.candidates)}",
        f"project_master_seed: {context.project.master_seed}",
        f"service_seed_tree_version: {identity.service_seed_tree_version}",
        f"independent_root_child: {identity.independent_root_child}",
        f"joint_root_child: {identity.joint_root_child}",
        f"optimizer_child_seeds: {_compact_json(result.child_seeds)}",
        f"mcmc_child_seed: {None if mcmc is None else mcmc.child_seed}",
    ]
    if context.uncertainty_absent_reason is not None:
        lines.append(f"uncertainty_absent_reason: {context.uncertainty_absent_reason}")
    lines.extend(f"warning: {value}" for value in result.warnings)
    lines.extend(
        "stage "
        f"{stage.stage}: candidate_ids={_compact_json(stage.candidate_ids)}; "
        f"best_objective={stage.best_objective}; total_nfev={stage.total_nfev}; "
        f"stop_reasons={_compact_json(stage.stop_reasons)}"
        for stage in result.stage_summaries
    )
    diagnostics = _persisted_diagnostics(context)
    lines.extend(_diagnostic_line(context, value) for value in diagnostics)
    if _rejected_surface_oxide(context, diagnostics):
        lines.append("疑似缺失自然氧化层（此前已拒绝建议）")
    return ("\n".join(lines) + "\n").encode("utf-8")
