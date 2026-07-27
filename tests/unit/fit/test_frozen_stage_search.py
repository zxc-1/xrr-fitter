"""Frozen staged-search replay contracts.

These focused checks replay the committed single-layer input through the live
R23 compiler and A-through-E search. Stage-A auditing, candidate order, work
counts, progress framing, and checkpoint identity remain frozen together.
The fixtures read reference input bytes but never import R22 production code.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.io.project_codec import project_from_bytes
from xrr_fitter.io.xy import read_xy_bytes
from xrr_fitter.model.data import with_fit_mask
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.parameters import ParameterSetting


def _frozen_single_layer_problem():
    root = Path(__file__).resolve().parents[3]
    inputs = root / "verification" / "r22" / "reference" / "xrr_fitter" / "examples"
    project = project_from_bytes((inputs / "single-layer.xrrproj.json").read_bytes())
    dataset = project.datasets[0]
    data = read_xy_bytes(
        (inputs / "single-layer.xy").read_bytes(),
        source_path=PurePosixPath("xrr_fitter/examples/single-layer.xy"),
        beam=dataset.beam,
        import_angle_offset_deg=dataset.import_angle_offset_deg,
        column_mapping=dataset.column_mapping,
    )
    data = with_fit_mask(data, np.asarray(dataset.fit_mask, dtype=np.bool_))
    base = compile_fit_problem(
        data,
        dataset.structure,
        dataset.instrument,
        project.fit_config,
        dataset.parameter_settings,
    )
    target = "component.0.thickness_a"
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name != target,
        )
        for definition in base.parameter_definitions
    )
    config = replace(
        FitConfig.fast(20260723),
        budget=SearchBudget(0, 0, 5, 1, 8),
        local_workers=1,
    )
    return compile_fit_problem(
        data,
        dataset.structure,
        dataset.instrument,
        config,
        settings,
    )


def _assert_frozen_candidates(result) -> None:
    assert result.best_index == 6
    assert result.warnings == (
        "全反射平台点不足，尺度弱先验已关闭",
        "stage_a_invalid_candidate_evaluation",
        "stage_a_fringe_candidate_rejected",
    )
    assert tuple(candidate.candidate_id for candidate in result.candidates) == (
        "B-0",
        "B-1",
        "C-0-0",
        "C-1-0",
        "D-0-0",
        "D-1-0",
        "E-0",
        "E-1",
        "E-2",
        "E-3",
    )
    assert tuple(candidate.nfev for candidate in result.candidates) == (
        33,
        33,
        2,
        2,
        2,
        2,
        112,
        112,
        112,
        112,
    )
    np.testing.assert_array_equal(
        np.asarray([candidate.unit_vector[0] for candidate in result.candidates]),
        np.asarray(
            [
                0.4870315924464057,
                0.8950138248837597,
                0.4870315924464057,
                0.8950138248837597,
                0.4870315924464057,
                0.8950138248837597,
                0.48695881459300366,
                0.48695881459300366,
                0.48695881459300366,
                0.48695881459300366,
            ]
        ),
    )
    assert tuple(summary.total_nfev for summary in result.stage_summaries) == (
        512,
        66,
        4,
        4,
        448,
    )


def _assert_frozen_progress(progress) -> None:
    assert tuple(value.total for value in progress[-10:]) == (
        2,
        2,
        6,
        6,
        6,
        6,
        4,
        4,
        4,
        4,
    )
    assert tuple(value.message for value in progress[-10:]) == (
        "completed short differential evolution 1",
        "completed short differential evolution 2",
        "full-resolution density refinement",
        "full-resolution density refinement",
        "full-resolution roughness/instrument refinement",
        "full-resolution roughness/instrument refinement",
        "completed final seed 1",
        "completed final seed 2",
        "completed final seed 3",
        "completed final seed 4",
    )


def _checkpoint_identities(checkpoints) -> set[tuple[str, ...]]:
    return {
        (
            checkpoint.data_sha256,
            checkpoint.structure_fingerprint,
            checkpoint.instrument_fingerprint,
            checkpoint.config_fingerprint,
            checkpoint.parameter_settings_fingerprint,
        )
        for checkpoint in checkpoints
    }


def _assert_frozen_checkpoints(checkpoints) -> None:
    assert tuple(checkpoint.runtime_warnings for checkpoint in checkpoints) == (
        (
            "stage_a_invalid_candidate_evaluation",
            "stage_a_fringe_candidate_rejected",
        ),
    ) * 4
    assert _checkpoint_identities(checkpoints) == {
        (
            "85729258067ff1c953257f6e784b6ec5a5c9e175e92f449ae0bc04680c1e42ea",
            "1f0681cfcc77d487b345d3739394e100597601782f7ae45f900a1cefa564a84f",
            "2e006dff3a7e489619e37403d3e58c9afb50642a06acd3b1aff9c2f392cc9120",
            "45a3c8b3fcabaa2eb54b5497a3ee45f02d2bb12ce0f216312ab3a499819c2828",
            "bab9ebdb6b2377582c6d3e5afddbec238d6b4c427be151500cbd19c18ff076f3",
        )
    }


def test_stage_a_replays_frozen_coarse_grid_selection_and_audit() -> None:
    stages = import_module("xrr_fitter.fit.stages")
    problem = _frozen_single_layer_problem()

    starts, summary, warnings = stages.run_stage_a(
        problem,
        problem.data.source_sha256,
        progress=None,
        cancelled=None,
    )

    assert tuple(start.feature_key for start in starts) == (
        "declared-baseline",
        "geometry-6",
    )
    assert summary.candidate_ids == ("declared-baseline", "geometry-6")
    assert summary.best_objective == pytest.approx(
        0.014555078379018142,
        rel=0.0,
        abs=1e-15,
    )
    assert summary.total_nfev == 512
    assert summary.stop_reasons == (
        "evaluated",
        "invalid_evaluation:108",
        "fringe_rejected:294",
    )
    assert warnings == (
        "stage_a_invalid_candidate_evaluation",
        "stage_a_fringe_candidate_rejected",
    )


def test_fit_search_replays_frozen_a_through_e_evidence() -> None:
    pipeline = import_module("xrr_fitter.fit.pipeline")
    problem = _frozen_single_layer_problem()
    progress = []
    checkpoints = []

    result = pipeline.run_fit_search(
        pipeline.FitSearchRequest(problem.data.source_sha256, problem),
        progress=progress.append,
        checkpoint=checkpoints.append,
    )

    _assert_frozen_candidates(result)
    _assert_frozen_progress(progress)
    _assert_frozen_checkpoints(checkpoints)
