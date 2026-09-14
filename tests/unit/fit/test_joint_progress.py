"""联合进度事件里的每数据集目标值分解。

联合运行只有一个全局 ``best_objective``，可界面要显示"哪条曲线拖后腿"。这个分解已经在
``JointEvaluation.local_evaluations`` 里现成躺着——每个 stage 投影候选时都要读它。所以
进度事件只是把它上报，不是另算一遍：另算一遍就有算出第二套数的风险，而两套数会在同一
屏上互相打脸。
"""

from __future__ import annotations

from importlib import import_module

import pytest
from tests.unit.fit.test_joint_pipeline import _joint_problem


def _candidate_objective(result, candidate_id: str) -> float:
    return next(item for item in result.candidates if item.candidate_id == candidate_id).objective


def _projected_objectives(results) -> dict[str, tuple[float, ...]]:
    """每个候选投影到各数据集上的目标值，按候选 id 索引。"""
    return {
        candidate.candidate_id: tuple(_candidate_objective(result, candidate.candidate_id) for result in results)
        for candidate in results[0].candidates
    }


def _reported_objectives(progress) -> dict[str, tuple[float, ...]]:
    """进度事件报出的分解，按它对应的候选 id 索引。

    ``completed`` 是阶段内从 1 起的序号，候选 id 由阶段名加从 0 起的下标组成，减一即可
    对上——阶段行的既有断言钉住的正是这个序号。
    """
    return {
        f"{value.stage}-{value.completed - 1}": tuple(objective for _id, objective in value.dataset_objectives)
        for value in progress
    }


def test_joint_progress_carries_the_per_dataset_objective_breakdown() -> None:
    """每个联合进度事件都带全部数据集的分解，数值与投影出来的候选逐一相等。"""
    api = import_module("xrr_fitter.fit.joint_pipeline")
    problem = _joint_problem()
    progress: list[object] = []

    results = api.run_joint_fit(api.JointFitRequest(problem), progress=progress.append)

    assert all(value.dataset_objectives is not None for value in progress)
    assert all(
        tuple(dataset_id for dataset_id, _objective in value.dataset_objectives) == problem.dataset_ids
        for value in progress
    )
    projected = _projected_objectives(results)
    for candidate_id, objectives in _reported_objectives(progress).items():
        assert objectives == pytest.approx(projected[candidate_id]), candidate_id
