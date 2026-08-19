"""Prove both progress renderings carry the full FitProgress contract."""

from __future__ import annotations

import json

import numpy as np

import xrr_fitter.api as api
from xrr_fitter.cli import progress as progress_module

SCALAR_FIELDS = {
    "best_objective",
    "completed",
    "dataset_id",
    "message",
    "stage",
    "total",
}


def _progress() -> api.FitProgress:
    return api.FitProgress(
        dataset_id="P1",
        stage="B",
        completed=3,
        total=5,
        best_objective=1.25,
        message="粗搜索完成",
        preview_qz_a_inv=np.array([0.01, 0.02]),
        preview_model_normalized=np.array([1.0, 0.5]),
    )


def test_text_rendering_is_one_line_without_preview_arrays() -> None:
    line = progress_module.render_text(_progress())

    assert "\n" not in line
    assert "P1" in line and "3/5" in line and "粗搜索完成" in line
    assert "0.01" not in line


def test_json_rendering_is_one_parsable_line_with_scalar_fields_only() -> None:
    rendered = progress_module.render_json(_progress())

    assert "\n" not in rendered
    assert json.loads(rendered) == {
        "dataset_id": "P1",
        "stage": "B",
        "completed": 3,
        "total": 5,
        "best_objective": 1.25,
        "message": "粗搜索完成",
    }


def test_json_rendering_survives_a_missing_dataset_id() -> None:
    bare = api.FitProgress(
        dataset_id=None,
        stage="A",
        completed=0,
        total=1,
        best_objective=float("inf"),
        message="开始",
    )

    payload = json.loads(progress_module.render_json(bare))

    assert payload["dataset_id"] is None
    assert payload["best_objective"] is None


def test_text_rendering_marks_an_unset_objective_and_dataset() -> None:
    bare = api.FitProgress(
        dataset_id=None,
        stage="A",
        completed=0,
        total=1,
        best_objective=float("inf"),
        message="开始",
    )

    line = progress_module.render_text(bare)

    assert "\n" not in line
    assert "Infinity" not in line and "inf" not in line
    assert "开始" in line


def test_json_rendering_keeps_chinese_readable_and_field_set_exact() -> None:
    rendered = progress_module.render_json(_progress())

    assert "粗搜索完成" in rendered
    assert set(json.loads(rendered)) == SCALAR_FIELDS
