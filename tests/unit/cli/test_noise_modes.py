"""CLI mode selection uses the public invalidation boundary before dispatch."""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace

import pytest
from tests.support.model_cases import dataset_project, final_fit_result, project

import xrr_fitter.api as api
from xrr_fitter.cli import commands, main
from xrr_fitter.model.fitting import FitCheckpoint


def test_parser_keeps_the_saved_mode_unless_explicitly_overridden() -> None:
    assert main.build_parser().parse_args(["fit", "project.json"]).noise_model is None


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
def test_parser_accepts_exact_noise_modes(mode: str) -> None:
    arguments = main.build_parser().parse_args(["fit", "project.json", "--noise-model", mode])

    assert arguments.noise_model == mode


def test_parser_explains_likelihood_input_requirements(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main.build_parser().parse_args(["fit", "--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--noise-model" in help_text
    assert "标准差" in help_text
    assert "原始" in help_text and "整数计数" in help_text


def _completed_project() -> api.XrrProject:
    result = final_fit_result()
    dataset = replace(
        dataset_project(result=result),
        checkpoint=FitCheckpoint("a" * 64, "b" * 64, "c" * 64, "E", result.candidates, (101,)),
        fit_mask=tuple(index != 5 for index in range(32)),
    )
    value = project(dataset)
    return replace(value, ui_state=replace(value.ui_state, selected_candidate_ids=(("curve", "candidate-0"),)))


def _assert_config_invalidation(original, updated, mode, setter_calls) -> None:
    if mode is None:
        assert setter_calls == []
        assert updated is original
    elif mode == original.fit_config.noise_model:
        assert updated is original
    else:
        assert setter_calls == [replace(original.fit_config, noise_model=mode)]
        assert (
            updated.datasets[0].checkpoint,
            updated.datasets[0].last_valid_result,
            updated.ui_state.selected_candidate_ids,
        ) == (None, None, ())


@pytest.mark.parametrize("automatic", (False, True))
@pytest.mark.parametrize("mode", (None, "robust_log", "gaussian", "poisson"))
def test_mode_reaches_the_selected_fit_route_and_saved_project(monkeypatch, tmp_path, automatic, mode) -> None:
    original = _completed_project()
    seen = []
    setter_calls = []
    saved = []
    setter = api.set_fit_config

    def record_config(value, config):
        setter_calls.append(config)
        return setter(value, config)

    def preflight(value):
        seen.append(("preflight", value))
        return api.FitReadiness(True, "ready")

    def manual_fit(value, *, progress_callback):
        seen.append(("manual", value))
        return api.ProjectFitResult("independent", (), (), value)

    def automatic_fit(value, *, progress_callback):
        seen.append(("automatic", value))
        return api.ProjectFitResult("automatic", (), (), value)

    monkeypatch.setattr(commands, "_load", lambda path: original)
    monkeypatch.setattr(commands, "_require_fresh_sources", lambda value: None)
    monkeypatch.setattr(api, "set_fit_config", record_config)
    monkeypatch.setattr(api, "preflight_fit", preflight)
    monkeypatch.setattr(api, "fit_project", manual_fit)
    monkeypatch.setattr(api, "fit_automatically", automatic_fit)
    monkeypatch.setattr(api, "save_project", lambda value, path: saved.append((value, path)))
    output = tmp_path / "updated.json"

    code = commands.run_fit(
        Namespace(project="project.json", output=output, json_progress=False, auto=automatic, noise_model=mode)
    )

    assert code == 0
    assert [event for event, _ in seen] == (["automatic"] if automatic else ["preflight", "manual"])
    updated = seen[-1][1]
    assert updated.fit_config.noise_model == (mode or original.fit_config.noise_model)
    assert all(value is updated for _, value in seen)
    assert saved == [(updated, output)]
    assert updated.datasets[0].fit_mask == original.datasets[0].fit_mask
    _assert_config_invalidation(original, updated, mode, setter_calls)


def _assert_mode_announcement(announcement, mode, requirements) -> None:
    assert announcement.out == ""
    assert announcement.err.startswith(f"噪声模式：{mode}")
    assert len(announcement.err.splitlines()) == 1
    assert all(requirement in announcement.err for requirement in requirements), announcement.err


def _assert_progress_streams(after_fit, as_json) -> None:
    if as_json:
        assert [json.loads(line) for line in after_fit.out.splitlines()] == [
            {
                "dataset_id": "curve",
                "stage": "B",
                "completed": 1,
                "total": 2,
                "best_objective": 1.25,
                "message": "progress",
            }
        ]
        assert after_fit.err == ""
    else:
        assert after_fit.out == ""
        assert "阶段 B 1/2" in after_fit.err
        assert "噪声模式" not in after_fit.err


@pytest.mark.parametrize("automatic", (False, True))
@pytest.mark.parametrize("as_json", (False, True))
@pytest.mark.parametrize("explicit_mode", (False, True))
@pytest.mark.parametrize(
    ("mode", "requirements"),
    (
        ("robust_log", ("稳健对数（探索）", "强度加稳定下限后大于 0", "允许零强度", "不是正式似然")),
        ("gaussian", ("Gaussian（已知标准差）", "每个拟合点", "有限", "严格为正", "同单位", "sigma")),
        (
            "poisson",
            ("Poisson（原始整数计数）", "非负整数", "未合并", "不能使用归一化强度、计数率或背景扣除值"),
        ),
    ),
)
def test_fit_reports_effective_mode_and_requirements_on_stderr_before_fitting(
    monkeypatch, capsys, automatic, as_json, explicit_mode, mode, requirements
) -> None:
    initial = project()
    previous_mode = "gaussian" if mode == "poisson" else "poisson"
    saved_mode = previous_mode if explicit_mode else mode
    original = replace(initial, fit_config=replace(initial.fit_config, noise_model=saved_mode))
    before_fit = []
    progress = api.FitProgress("curve", "B", 1, 2, 1.25, "progress")

    def fit(value, *, progress_callback):
        before_fit.append(capsys.readouterr())
        progress_callback(progress)
        return api.ProjectFitResult("automatic" if automatic else "independent", (), (), value)

    monkeypatch.setattr(commands, "_load", lambda path: original)
    monkeypatch.setattr(commands, "_require_fresh_sources", lambda value: None)
    monkeypatch.setattr(api, "preflight_fit", lambda value: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "fit_project", fit)
    monkeypatch.setattr(api, "fit_automatically", fit)

    code = commands.run_fit(
        Namespace(
            project="project.json",
            output=None,
            json_progress=as_json,
            auto=automatic,
            noise_model=mode if explicit_mode else None,
        )
    )

    assert code == 0
    assert len(before_fit) == 1
    _assert_mode_announcement(before_fit[0], mode, requirements)
    _assert_progress_streams(capsys.readouterr(), as_json)
