from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.support.model_cases import final_fit_result, simple_structure
from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.projects import new_project
from xrr_fitter.services.structures import set_structure
from xrr_fitter.services import fitting


def _source(path: Path) -> Path:
    angles = np.linspace(0.1, 3.2, 48)
    path.write_bytes(xy_bytes(angles, np.geomspace(1.0, 1e-5, angles.size)))
    return path


def _project(tmp_path: Path):
    value = add_dataset(
        new_project(),
        _source(tmp_path / "curve.xy"),
        InstrumentSpec(instrument_id="fitting-service", footprint_mode="none"),
    )
    value = set_structure(value, "curve", simple_structure())
    return replace(value, fit_config=replace(value.fit_config, scale_prior_enabled=False))


def test_preflight_loads_current_sources_and_compiles_declared_structure(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)

    ready = fitting.preflight_fit(value)
    missing_structure = fitting.preflight_fit(
        replace(value, datasets=(replace(value.datasets[0], structure=None),))
    )

    assert ready.ready is True
    assert ready.message == "ready"
    assert missing_structure.ready is False
    assert "structure" in missing_structure.message


def test_fitting_composes_search_profile_recovery_and_analysis_in_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    value = _project(tmp_path)
    calls: list[object] = []
    initial_search = SimpleNamespace(best_candidate=object())
    continued_search = object()
    decision = SimpleNamespace(
        parameter_name="component.0.thickness_a",
        unit_vector=np.array([0.25]),
    )
    analyzed = final_fit_result()

    def run_search(request, **kwargs):
        calls.append(("search", request.dataset_id, kwargs["checkpoint"] is not None))
        return initial_search

    def recover(problem, candidate, **_kwargs):
        calls.append(("recover", problem, candidate))
        return decision

    def continue_search(problem, search, center, **kwargs):
        calls.append(
            (
                "continue",
                problem,
                search,
                tuple(center),
                kwargs["parameter_name"],
            )
        )
        return continued_search

    def analysis_request(dataset_id, problem, search):
        calls.append(("analysis-request", dataset_id, problem, search))
        return "analysis-request"

    def run_analysis(request, **_kwargs):
        calls.append(("analysis", request))
        return analyzed

    monkeypatch.setattr(fitting, "run_fit_search", run_search)
    monkeypatch.setattr(fitting, "recover_profile_basin", recover)
    monkeypatch.setattr(fitting, "continue_profile_basin", continue_search)
    monkeypatch.setattr(fitting, "AnalysisRequest", analysis_request)
    monkeypatch.setattr(fitting, "run_analysis", run_analysis)

    result = fitting.fit_project(value, checkpoint_callback=lambda _project: None)

    assert result.datasets[0].fit_result is analyzed
    assert [call[0] for call in calls] == [
        "search",
        "recover",
        "continue",
        "analysis-request",
        "analysis",
    ]
    assert calls[2][-1] == decision.parameter_name
    assert calls[3][-1] is continued_search
