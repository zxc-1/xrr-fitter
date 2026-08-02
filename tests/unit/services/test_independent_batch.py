from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.model_cases import final_fit_result, simple_structure
from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.services import batch, fitting
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.projects import new_project
from xrr_fitter.services.structures import set_structure


def _source(path: Path, scale: float) -> Path:
    angles = np.linspace(0.1, 3.2, 48)
    intensity = scale * np.geomspace(1.0, 1e-5, angles.size)
    path.write_bytes(xy_bytes(angles, intensity))
    return path


def _project(tmp_path: Path):
    value = new_project()
    for name, scale in (("zeta", 1.0), ("alpha", 0.9)):
        value = add_dataset(
            value,
            _source(tmp_path / f"{name}.xy", scale),
            InstrumentSpec(instrument_id=name, footprint_mode="none"),
        )
        value = set_structure(value, name, simple_structure())
    return replace(value, fit_config=replace(value.fit_config, scale_prior_enabled=False))


def test_independent_dispatch_uses_sorted_seed_identity_but_project_result_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _project(tmp_path)
    observed: list[tuple[str, int]] = []

    def prepare(project, dataset_id, seed):
        index = next(
            index for index, dataset in enumerate(project.datasets) if dataset.dataset_id == dataset_id
        )
        observed.append((dataset_id, seed))
        return SimpleNamespace(
            dataset_id=dataset_id,
            dataset_index=index,
            updated_dataset=project.datasets[index],
        )

    monkeypatch.setattr(fitting, "prepare_dataset_fit", prepare)
    monkeypatch.setattr(
        fitting,
        "fit_prepared_dataset",
        lambda *_args, **_kwargs: final_fit_result(),
    )

    result = fitting.fit_project(value)
    expected = fitting.service_seed_branches(value)[0]

    assert observed == [("zeta", expected["zeta"]), ("alpha", expected["alpha"])]
    assert tuple(item.dataset_id for item in result.datasets) == ("zeta", "alpha")
    assert all(dataset.last_valid_result is not None for dataset in result.updated_project.datasets)


def test_independent_prepare_failure_clears_only_that_dataset_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _project(tmp_path)
    prior = final_fit_result()
    value = replace(
        value,
        datasets=tuple(replace(dataset, last_valid_result=prior) for dataset in value.datasets),
    )

    def prepare(project, dataset_id, _seed):
        if dataset_id == "zeta":
            raise ValueError("compile failed")
        return SimpleNamespace(
            dataset_id=dataset_id,
            dataset_index=1,
            updated_dataset=project.datasets[1],
        )

    completed = final_fit_result()
    monkeypatch.setattr(fitting, "prepare_dataset_fit", prepare)
    monkeypatch.setattr(
        fitting,
        "fit_prepared_dataset",
        lambda *_args, **_kwargs: completed,
    )

    result = fitting.fit_project(value)

    assert result.updated_project.datasets[0].last_valid_result is None
    assert result.updated_project.datasets[1].last_valid_result is completed
    assert "compile failed" in result.datasets[0].fit_result.warnings[0]
    assert result.datasets[1].fit_result is completed


def test_independent_project_warnings_preserve_dataset_order_and_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _project(tmp_path)
    warned = replace(final_fit_result(), warnings=("first", "second"))

    def prepare(project, dataset_id, _seed):
        index = next(
            index
            for index, dataset in enumerate(project.datasets)
            if dataset.dataset_id == dataset_id
        )
        return SimpleNamespace(
            dataset_id=dataset_id,
            dataset_index=index,
            updated_dataset=project.datasets[index],
        )

    monkeypatch.setattr(fitting, "prepare_dataset_fit", prepare)
    monkeypatch.setattr(
        fitting,
        "fit_prepared_dataset",
        lambda *_args, **_kwargs: warned,
    )

    result = fitting.fit_project(value)

    assert result.warnings == ("first", "second", "first", "second")


def test_independent_batch_runs_concurrently_with_one_total_worker_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _project(tmp_path)
    value = replace(value, fit_config=replace(value.fit_config, local_workers=4))
    rendezvous = Barrier(2, timeout=2.0)
    alpha_finished = Event()
    allocations: list[tuple[str, int]] = []
    completion: list[str] = []
    published: list[tuple[str, str]] = []

    def prepare(project, dataset_id, _seed):
        index = next(
            index
            for index, dataset in enumerate(project.datasets)
            if dataset.dataset_id == dataset_id
        )
        return SimpleNamespace(
            dataset_id=dataset_id,
            dataset_index=index,
            updated_dataset=project.datasets[index],
        )

    def fit(prepared, *, progress, checkpoint, local_workers, **_options):
        dataset_id = prepared.dataset_id
        allocations.append((dataset_id, local_workers))
        progress(SimpleNamespace(message=f"{dataset_id}:start"))
        rendezvous.wait()
        if dataset_id == "alpha":
            completion.append(dataset_id)
            alpha_finished.set()
        else:
            assert alpha_finished.wait(timeout=2.0)
            completion.append(dataset_id)
        checkpoint(None)
        progress(SimpleNamespace(message=f"{dataset_id}:finish"))
        return final_fit_result()

    monkeypatch.setattr(fitting, "prepare_dataset_fit", prepare)
    monkeypatch.setattr(fitting, "fit_prepared_dataset", fit)

    result = fitting.fit_project(
        value,
        progress_callback=lambda event: published.append(("progress", event.message)),
        checkpoint_callback=lambda _project: published.append(("checkpoint", "saved")),
    )

    assert completion == ["alpha", "zeta"]
    assert sorted(allocations) == [("alpha", 2), ("zeta", 2)]
    assert published == [
        ("progress", "zeta:start"),
        ("checkpoint", "saved"),
        ("progress", "zeta:finish"),
        ("progress", "alpha:start"),
        ("checkpoint", "saved"),
        ("progress", "alpha:finish"),
    ]
    assert tuple(item.dataset_id for item in result.datasets) == ("zeta", "alpha")
    assert all(dataset.last_valid_result is not None for dataset in result.updated_project.datasets)
