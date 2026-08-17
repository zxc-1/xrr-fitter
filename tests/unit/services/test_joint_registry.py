from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

from xrr_fitter.services import batch, fitting


def _expected_seed_tree(value):
    independent, joint, mcmc = np.random.SeedSequence(
        value.master_seed,
        spawn_key=(fitting.SERVICE_SEED_TREE_VERSION,),
    ).spawn(3)
    identifiers = tuple(sorted(dataset.dataset_id for dataset in value.datasets))
    independent_values = {
        dataset_id: int(child.generate_state(1, dtype=np.uint64)[0])
        for dataset_id, child in zip(
            identifiers,
            independent.spawn(len(identifiers)),
            strict=True,
        )
    }
    mcmc_values = dict(zip(identifiers, mcmc.spawn(len(identifiers)), strict=True))
    return independent_values, int(joint.generate_state(1, dtype=np.uint64)[0]), mcmc_values


def test_service_seed_registry_is_versioned_and_independent_of_project_order() -> None:
    forward = project(dataset_project("zeta"), dataset_project("alpha"))
    backward = replace(forward, datasets=tuple(reversed(forward.datasets)))

    first = fitting.service_seed_branches(forward)
    second = fitting.service_seed_branches(backward)
    expected = _expected_seed_tree(forward)

    assert fitting.SERVICE_SEED_TREE_VERSION == 1
    assert first[0] == second[0] == expected[0]
    assert first[1] == second[1] == expected[1]
    assert tuple(first[2]) == tuple(second[2]) == ("alpha", "zeta")


def test_joint_failure_invalidates_the_whole_graph_without_independent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = final_fit_result(replace(fit_candidate(), ranking_objective=1.0))
    value = replace(
        project(
            dataset_project("first", result=prior),
            dataset_project("second", result=prior),
        ),
        batch_mode="joint",
    )
    monkeypatch.setattr(fitting, "preflight_fit", lambda _project: SimpleNamespace(ready=True, message="ready"))
    monkeypatch.setattr(batch, "inspect_sources", lambda _project: SimpleValidation())
    monkeypatch.setattr(
        fitting,
        "prepare_dataset_fit",
        lambda project, dataset_id, seed: SimpleNamespace(
            updated_dataset=next(dataset for dataset in project.datasets if dataset.dataset_id == dataset_id),
            dataset_id=dataset_id,
            seed=seed,
        ),
    )
    monkeypatch.setattr(
        fitting,
        "fit_joint_datasets",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("joint compile failed")),
    )
    monkeypatch.setattr(
        fitting,
        "fit_prepared_dataset",
        lambda *_args, **_kwargs: pytest.fail("joint failure fell back to independent"),
    )

    result = fitting.fit_project(value)

    assert result.mode == "joint"
    assert all(dataset.last_valid_result is None for dataset in result.updated_project.datasets)
    assert all(dataset.checkpoint is None for dataset in result.updated_project.datasets)
    assert all(item.fit_result.confidence.value == "不可信" for item in result.datasets)
    assert "joint compile failed" in result.warnings[0]


class SimpleValidation:
    valid = True
    datasets = ()
    issues = ()
