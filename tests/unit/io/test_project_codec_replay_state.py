from __future__ import annotations

from dataclasses import replace

from tests.unit.io.test_project_codec import _project_with_prior_conflicts

from xrr_fitter.io.project_codec import project_from_dict, project_to_dict


def test_project_roundtrip_preserves_mcmc_fixed_parameter_values() -> None:
    original = _project_with_prior_conflicts()
    mcmc = replace(
        original.datasets[0].last_valid_result.uncertainty.mcmc,
        fixed_parameter_values=(("component.0.roughness_a", 3.5),),
    )
    uncertainty = replace(
        original.datasets[0].last_valid_result.uncertainty,
        mcmc=mcmc,
    )
    result = replace(original.datasets[0].last_valid_result, uncertainty=uncertainty)
    value = replace(original, datasets=(replace(original.datasets[0], last_valid_result=result),))

    restored = project_from_dict(project_to_dict(value))
    report = restored.datasets[0].last_valid_result.uncertainty.mcmc

    assert report.fixed_parameter_values == (("component.0.roughness_a", 3.5),)


def test_project_roundtrip_preserves_mcmc_gradient_slab_counts() -> None:
    original = _project_with_prior_conflicts()
    mcmc = replace(
        original.datasets[0].last_valid_result.uncertainty.mcmc,
        gradient_slab_counts=(("component.0", 112),),
    )
    uncertainty = replace(
        original.datasets[0].last_valid_result.uncertainty,
        mcmc=mcmc,
    )
    result = replace(original.datasets[0].last_valid_result, uncertainty=uncertainty)
    value = replace(original, datasets=(replace(original.datasets[0], last_valid_result=result),))

    restored = project_from_dict(project_to_dict(value))
    report = restored.datasets[0].last_valid_result.uncertainty.mcmc

    assert report.gradient_slab_counts == (("component.0", 112),)
