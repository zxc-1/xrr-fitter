"""Changed interval and provenance contracts must not impersonate frozen RMS."""

import pytest

import xrr_fitter.api as api
from xrr_fitter.io.project_codec import project_from_dict, project_to_dict


def test_current_joint_contract_has_new_schema_and_algorithm_but_same_diagnostic_estimator():
    project = api.new_project()
    assert project.schema_version == 5
    assert project.algorithm_version == "xrr-fit-v2-poisson-5"
    assert project.fit_config.diagnostic_version == "poisson-refit-null-v4"
    assert project.fit_config.objective_version == "2"


def test_current_codec_rejects_old_schema_instead_of_filling_new_seals():
    payload = project_to_dict(api.new_project())
    payload["schema_version"] = 4
    with pytest.raises(ValueError, match="schema"):
        project_from_dict(payload)


def test_current_runtime_cannot_restart_the_finished_rms_campaign(load_tool_module):
    tool = load_tool_module("check_poisson_diagnostics")
    assert tool.protocol.EXPECTED_VERSIONS["schema_version"] == 4
    with pytest.raises(ValueError, match="integrated version"):
        tool._check_versions()
