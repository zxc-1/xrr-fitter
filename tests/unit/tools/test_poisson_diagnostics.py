"""Fixed population denominators and the eight preregistered CP assertions."""

from __future__ import annotations

from copy import deepcopy

import pytest
from tests.unit.tools.poisson_diagnostic_fixtures import _row
from tests.unit.tools.poisson_diagnostic_fixtures import tool as tool


def test_summary_keeps_unavailable_in_all_fixed_seed_denominators(tool):
    report = tool.protocol.summarize([_row(), _row("unavailable", covered=None)], required_count=2)

    assert report["attempted"] == 2
    assert report["unavailable"] == 1
    assert report["failure_or_rejection"] == 1
    assert report["all_seed_coverage"] == 0.5
    assert report["conditional_coverage"] == 1.0
    assert report["profile"]["availability"] == 0.5


def test_untriggered_screen_is_neither_mc_unavailable_nor_validated_null(tool):
    report = tool.protocol.summarize([_row(), _row("not_rejected")], required_count=2)

    assert report["unavailable"] == 0
    assert report["not_triggered"] == 1
    assert report["calibration_available"] == 1
    assert report["raw_triggered"] == 1


def test_product_failure_counts_as_u_without_discarding_fixed_seed(tool):
    report = tool.protocol.summarize([_row(fit=False, covered=None)], required_count=2)

    assert report["unavailable"] == report["fit_failed"] == 1
    assert report["failure_or_rejection_rate"] == 0.5
    assert report["missing"] == 1
    assert report["protocol_complete"] is False
    assert report["conditional_coverage"] is None


@pytest.mark.parametrize(
    ("count", "limit", "passed"), ((24, "F", True), (25, "F", False), (9, "U", True), (10, "U", False))
)
def test_null_cp_boundaries_are_fixed_not_data_dependent(tool, count, limit, passed):
    endpoint = tool.protocol.endpoint(count, 2000, limit)

    assert endpoint["passed"] is passed
    assert endpoint["gamma"] == 0.00625
    assert endpoint["bound"] == pytest.approx(tool.protocol.cp_bound(count, 2000, "upper"))


@pytest.mark.parametrize(("count", "passed"), ((89, False), (90, True)))
def test_target_column_power_uses_predeclared_cp_lower_bound(tool, count, passed):
    endpoint = tool.protocol.endpoint(count, 100, "power")

    assert endpoint["passed"] is passed
    assert endpoint["threshold"] == 0.8
    assert endpoint["bound"] == pytest.approx(tool.protocol.cp_bound(count, 100, "lower"))


@pytest.mark.parametrize(("successes", "trials", "side", "bound"), ((0, 10, "lower", 0.0), (10, 10, "upper", 1.0)))
def test_cp_degenerate_endpoints_are_exact(tool, successes, trials, side, bound):
    assert tool.protocol.cp_bound(successes, trials, side) == bound


def test_target_power_is_not_omnibus_or_available_subset_power(tool):
    rows = [_row("rejected", target=True), _row("rejected"), _row("unavailable", covered=None), _row()]
    report = tool.protocol.summarize(rows, required_count=4)

    assert report["rejected"] == 2
    assert report["target_detected"] == 1
    assert report["target_detection_rate"] == 0.25
    assert report["rejected_columns"] == {"background": 1}


def test_protocol_freezes_all_4400_new_and_200_replay_seeds(tool):
    protocol = tool.protocol.definition()
    groups = protocol["scenarios"]

    assert tuple(groups) == ("null_single", "null_joint", "background", "footprint", "surface", "acf", "replay_low")
    assert groups["null_single"]["seeds"] == list(range(3000000, 3002000))
    assert groups["null_joint"]["seeds"] == list(range(3010000, 3012000))
    assert groups["replay_low"]["seeds"] == list(range(200))
    for index, name in enumerate(("background", "footprint", "surface", "acf")):
        assert groups[name]["seeds"] == list(range(4000000 + 10000 * index, 4000100 + 10000 * index))


def test_protocol_freezes_rng_budget_and_family_alpha(tool):
    protocol = tool.protocol.definition()
    assert protocol["gamma"] == 0.00625
    assert protocol["diagnostic_samples"] == 999
    assert protocol["diagnostic_alpha"] == 0.01
    assert protocol["diagnostic_seed_domain"] == 0x504F495344494147
    assert protocol["observation_rng"] == "Generator(PCG64(SeedSequence([20260912, scenario_id, seed, member_id])))"


def test_protocol_declares_physics_joint_nuisances_and_strong_injections(tool):
    definition = tool.protocol.definition()
    groups = definition["scenarios"]

    assert groups["null_joint"]["raw_amplitudes"] == [400.0, 1600.0]
    assert groups["null_joint"]["scale_initial_factor"] == 0.9
    assert groups["null_joint"]["scale_bound_factors"] == [0.5, 1.5]
    assert groups["null_single"]["theta_deg_start_stop_count"] == [0.15, 1.8, 80]
    assert groups["surface"]["theta_deg_start_stop_count"] == [0.03, 3.0, 520]


def test_protocol_freezes_all_strong_injection_parameters(tool):
    definition = tool.protocol.definition()
    assert definition["injections"]["background"]["slope_q_over_qmax"] == 2000.0
    assert definition["injections"]["footprint"]["spill_angle_deg"] == 0.30
    assert definition["injections"]["surface"] == {"thickness_a": 30.0, "sld_real_a2": 18.9e-6, "roughness_a": 1.0}
    assert definition["injections"]["acf"]["block_size"] == 8
    assert definition["injections"]["acf"]["shared_fraction"] == 0.9


def test_source_identity_hashes_dirty_executable_source_not_only_head(tool, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "src" / "sample.py").write_text("value = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sample'\n")
    first = tool.protocol.source_identity(tmp_path)
    (tmp_path / "src" / "sample.py").write_text("value = 2\n")
    second = tool.protocol.source_identity(tmp_path)

    assert first["sha256"] != second["sha256"]
    assert {row["path"] for row in first["files"]} == {"src/sample.py", "pyproject.toml"}
    assert first["head"] is None


def test_source_identity_includes_the_real_platform_dependency_lock_files(tool, tmp_path):
    lock = tmp_path / "requirements-macos-arm64-py312.lock"
    lock.write_text("numpy==2.0.0\n")
    before = tool.protocol.source_identity(tmp_path)
    lock.write_text("numpy==2.0.1\n")

    assert tool.protocol.source_identity(tmp_path)["sha256"] != before["sha256"]


def test_protocol_definition_cannot_mutate_the_frozen_injection_recipe(tool, monkeypatch):
    monkeypatch.setattr(tool.protocol, "INJECTIONS", deepcopy(tool.protocol.INJECTIONS))
    definition = tool.protocol.definition()
    definition["injections"]["acf"]["shared_fraction"] = 0.1

    assert tool.protocol.definition()["injections"]["acf"]["shared_fraction"] == 0.9


def test_raw_rate_and_both_interval_yields_remain_separate(tool):
    row = _row("not_rejected", covered=None)
    row["bootstrap"] = {"available": True, "bounds": [90.0, 110.0]}
    report = tool.protocol.summarize([row], required_count=2)

    assert report["raw_trigger_rate"] == 0.5
    assert report["profile"]["availability"] == 0.0
    assert report["bootstrap"]["all_seed_coverage"] == 0.5
    assert report["all_seed_coverage"] == 0.0


def test_unknown_diagnostic_status_cannot_silently_shrink_r_or_u(tool):
    with pytest.raises(ValueError, match="diagnostic status"):
        tool.protocol.summarize([_row("typo")], required_count=1)


def test_incomplete_populations_have_no_cp_acceptance_even_if_observed_rows_pass(tool):
    summary = tool.protocol.summarize([_row()], required_count=2000)

    assert tool.protocol.acceptance(summary, "null") is None


@pytest.mark.parametrize(
    ("kind", "rejected", "unavailable", "target", "passed"),
    (
        ("null", 15, 9, 0, True),
        ("null", 16, 9, 0, False),
        ("null", 0, 10, 0, False),
        ("power", 100, 0, 89, False),
        ("power", 90, 10, 90, True),
    ),
)
def test_acceptance_uses_disjoint_r_plus_u_and_target_detection(tool, kind, rejected, unavailable, target, passed):
    total = 2000 if kind == "null" else 100
    rows = [_row("rejected", target=index < target) for index in range(rejected)]
    rows += [_row("unavailable", covered=None)] * unavailable
    rows += [_row()] * (total - len(rows))
    gates = tool.protocol.acceptance(tool.protocol.summarize(rows, required_count=total), kind)

    assert all(gate["passed"] for gate in gates) is passed


def test_null_rejection_interval_reuses_f_bound_without_adding_a_ninth_claim(tool):
    rows = [_row("rejected")] * 15 + [_row("unavailable", covered=None)] * 9 + [_row()] * 1976
    summary = tool.protocol.summarize(rows, required_count=2000)
    interval = tool.protocol.rejection_interval(summary)

    assert interval["count"] == 15
    assert interval["bounds"] == [0.0, tool.protocol.cp_bound(24, 2000, "upper")]
    assert interval["basis"] == "R <= F = R + U; same predeclared F endpoint, no additional claim"
    assert tool.protocol.rejection_interval(tool.protocol.summarize([], required_count=2000)) is None


def test_preregistration_cannot_reuse_a_review_for_the_previous_runtime(tool, monkeypatch):
    stale = dict(tool.protocol.EXPECTED_VERSIONS)
    stale.update(algorithm_version="xrr-fit-v2-poisson-1", diagnostic_version="poisson-refit-null-v1")
    monkeypatch.setattr(tool.protocol, "EXPECTED_VERSIONS", stale)
    with pytest.raises(ValueError, match="integrated version"):
        tool._check_versions()
