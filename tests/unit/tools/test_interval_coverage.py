"""Fixed-seed accounting, exact tests, and formal/representative protocol separation."""

from __future__ import annotations

import pytest
from scipy.stats import binomtest


def _payload(_module, *, available=True, covered=True):
    """Inject only report-control inputs; real fits live in the case tests."""
    bounds = [90.0, 110.0] if covered else [101.0, 103.0]
    interval = {
        "available": available,
        "bounds": bounds if available else None,
        "kind": "likelihood_ratio" if available else "loss_support",
        "confidence_level": 0.95 if available else None,
        "method": "chi_square_1df_asymptotic" if available else "objective_tolerance",
        "unavailable_reason": None if available else "prior_present",
    }
    return {"estimate": 100.0, "fit_available": True, "interval": interval}


def test_fixed_seeds_retain_failure_unavailable_and_noncovers(load_tool_module):
    module = load_tool_module("check_interval_coverage")
    calls = []

    def case(group, seed):
        calls.append((group, seed))
        if seed == 0:
            raise RuntimeError("deliberate fit failure")
        return _payload(module, available=seed != 1, covered=seed != 2)

    report = module.run_group("gaussian_regular", case_runner=case)

    assert calls == [("gaussian_regular", seed) for seed in range(200)]
    assert report["seeds"] == list(range(200))
    assert report["seed_count"] == len(report["cases"]) == 200
    assert [row["seed"] for row in report["cases"]] == list(range(200))
    assert report["nominal_coverage"] == 0.95
    assert report["failed_count"] == sum(not row["success"] for row in report["cases"]) == 2
    _assert_seed_outcomes(report["cases"])


def _assert_seed_outcomes(rows):
    assert rows[0]["failure_reason"] == "RuntimeError: deliberate fit failure"
    assert rows[1]["failure_reason"] == "prior_present"
    assert rows[2]["success"] is True
    assert rows[2]["covered"] is False


def test_coverage_denominators_and_exact_two_sided_binomial_are_explicit(load_tool_module):
    module = load_tool_module("check_interval_coverage")

    def case(_group, seed):
        return _payload(module, available=seed >= 2, covered=seed != 2)

    report = module.run_group("poisson_low", case_runner=case)

    assert report["interval_available_count"] == 198
    assert report["covered_count"] == 197
    assert report["interval_yield"] == pytest.approx(198 / 200)
    assert report["conditional_coverage"] == pytest.approx(197 / 198)
    assert report["all_seed_coverage"] == pytest.approx(197 / 200)
    for key, count in (("conditional_binomial_test", 198), ("all_seed_binomial_test", 200)):
        _assert_binomial_result(report[key], count)


def _assert_binomial_result(result, count):
    assert result["trials"] == count
    assert result["successes"] == 197
    assert result["alternative"] == "two-sided"
    assert result["null_probability"] == 0.95
    assert result["pvalue"] == pytest.approx(binomtest(197, count, 0.95).pvalue, abs=1e-15)


def test_zero_available_intervals_cannot_pass_conditional_binomial(load_tool_module):
    module = load_tool_module("check_interval_coverage")
    report = module.run_group("gaussian_regular", case_runner=lambda _group, _seed: _payload(module, available=False))

    assert report["interval_available_count"] == 0
    assert report["conditional_coverage"] is None
    assert report["conditional_binomial_test"] is None
    assert report["conditional_test_unavailable_reason"] == "no_available_intervals"
    assert report["all_seed_coverage"] == 0.0
    assert report["failed_count"] == 200


def test_each_group_has_its_own_complete_seed_axis(load_tool_module):
    module = load_tool_module("check_interval_coverage")
    calls = []

    def case(group, seed):
        calls.append((group, seed))
        return _payload(module, available=group != "poisson_low")

    reports = [module.run_group(group, case_runner=case) for group in module.GROUPS]

    assert module.GROUPS == ("gaussian_regular", "poisson_low", "poisson_regular", "shared_gaussian")
    assert len(calls) == 800
    assert [report["failed_count"] for report in reports] == [0, 200, 0, 0]
    assert all(report["seeds"] == list(range(200)) for report in reports)


def test_unknown_group_is_rejected_before_running_any_seed(load_tool_module):
    module = load_tool_module("check_interval_coverage")
    with pytest.raises(ValueError, match="group"):
        module.run_group("picked_seeds", case_runner=lambda *_args: pytest.fail("runner was called"))


def test_interrupts_are_not_swallowed_as_bad_seeds(load_tool_module):
    module = load_tool_module("check_interval_coverage")

    def interrupt(*_args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        module.run_group("gaussian_regular", case_runner=interrupt)


def test_pilot_is_fixed_seed_zero_and_explicitly_not_complete(load_tool_module):
    module = load_tool_module("check_interval_coverage")
    report = module.run_group("gaussian_regular", pilot=True, case_runner=lambda *_args: _payload(module))

    assert report["seeds"] == [0]
    assert report["seed_count"] == 1
    assert report["required_seed_count"] == 200
    assert report["protocol_complete"] is False


def test_report_keeps_robust_representative_out_of_formal_group_statistics(load_tool_module, monkeypatch):
    module = load_tool_module("check_interval_coverage")
    calls = []

    def case(group, seed):
        calls.append((group, seed))
        return _payload(module)

    monkeypatch.setattr(module.cases, "run_case", case)
    robust = _payload(module)
    robust["interval"]["method"] = "robust_log_moving_block"
    monkeypatch.setattr(module, "run_robust_representative", lambda: robust)
    report = module.run_experiment(module.parse_args(("--output", "/tmp/report.json", "--pilot")))

    assert calls == [(group, 0) for group in module.GROUPS]
    assert len(report["groups"]) == 4
    assert report["robust_representative"]["interval"]["method"] == "robust_log_moving_block"
    assert report["robust_representative"]["coverage_calibration_claim"] is False
    assert report["protocol_complete"] is False
    assert report["algorithm_version"] == "xrr-fit-v2-poisson-5"


def test_invalid_fit_never_gets_interval_success(load_tool_module, monkeypatch):
    module = load_tool_module("check_interval_coverage")
    payload = _payload(module) | {"fit_available": False, "failure_reason": "invalid_candidate"}
    report = module.run_group("gaussian_regular", pilot=True, case_runner=lambda *_args: payload)

    assert report["cases"][0]["success"] is False
    assert report["cases"][0]["covered"] is None
    assert report["fit_failed_count"] == 1


def _report_without_real_fits(module, monkeypatch, robust):
    monkeypatch.setattr(module, "run_group", lambda group, **_kwargs: {"group": group, "protocol_complete": True})
    monkeypatch.setattr(module, "run_robust_representative", lambda: robust)
    return module.run_experiment(module.parse_args(("--output", "/tmp/report.json")))


def test_full_seed_groups_do_not_hide_an_unexecuted_robust_bootstrap(load_tool_module, monkeypatch):
    module = load_tool_module("check_interval_coverage")
    robust = _payload(module, available=False)
    report = _report_without_real_fits(module, monkeypatch, robust)

    assert report.get("seed_protocol_complete") is True
    assert report["protocol_complete"] is False
    assert report["protocol_incomplete_reason"] == "robust_bootstrap_not_completed"


def test_report_records_the_full_fixed_physics_and_fit_configuration(load_tool_module, monkeypatch):
    module = load_tool_module("check_interval_coverage")
    report = _report_without_real_fits(module, monkeypatch, _payload(module))
    assert "experiment_definition" in report
    definition = report["experiment_definition"]

    assert definition["target"] == "component.0.thickness_a"
    assert definition["true_value"] == 100.0
    assert definition["parameter_bounds"] == [75.0, 125.0]
    assert definition["all_non_target_parameters_known"] is True
    assert definition["groups"]["poisson_regular"]["raw_amplitudes"] == [40000.0]
    assert definition["groups"]["shared_gaussian"]["fit_config_seed_0"]["budget"]["bootstrap_samples"] == 200
