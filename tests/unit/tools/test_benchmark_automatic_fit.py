from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "arguments",
    (
        (),
        ("--single", "--batch-size", "2"),
        ("--adaptive", "--batch-size", "3"),
        ("--single", "--repeat", "0"),
        ("--batch-size", "1"),
        ("--batch-size", "5"),
    ),
)
def test_cli_requires_exactly_one_valid_mode(load_tool_module, arguments) -> None:
    module = load_tool_module("benchmark_automatic_fit")

    with pytest.raises(SystemExit):
        module.parse_args(arguments)


def test_json_report_has_stable_schema(load_tool_module) -> None:
    module = load_tool_module("benchmark_automatic_fit")
    run = module.BenchmarkRun(
        case_id="single",
        repeat_index=0,
        elapsed_seconds=1.25,
        stage_nfev=(("A", 10), ("E", 20)),
        total_nfev=30,
        bootstrap_count=0,
        profile_count=2,
        statuses=(("point-1", "passed"),),
        recovery_errors=(("point-1", 0.012),),
    )

    report = module.build_report("single", None, 1, (run,))

    assert report == {
        "schema": "xrr-automatic-benchmark-v1",
        "mode": "single",
        "batch_size": None,
        "repeat": 1,
        "median_seconds": 1.25,
        "maximum_seconds": 1.25,
        "runs": [
            {
                "case_id": "single",
                "repeat_index": 0,
                "elapsed_seconds": 1.25,
                "stage_nfev": {"A": 10, "E": 20},
                "total_nfev": 30,
                "bootstrap_count": 0,
                "profile_count": 2,
                "status": {"point-1": "passed"},
                "recovery_error": {"point-1": 0.012},
            }
        ],
    }
    assert json.loads(module.canonical_json(report)) == report


def test_work_signature_requires_consistent_stage_total(load_tool_module) -> None:
    module = load_tool_module("benchmark_automatic_fit")

    with pytest.raises(ValueError, match="total_nfev"):
        module.BenchmarkRun(
            "bad",
            0,
            1.0,
            (("A", 10),),
            11,
            0,
            0,
            (("point-1", "review"),),
            (("point-1", 0.1),),
        )


@pytest.mark.parametrize("count", (1, 2, 3, 4))
def test_bounded_benchmark_fixture_fixes_seeds_and_search_budget(
    load_tool_module,
    tmp_path,
    count,
) -> None:
    module = load_tool_module("benchmark_automatic_fit")

    fixture = module._benchmark_fixture(tmp_path, "deterministic", count)
    project = fixture.project

    assert len(project.datasets) == count
    assert len(fixture.targets) == count
    assert project.master_seed == module.MASTER_SEED
    assert project.fit_config.master_seed == module.MASTER_SEED
    assert project.fit_config.budget.short_de_maxiter == 0
    assert project.fit_config.budget.full_de_maxiter == 0


@pytest.mark.parametrize(
    ("arguments", "count", "case_id"),
    (
        (("--single",), 1, "single"),
        (("--batch-size", "2"), 2, "batch-2"),
        (("--batch-size", "4"), 4, "batch-4"),
    ),
)
def test_single_and_batch_modes_use_the_bounded_synthetic_fixture(
    load_tool_module,
    monkeypatch,
    arguments,
    count,
    case_id,
) -> None:
    module = load_tool_module("benchmark_automatic_fit")
    built = []
    fixture = SimpleNamespace(case_id=case_id)

    def build(root, observed_case_id, observed_count):
        built.append((root.name, observed_case_id, observed_count))
        return fixture

    monkeypatch.setattr(module, "_benchmark_fixture", build, raising=False)

    def benchmark(observed, repeat_index):
        assert observed is fixture
        return module.BenchmarkRun(
            case_id,
            repeat_index,
            0.0,
            (),
            0,
            0,
            0,
            ((case_id, "passed"),),
            ((case_id, 0.0),),
        )

    monkeypatch.setattr(module, "_benchmark_recovery_case", benchmark)

    def reject_copied_source(*_args, **_kwargs):
        pytest.fail("single/batch mode copied the unbounded example source")

    monkeypatch.setattr(
        module,
        "_benchmark_case",
        reject_copied_source,
        raising=False,
    )

    report = module.run_benchmark(module.parse_args(arguments))

    assert built == [(f"0-{case_id}", case_id, count)]
    assert report["runs"][0]["case_id"] == case_id


def test_adaptive_mode_uses_every_deterministic_recovery_fixture(
    load_tool_module,
    monkeypatch,
) -> None:
    module = load_tool_module("benchmark_automatic_fit")
    called = []
    case_ids = (
        "direct-sld",
        "shared-local",
        "isolated-outlier",
        "roughness-release",
    )
    fixtures = tuple(
        (
            case_id,
            lambda root, case_id=case_id: SimpleNamespace(
                case_id=case_id,
                root=root,
            ),
        )
        for case_id in case_ids
    )

    monkeypatch.setattr(module, "_adaptive_cases", lambda: fixtures, raising=False)

    def benchmark_fixture(fixture, repeat_index):
        called.append((fixture.root.name, fixture.case_id, repeat_index))
        return module.BenchmarkRun(
            fixture.case_id,
            repeat_index,
            0.0,
            (),
            0,
            0,
            0,
            ((fixture.case_id, "passed"),),
            ((fixture.case_id, 0.0),),
        )

    monkeypatch.setattr(
        module,
        "_benchmark_recovery_case",
        benchmark_fixture,
        raising=False,
    )

    def reject_fake_source_case(*_args, **_kwargs):
        pytest.fail("adaptive mode used the copied single-layer source")

    monkeypatch.setattr(
        module,
        "_benchmark_case",
        reject_fake_source_case,
        raising=False,
    )

    report = module.run_benchmark(module.parse_args(("--adaptive", "--repeat", "2")))

    assert [item[1] for item in called] == list(case_ids) * 2
    assert [run["case_id"] for run in report["runs"]] == list(case_ids) * 2


def test_adaptive_registry_builds_the_real_recovery_contract(
    load_tool_module,
    tmp_path,
) -> None:
    module = load_tool_module("benchmark_automatic_fit")
    registered = module._adaptive_cases()
    fixtures = tuple(builder(tmp_path / case_id) for case_id, builder in registered)

    assert tuple(case_id for case_id, _builder in registered) == (
        "direct-sld",
        "shared-local",
        "isolated-outlier",
        "roughness-release",
    )
    assert tuple(tuple(target.dataset_id for target in fixture.targets) for fixture in fixtures) == (
        ("D1",),
        ("P1", "P2", "P3", "P4"),
        ("P1", "P2", "P3", "P4"),
        ("R1", "R2", "R3", "R4"),
    )
    assert tuple(tuple(target.parameters for target in fixture.targets) for fixture in fixtures) == (
        ((("component.0.thickness_a", 130.0), ("component.0.sld_real_a2", 24e-6)),),
        tuple(
            (
                ("component.0.thickness_a", thickness),
                ("component.0.density_scale", 0.93),
            )
            for thickness in (90.0, 100.0, 110.0, 120.0)
        ),
        ((("component.0.thickness_a", 100.0),),) * 4,
        tuple((("component.0.roughness_a", roughness),) for roughness in (2.0, 3.0, 8.0, 9.0)),
    )
    assert all(fixture.project.fit_config.master_seed == module.MASTER_SEED for fixture in fixtures)


@pytest.mark.parametrize(
    ("fitted", "truths", "expected"),
    (
        (
            (("component.0.thickness_a", 130.0), ("component.0.sld_real_a2", 21.6e-6)),
            (("component.0.thickness_a", 130.0), ("component.0.sld_real_a2", 24e-6)),
            0.10,
        ),
        (
            (("component.0.thickness_a", 90.0), ("component.0.density_scale", 0.837)),
            (("component.0.thickness_a", 90.0), ("component.0.density_scale", 0.93)),
            0.10,
        ),
        (
            (("component.0.roughness_a", 2.5),),
            (("component.0.roughness_a", 2.0),),
            0.25,
        ),
    ),
)
def test_adaptive_recovery_error_tracks_case_defining_parameters(
    load_tool_module,
    fitted,
    truths,
    expected,
) -> None:
    module = load_tool_module("benchmark_automatic_fit")
    candidate = SimpleNamespace(parameters=tuple(SimpleNamespace(name=name, value=value) for name, value in fitted))
    dataset = SimpleNamespace(last_valid_result=SimpleNamespace(best_candidate=candidate))
    target = SimpleNamespace(parameters=truths)

    assert module._recovery_error(dataset, target) == pytest.approx(expected)


def _work_result(stages, *, profiles=0, bootstrap=False):
    return SimpleNamespace(
        stage_summaries=tuple(
            SimpleNamespace(
                stage=stage,
                candidate_ids=(f"{stage}-0",),
                total_nfev=nfev,
            )
            for stage, nfev in stages
        ),
        uncertainty=SimpleNamespace(
            profiles=(object(),) * profiles,
            bootstrap_performed=bootstrap,
        ),
    )


def _work_dataset(
    module,
    dataset_id,
    result,
    *,
    role,
    status,
    checkpoint=None,
    group_id="group-1",
):
    return SimpleNamespace(
        dataset_id=dataset_id,
        last_valid_result=result,
        checkpoint=checkpoint,
        automation=SimpleNamespace(
            fit_group_id=group_id,
            role=role,
            status=status,
        ),
    )


def test_work_ledger_counts_prefits_and_joint_projection_once(
    load_tool_module,
) -> None:
    module = load_tool_module("benchmark_automatic_fit")
    ledger = module._recovery_support().AutomaticWorkLedger()
    prefit_a = _work_result((("A", 10),), profiles=1)
    prefit_b = _work_result((("A", 10),), profiles=2)
    ledger.observe(
        SimpleNamespace(
            datasets=(
                _work_dataset(
                    module,
                    "a",
                    prefit_a,
                    role=module.api.AutomaticRole.JOINT,
                    status=module.api.AutomaticStatus.REFINING,
                ),
                _work_dataset(
                    module,
                    "b",
                    None,
                    role=module.api.AutomaticRole.JOINT,
                    status=module.api.AutomaticStatus.REFINING,
                ),
            )
        )
    )
    ledger.observe(
        SimpleNamespace(
            datasets=(
                _work_dataset(
                    module,
                    "a",
                    prefit_a,
                    role=module.api.AutomaticRole.JOINT,
                    status=module.api.AutomaticStatus.REFINING,
                ),
                _work_dataset(
                    module,
                    "b",
                    prefit_b,
                    role=module.api.AutomaticRole.JOINT,
                    status=module.api.AutomaticStatus.REFINING,
                ),
            )
        )
    )
    joint_a = _work_result((("B", 30),), profiles=4, bootstrap=True)
    joint_b = _work_result((("B", 30),), profiles=4, bootstrap=True)
    ledger.observe(
        SimpleNamespace(
            datasets=tuple(
                _work_dataset(
                    module,
                    dataset_id,
                    result,
                    role=module.api.AutomaticRole.JOINT,
                    status=module.api.AutomaticStatus.PASSED,
                )
                for dataset_id, result in (("a", joint_a), ("b", joint_b))
            )
        )
    )

    assert ledger.metrics() == ((("A", 20), ("B", 30)), 1, 7)


def test_work_ledger_counts_real_isolated_retry_but_not_a_relabel(
    load_tool_module,
) -> None:
    module = load_tool_module("benchmark_automatic_fit")
    recovery = module._recovery_support()
    prefit = _work_result((("A", 11),), profiles=1)

    retried = recovery.AutomaticWorkLedger()
    retried.observe(
        SimpleNamespace(
            datasets=(
                _work_dataset(
                    module,
                    "outlier",
                    prefit,
                    role=module.api.AutomaticRole.JOINT,
                    status=module.api.AutomaticStatus.REFINING,
                ),
            )
        )
    )
    retry = _work_result((("A", 11),), profiles=3)
    retried.observe(
        SimpleNamespace(
            datasets=(
                _work_dataset(
                    module,
                    "outlier",
                    retry,
                    role=module.api.AutomaticRole.ISOLATED_RETRY,
                    status=module.api.AutomaticStatus.REVIEW,
                ),
            )
        )
    )

    relabelled = recovery.AutomaticWorkLedger()
    relabelled.observe(
        SimpleNamespace(
            datasets=(
                _work_dataset(
                    module,
                    "outlier",
                    prefit,
                    role=module.api.AutomaticRole.JOINT,
                    status=module.api.AutomaticStatus.REFINING,
                ),
            )
        )
    )
    relabelled.observe(
        SimpleNamespace(
            datasets=(
                _work_dataset(
                    module,
                    "outlier",
                    prefit,
                    role=module.api.AutomaticRole.ISOLATED_RETRY,
                    status=module.api.AutomaticStatus.REVIEW,
                ),
            )
        )
    )

    assert retried.metrics() == ((("A", 22),), 0, 4)
    assert relabelled.metrics() == ((("A", 11),), 0, 1)


def test_work_ledger_retains_discarded_joint_attempts_without_checkpoint_duplication(
    load_tool_module,
) -> None:
    module = load_tool_module("benchmark_automatic_fit")
    ledger = module._recovery_support().AutomaticWorkLedger()

    def snapshot(layout, stages):
        checkpoint = SimpleNamespace(
            joint_layout_fingerprint=layout,
            stage_summaries=tuple(
                SimpleNamespace(
                    stage=stage,
                    candidate_ids=(f"{stage}-0",),
                    total_nfev=nfev,
                )
                for stage, nfev in stages
            ),
        )
        return SimpleNamespace(
            datasets=tuple(
                _work_dataset(
                    module,
                    dataset_id,
                    None,
                    role=module.api.AutomaticRole.JOINT,
                    status=module.api.AutomaticStatus.REFINING,
                    checkpoint=checkpoint,
                )
                for dataset_id in ("a", "b")
            )
        )

    ledger.observe(snapshot("roughness", (("B", 10),)))
    ledger.observe(snapshot("roughness", (("B", 10), ("C", 20))))
    ledger.observe(snapshot("material-only", (("B", 15),)))
    final_a = _work_result((("A", 1), ("B", 15)))
    final_b = _work_result((("A", 1), ("B", 15)))
    ledger.observe(
        SimpleNamespace(
            datasets=tuple(
                _work_dataset(
                    module,
                    dataset_id,
                    result,
                    role=module.api.AutomaticRole.JOINT,
                    status=module.api.AutomaticStatus.PASSED,
                )
                for dataset_id, result in (("a", final_a), ("b", final_b))
            )
        )
    )

    assert ledger.metrics() == ((("A", 2), ("B", 25), ("C", 20)), 0, 0)


def test_benchmark_times_only_public_fit_and_uses_intermediate_work_ledger(
    load_tool_module,
    monkeypatch,
) -> None:
    module = load_tool_module("benchmark_automatic_fit")
    observed = []

    class Ledger:
        def observe(self, project):
            observed.append(project)

        def metrics(self):
            return (("A", 20),), 0, 6

    ledger = Ledger()
    monkeypatch.setattr(
        module,
        "_recovery_support",
        lambda: SimpleNamespace(AutomaticWorkLedger=lambda: ledger),
    )
    candidate = SimpleNamespace(parameters=(SimpleNamespace(name="component.0.thickness_a", value=130.0),))
    dataset = SimpleNamespace(
        dataset_id="D1",
        automation=SimpleNamespace(status=SimpleNamespace(value="passed")),
        last_valid_result=SimpleNamespace(best_candidate=candidate),
    )
    updated = SimpleNamespace(datasets=(dataset,))
    fit_calls = []

    def fit(project, batch_id, *, checkpoint_callback):
        fit_calls.append((project, batch_id, checkpoint_callback))
        return SimpleNamespace(updated_project=updated)

    monkeypatch.setattr(module.api, "fit_automatically", fit)
    times = iter((10.0, 12.5))
    monkeypatch.setattr(module, "monotonic", lambda: next(times))
    target = SimpleNamespace(
        dataset_id="D1",
        parameters=(("component.0.thickness_a", 130.0),),
    )
    fixture = SimpleNamespace(
        case_id="direct-sld",
        project=object(),
        import_batch_id="batch-1",
        targets=(target,),
    )

    run = module._benchmark_recovery_case(fixture, 0)

    assert fit_calls == [(fixture.project, "batch-1", ledger.observe)]
    assert observed == [updated]
    assert run.elapsed_seconds == pytest.approx(2.5)
    assert run.stage_nfev == (("A", 20),)
    assert run.total_nfev == 20
    assert run.profile_count == 6
