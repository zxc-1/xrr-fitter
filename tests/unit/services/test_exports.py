from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from inspect import Parameter, signature
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from tests.support.model_cases import final_fit_result, fit_candidate, simple_structure

from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.analysis import UncertaintyReport
from xrr_fitter.model.export import (
    DEFAULT_FORMATS,
    DatasetExportManifest,
    ExportFileRecord,
    ExportFormat,
    ExportManifest,
)
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.services import exports
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.projects import inspect_sources, new_project
from xrr_fitter.services.structures import set_structure

DATASET_FILES = (
    "fit_overview.png",
    "fit_result.json",
    "fit_result.xlsx",
    "residuals.png",
    "run_log.txt",
    "sld_profile.png",
)

# ``DatasetArtifacts`` sorts payloads by path, so the opt-in ``.ort`` lands
# between ``fit_result.json`` and ``fit_result.xlsx`` (j < o < x).
DATASET_FILES_WITH_ORT = (
    "fit_overview.png",
    "fit_result.json",
    "fit_result.ort",
    "fit_result.xlsx",
    "residuals.png",
    "run_log.txt",
    "sld_profile.png",
)

# Every opt-in format at once. The same path sort puts ``parameters.csv`` after
# the ``fit_result.*`` group and ``sld_profile.svg`` after its PNG sibling.
DATASET_FILES_WITH_EVERY_FORMAT = (
    "fit_overview.png",
    "fit_result.json",
    "fit_result.ort",
    "fit_result.xlsx",
    "parameters.csv",
    "residuals.png",
    "run_log.txt",
    "sld_profile.png",
    "sld_profile.svg",
)


def _fitted_project(tmp_path: Path):
    source = tmp_path / "curve.xy"
    angles = np.linspace(0.1, 3.2, 40)
    source.write_bytes(xy_bytes(angles, np.geomspace(1.0, 1e-5, angles.size)))
    value = add_dataset(
        new_project(),
        source,
        InstrumentSpec(instrument_id="export-service", footprint_mode="none"),
    )
    value = set_structure(value, "curve", simple_structure())
    data = exports.load_export_data(value, value.datasets[0])
    candidate = replace(
        fit_candidate(),
        qz_a_inv=data.qz_a_inv,
        model_normalized=data.intensity_normalized,
        log_residuals_decades=np.zeros(data.qz_a_inv.size),
        residuals=np.zeros(data.qz_a_inv.size),
        weighted_residuals=np.zeros(data.qz_a_inv.size),
    )
    result = final_fit_result(candidate)
    return replace(value, datasets=(replace(value.datasets[0], last_valid_result=result),))


def _stub_serializers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "dataset_json_bytes",
        "dataset_workbook_bytes",
        "fit_overview_png",
        "sld_profile_png",
        "residuals_png",
        "run_log_bytes",
        "compatibility_workbook_bytes",
        "batch_workbook_bytes",
        "parameter_trends_png",
    ):
        monkeypatch.setattr(exports, name, lambda *_args, _name=name: _name.encode())


def _published_manifest() -> ExportManifest:
    """Minimal stand-in honouring ``publish_export_run``'s return contract.

    The publication step is what these tests replace; returning a real manifest
    keeps the caller's post-publication handling exercised instead of skipped.
    """
    dataset = DatasetExportManifest("curve", "curve", (ExportFileRecord("curve/fit_result.json", 12, "b" * 64),))
    return ExportManifest(Path("run-1"), (dataset,), (ExportFileRecord("manifest.json", 10, "a" * 64),))


def test_export_builds_the_fixed_artifact_set_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    captured: dict[str, object] = {}
    _stub_serializers(monkeypatch)

    def publish(output_dir, datasets, root_files):
        captured.update(output_dir=output_dir, datasets=datasets, root_files=root_files)
        return _published_manifest()

    monkeypatch.setattr(exports, "publish_export_run", publish)

    result = exports.export_result(value, tmp_path / "exports")

    assert result.run_directory == Path("run-1")
    dataset = captured["datasets"][0]
    assert tuple(item.path for item in dataset.files) == DATASET_FILES
    assert tuple(item.path for item in captured["root_files"]) == (
        "project_snapshot.xrrproj.json",
        "compatibility_summary.xlsx",
    )


def test_export_defers_serializers_until_artifact_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    captured: dict[str, object] = {}
    calls: list[str] = []
    for name in (
        "dataset_json_bytes",
        "parameters_csv_bytes",
        "dataset_workbook_bytes",
        "fit_overview_png",
        "sld_profile_png",
        "residuals_png",
        "run_log_bytes",
        "compatibility_workbook_bytes",
    ):
        monkeypatch.setattr(
            exports,
            name,
            lambda *_args, _name=name: (calls.append(_name), _name.encode())[1],
        )
    monkeypatch.setattr(exports, "publish_export_run", _capture_publish(captured))

    exports.export_result(value, tmp_path / "exports")

    assert calls == []
    producers = (
        *captured["root_files"],
        *captured["datasets"][0].files,
    )
    rendered = {producer.path: producer.render() for producer in producers}
    assert set(calls) == {
        "dataset_json_bytes",
        "dataset_workbook_bytes",
        "fit_overview_png",
        "sld_profile_png",
        "residuals_png",
        "run_log_bytes",
        "compatibility_workbook_bytes",
    }
    assert rendered["project_snapshot.xrrproj.json"].startswith(b"{")


def test_export_rechecks_source_after_initial_inspection_before_allocating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    source = Path(value.datasets[0].source_path)
    original_inspect = inspect_sources

    def inspect_then_mutate(project):
        validation = original_inspect(project)
        source.write_bytes(source.read_bytes() + b"\n0.0 1.0\n")
        return validation

    monkeypatch.setattr(exports, "inspect_sources", inspect_then_mutate)
    monkeypatch.setattr(
        exports,
        "publish_export_run",
        lambda *_args, **_kwargs: pytest.fail("stale export allocated a run"),
    )

    with pytest.raises(ValueError, match="source|hash|changed"):
        exports.export_result(value, tmp_path / "exports")

    assert not (tmp_path / "exports").exists()


def _capture_publish(captured: dict[str, object]):
    def publish(output_dir, datasets, root_files):
        captured.update(output_dir=output_dir, datasets=datasets, root_files=root_files)
        return _published_manifest()

    return publish


def test_export_omits_ort_when_not_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    captured: dict[str, object] = {}
    _stub_serializers(monkeypatch)
    monkeypatch.setattr(exports, "publish_export_run", _capture_publish(captured))

    exports.export_result(value, tmp_path / "exports", formats=DEFAULT_FORMATS)

    dataset = captured["datasets"][0]
    assert tuple(item.path for item in dataset.files) == DATASET_FILES


def test_export_appends_single_ort_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    captured: dict[str, object] = {}
    _stub_serializers(monkeypatch)
    monkeypatch.setattr(exports, "publish_export_run", _capture_publish(captured))
    monkeypatch.setattr(exports, "orso_bytes", lambda *_args, **_kwargs: b"orso-document", raising=False)

    exports.export_result(value, tmp_path / "exports", formats=(*DEFAULT_FORMATS, ExportFormat.ORT))

    dataset = captured["datasets"][0]
    assert tuple(item.path for item in dataset.files) == DATASET_FILES_WITH_ORT
    assert tuple(item.path for item in captured["root_files"]) == (
        "project_snapshot.xrrproj.json",
        "compatibility_summary.xlsx",
    )


def test_export_appends_csv_and_svg_only_when_each_is_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each optional format is a separate opt-in: requesting one must not drag the
    # other in, or a caller asking for a vector figure would silently also publish
    # a parameter table it never asked for.
    value = _fitted_project(tmp_path)
    _stub_serializers(monkeypatch)

    def paths(*optional: ExportFormat) -> tuple[str, ...]:
        captured: dict[str, object] = {}
        monkeypatch.setattr(exports, "publish_export_run", _capture_publish(captured))
        exports.export_result(value, tmp_path / "exports", formats=(*DEFAULT_FORMATS, *optional))
        return tuple(item.path for item in captured["datasets"][0].files)

    assert paths(ExportFormat.CSV) == (*DATASET_FILES[:3], "parameters.csv", *DATASET_FILES[3:])
    assert paths(ExportFormat.SVG) == (*DATASET_FILES, "sld_profile.svg")
    assert paths() == DATASET_FILES


def test_export_emits_every_optional_format_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    captured: dict[str, object] = {}
    _stub_serializers(monkeypatch)
    monkeypatch.setattr(exports, "publish_export_run", _capture_publish(captured))
    monkeypatch.setattr(exports, "orso_bytes", lambda *_args, **_kwargs: b"orso-document", raising=False)

    exports.export_result(
        value,
        tmp_path / "exports",
        formats=(*DEFAULT_FORMATS, ExportFormat.ORT, ExportFormat.CSV, ExportFormat.SVG),
    )

    dataset = captured["datasets"][0]
    assert tuple(item.path for item in dataset.files) == DATASET_FILES_WITH_EVERY_FORMAT
    # Producers render lazily, so the path list alone cannot tell a correctly wired
    # renderer from one paired with the wrong path. ``_stub_serializers`` leaves the
    # two new serializers real, so rendering here exercises the actual wiring.
    rendered = {
        item.path: item.render() for item in dataset.files if item.path in {"parameters.csv", "sld_profile.svg"}
    }
    header = rendered["parameters.csv"].decode("utf-8").splitlines()[0].split(",")
    assert header[:7] == ["parameter_name", "value", "lower", "upper", "noise_model", "residual_name", "residual_unit"]
    assert "sigma" in header
    assert header[-1] == "inference_json"
    assert rendered["sld_profile.svg"].startswith(b"<?xml")


def test_export_omits_ort_covariance_owned_by_another_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _fitted_project(tmp_path)
    first = value.datasets[0].last_valid_result.best_candidate
    selected = replace(first, candidate_id="candidate-b")
    uncertainty = UncertaintyReport(
        correlation_names=("scale",),
        correlation_matrix=np.ones((1, 1)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=first.candidate_id,
        parameter_sigma=np.array([0.1]),
    )
    fit_result = replace(
        value.datasets[0].last_valid_result,
        candidates=(first, selected),
        uncertainty=uncertainty,
    )
    dataset = replace(value.datasets[0], last_valid_result=fit_result)
    ui_state = replace(
        value.ui_state,
        selected_candidate_ids=((dataset.dataset_id, selected.candidate_id),),
    )
    value = replace(value, datasets=(dataset,), ui_state=ui_state)
    context = exports._contexts(
        value,
        ExportFileRecord("project_snapshot.xrrproj.json", 123, "b" * 64),
    )[0]
    captured: dict[str, object] = {}
    _stub_serializers(monkeypatch)

    def serialize_orso(export_context):
        captured["context"] = export_context
        return b"orso-document"

    monkeypatch.setattr(exports, "orso_bytes", serialize_orso)

    artifacts = exports._dataset_artifacts(context, formats=frozenset({ExportFormat.ORT}))
    next(item for item in artifacts.files if item.path == "fit_result.ort").render()

    assert captured["context"] is context
    assert captured["context"].selected_uncertainty is None


def _published_tree(root: Path) -> dict[str, tuple[int, str]]:
    """Map every published file to its size and digest, keyed by run-relative path.

    A run directory name embeds a UTC timestamp and ``secrets.token_hex(4)``, and
    ``export_result`` never forwards ``publish_export_run``'s ``run_timestamp``, so two
    runs cannot share an absolute tree by construction. The run-relative path is the
    strongest identity actually available for comparing two publications.
    """
    return {
        PurePosixPath(path.relative_to(root).as_posix()).as_posix(): (
            path.stat().st_size,
            sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_default_publication_is_byte_identical_when_optional_formats_are_added(tmp_path: Path) -> None:
    """Really publish twice and compare bytes, not just manifest composition.

    Every other format test replaces ``publish_export_run`` with a capture double, so
    the frozen path tuples pin *which* artifacts a run would contain and never compare
    what lands on disk. This test writes both trees for real, so a default run keeps
    producing the same bytes it produced before any optional format existed.

    Comparing the default tree against an explicitly-spelled ``DEFAULT_FORMATS`` run
    would be tautological -- both reach ``normalize_export_formats`` with the same set.
    Asking for the three optional formats *on top of* the default set is not: it adds
    three renderers to the run, and the promise is that the six files the default set
    already publishes come out unchanged. ``export_manifest.json`` is excluded because
    it inventories the tree, so it is *supposed* to differ once the tree grows.
    """
    value = _fitted_project(tmp_path)

    default = exports.export_result(value, tmp_path / "default")
    augmented = exports.export_result(
        value,
        tmp_path / "augmented",
        formats=(*DEFAULT_FORMATS, ExportFormat.ORT, ExportFormat.CSV, ExportFormat.SVG),
    )

    published = _published_tree(default.run_directory)
    # Guard against a vacuous pass: two empty trees would also compare equal.
    assert tuple(published) == (
        *(f"{default.datasets[0].directory}/{name}" for name in DATASET_FILES),
        "compatibility_summary.xlsx",
        "export_manifest.json",
        "project_snapshot.xrrproj.json",
    )
    grown = _published_tree(augmented.run_directory)
    inventory = "export_manifest.json"
    assert {path: value for path, value in grown.items() if path in published and path != inventory} == {
        path: value for path, value in published.items() if path != inventory
    }
    assert tuple(path for path in grown if path not in published) == tuple(
        f"{augmented.datasets[0].directory}/{name}" for name in ("fit_result.ort", "parameters.csv", "sld_profile.svg")
    )


def test_dataset_artifacts_takes_its_format_set_without_a_default() -> None:
    """The format set may not carry a default the only caller always overrides.

    ``export_result`` passes ``formats`` by keyword on every path, so a default here is
    unreachable: changing it moves no published byte and no test goes red. An
    unreachable default reads like a safe fallback while guaranteeing nothing, so the
    selection stays the caller's to state.
    """
    parameters = signature(exports._dataset_artifacts).parameters

    assert parameters["formats"].kind is Parameter.KEYWORD_ONLY
    assert parameters["formats"].default is Parameter.empty
    # A leftover boolean would keep working silently beside the set, so the switch era
    # has to be gone from the signature rather than merely unused by the caller.
    assert [name for name in parameters if name.startswith("include_")] == []


def test_export_result_selects_artifacts_by_format_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """入参是格式集合，不是逐格式布尔叠加。

    布尔叠加下「只要 CSV」这句话说不出来：调用者得先知道另外五个开关各自的默认值，才
    能算出自己会拿到什么。集合把这句话变成它字面的样子。代价是默认集合必须写进签名，
    于是「默认发布哪几种」成了可读的契约而不是六个散落的 ``False``。
    """
    value = _fitted_project(tmp_path)
    _stub_serializers(monkeypatch)
    monkeypatch.setattr(exports, "orso_bytes", lambda *_args, **_kwargs: b"orso-document", raising=False)

    def paths(*formats: ExportFormat) -> tuple[str, ...]:
        captured: dict[str, object] = {}
        monkeypatch.setattr(exports, "publish_export_run", _capture_publish(captured))
        exports.export_result(value, tmp_path / "exports", formats=formats)
        return tuple(item.path for item in captured["datasets"][0].files)

    assert paths(*DEFAULT_FORMATS) == DATASET_FILES
    # 只点一种格式就只发那一种，加上与格式无关的运行日志——它记录这次发布本身，任何
    # 格式选择都关不掉它。
    assert paths(ExportFormat.CSV) == ("parameters.csv", "run_log.txt")
    assert paths(ExportFormat.JSON, ExportFormat.ORT) == (
        "fit_result.json",
        "fit_result.ort",
        "run_log.txt",
    )
    # 集合语义：给的顺序不进产物，重复也不多发一份。``DatasetArtifacts`` 按路径排序。
    assert paths(ExportFormat.SVG, ExportFormat.PNG) == paths(ExportFormat.PNG, ExportFormat.SVG)
    assert paths(ExportFormat.CSV, ExportFormat.CSV) == paths(ExportFormat.CSV)


def test_export_result_rejects_an_empty_or_unknown_format_set(tmp_path: Path) -> None:
    """空集合与拼错的格式名都当场报错，不静默发一棵只有日志的树。

    默认值糊不过去的正是这两种：空集合在布尔时代根本无法表达，而拼错的名字在布尔时代
    是 ``TypeError``。集合入参把它们收进同一处校验。
    """
    value = _fitted_project(tmp_path)

    with pytest.raises(ValueError, match="at least one export format"):
        exports.export_result(value, tmp_path / "empty", formats=())
    with pytest.raises(ValueError):
        exports.export_result(value, tmp_path / "typo", formats=("xls",))
