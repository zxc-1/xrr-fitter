from __future__ import annotations

from pathlib import Path

import pytest

import xrr_fitter.api as api


AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}"
            for index in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _panel(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.structure.panel import StructurePanel

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(),
    )
    panel = StructurePanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    return panel


def test_structure_editor_accepts_backing_oxide_with_exact_project_provenance(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (), SI))
    suggestion = panel.current_oxide_suggestion()

    panel.accept_current_oxide()

    dataset = panel.document.project.datasets[0]
    inserted = dataset.structure.components[-1]
    assert suggestion.location == "backing"
    assert inserted.material == suggestion.oxide_material
    assert dataset.oxide_decisions[-1] == api.OxideDecision(
        "Si",
        "SiO2",
        "backing",
        True,
        suggestion.oxide_table_version,
    )
    settings = {item.name: item for item in dataset.parameter_settings}
    assert settings["component.0.density_scale"].locked is True


def test_surface_oxide_acceptance_inserts_at_component_zero(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    backing = api.MaterialSpec("backing", None, None, 2e-6 + 0.1e-6j)
    silicon = api.LayerSpec("surface Si", SI, 20.0)
    panel.set_structure(api.StructureSpec(AIR, (silicon,), backing))

    panel.accept_current_oxide()

    components = panel.structure.components
    assert components[0].material.formula == "SiO2"
    assert components[1] == silicon


def test_structure_editor_refuses_exact_oxide_without_changing_structure(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    structure = api.StructureSpec(AIR, (), SI)
    panel.set_structure(structure)
    suggestion = panel.current_oxide_suggestion()

    panel.refuse_current_oxide()

    dataset = panel.document.project.datasets[0]
    assert dataset.structure is structure
    assert dataset.oxide_decisions[-1] == api.OxideDecision(
        suggestion.base_material,
        suggestion.oxide_material.formula,
        suggestion.location,
        False,
        suggestion.oxide_table_version,
    )
    assert panel.current_oxide_suggestion() is None


def test_oxide_acceptance_commits_only_through_public_api(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (), SI))
    original = panel.document.project
    suggestion = panel.current_oxide_suggestion()
    updated = api.accept_oxide_suggestion(original, "sample", suggestion)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "accept_oxide_suggestion",
        lambda project, dataset_id, value: (
            calls.append((project, dataset_id, value)),
            updated,
        )[1],
    )

    panel.accept_current_oxide()

    assert calls == [(original, "sample", suggestion)]
    assert panel.document.project is updated


def test_oxide_workflow_failure_is_atomic_and_emits_no_success(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (), SI))
    before = panel.document.project
    events: list[object] = []
    panel.oxide_decision_changed.connect(events.append)
    monkeypatch.setattr(
        api,
        "accept_oxide_suggestion",
        lambda *_args: (_ for _ in ()).throw(ValueError("oxide rejected")),
    )

    with pytest.raises(ValueError, match="oxide rejected"):
        panel.accept_current_oxide()

    assert panel.document.project is before
    assert panel.current_oxide_suggestion() is not None
    assert events == []
