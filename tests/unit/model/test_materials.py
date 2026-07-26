from __future__ import annotations

from dataclasses import fields

import pytest

from xrr_fitter.model.structure import MaterialSpec, OxideSuggestion


@pytest.mark.parametrize("density", [0.0, -1.0, float("nan"), float("inf")])
def test_material_requires_positive_density(density: float) -> None:
    with pytest.raises(ValueError, match="bulk density"):
        MaterialSpec("Si", "Si", density)


def test_material_requires_exactly_one_finite_sld_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MaterialSpec("missing", None, None)
    with pytest.raises(ValueError, match="exactly one"):
        MaterialSpec("double", "Si", 2.329, 2e-5j)
    with pytest.raises(ValueError, match="nonnegative absorption"):
        MaterialSpec("bad", None, None, 2e-5 - 1e-6j)


def test_oxide_suggestion_preserves_public_schema() -> None:
    oxide = MaterialSpec("SiO2", "SiO2", 2.2)
    suggestion = OxideSuggestion(
        base_material="Si",
        oxide_material=oxide,
        density_locked=True,
        thickness_initial_a=10.0,
        thickness_bounds_a=(2.0, 50.0),
        oxide_table_version="oxide-table-v1",
        location="surface",
    )

    assert [field.name for field in fields(OxideSuggestion)] == [
        "base_material",
        "oxide_material",
        "density_locked",
        "thickness_initial_a",
        "thickness_bounds_a",
        "oxide_table_version",
        "location",
    ]
    assert suggestion.oxide_material is oxide
