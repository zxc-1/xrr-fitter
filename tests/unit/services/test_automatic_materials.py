from xrr_fitter.services.materials import automatic_structure, material_from_token


def test_unknown_compound_tokens_are_direct_sld_not_fake_formulas() -> None:
    for token in ("CrSiC", "SiCMo", "AlScN", "custom-4element"):
        material = material_from_token(token)
        assert material.name == token
        assert material.formula is None
        assert material.bulk_density_g_cm3 is None
        assert material.sld_override_a2 == 20e-6 + 0j


def test_known_material_retains_formula_and_nominal_density() -> None:
    material = material_from_token("Si3N4")
    assert (material.formula, material.bulk_density_g_cm3) == ("Si3N4", 3.17)


def test_si_backing_inserts_one_locked_native_oxide() -> None:
    structure, settings = automatic_structure(("Zr", "Si3N4"), "Si")
    assert tuple(layer.name for layer in structure.components) == (
        "Zr",
        "Si3N4",
        "SiO2 native oxide",
    )
    assert structure.components[-1].material.bulk_density_g_cm3 == 2.20
    by_name = {setting.name: setting for setting in settings}
    assert by_name["component.2.thickness_a"].initial == 10.0
    assert (
        by_name["component.2.thickness_a"].lower,
        by_name["component.2.thickness_a"].upper,
    ) == (2.0, 50.0)
    assert by_name["component.2.density_scale"].locked is True


def test_existing_backing_adjacent_exact_sio2_is_not_duplicated() -> None:
    structure, _settings = automatic_structure(("Zr", "SiO2"), "Si")
    assert tuple(layer.name for layer in structure.components) == ("Zr", "SiO2")


def test_unknown_direct_sld_density_scale_is_locked_to_one() -> None:
    _structure, settings = automatic_structure(("CrSiC",), "sapphire")
    density = next(
        value for value in settings if value.name == "component.0.density_scale"
    )
    assert (density.initial, density.lower, density.upper, density.locked) == (
        1.0,
        1.0,
        1.0,
        True,
    )
