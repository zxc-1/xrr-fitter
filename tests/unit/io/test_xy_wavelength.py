"""Explicit source wavelengths must not silently turn into a Cu default."""

import numpy as np
import pytest

import xrr_fitter.api as api


def _source(tmp_path, header):
    path = tmp_path / "scan.xy"
    path.write_text(header + "\n" + "\n".join(f"{0.1 + i * 0.01} {100 / (i + 1)}" for i in range(32)))
    return path


def test_declared_synchrotron_wavelength_rejects_copper_default(tmp_path):
    path = _source(tmp_path, "# 2theta_deg intensity; wavelength=0.6889 A")
    with pytest.raises(ValueError, match="wavelength.*0.6889"):
        api.import_data(path, api.BeamSpec("monochromatic", wavelength_a=1.5406))


def test_declared_wavelength_keeps_correct_full_resolution_qz(tmp_path):
    path = _source(tmp_path, "# wavelength=0.6889 A")
    data = api.import_data(path, api.BeamSpec("monochromatic", wavelength_a=0.6889))
    assert len(data.two_theta_deg) == 32
    np.testing.assert_allclose(data.qz_a_inv, 4 * np.pi * np.sin(np.radians(data.two_theta_deg / 2)) / 0.6889)


@pytest.mark.parametrize(
    "header", ("# wavelength=0.6889 A", "# wavelength = 0.6889 Å", "# lambda: 0.06889 nm", "# 波长 = 6.889e-1 angstrom")
)
def test_detect_wavelength_reads_explicit_units(tmp_path, header):
    detector = getattr(api, "detect_wavelength", None)
    assert callable(detector), "public wavelength evidence entry point is missing"
    assert detector(_source(tmp_path, header)) == pytest.approx(0.6889)


@pytest.mark.parametrize(
    "header",
    (
        "# wavelength=0 A",
        "# wavelength=-1 A",
        "# wavelength=nan A",
        "# wavelength=0.6889",
        "# wavelength=1 keV",
        "# wavelength=0.6889 A; wavelength=1.5406 A",
    ),
)
def test_explicit_invalid_or_conflicting_wavelength_is_not_guessed(tmp_path, header):
    with pytest.raises(ValueError, match="wavelength"):
        api.import_data(_source(tmp_path, header), api.BeamSpec("monochromatic", wavelength_a=0.6889))


def test_header_wavelength_requires_monochromatic_beam(tmp_path):
    with pytest.raises(ValueError, match="wavelength"):
        api.import_data(_source(tmp_path, "# wavelength=0.6889 A"), api.BeamSpec("mixed_kalpha"))


def test_absent_wavelength_does_not_invent_metadata(tmp_path):
    detector = getattr(api, "detect_wavelength", None)
    assert callable(detector), "public wavelength evidence entry point is missing"
    path = _source(tmp_path, "# 2theta_deg intensity; PETRA1046")
    assert detector(path) is None
    data = api.import_data(path, api.BeamSpec("monochromatic", wavelength_a=1.2345))
    assert data.beam.wavelength_a == 1.2345


def test_footer_wavelength_is_not_detected_as_header_metadata(tmp_path):
    path = _source(tmp_path, "# 2theta_deg intensity")
    with path.open("a", encoding="utf-8") as source:
        source.write("\n# wavelength=0.6889 A\n")

    assert api.detect_wavelength(path) is None
    data = api.import_data(path, api.BeamSpec("monochromatic", wavelength_a=1.5406))
    assert data.beam.wavelength_a == 1.5406


def test_repeated_equivalent_wavelength_declarations_are_consistent(tmp_path):
    path = _source(tmp_path, "# wavelength=0.6889 A; lambda=0.06889 nm")
    data = api.import_data(path, api.BeamSpec("monochromatic", wavelength_a=0.6889))
    assert data.fit_ready


@pytest.mark.parametrize(
    "header",
    (
        "wavelength = 0.6889 A\n2theta_deg intensity",
        "lambda: 0.06889 nm",
        "  λ = 6.889e-1 Å",
        "\ufeff波长 = 0.6889 angstrom",
    ),
)
def test_plain_header_wavelength_is_detected_and_preserved(tmp_path, header):
    path = _source(tmp_path, header)

    assert api.detect_wavelength(path) == pytest.approx(0.6889)
    data = api.import_data(path, api.BeamSpec("monochromatic", wavelength_a=0.6889))
    assert data.fit_ready
    assert len(data.two_theta_deg) == 32
    np.testing.assert_allclose(data.qz_a_inv, 4 * np.pi * np.sin(np.radians(data.two_theta_deg / 2)) / 0.6889)


def test_plain_header_wavelength_rejects_copper_default(tmp_path):
    path = _source(tmp_path, "wavelength = 0.6889 A\n2theta_deg intensity")

    with pytest.raises(ValueError, match="wavelength.*0.6889"):
        api.import_data(path, api.BeamSpec("monochromatic", wavelength_a=1.5406))


@pytest.mark.parametrize("declaration", ("0 A", "-1 A", "nan A", "0.6889", "1 keV", "0.6889 A; lambda=1.5406 A"))
def test_invalid_plain_header_wavelength_is_not_silently_ignored(tmp_path, declaration):
    path = _source(tmp_path, f"wavelength = {declaration}")

    with pytest.raises(ValueError, match="wavelength"):
        api.detect_wavelength(path)
    with pytest.raises(ValueError, match="wavelength"):
        api.import_data(path, api.BeamSpec("monochromatic", wavelength_a=0.6889))


@pytest.mark.parametrize("footer", ("wavelength = 0.6889 A", "wavelength = invalid", "# wavelength = invalid"))
def test_wavelength_detection_and_import_ignore_plain_or_invalid_footer(tmp_path, footer):
    path = _source(tmp_path, "2theta_deg intensity")
    with path.open("a", encoding="utf-8") as source:
        source.write(f"\n{footer}\n")

    assert api.detect_wavelength(path) is None
    data = api.import_data(path, api.BeamSpec("monochromatic", wavelength_a=1.5406))
    assert data.fit_ready
    assert data.raw_parse_status[-1] == "footer"
