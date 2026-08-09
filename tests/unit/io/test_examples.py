from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from xrr_fitter.io import export_run
from xrr_fitter.io.examples import (
    build_mo_si_periodic_example,
    build_single_layer_example,
    write_examples,
)
from xrr_fitter.io.project_codec import load_project
from xrr_fitter.io.xy import read_xy
from xrr_fitter.model.automation import AutomaticRole, AutomaticStatus
from xrr_fitter.physics.reflectivity import instrument_reflectivity
from xrr_fitter.physics.stack import expand_structure
from xrr_fitter.services.fitting import preflight_automatic_fit

CANONICAL_FILES = (
    "mo-si-periodic.xrrproj.json",
    "mo-si-periodic.xy",
    "single-layer.xrrproj.json",
    "single-layer.xy",
)


def _assert_project_root(value, seed: int) -> None:
    assert value.master_seed == seed
    assert value.base_directory is None
    assert len(value.datasets) == 1
    assert value.sharing_rules == ()
    assert value.ui_state.selected_candidate_ids == ()


def _assert_dataset_source(dataset, stem: str, size: int) -> None:
    assert dataset.dataset_id == stem
    assert dataset.source_path == f"{stem}.xy"
    assert not Path(dataset.source_path).is_absolute()
    assert dataset.fit_mask == (True,) * size


def _assert_dataset_is_unfitted(dataset) -> None:
    assert dataset.structure_evidence is None
    assert dataset.scale_prior.enabled is False
    assert dataset.oxide_decisions == ()
    assert dataset.parameter_settings == ()
    assert dataset.last_valid_result is None
    assert dataset.checkpoint is None


def _assert_unfitted_example(value, stem: str, seed: int, size: int) -> None:
    _assert_project_root(value, seed)
    dataset = value.datasets[0]
    _assert_dataset_source(dataset, stem, size)
    _assert_dataset_is_unfitted(dataset)


def test_example_builders_return_unfitted_relocatable_model_values() -> None:
    _assert_unfitted_example(
        build_single_layer_example(),
        "single-layer",
        1201,
        1200,
    )
    _assert_unfitted_example(
        build_mo_si_periodic_example(),
        "mo-si-periodic",
        2301,
        1800,
    )


def test_examples_carry_the_measurement_preset_their_datasets_declare() -> None:
    """An example without a preset cannot reach the automatic fit path.

    ``preflight_automatic_fit`` requires ``project.measurement_preset``, which
    only an import produces. Shipping examples without it left the headline
    automatic action permanently disabled on the very projects meant to
    demonstrate it, so each example declares the preset matching the beam and
    instrument its dataset already records.
    """
    for value in (build_single_layer_example(), build_mo_si_periodic_example()):
        preset = value.measurement_preset
        assert preset is not None
        dataset = value.datasets[0]
        assert preset.beam == dataset.beam
        assert preset.instrument == dataset.instrument
        assert preset.preset_id == dataset.instrument.instrument_id


def test_examples_are_runnable_through_the_automatic_fit_route() -> None:
    """A preset alone still leaves the automatic action disabled.

    ``preflight_automatic_fit`` skips every dataset whose automation role is
    ``manual``, so examples carrying only the preset reported "no runnable
    automatic datasets". Each example therefore ships the unrouted/pending
    markers an import would produce, under a fixed batch id that keeps the
    published files byte reproducible.
    """
    for value in (build_single_layer_example(), build_mo_si_periodic_example()):
        automation = value.datasets[0].automation
        assert automation.role is AutomaticRole.UNROUTED
        assert automation.status is AutomaticStatus.PENDING
        assert automation.import_batch_id == f"example-{value.datasets[0].dataset_id}"
        assert automation.fit_group_id is None
        assert automation.statistics_member is False


def test_published_examples_pass_the_automatic_fit_preflight(
    tmp_path: Path,
) -> None:
    """The published examples must satisfy the preflight a user's click runs.

    The builders return relocatable values whose sources only resolve once the
    file is loaded from disk, so readiness is only meaningful on the published
    tree. This is the exact state the GUI holds after opening an example.
    """
    destination = tmp_path / "examples"
    write_examples(destination)

    for stem in ("single-layer", "mo-si-periodic"):
        loaded = load_project(destination / f"{stem}.xrrproj.json")
        readiness = preflight_automatic_fit(loaded)
        assert readiness.ready, readiness.message


def _file_bytes(paths: tuple[Path, ...]) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in paths}


def test_example_generation_is_byte_reproducible_and_global_rng_neutral(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    before = np.random.get_state()

    first_paths = write_examples(first)
    after = np.random.get_state()
    second_paths = write_examples(second)

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert tuple(path.name for path in first_paths) == CANONICAL_FILES
    assert tuple(path.name for path in second_paths) == CANONICAL_FILES
    assert _file_bytes(first_paths) == _file_bytes(second_paths)
    assert write_examples(first) == first_paths
    for name in ("single-layer.xy", "mo-si-periodic.xy"):
        raw = (first / name).read_bytes()
        assert raw.endswith(b"\n") and b"\r" not in raw
        lines = raw.decode("ascii").splitlines()
        assert lines[0] == "# 2theta_deg intensity"


def test_example_physics_and_provenance_match_canonical_contract(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "examples"
    write_examples(destination)
    cases = (
        (
            "single-layer",
            np.linspace(0.08, 8.0, 1200),
            1101,
            1201,
            0.86,
            2e-7,
            0.004,
        ),
        (
            "mo-si-periodic",
            np.linspace(0.08, 12.0, 1800),
            2201,
            2301,
            0.91,
            1e-8,
            0.003,
        ),
    )

    for stem, angles, generation_seed, project_seed, scale, background, sigma in cases:
        source = destination / f"{stem}.xy"
        source_bytes = source.read_bytes()
        values = np.loadtxt(source)
        loaded = load_project(destination / f"{stem}.xrrproj.json")
        dataset = loaded.datasets[0]
        assert loaded.base_directory == str(destination.resolve())
        assert loaded.master_seed == project_seed
        assert dataset.source_path == source.name
        assert dataset.source_sha256 == sha256(source_bytes).hexdigest()
        imported = read_xy(source, dataset.beam)
        assert imported.source_sha256 == dataset.source_sha256
        assert tuple(bool(value) for value in imported.fit_mask) == dataset.fit_mask

        assert dataset.structure is not None
        stack = expand_structure(
            dataset.structure,
            wavelength_a=dataset.beam.effective_wavelength_a,
        )
        ideal = instrument_reflectivity(
            angles / 2.0,
            stack,
            beam=dataset.beam,
            scale=scale,
            background=background,
            relative_sigma=sigma,
        )
        sequence = np.random.SeedSequence(
            generation_seed,
            spawn_key=(1,),
        ).spawn(1)[0]
        noise = np.random.default_rng(sequence).lognormal(
            mean=0.0,
            sigma=0.01,
            size=angles.size,
        )
        np.testing.assert_allclose(values[:, 0], angles, rtol=0.0, atol=5e-11)
        np.testing.assert_allclose(values[:, 1], ideal * noise, rtol=1e-15, atol=0.0)


def test_example_publication_rejects_unknown_members_without_mutation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "examples"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="unexpected|conflicting"):
        write_examples(destination)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert tuple(destination.iterdir()) == (marker,)
    assert not tuple(tmp_path.glob(".partial-*"))


def test_example_write_failure_leaves_no_partial_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "examples"
    real_write = export_run._write_payload
    calls = 0

    def fail_second(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("example write failed")
        real_write(path, content)

    monkeypatch.setattr(export_run, "_write_payload", fail_second)

    with pytest.raises(OSError, match="example write failed"):
        write_examples(destination)

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".partial-*"))
