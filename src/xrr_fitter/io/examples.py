"""Deterministic unfitted example project construction and publication."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from xrr_fitter.io.export_run import ArtifactPayload, publish_exact_tree
from xrr_fitter.io.project_codec import project_to_bytes
from xrr_fitter.io.xy import read_xy_bytes, xy_bytes
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    MeasurementPreset,
)
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.project import DatasetProject, XrrProject
from xrr_fitter.model.structure import (
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.physics.reflectivity import instrument_reflectivity
from xrr_fitter.physics.stack import expand_structure

BEAM = BeamSpec(kind="monochromatic", wavelength_a=1.5406)
INSTRUMENT = InstrumentSpec(
    instrument_id="example-lab-xrr",
    footprint_mode="fit",
    footprint_spill_angle_deg=0.0,
    background_kind="constant",
    resolution_domain="q",
)
AIR = MaterialSpec("Air", None, None, 0.0j)
SILICON = MaterialSpec("Si", "Si", 2.329)
SILICA = MaterialSpec("SiO2", "SiO2", 2.20)
MOLYBDENUM = MaterialSpec("Mo", "Mo", 10.28)
SEED_TREE_VERSION = 1


def _single_layer_structure() -> StructureSpec:
    return StructureSpec(
        AIR,
        (
            LayerSpec(
                "SiO2 film",
                SILICA,
                173.0,
                density_scale=0.94,
                roughness_a=3.0,
            ),
        ),
        SILICON,
        backing_roughness_a=4.0,
    )


def _mo_si_periodic_structure() -> StructureSpec:
    return StructureSpec(
        AIR,
        (
            PeriodicBlock(
                "Mo/Si",
                (
                    LayerSpec(
                        "Mo",
                        MOLYBDENUM,
                        28.0,
                        density_scale=0.96,
                        roughness_a=3.0,
                    ),
                    LayerSpec(
                        "Si",
                        SILICON,
                        42.0,
                        density_scale=0.98,
                        roughness_a=4.0,
                    ),
                ),
                repeats=20,
                top_roughness_a=2.0,
            ),
        ),
        SILICON,
        backing_roughness_a=5.0,
    )


def _curve_bytes(
    structure: StructureSpec,
    two_theta_deg: np.ndarray,
    *,
    seed: int,
    scale: float,
    background: float,
    relative_sigma: float,
) -> bytes:
    stack = expand_structure(
        structure,
        wavelength_a=BEAM.effective_wavelength_a,
    )
    ideal = instrument_reflectivity(
        two_theta_deg / 2.0,
        stack,
        beam=BEAM,
        scale=scale,
        background=background,
        relative_sigma=relative_sigma,
    )
    sequence = np.random.SeedSequence(
        seed,
        spawn_key=(SEED_TREE_VERSION,),
    ).spawn(1)[0]
    noise = np.random.default_rng(sequence).lognormal(
        mean=0.0,
        sigma=0.01,
        size=two_theta_deg.size,
    )
    return xy_bytes(two_theta_deg, ideal * noise)


def _project(
    stem: str,
    content: bytes,
    structure: StructureSpec,
    seed: int,
) -> XrrProject:
    source_name = f"{stem}.xy"
    data = read_xy_bytes(
        content,
        source_path=source_name,
        beam=BEAM,
    )
    dataset = DatasetProject(
        dataset_id=stem,
        source_path=source_name,
        source_sha256=data.source_sha256,
        beam=data.beam,
        import_angle_offset_deg=data.import_angle_offset_deg,
        column_mapping=data.column_mapping,
        fit_mask=tuple(bool(value) for value in data.fit_mask),
        fit_range_two_theta_deg=(
            float(data.two_theta_deg[0]),
            float(data.two_theta_deg[-1]),
        ),
        structure=structure,
        instrument=INSTRUMENT,
        # The automatic route only considers datasets an import has marked, so a
        # default manual role keeps them out of the preflight entirely. The batch
        # id is derived from the stem rather than generated, because published
        # example files must stay byte reproducible.
        automation=DatasetAutomation(
            import_batch_id=f"example-{stem}",
            role=AutomaticRole.UNROUTED,
            status=AutomaticStatus.PENDING,
        ),
    )
    # The automatic fit preflight requires a project-level preset, which only an
    # import normally produces. Without it the headline automatic action stays
    # disabled on the examples meant to demonstrate it, so each example declares
    # the preset matching the beam and instrument its dataset already records.
    return replace(
        XrrProject.new((dataset,), master_seed=seed),
        measurement_preset=MeasurementPreset(
            INSTRUMENT.instrument_id,
            BEAM,
            INSTRUMENT,
        ),
    )


def _single_layer_example() -> tuple[bytes, XrrProject]:
    structure = _single_layer_structure()
    curve = _curve_bytes(
        structure,
        np.linspace(0.08, 8.0, 1200),
        seed=1101,
        scale=0.86,
        background=2e-7,
        relative_sigma=0.004,
    )
    return curve, _project("single-layer", curve, structure, 1201)


def _mo_si_periodic_example() -> tuple[bytes, XrrProject]:
    structure = _mo_si_periodic_structure()
    curve = _curve_bytes(
        structure,
        np.linspace(0.08, 12.0, 1800),
        seed=2201,
        scale=0.91,
        background=1e-8,
        relative_sigma=0.003,
    )
    return curve, _project("mo-si-periodic", curve, structure, 2301)


def build_single_layer_example() -> XrrProject:
    """Build the deterministic unfitted single-layer project value."""
    return _single_layer_example()[1]


def build_mo_si_periodic_example() -> XrrProject:
    """Build the deterministic unfitted Mo/Si periodic project value."""
    return _mo_si_periodic_example()[1]


def write_examples(destination: Path) -> tuple[Path, ...]:
    """Atomically publish exactly the four canonical example files."""
    single_curve, single_project = _single_layer_example()
    periodic_curve, periodic_project = _mo_si_periodic_example()
    files = (
        ArtifactPayload(
            "single-layer.xrrproj.json",
            project_to_bytes(single_project),
        ),
        ArtifactPayload("single-layer.xy", single_curve),
        ArtifactPayload(
            "mo-si-periodic.xrrproj.json",
            project_to_bytes(periodic_project),
        ),
        ArtifactPayload("mo-si-periodic.xy", periodic_curve),
    )
    return publish_exact_tree(Path(destination), files)
