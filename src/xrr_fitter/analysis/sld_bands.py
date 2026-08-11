"""Replay retained MCMC samples into aligned SLD depth-profile quantile bands.

Each retained sample is a complete physical parameter vector, so a band is
obtained by replaying samples through the same three public steps the forward
model uses -- ``rebuild_structure``, ``expand_structure``, ``sld_depth_profile``
-- and taking quantiles across the replayed profiles. No linearized covariance
propagation happens here: the depth profile is a nonlinear function of thickness
and roughness, and a linear band would understate the envelope exactly where
layers are thin enough for adjacent interfaces to merge.

Replayed profiles do not share a depth axis. Thickness varies from sample to
sample, so both the grid extent and the interface positions move. Profiles are
therefore shifted so the chosen interface sits at depth zero, then interpolated
onto the intersection of the shifted axes. The intersection rather than the
union is deliberate: outside it some samples contribute no value at all, and
quantiles taken over a varying sample count would put a visible discontinuity in
the band that no physical uncertainty produced.

Thinning is index-based rather than random. Evenly spaced indices over the
retained chain give the same subset for the same report without carrying a seed
through the call, which is what makes exported figures reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from math import isfinite

import numpy as np

from xrr_fitter.model.analysis import McmcReport, SldUncertaintyBands
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureComponent,
    StructureSpec,
)
from xrr_fitter.physics.sld_profile import sld_depth_profile
from xrr_fitter.physics.stack import expand_structure, rebuild_structure

MAX_REPLAY_SAMPLES = 500
QUANTILE_LEVELS = (0.025, 0.16, 0.5, 0.84, 0.975)
ALIGN_CHOICES = ("backing", "surface")
ALIGN_LABELS = {"backing": "基底界面", "surface": "表面界面"}
MAX_FAILURE_RATE = 0.05

Profile = tuple[np.ndarray, np.ndarray]


def _material_values(material: MaterialSpec, prefix: str) -> dict[str, float]:
    """Emit override coordinates only, matching ``_replace_material``.

    A formula-backed material carries no override keys, and emitting them would
    make reconstruction replace an identity it was asked to retain.
    """
    override = material.sld_override_a2
    if override is None:
        return {}
    return {f"{prefix}.sld_real_a2": override.real, f"{prefix}.sld_imag_a2": override.imag}


def _layer_values(layer: LayerSpec, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}.thickness_a": layer.thickness_a,
        f"{prefix}.density_scale": layer.density_scale,
        f"{prefix}.roughness_a": layer.roughness_a,
        **_material_values(layer.material, prefix),
    }


def _periodic_values(block: PeriodicBlock, prefix: str) -> dict[str, float]:
    """Emit shared cell coordinates without materializing the top sentinel.

    ``top_roughness_a`` stays absent when the block inherits it, because
    ``_replace_periodic`` reads that key only when the declaration overrode it.
    """
    values: dict[str, float] = {f"{prefix}.repeats": float(block.repeats)}
    for index, layer in enumerate(block.layers):
        values.update(_layer_values(layer, f"{prefix}.layer.{index}"))
    if block.top_roughness_a is not None:
        values[f"{prefix}.top_roughness_a"] = block.top_roughness_a
    return values


def _gradient_values(layer: GradientLayerSpec, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}.upper_sld_real_a2": layer.upper_sld_a2.real,
        f"{prefix}.upper_sld_imag_a2": layer.upper_sld_a2.imag,
        f"{prefix}.lower_sld_real_a2": layer.lower_sld_a2.real,
        f"{prefix}.lower_sld_imag_a2": layer.lower_sld_a2.imag,
        f"{prefix}.thickness_a": layer.thickness_a,
        f"{prefix}.roughness_a": layer.roughness_a,
        f"{prefix}.microslab_max_a": layer.microslab_max_a,
    }


COMPONENT_VALUES: tuple[tuple[type, Callable[..., dict[str, float]]], ...] = (
    (LayerSpec, _layer_values),
    (PeriodicBlock, _periodic_values),
    (GradientLayerSpec, _gradient_values),
)


def _component_values(component: StructureComponent, prefix: str) -> dict[str, float]:
    for kind, builder in COMPONENT_VALUES:
        if isinstance(component, kind):
            return builder(component, prefix)
    raise TypeError(f"unsupported structure component: {type(component).__name__}")


def _baseline_values(structure: StructureSpec) -> dict[str, float]:
    """Flatten a declaration into the complete map ``rebuild_structure`` reads.

    Reconstruction indexes its coordinate names directly rather than tolerating
    absent ones, while retained samples cover free parameters only. The baseline
    supplies every remaining coordinate at its declared value, so a sample that
    varies one thickness rebuilds a structure identical elsewhere.

    This lives here rather than being borrowed from the fit layer: ``analysis``
    may not import ``fit``, and the compiled definitions there additionally
    require prepared data and an instrument specification that a replay of an
    already-finished fit has no reason to reconstruct.
    """
    values: dict[str, float] = {"backing.roughness_a": structure.backing_roughness_a}
    values.update(_material_values(structure.backing, "backing"))
    for index, component in enumerate(structure.components):
        values.update(_component_values(component, f"component.{index}"))
    return values


def _value_map(baseline: dict[str, float], names: Sequence[str], row: np.ndarray) -> dict[str, float]:
    """Overlay one sample onto the baseline, matching coordinates by name.

    Names rather than positions carry the association: retained columns follow
    the sampler's free-parameter order, which has no reason to agree with the
    declaration order the baseline was built from.
    """
    values = dict(baseline)
    for name, value in zip(names, row, strict=True):
        if name not in values:
            raise ValueError(f"sample parameter {name} is not a structure coordinate")
        values[name] = float(value)
    return values


def _thinned_indices(total: int, limit: int) -> np.ndarray:
    """Pick evenly spaced retained indices, keeping both chain endpoints.

    Rounding an even split is deterministic without a seed, so the same report
    always yields the same subset and exported figures stay reproducible.
    """
    if total <= limit:
        return np.arange(total)
    return np.unique(np.rint(np.linspace(0.0, total - 1, limit)).astype(int))


def _alignment_offset(depth: np.ndarray, interfaces_end: float, align: str) -> float:
    """Return the shift placing the requested interface at depth zero.

    ``sld_depth_profile`` starts its interface list at zero, so the surface
    interface needs no shift and the backing interface needs the total finite
    thickness removed.
    """
    del depth
    return -interfaces_end if align == "backing" else 0.0


def _replay_one(
    structure: StructureSpec,
    values: dict[str, float],
    wavelength_a: float,
    step_a: float,
    align: str,
) -> Profile | None:
    """Replay one sample, returning ``None`` when its structure is unphysical.

    A sample can leave the physically expandable region -- roughness above the
    dynamic interface limit is the common case -- and such a draw carries no
    profile to average. It is dropped here and counted by the caller so the
    failure rate stays visible instead of silently narrowing the band.
    """
    try:
        stack = expand_structure(rebuild_structure(structure, values), wavelength_a)
        depth, profile = sld_depth_profile(stack, step_a=step_a)
    except (ValueError, TypeError):
        return None
    total = float(np.sum(stack.thickness_a[1:-1]))
    return depth + _alignment_offset(depth, total, align), profile


def _replayed_profiles(
    structure: StructureSpec,
    report: McmcReport,
    indices: Iterable[int],
    wavelength_a: float,
    step_a: float,
    align: str,
) -> Iterator[Profile | None]:
    baseline = _baseline_values(structure)
    names = report.parameter_names
    for index in indices:
        values = _value_map(baseline, names, report.samples_physical[index])
        yield _replay_one(structure, values, wavelength_a, step_a, align)


def _common_grid(profiles: Sequence[Profile], step_a: float) -> np.ndarray:
    """Build the shared axis over the intersection of the shifted depth axes.

    Every replayed profile covers the whole intersection, so each quantile is
    taken over the same sample count at every depth.
    """
    start = max(float(depth[0]) for depth, _ in profiles)
    stop = min(float(depth[-1]) for depth, _ in profiles)
    if not isfinite(start) or not isfinite(stop) or stop <= start:
        raise ValueError("aligned sample profiles share no overlapping depth range")
    count = max(2, int(np.ceil((stop - start) / step_a)) + 1)
    return np.linspace(start, stop, count)


def _interpolated(profiles: Sequence[Profile], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real = np.asarray([np.interp(grid, depth, profile.real) for depth, profile in profiles])
    imaginary = np.asarray([np.interp(grid, depth, profile.imag) for depth, profile in profiles])
    return real, imaginary


def _validated_align(align: str) -> str:
    if align not in ALIGN_CHOICES:
        raise ValueError(f"align must be one of {ALIGN_CHOICES}")
    return align


def _validated_failure_rate(attempted: int, succeeded: int) -> float:
    if succeeded == 0:
        raise ValueError("no retained sample replayed into a usable SLD profile")
    rate = (attempted - succeeded) / attempted
    if rate > MAX_FAILURE_RATE:
        raise ValueError(f"replay failure rate {rate:.3f} exceeds {MAX_FAILURE_RATE:.3f}")
    return rate


def sld_uncertainty_bands(
    structure: StructureSpec,
    report: McmcReport,
    *,
    wavelength_a: float,
    step_a: float = 0.5,
    align: str = "backing",
    max_samples: int = MAX_REPLAY_SAMPLES,
) -> SldUncertaintyBands:
    """Replay retained samples into aligned real and imaginary SLD bands."""
    chosen = _validated_align(align)
    total = int(report.samples_physical.shape[0])
    indices = _thinned_indices(total, max_samples)
    replayed = _replayed_profiles(structure, report, indices, wavelength_a, step_a, chosen)
    profiles = [item for item in replayed if item is not None]
    failure_rate = _validated_failure_rate(indices.size, len(profiles))
    grid = _common_grid(profiles, step_a)
    real, imaginary = _interpolated(profiles, grid)
    levels = np.asarray(QUANTILE_LEVELS, dtype=float)
    return SldUncertaintyBands(
        depth_a=grid,
        quantiles=QUANTILE_LEVELS,
        real=np.quantile(real, levels, axis=0),
        imaginary=np.quantile(imaginary, levels, axis=0),
        align_label=ALIGN_LABELS[chosen],
        sample_count=len(profiles),
        total_samples=total,
        failure_rate=failure_rate,
    )
