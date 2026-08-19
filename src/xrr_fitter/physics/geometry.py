"""Material-independent structure geometry for evaluation constraints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from xrr_fitter.model.slab_stack import SlabStack
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    PeriodicBlock,
    StructureSpec,
    _ExpandedDriftBlock,
)
from xrr_fitter.physics.materials import material_sld
from xrr_fitter.physics.stack import expand_structure, gradient_slab_count, rebuild_structure
from xrr_fitter.physics.transitions import (
    transition_fractions,
    transition_slab_count,
    transition_width,
)

GRADIENT_INTERNAL_INTERFACE = "__gradient_internal_zero__"


def _transition_count(layer: LayerSpec) -> int:
    """Resolve one layer's microslab count from the shared primal source."""
    transition = layer.transition
    assert transition is not None
    return transition_slab_count(transition_width(transition), transition.microslab_max_a)


def _expanded_thickness_name(
    prefix: str,
    repeat_index: int,
    layer_index: int,
    *,
    drifts_thickness: bool,
) -> str:
    """Thickness source coordinate for one expanded periodic or drift layer.

    Copy zero is the free base cell. A thickness-drift block's later copies read
    their own ``repeat.{k}`` coordinate, which the desugared drift rules populate;
    every other cell shares the base declaration so the mapping stays unchanged.
    """
    if drifts_thickness and repeat_index > 0:
        return f"{prefix}.repeat.{repeat_index}.layer.{layer_index}.thickness_a"
    return f"{prefix}.layer.{layer_index}.thickness_a"


def _expanded_roughness_name(
    prefix: str,
    repeat_index: int,
    layer_index: int,
    *,
    top_present: bool,
    drifts_roughness: bool,
) -> str:
    """Roughness source coordinate for one expanded periodic or drift interface.

    The block's very first interface honors an explicit top-roughness override.
    A roughness-drift block's later copies resolve to their own ``repeat.{k}``
    coordinate; every other cell keeps the shared base declaration.
    """
    if (repeat_index, layer_index) == (0, 0) and top_present:
        return f"{prefix}.top_roughness_a"
    if drifts_roughness and repeat_index > 0:
        return f"{prefix}.repeat.{repeat_index}.layer.{layer_index}.roughness_a"
    return f"{prefix}.layer.{layer_index}.roughness_a"


def _periodic_interface_names(prefix: str, block: PeriodicBlock) -> tuple[str, ...]:
    """Mirror repeated interfaces back to their shared base source coordinates.

    Only non-drift blocks reach this helper: a drifted block is baked into an
    ``_ExpandedDriftBlock`` upstream and names its per-copy interfaces separately,
    so every cell here keeps the shared declaration and stays byte-identical.
    """
    top_present = block.top_roughness_a is not None
    return tuple(
        _expanded_roughness_name(
            prefix,
            repeat_index,
            layer_index,
            top_present=top_present,
            drifts_roughness=False,
        )
        for repeat_index in range(block.repeats)
        for layer_index in range(len(block.layers))
    )


def _drift_interface_names(prefix: str, block: _ExpandedDriftBlock) -> tuple[str, ...]:
    """Name every baked drift interface by its resolved per-copy source.

    The flattened copy-major layout recovers ``(copy, layer)`` through
    ``divmod(flat_index, layer_count)``. Roughness-drift copies resolve to their
    own ``repeat.{k}`` coordinate; a thickness-drift block keeps the shared base
    roughness declaration on every copy.
    """
    top_present = block.top_roughness_a is not None
    drifts_roughness = block.target == "roughness"
    names: list[str] = []
    for flat_index in range(len(block.layers)):
        repeat_index, layer_index = divmod(flat_index, block.layer_count)
        names.append(
            _expanded_roughness_name(
                prefix,
                repeat_index,
                layer_index,
                top_present=top_present,
                drifts_roughness=drifts_roughness,
            )
        )
    return tuple(names)


def _expanded_interface_names(
    structure: StructureSpec,
    gradient_slab_counts: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    """Return one source name for every expanded roughness position."""
    names: list[str] = []
    for component_index, component in enumerate(structure.components):
        prefix = f"component.{component_index}"
        if isinstance(component, GradientLayerSpec):
            count = gradient_slab_count(component, prefix, gradient_slab_counts)
            names.append(f"{prefix}.roughness_a")
            names.extend((GRADIENT_INTERNAL_INTERFACE,) * (count - 1))
        elif isinstance(component, LayerSpec):
            names.append(f"{prefix}.roughness_a")
            if component.transition is not None:
                # Microslab boundaries are numerical subdivisions, not physical
                # interfaces, so they stay out of the dynamic roughness limits.
                names.extend((GRADIENT_INTERNAL_INTERFACE,) * _transition_count(component))
        elif isinstance(component, _ExpandedDriftBlock):
            names.extend(_drift_interface_names(prefix, component))
        else:
            names.extend(_periodic_interface_names(prefix, component))
    names.append("backing.roughness_a")
    return tuple(names)


@dataclass(frozen=True, slots=True)
class GeometryExpansion:
    """Expanded finite-layer geometry used by dynamic roughness constraints."""

    thickness_a: np.ndarray
    thickness_jacobian: np.ndarray | None
    interface_names: tuple[str, ...]
    limit_thickness_a: np.ndarray
    limit_thickness_jacobian: np.ndarray | None


def _append_geometry(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    value: float,
    tangent: np.ndarray | None,
) -> None:
    # Keep value and tangent rows synchronized for analytic bound derivatives.
    thickness.append(value)
    if tangents is not None:
        assert tangent is not None
        tangents.append(np.asarray(tangent, dtype=float))


def _geometry_tangent(
    value_jacobians: dict[str, np.ndarray] | None,
    name: str,
    divisor: int = 1,
) -> np.ndarray | None:
    # A gradient microslab shares the parent derivative after equal division.
    if value_jacobians is None:
        return None
    tangent = value_jacobians[name]
    return tangent if divisor == 1 else tangent / divisor


def _append_transition_geometry(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    component: LayerSpec,
    prefix: str,
    value_jacobians: dict[str, np.ndarray] | None,
) -> None:
    # Transition widths are declared constants, so only the body carries the
    # thickness tangent; the microslabs contribute exactly zero.
    transition = component.transition
    assert transition is not None
    width = transition_width(transition)
    count = _transition_count(component)
    zero = None if tangents is None else np.zeros(len(tangents[0]), dtype=float)
    for _ in range(count):
        _append_geometry(thickness, tangents, width / count, zero)
    _append_geometry(
        thickness,
        tangents,
        component.thickness_a - width,
        _geometry_tangent(value_jacobians, f"{prefix}.thickness_a"),
    )


def _append_layer_geometry(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    component: LayerSpec,
    prefix: str,
    value_jacobians: dict[str, np.ndarray] | None,
) -> None:
    # Ordinary layers contribute one finite medium and one public interface.
    if component.transition is not None:
        _append_transition_geometry(
            thickness,
            tangents,
            component,
            prefix,
            value_jacobians,
        )
        return
    _append_geometry(
        thickness,
        tangents,
        component.thickness_a,
        _geometry_tangent(value_jacobians, f"{prefix}.thickness_a"),
    )


def _append_periodic_geometry(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    component: PeriodicBlock,
    prefix: str,
    value_jacobians: dict[str, np.ndarray] | None,
) -> None:
    # Only non-drift blocks reach here; every repeated cell shares the base
    # thickness coordinate, so the expansion stays byte-identical across repeats.
    for repeat_index in range(component.repeats):
        for layer_index, layer in enumerate(component.layers):
            name = _expanded_thickness_name(prefix, repeat_index, layer_index, drifts_thickness=False)
            _append_geometry(
                thickness,
                tangents,
                layer.thickness_a,
                _geometry_tangent(value_jacobians, name),
            )


def _append_drift_geometry(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    component: _ExpandedDriftBlock,
    prefix: str,
    value_jacobians: dict[str, np.ndarray] | None,
) -> None:
    # Per-copy thicknesses are already baked into the layers, so only the tangent
    # source name distinguishes a thickness-drift copy from the shared base cell.
    drifts_thickness = component.target == "thickness"
    for flat_index, layer in enumerate(component.layers):
        repeat_index, layer_index = divmod(flat_index, component.layer_count)
        name = _expanded_thickness_name(prefix, repeat_index, layer_index, drifts_thickness=drifts_thickness)
        _append_geometry(
            thickness,
            tangents,
            layer.thickness_a,
            _geometry_tangent(value_jacobians, name),
        )


def _append_gradient_geometry(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    component: GradientLayerSpec,
    prefix: str,
    value_jacobians: dict[str, np.ndarray] | None,
    gradient_slab_counts: Mapping[str, int] | None,
) -> None:
    # Internal gradient boundaries are numerical subdivisions, not fit axes.
    count = gradient_slab_count(component, prefix, gradient_slab_counts)
    value = component.thickness_a / count
    tangent = _geometry_tangent(
        value_jacobians,
        f"{prefix}.thickness_a",
        divisor=count,
    )
    for _ in range(count):
        _append_geometry(thickness, tangents, value, tangent)


def _append_component_geometry(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    component: LayerSpec | PeriodicBlock | GradientLayerSpec | _ExpandedDriftBlock,
    prefix: str,
    value_jacobians: dict[str, np.ndarray] | None,
    gradient_slab_counts: Mapping[str, int] | None,
) -> None:
    # Rebuilt structures close this dispatch over ordinary, baked-drift, periodic,
    # and gradient components; drift geometry is already flattened per copy.
    if isinstance(component, LayerSpec):
        _append_layer_geometry(thickness, tangents, component, prefix, value_jacobians)
        return
    if isinstance(component, _ExpandedDriftBlock):
        _append_drift_geometry(thickness, tangents, component, prefix, value_jacobians)
        return
    if isinstance(component, PeriodicBlock):
        _append_periodic_geometry(thickness, tangents, component, prefix, value_jacobians)
        return
    _append_gradient_geometry(
        thickness,
        tangents,
        component,
        prefix,
        value_jacobians,
        gradient_slab_counts,
    )


def _append_limit_rows(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    value: float,
    tangent: np.ndarray | None,
    count: int,
) -> None:
    """Repeat one declared roughness-limit thickness across expanded slabs."""
    for _ in range(count):
        _append_geometry(thickness, tangents, value, tangent)


def _append_component_limit_geometry(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    component: LayerSpec | PeriodicBlock | GradientLayerSpec | _ExpandedDriftBlock,
    prefix: str,
    value_jacobians: dict[str, np.ndarray] | None,
    gradient_slab_counts: Mapping[str, int] | None,
) -> None:
    """Mirror ``physics.stack`` roughness limits without evaluating material SLDs."""
    if isinstance(component, LayerSpec):
        name = f"{prefix}.thickness_a"
        count = _transition_count(component) + 1 if component.transition is not None else 1
        _append_limit_rows(
            thickness,
            tangents,
            component.thickness_a,
            _geometry_tangent(value_jacobians, name),
            count,
        )
        return
    if isinstance(component, _ExpandedDriftBlock):
        drifts_thickness = component.target == "thickness"
        for flat_index, layer in enumerate(component.layers):
            repeat_index, layer_index = divmod(flat_index, component.layer_count)
            name = _expanded_thickness_name(
                prefix,
                repeat_index,
                layer_index,
                drifts_thickness=drifts_thickness,
            )
            _append_limit_rows(
                thickness,
                tangents,
                layer.thickness_a,
                _geometry_tangent(value_jacobians, name),
                1,
            )
        return
    if isinstance(component, PeriodicBlock):
        for repeat_index in range(component.repeats):
            for layer_index, layer in enumerate(component.layers):
                name = _expanded_thickness_name(
                    prefix,
                    repeat_index,
                    layer_index,
                    drifts_thickness=False,
                )
                _append_limit_rows(
                    thickness,
                    tangents,
                    layer.thickness_a,
                    _geometry_tangent(value_jacobians, name),
                    1,
                )
        return
    count = gradient_slab_count(component, prefix, gradient_slab_counts)
    _append_limit_rows(
        thickness,
        tangents,
        component.thickness_a,
        _geometry_tangent(value_jacobians, f"{prefix}.thickness_a"),
        count,
    )


def _expand_limit_geometry(
    structure: StructureSpec,
    parameter_count: int | None,
    value_jacobians: dict[str, np.ndarray] | None,
    gradient_slab_counts: Mapping[str, int] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Expand the declared thicknesses used by stack roughness validation."""
    thickness: list[float] = [0.0]
    tangents = None if value_jacobians is None else [np.zeros(parameter_count, dtype=float)]
    for component_index, component in enumerate(structure.components):
        _append_component_limit_geometry(
            thickness,
            tangents,
            component,
            f"component.{component_index}",
            value_jacobians,
            gradient_slab_counts,
        )
    _append_geometry(
        thickness,
        tangents,
        0.0,
        None if value_jacobians is None else np.zeros(parameter_count, dtype=float),
    )
    return (
        np.asarray(thickness, dtype=float),
        None if tangents is None else np.asarray(tangents, dtype=float),
    )


def expand_geometry(
    structure: StructureSpec,
    parameter_count: int | None = None,
    value_jacobians: dict[str, np.ndarray] | None = None,
    gradient_slab_counts: Mapping[str, int] | None = None,
) -> GeometryExpansion:
    """Expand thickness and interface names without evaluating material SLDs.

    This is a pure function of the (already rebuilt) structure: any per-copy
    drift geometry is baked into ``_ExpandedDriftBlock`` components upstream, so
    non-drift expansion stays byte-identical and no value map is threaded here.
    """
    if (parameter_count is None) != (value_jacobians is None):
        raise ValueError("geometry Jacobian inputs must be supplied together")
    if value_jacobians is not None:
        assert parameter_count is not None
    thickness: list[float] = [0.0]
    # The leading zero is the incident medium; backing is appended below.
    tangents = None if value_jacobians is None else [np.zeros(parameter_count, dtype=float)]
    for component_index, component in enumerate(structure.components):
        _append_component_geometry(
            thickness,
            tangents,
            component,
            f"component.{component_index}",
            value_jacobians,
            gradient_slab_counts,
        )
    _append_geometry(
        thickness,
        tangents,
        0.0,
        None if value_jacobians is None else np.zeros(parameter_count, dtype=float),
    )
    names = _expanded_interface_names(structure, gradient_slab_counts)
    limit_thickness, limit_tangents = _expand_limit_geometry(
        structure,
        parameter_count,
        value_jacobians,
        gradient_slab_counts,
    )
    # Source labels and expanded media must remain positionally aligned.
    if len(names) != len(thickness) - 1 or limit_thickness.size != len(thickness):
        raise RuntimeError("expanded geometry interface mapping mismatch")
    return GeometryExpansion(
        np.asarray(thickness, dtype=float),
        None if tangents is None else np.asarray(tangents, dtype=float),
        names,
        limit_thickness,
        limit_tangents,
    )


@dataclass(frozen=True, slots=True)
class DifferentiableStack:
    """One expanded stack with aligned unit-coordinate forward tangents.

    Each Jacobian appends the same parameter axis to its corresponding stack
    array. The value stack remains the authoritative primal representation used
    by the low-level physics kernels.
    """

    stack: SlabStack
    thickness_jacobian: np.ndarray
    sld_jacobian: np.ndarray
    roughness_jacobian: np.ndarray


def _layer_sld_jacobian(
    layer: LayerSpec,
    prefix: str,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
    wavelength_a: float,
) -> np.ndarray:
    """Differentiate formula or explicit-SLD material density scaling.

    Formula SLD varies only through density. An explicit complex override also
    carries real and imaginary coordinate tangents before density multiplication,
    preserving the primal material convention at the requested wavelength.
    """
    density_name = f"{prefix}.density_scale"
    density = values[density_name]
    density_jacobian = value_jacobians[density_name]
    sld = material_sld(layer.material, density, wavelength_a)
    # Formula materials expose density only, so SLD/density is the local slope.
    # Explicit materials also carry complex override coordinates through density.
    if layer.material.sld_override_a2 is None:
        return (sld / density) * density_jacobian
    real_name = f"{prefix}.sld_real_a2"
    imaginary_name = f"{prefix}.sld_imag_a2"
    override_jacobian = value_jacobians.get(
        real_name,
        np.zeros_like(density_jacobian),
    ) + 1j * value_jacobians.get(
        imaginary_name,
        np.zeros_like(density_jacobian),
    )
    return layer.material.sld_override_a2 * density_jacobian + density * override_jacobian


@dataclass(slots=True)
class _StackJacobianBuilder:
    """Accumulate expanded slab tangents in primal stack order.

    Fronting and backing media are inserted explicitly, while layer components
    append their incident interface roughness. The per-expansion SLD cache lets
    repeated periodic cells share an identical source tangent without leaking
    state between wavelengths or candidates.
    """

    parameter_count: int
    thickness: list[np.ndarray] = field(default_factory=list)
    sld: list[np.ndarray] = field(default_factory=list)
    roughness: list[np.ndarray] = field(default_factory=list)
    sld_cache: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def create(cls, parameter_count: int) -> _StackJacobianBuilder:
        """Start with the semi-infinite fronting medium tangent rows.

        Fronting thickness and SLD are fixed by the compiled model and therefore
        begin as zero real and complex vectors respectively.
        """
        builder = cls(parameter_count)
        builder.thickness.append(builder.zero_real())
        builder.sld.append(builder.zero_complex())
        return builder

    def zero_real(self) -> np.ndarray:
        return np.zeros(self.parameter_count, dtype=float)

    def zero_complex(self) -> np.ndarray:
        return np.zeros(self.parameter_count, dtype=np.complex128)

    def append_layer(
        self,
        layer: LayerSpec,
        prefix: str,
        thickness_name: str,
        roughness_name: str,
        values: dict[str, float],
        value_jacobians: dict[str, np.ndarray],
        wavelength_a: float,
    ) -> None:
        """Append one finite layer and the roughness of its incident interface.

        The caller supplies thickness and roughness source names separately from
        ``prefix``: a drifted periodic copy differentiates its own ``repeat.{k}``
        coordinate, while SLD and density stay bound to the shared base cell that
        ``prefix`` still keys, so the per-expansion cache remains correct.
        """
        self.thickness.append(np.asarray(value_jacobians[thickness_name], dtype=float))
        # Cache scope is one builder and therefore one candidate/wavelength pair.
        # Repeated cells reuse values without sharing mutable state across calls.
        if prefix not in self.sld_cache:
            self.sld_cache[prefix] = np.asarray(
                _layer_sld_jacobian(
                    layer,
                    prefix,
                    values,
                    value_jacobians,
                    wavelength_a,
                ),
                dtype=np.complex128,
            )
        self.sld.append(self.sld_cache[prefix])
        self.roughness.append(np.asarray(value_jacobians[roughness_name], dtype=float))

    def append_periodic(
        self,
        block: PeriodicBlock,
        prefix: str,
        values: dict[str, float],
        value_jacobians: dict[str, np.ndarray],
        wavelength_a: float,
    ) -> None:
        """Expand repeated cells while retaining their shared tangent sources.

        Only non-drift blocks reach here; a drifted block is baked into an
        ``_ExpandedDriftBlock`` upstream and differentiated by ``append_drift_block``.
        Flat traversal is equivalent to repeat-major nested loops and preserves
        exact slab order. It avoids an additional control-flow axis without
        materializing a potentially large Cartesian product.
        """
        top_present = block.top_roughness_a is not None
        layer_count = len(block.layers)
        for flat_index in range(block.repeats * layer_count):
            repeat_index, layer_index = divmod(flat_index, layer_count)
            layer_prefix = f"{prefix}.layer.{layer_index}"
            self.append_layer(
                block.layers[layer_index],
                layer_prefix,
                _expanded_thickness_name(prefix, repeat_index, layer_index, drifts_thickness=False),
                _expanded_roughness_name(
                    prefix,
                    repeat_index,
                    layer_index,
                    top_present=top_present,
                    drifts_roughness=False,
                ),
                values,
                value_jacobians,
                wavelength_a,
            )

    def append_drift_block(
        self,
        block: _ExpandedDriftBlock,
        prefix: str,
        values: dict[str, float],
        value_jacobians: dict[str, np.ndarray],
        wavelength_a: float,
    ) -> None:
        """Differentiate baked per-copy drift layers in flattened copy-major order.

        The flattened layout recovers ``(copy, layer)`` through
        ``divmod(flat_index, layer_count)``. Thickness- or roughness-drift copies
        differentiate their own ``repeat.{k}`` coordinate, while SLD and density
        stay bound to the shared base ``layer_prefix`` — those families never drift,
        so the per-expansion SLD cache correctly shares across copies.
        """
        top_present = block.top_roughness_a is not None
        drifts_thickness = block.target == "thickness"
        drifts_roughness = block.target == "roughness"
        for flat_index, layer in enumerate(block.layers):
            repeat_index, layer_index = divmod(flat_index, block.layer_count)
            layer_prefix = f"{prefix}.layer.{layer_index}"
            self.append_layer(
                layer,
                layer_prefix,
                _expanded_thickness_name(prefix, repeat_index, layer_index, drifts_thickness=drifts_thickness),
                _expanded_roughness_name(
                    prefix,
                    repeat_index,
                    layer_index,
                    top_present=top_present,
                    drifts_roughness=drifts_roughness,
                ),
                values,
                value_jacobians,
                wavelength_a,
            )

    def append_gradient(
        self,
        layer: GradientLayerSpec,
        prefix: str,
        value_jacobians: dict[str, np.ndarray],
        gradient_slab_counts: Mapping[str, int] | None,
    ) -> None:
        """Differentiate equal-width microslabs and linear complex-SLD centers.

        Total thickness tangent is divided equally across slabs. Only the first
        incident interface carries declared gradient roughness; internal
        numerical boundaries retain exact zero tangents.
        """
        count = gradient_slab_count(layer, prefix, gradient_slab_counts)
        thickness_jacobian = value_jacobians[f"{prefix}.thickness_a"] / count
        upper_jacobian = (
            value_jacobians[f"{prefix}.upper_sld_real_a2"] + 1j * value_jacobians[f"{prefix}.upper_sld_imag_a2"]
        )
        lower_jacobian = (
            value_jacobians[f"{prefix}.lower_sld_real_a2"] + 1j * value_jacobians[f"{prefix}.lower_sld_imag_a2"]
        )
        delta_jacobian = lower_jacobian - upper_jacobian
        finite_delta = np.all(np.isfinite(delta_jacobian.real)) and np.all(np.isfinite(delta_jacobian.imag))
        roughness_jacobians = [self.zero_real()] * count
        roughness_jacobians[0] = value_jacobians[f"{prefix}.roughness_a"].copy()
        for slab_index in range(count):
            fraction = (slab_index + 0.5) / count
            self.thickness.append(thickness_jacobian.copy())
            self.sld.append(
                upper_jacobian + fraction * delta_jacobian
                if finite_delta
                else (1.0 - fraction) * upper_jacobian + fraction * lower_jacobian
            )
            self.roughness.append(roughness_jacobians[slab_index])

    def append_transition_layer(
        self,
        layer: LayerSpec,
        prefix: str,
        values: dict[str, float],
        value_jacobians: dict[str, np.ndarray],
        wavelength_a: float,
    ) -> None:
        """Differentiate a graded incident interface into microslabs plus body.

        The blend fractions are declared constants, so each microslab tangent is
        a fixed linear combination of the upper medium and this layer. The upper
        tangent must be read before any row is appended, mirroring the primal
        expansion's ``state.sld[-1]``.
        """
        transition = layer.transition
        assert transition is not None
        upper_jacobian = self.sld[-1]
        lower_jacobian = np.asarray(
            _layer_sld_jacobian(layer, prefix, values, value_jacobians, wavelength_a),
            dtype=np.complex128,
        )
        count = _transition_count(layer)
        fractions = transition_fractions(transition, count)
        for index, fraction in enumerate(fractions):
            self.thickness.append(self.zero_real())
            self.sld.append((1.0 - fraction) * upper_jacobian + fraction * lower_jacobian)
            self.roughness.append(
                np.asarray(value_jacobians[f"{prefix}.roughness_a"], dtype=float).copy()
                if index == 0
                else self.zero_real()
            )
        self.thickness.append(np.asarray(value_jacobians[f"{prefix}.thickness_a"], dtype=float))
        self.sld.append(lower_jacobian)
        self.roughness.append(self.zero_real())

    def append_component(
        self,
        component: LayerSpec | PeriodicBlock | GradientLayerSpec | _ExpandedDriftBlock,
        prefix: str,
        values: dict[str, float],
        value_jacobians: dict[str, np.ndarray],
        wavelength_a: float,
        gradient_slab_counts: Mapping[str, int] | None,
    ) -> None:
        """Dispatch one validated structure component to its concrete expander.

        A rebuilt structure closes this dispatch over ordinary, baked-drift,
        periodic, and gradient components; drift geometry arrives already
        flattened per copy as an ``_ExpandedDriftBlock``.
        """
        if isinstance(component, LayerSpec) and component.transition is not None:
            self.append_transition_layer(
                component,
                prefix,
                values,
                value_jacobians,
                wavelength_a,
            )
        elif isinstance(component, LayerSpec):
            self.append_layer(
                component,
                prefix,
                f"{prefix}.thickness_a",
                f"{prefix}.roughness_a",
                values,
                value_jacobians,
                wavelength_a,
            )
        elif isinstance(component, _ExpandedDriftBlock):
            self.append_drift_block(
                component,
                prefix,
                values,
                value_jacobians,
                wavelength_a,
            )
        elif isinstance(component, PeriodicBlock):
            self.append_periodic(
                component,
                prefix,
                values,
                value_jacobians,
                wavelength_a,
            )
        else:
            self.append_gradient(component, prefix, value_jacobians, gradient_slab_counts)

    def finish(
        self,
        stack: SlabStack,
        backing_sld_jacobian: np.ndarray,
    ) -> DifferentiableStack:
        """Append backing rows and prove positional alignment with the stack.

        A shape mismatch would attach a derivative to the wrong physical slab
        or interface. It is rejected before any reflectivity kernel consumes the
        constructed tangent arrays.
        """
        self.thickness.append(self.zero_real())
        self.sld.append(np.asarray(backing_sld_jacobian, dtype=np.complex128))
        differentiable = DifferentiableStack(
            stack,
            np.asarray(self.thickness, dtype=float),
            np.asarray(self.sld, dtype=np.complex128),
            np.asarray(self.roughness, dtype=float),
        )
        # Every stack axis must gain exactly one trailing free-parameter axis.
        # A mismatch here would silently differentiate the wrong physical layer.
        expected = (self.parameter_count,)
        valid = all(
            (
                differentiable.thickness_jacobian.shape == stack.thickness_a.shape + expected,
                differentiable.sld_jacobian.shape == stack.sld_a2.shape + expected,
                differentiable.roughness_jacobian.shape == stack.roughness_a.shape + expected,
            )
        )
        if not valid:
            raise RuntimeError("expanded structure Jacobian mapping mismatch")
        return differentiable


def expand_structure_with_jacobian(
    structure: StructureSpec,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
    wavelength_a: float,
    parameter_count: int,
    gradient_slab_counts: Mapping[str, int] | None = None,
) -> DifferentiableStack:
    """Expand one wavelength-specific structure and its aligned tangents."""
    rebuilt = rebuild_structure(structure, values)
    stack = expand_structure(rebuilt, wavelength_a, gradient_slab_counts)
    builder = _StackJacobianBuilder.create(parameter_count)
    for component_index, component in enumerate(rebuilt.components):
        builder.append_component(
            component,
            f"component.{component_index}",
            values,
            value_jacobians,
            wavelength_a,
            gradient_slab_counts,
        )
    builder.roughness.append(np.asarray(value_jacobians["backing.roughness_a"], dtype=float))
    backing_sld_jacobian = builder.zero_complex()
    if rebuilt.backing.sld_override_a2 is not None:
        backing_sld_jacobian = value_jacobians["backing.sld_real_a2"] + 1j * value_jacobians["backing.sld_imag_a2"]
    return builder.finish(stack, backing_sld_jacobian)
