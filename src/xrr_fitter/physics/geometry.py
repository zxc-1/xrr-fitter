"""Material-independent structure geometry for evaluation constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    PeriodicBlock,
    SlabStack,
    StructureSpec,
)
from xrr_fitter.physics.materials import material_sld
from xrr_fitter.physics.stack import expand_structure, rebuild_structure


GRADIENT_INTERNAL_INTERFACE = "__gradient_internal_zero__"


def _periodic_interface_names(prefix: str, block: PeriodicBlock) -> tuple[str, ...]:
    """Mirror repeated interfaces back to their shared source coordinates."""
    ordinary = tuple(
        map("layer.{}.roughness_a".format, range(len(block.layers)))
    )
    first = ordinary[0] if block.top_roughness_a is None else "top_roughness_a"
    suffixes = (first, *ordinary[1:], *(ordinary * (block.repeats - 1)))
    return tuple(map(f"{prefix}.{{}}".format, suffixes))


def _expanded_interface_names(structure: StructureSpec) -> tuple[str, ...]:
    """Return one source name for every expanded roughness position."""
    names: list[str] = []
    for component_index, component in enumerate(structure.components):
        prefix = f"component.{component_index}"
        if isinstance(component, GradientLayerSpec):
            count = int(np.ceil(component.thickness_a / component.microslab_max_a))
            names.append(f"{prefix}.roughness_a")
            names.extend((GRADIENT_INTERNAL_INTERFACE,) * (count - 1))
        elif isinstance(component, LayerSpec):
            names.append(f"{prefix}.roughness_a")
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


def _append_layer_geometry(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    component: LayerSpec,
    prefix: str,
    value_jacobians: dict[str, np.ndarray] | None,
) -> None:
    # Ordinary layers contribute one finite medium and one public interface.
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
    # Every repeated cell keeps the same source-coordinate thickness tangent.
    for _ in range(component.repeats):
        for layer_index, layer in enumerate(component.layers):
            _append_geometry(
                thickness,
                tangents,
                layer.thickness_a,
                _geometry_tangent(
                    value_jacobians,
                    f"{prefix}.layer.{layer_index}.thickness_a",
                ),
            )


def _append_gradient_geometry(
    thickness: list[float],
    tangents: list[np.ndarray] | None,
    component: GradientLayerSpec,
    prefix: str,
    value_jacobians: dict[str, np.ndarray] | None,
) -> None:
    # Internal gradient boundaries are numerical subdivisions, not fit axes.
    count = int(np.ceil(component.thickness_a / component.microslab_max_a))
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
    component: LayerSpec | PeriodicBlock | GradientLayerSpec,
    prefix: str,
    value_jacobians: dict[str, np.ndarray] | None,
) -> None:
    # Structure validation closes this dispatch over the three component kinds.
    if isinstance(component, LayerSpec):
        _append_layer_geometry(thickness, tangents, component, prefix, value_jacobians)
        return
    if isinstance(component, PeriodicBlock):
        _append_periodic_geometry(
            thickness,
            tangents,
            component,
            prefix,
            value_jacobians,
        )
        return
    _append_gradient_geometry(thickness, tangents, component, prefix, value_jacobians)


def expand_geometry(
    structure: StructureSpec,
    parameter_count: int | None = None,
    value_jacobians: dict[str, np.ndarray] | None = None,
) -> GeometryExpansion:
    """Expand thickness and interface names without evaluating material SLDs."""
    if (parameter_count is None) != (value_jacobians is None):
        raise ValueError("geometry Jacobian inputs must be supplied together")
    if value_jacobians is not None:
        assert parameter_count is not None
    thickness: list[float] = [0.0]
    # The leading zero is the incident medium; backing is appended below.
    tangents = (
        None
        if value_jacobians is None
        else [np.zeros(parameter_count, dtype=float)]
    )
    for component_index, component in enumerate(structure.components):
        _append_component_geometry(
            thickness,
            tangents,
            component,
            f"component.{component_index}",
            value_jacobians,
        )
    _append_geometry(
        thickness,
        tangents,
        0.0,
        None
        if value_jacobians is None
        else np.zeros(parameter_count, dtype=float),
    )
    names = _expanded_interface_names(structure)
    # Source labels and expanded media must remain positionally aligned.
    if len(names) != len(thickness) - 1:
        raise RuntimeError("expanded geometry interface mapping mismatch")
    return GeometryExpansion(
        np.asarray(thickness, dtype=float),
        None if tangents is None else np.asarray(tangents, dtype=float),
        names,
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
    return (
        layer.material.sld_override_a2 * density_jacobian
        + density * override_jacobian
    )


def _periodic_roughness_name(
    prefix: str,
    block: PeriodicBlock,
    repeat_index: int,
    layer_index: int,
) -> str:
    top_interface = (repeat_index, layer_index, block.top_roughness_a is not None)
    if top_interface == (0, 0, True):
        return f"{prefix}.top_roughness_a"
    return f"{prefix}.layer.{layer_index}.roughness_a"


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
        roughness_name: str,
        values: dict[str, float],
        value_jacobians: dict[str, np.ndarray],
        wavelength_a: float,
    ) -> None:
        """Append one finite layer and the roughness of its incident interface.

        The caller supplies the roughness source name separately because the
        first periodic interface may use a block override while all later cells
        share their ordinary layer declaration.
        """
        self.thickness.append(
            np.asarray(value_jacobians[f"{prefix}.thickness_a"], dtype=float)
        )
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
        self.roughness.append(
            np.asarray(value_jacobians[roughness_name], dtype=float)
        )

    def append_periodic(
        self,
        block: PeriodicBlock,
        prefix: str,
        values: dict[str, float],
        value_jacobians: dict[str, np.ndarray],
        wavelength_a: float,
    ) -> None:
        """Expand repeated cells while retaining their shared tangent sources.

        Flat traversal is equivalent to repeat-major nested loops and preserves
        exact slab order. It avoids an additional control-flow axis without
        materializing a potentially large Cartesian product.
        """
        layer_count = len(block.layers)
        for flat_index in range(block.repeats * layer_count):
            repeat_index, layer_index = divmod(flat_index, layer_count)
            layer_prefix = f"{prefix}.layer.{layer_index}"
            self.append_layer(
                block.layers[layer_index],
                layer_prefix,
                _periodic_roughness_name(
                    prefix,
                    block,
                    repeat_index,
                    layer_index,
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
    ) -> None:
        """Differentiate equal-width microslabs and linear complex-SLD centers.

        Total thickness tangent is divided equally across slabs. Only the first
        incident interface carries declared gradient roughness; internal
        numerical boundaries retain exact zero tangents.
        """
        count = int(np.ceil(layer.thickness_a / layer.microslab_max_a))
        thickness_jacobian = value_jacobians[f"{prefix}.thickness_a"] / count
        upper_jacobian = (
            value_jacobians[f"{prefix}.upper_sld_real_a2"]
            + 1j * value_jacobians[f"{prefix}.upper_sld_imag_a2"]
        )
        lower_jacobian = (
            value_jacobians[f"{prefix}.lower_sld_real_a2"]
            + 1j * value_jacobians[f"{prefix}.lower_sld_imag_a2"]
        )
        roughness_jacobians = [self.zero_real()] * count
        roughness_jacobians[0] = value_jacobians[f"{prefix}.roughness_a"].copy()
        for slab_index in range(count):
            fraction = (slab_index + 0.5) / count
            self.thickness.append(thickness_jacobian.copy())
            self.sld.append(
                upper_jacobian + fraction * (lower_jacobian - upper_jacobian)
            )
            self.roughness.append(roughness_jacobians[slab_index])

    def append_component(
        self,
        component: LayerSpec | PeriodicBlock | GradientLayerSpec,
        prefix: str,
        values: dict[str, float],
        value_jacobians: dict[str, np.ndarray],
        wavelength_a: float,
    ) -> None:
        """Dispatch one validated structure component to its concrete expander.

        ``StructureSpec`` validation closes this union to ordinary, periodic,
        and gradient components, so no fallback component semantics are needed.
        """
        if isinstance(component, LayerSpec):
            self.append_layer(
                component,
                prefix,
                f"{prefix}.roughness_a",
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
            self.append_gradient(component, prefix, value_jacobians)

    def finish(self, stack: SlabStack) -> DifferentiableStack:
        """Append backing rows and prove positional alignment with the stack.

        A shape mismatch would attach a derivative to the wrong physical slab
        or interface. It is rejected before any reflectivity kernel consumes the
        constructed tangent arrays.
        """
        self.thickness.append(self.zero_real())
        self.sld.append(self.zero_complex())
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
                differentiable.thickness_jacobian.shape
                == stack.thickness_a.shape + expected,
                differentiable.sld_jacobian.shape
                == stack.sld_a2.shape + expected,
                differentiable.roughness_jacobian.shape
                == stack.roughness_a.shape + expected,
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
) -> DifferentiableStack:
    """Expand one wavelength-specific structure and its aligned tangents."""
    rebuilt = rebuild_structure(structure, values)
    stack = expand_structure(rebuilt, wavelength_a)
    builder = _StackJacobianBuilder.create(parameter_count)
    for component_index, component in enumerate(rebuilt.components):
        builder.append_component(
            component,
            f"component.{component_index}",
            values,
            value_jacobians,
            wavelength_a,
        )
    builder.roughness.append(
        np.asarray(value_jacobians["backing.roughness_a"], dtype=float)
    )
    return builder.finish(stack)
