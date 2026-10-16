"""Strict parsing of scalar wavelengths explicitly declared by source files."""

from __future__ import annotations

import re
from math import isclose, isfinite

from xrr_fitter.model.data import BeamSpec

WAVELENGTH_LABEL_PATTERN = re.compile(r"(?<![\w])(?:wavelength|lambda|λ|波长)\s*[:=]\s*", re.IGNORECASE)
WAVELENGTH_VALUE_PATTERN = re.compile(
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(angstroms?|Å|Å|A|nm)(?![\w])",
    re.IGNORECASE,
)


def _wavelength_value(row: str, start: int) -> float:
    """Validate one explicitly labelled scalar and normalize its unit to Å."""
    match = WAVELENGTH_VALUE_PATTERN.match(row, start)
    if match is None:
        raise ValueError(f"wavelength 声明无效：请明确数值及 A/Å/angstrom 或 nm 单位：{row}")
    value = float(match[1]) * (10.0 if match[2].lower() == "nm" else 1.0)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"wavelength 必须为正的有限值：{row}")
    return value


def _row_wavelengths(row: str) -> tuple[float, ...]:
    """Parse every explicit declaration, including repeated labels on one row."""
    return tuple(_wavelength_value(row, label.end()) for label in WAVELENGTH_LABEL_PATTERN.finditer(row))


def declared_wavelength(rows: tuple[str, ...]) -> float | None:
    """Parse the complete pre-data header supplied by the authoritative reader."""
    values = tuple(value for row in rows for value in _row_wavelengths(row))
    if not values:
        return None
    if not all(isclose(value, values[0], rel_tol=1e-7) for value in values):
        raise ValueError(f"wavelength 表头声明相互冲突（Å）：{values}")
    return values[0]


def validate_declared_wavelength(rows: tuple[str, ...], beam: BeamSpec) -> None:
    """Reject a beam that contradicts explicit source metadata."""
    declared = declared_wavelength(rows)
    if declared is None:
        return
    if beam.kind != "monochromatic" or not isclose(beam.wavelength_a, declared, rel_tol=1e-7):
        raise ValueError(
            f"wavelength 与源文件表头不一致：文件声明单色 {declared:g} Å；"
            "请核对光路和波长，已有预设请使用主页的更换测量预设。"
        )
