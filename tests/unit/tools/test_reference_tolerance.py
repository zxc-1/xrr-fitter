from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


TOLERANCE = {"absolute": 1e-12, "relative": 1e-10}
EXPORT_POLICY = {
    "kind": "normalized_export",
    "mutable_suffixes": ["fit_result.xlsx"],
    "maximum_size_delta": 32,
}


def _export(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "timestamp": "20260726T120000Z",
        "token": "fixed",
        "manifest": {"dataset_order": ["curve"]},
        "artifacts": [
            {"path": "curve/plot.png", "size": 10, "sha256": "a" * 64},
            artifact,
        ],
    }


def _workbook(digest: str, member_digest: str, size: int = 100) -> dict[str, object]:
    return {
        "path": "curve/fit_result.xlsx",
        "sha256": digest * 64,
        "members": [
            {
                "path": "xl/sheet1.xml",
                "size": size,
                "sha256": member_digest * 64,
            }
        ],
    }


def test_version_tolerance_accepts_bounded_arrays_float_trees_and_exports(
    load_tool_module,
) -> None:
    module = load_tool_module("compare_r22_reference")
    module.compare_value(
        np.asarray([1.0, np.nan, np.inf]),
        np.asarray([1.0 + 1e-13, np.nan, np.inf]),
        {"kind": "array_tolerance", **TOLERANCE},
    )
    module.compare_value(
        {"candidate_id": "E-0", "nfev": 7, "values": [1.0, True]},
        {"candidate_id": "E-0", "nfev": 7, "values": [1.0 + 1e-13, True]},
        {"kind": "float_tree_tolerance", **TOLERANCE},
    )
    module.compare_value(
        _export(_workbook("b", "c")),
        _export(_workbook("d", "e", 112)),
        EXPORT_POLICY,
    )


def test_committed_reference_limits_fit_array_tolerances_to_verified_float_drift() -> None:
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads(
        (root / "verification/r22/reference/manifest.json").read_text(encoding="utf-8")
    )
    groups = manifest["groups"]
    compile_fields = groups["fit_compile"]["comparison_policy"]["fields"]
    compile_arrays = compile_fields["golden/fit_compile.npz"]["fields"]
    jacobian_tolerance = {
        "kind": "array_tolerance",
        "absolute": 1e-12,
        "relative": 2e-10,
    }
    assert compile_arrays["single_jacobian"] == jacobian_tolerance
    assert compile_arrays["mo_si_jacobian"] == jacobian_tolerance

    search_fields = groups["fit_search"]["comparison_policy"]["fields"]
    search_arrays = search_fields["golden/fit_search.npz"]["fields"]
    candidate_tolerance = {
        "kind": "array_tolerance",
        "absolute": 1e-13,
        "relative": 1e-12,
    }
    derived_suffixes = (
        "log_residuals",
        "sld_imag",
        "sld_real",
        "unit",
        "weighted_residuals",
    )
    for index in range(10):
        prefix = f"candidate_{index:02d}"
        expected = candidate_tolerance if index >= 6 else "array_equal"
        for suffix in derived_suffixes:
            assert search_arrays[f"{prefix}_{suffix}"] == expected
        assert search_arrays[f"{prefix}_qz_a_inv"] == "array_equal"
        assert search_arrays[f"{prefix}_sld_depth"] == "array_equal"
        assert search_arrays[f"{prefix}_model"] == "physics_reflectivity"


@pytest.mark.parametrize(
    ("reference", "actual", "policy"),
    (
        (
            np.ones(2),
            np.asarray([1.0, 1.1]),
            {"kind": "array_tolerance", **TOLERANCE},
        ),
        (
            {"nfev": 7, "value": 1.0},
            {"nfev": 8, "value": 1.0},
            {"kind": "float_tree_tolerance", **TOLERANCE},
        ),
        (
            _export(_workbook("b", "c")),
            {
                **_export(_workbook("b", "c")),
                "artifacts": [
                    {"path": "curve/plot.png", "size": 10, "sha256": "f" * 64},
                    _workbook("b", "c"),
                ],
            },
            EXPORT_POLICY,
        ),
    ),
)
def test_version_tolerance_rejects_structural_integer_and_unlisted_artifact_drift(
    load_tool_module,
    reference: object,
    actual: object,
    policy: object,
) -> None:
    module = load_tool_module("compare_r22_reference")
    with pytest.raises(ValueError):
        module.compare_value(reference, actual, policy)
