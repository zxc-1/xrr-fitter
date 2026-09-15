"""Fixed Poisson experiment recipe and dataset-level binomial summaries.

These CP intervals concern independent observed datasets, never the correlated
rows of a single symmetric Monte Carlo calibration matrix. Missing attempts
remain visible, and no endpoint is eligible for acceptance until all its fixed
seeds have records. The 200 historical seed numbers are a new current-API replay,
not an input-file migration or an additional independent holdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta

import interval_coverage_cases as coverage
import xrr_fitter.api as api

# Fail closed until the revised protocol has its own reviewed approval artifact.
REVIEW_SHA256 = "9026904726aff644516b552f40095e1f52d897397e0d3db0254eef13df297b9b"
EXPECTED_VERSIONS = {
    "schema_version": 4,
    "algorithm_version": "xrr-fit-v2-poisson-4",
    "objective_version": "2",
    "diagnostic_version": "poisson-refit-null-v4",
}
GROUPS = ("null_single", "null_joint", "background", "footprint", "surface", "acf", "replay_low")
TARGETS = ("background", "footprint", "surface", "acf")
GAMMA = 0.00625
ALPHA = 0.01
INJECTIONS = {
    "background": {"slope_q_over_qmax": 2000.0},
    "footprint": {"spill_angle_deg": 0.30},
    "surface": {"thickness_a": 30.0, "sld_real_a2": 18.9e-6, "roughness_a": 1.0},
    "acf": {
        "block_size": 8,
        "shared_fraction": 0.9,
        "formula": "Y_i=C_b+E_i; C_b~Poisson(.9*min(mu_block)); E_i~Poisson(mu_i-c_b)",
    },
}


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def config(group: str, seed: int) -> api.FitConfig:
    """All free paths use the existing coverage solver budget, with fixed B."""
    if group == "replay_low":
        return coverage._config("poisson_low", seed)
    base = coverage._config("poisson_low", seed + 10000000)
    budget = replace(base.budget, diagnostic_samples=999, bootstrap_samples=200 if group == "null_joint" else 1)
    return replace(base, budget=budget)


def _scale_declaration(group: str) -> dict:
    if group == "null_joint":
        return {
            "free_parameters": [coverage.TARGET, "instrument.scale"],
            "shared_parameters": [coverage.TARGET],
            "scale_initial_factor": 0.9,
            "scale_bound_factors": [0.5, 1.5],
        }
    return {
        "free_parameters": [coverage.TARGET],
        "shared_parameters": [],
        "scale_initial_factor": 1.0,
        "scale_bound_factors": [1.0, 1.0],
    }


def scenario(group: str) -> dict:
    if group not in GROUPS:
        raise ValueError(f"unknown scenario: {group}")
    index = GROUPS.index(group)
    kind = "null" if index < 2 else "power"
    start = (3000000, 3010000, 4000000, 4010000, 4020000, 4030000, 0)[index]
    count = 2000 if index < 2 else 100
    if group == "replay_low":
        count, kind = 200, "replay"
    return {
        "scenario_id": index,
        "kind": kind,
        "target_column": group if kind == "power" else None,
        "seeds": list(range(start, start + count)),
        "raw_amplitudes": [400.0, 1600.0] if group == "null_joint" else [1e7 if kind == "power" else 400.0],
        "theta_deg_start_stop_count": [0.03, 3.0, 520] if kind == "power" else [0.15, 1.8, 80],
        "fit_config_first_seed": asdict(config(group, start)),
    } | _scale_declaration(group)


def definition() -> dict:
    return {
        "scenarios": {name: scenario(name) for name in GROUPS},
        "holdout_seed_count": 4400,
        "replay_seed_count": 200,
        "gamma": GAMMA,
        "simultaneous_one_sided_claim_count": 8,
        "diagnostic_samples": 999,
        "diagnostic_alpha": ALPHA,
        "diagnostic_seed_domain": 0x504F495344494147,
        "diagnostic_method": "poisson_refit_null_rms_v4",
        "refit_policy": "declared_sobol4_lbfgsb_trf_v3",
        "family_standardization": {
            "center": "all-row median",
            "scale": "all-row median-centered RMS; ddof=0",
            "score": "peak-factorized z; S_i=max(0,max_j(z_ij))",
            "reduction": "per-column contiguous float64; sorted squared relatives; math.fsum",
            "constant": "exact all-row equality; scale=0; z=0",
            "tails": "inclusive >=; no jitter",
            "numeric_failure": "unavailable; no epsilon or fallback",
        },
        "observation_rng": "Generator(PCG64(SeedSequence([20260912, scenario_id, seed, member_id])))",
        "observation_rng_member_order": "0-based; independent member streams; blocks drawn left-to-right C_b then E_i",
        "fit_master_seed": "seed+10000000 for holdout; seed for replay_low",
        "replay_rng": "current interval_coverage_cases.build_case(root, 'poisson_low', seed): SeedSequence([seed,1,0])",
        "normalization": coverage.experiment_definition()["normalization"],
        "structure": coverage._structure_definition(),
        "beam": asdict(coverage.BEAM),
        "fit_instrument": asdict(coverage.INSTRUMENT),
        "thickness": {"parameter": coverage.TARGET, "truth": 100.0, "initial": 95.0, "bounds": [75.0, 125.0]},
        "all_points_fit": True,
        "scale_prior_enabled": False,
        "injections": json.loads(canonical(INJECTIONS)),
        "null_gates": {"F_upper_max": 0.02, "F_count_max": 24, "U_upper_max": 0.01, "U_count_max": 9},
        "power_gate": {"target_lower_min": 0.8, "target_count_min": 90},
        "failure_policy": (
            "U includes product/numerical/persistence failures; no replacement or retuning; screen-negative is not U"
        ),
        "interruption_policy": (
            "keep incomplete; explicit seal-interrupted records U without refitting or replacing the seed"
        ),
        "joint_interval_scope": (
            "no joint profile is provided; record its absence and actual saved bootstrap separately"
        ),
        "shard_assignment": "zero-based index of (GROUPS order, ascending seed) modulo shard_count",
    }


def version_identity() -> dict:
    project = api.new_project()
    fit = api.FitConfig.fast(0)
    return {
        "schema_version": project.schema_version,
        "algorithm_version": project.algorithm_version,
        "objective_version": fit.objective_version,
        "diagnostic_version": fit.diagnostic_version,
    }


def runtime_identity() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def _source_paths(root: Path) -> list[Path]:
    paths = []
    for folder in ("src", "tools", "requirements"):
        for path in (root / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                paths.append(path)
    paths.extend(
        path for path in (root / "pyproject.toml", root / "requirements.txt", root / "uv.lock") if path.is_file()
    )
    paths.extend(root.glob("requirements*.lock"))
    return sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())


def source_identity(root: Path) -> dict:
    files = [
        {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in _source_paths(root)
    ]
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=root, capture_output=True, text=True, check=False)
    return {"sha256": digest(files), "head": head.stdout.strip() if head.returncode == 0 else None, "files": files}


def cp_bound(successes: int, trials: int, side: str) -> float:
    if not 0 <= successes <= trials or trials < 1 or side not in {"lower", "upper"}:
        raise ValueError("invalid Clopper-Pearson counts or side")
    if side == "lower":
        return 0.0 if successes == 0 else float(beta.ppf(GAMMA, successes, trials - successes + 1))
    return 1.0 if successes == trials else float(beta.ppf(1.0 - GAMMA, successes + 1, trials - successes))


def endpoint(count: int, trials: int, kind: str) -> dict:
    side, threshold = {"F": ("upper", 0.02), "U": ("upper", 0.01), "power": ("lower", 0.8)}[kind]
    bound = cp_bound(count, trials, side)
    return {
        "count": count,
        "trials": trials,
        "gamma": GAMMA,
        "side": side,
        "bound": bound,
        "threshold": threshold,
        "passed": bound >= threshold if side == "lower" else bound <= threshold,
    }


def _state(row: dict) -> str:
    state = row["diagnostic"]["status"]
    if state not in {"not_triggered", "not_rejected", "rejected", "unavailable"}:
        raise ValueError(f"unknown diagnostic status: {state}")
    return state if row["fit_available"] else "unavailable"


def _interval_counts(rows: list[dict], key: str, count: int) -> dict:
    available = [row[key] for row in rows if row["fit_available"] and row[key]["available"]]
    covered = sum(interval["bounds"][0] <= coverage.TRUTH <= interval["bounds"][1] for interval in available)
    return {
        "available": len(available),
        "unavailable_completed": len(rows) - len(available),
        "covered": covered,
        "availability": len(available) / count,
        "all_seed_coverage": covered / count,
        "conditional_coverage": None if not available else covered / len(available),
    }


def _reason_counts(rows: list[dict]) -> dict:
    reasons = [
        row["diagnostic"]["unavailable_reason"] or row.get("failure_reason") or "unspecified_failure"
        for row in rows
        if _state(row) == "unavailable"
    ]
    return dict(sorted(Counter(reasons).items()))


def _descriptive_counts(rows: list[dict], count: int) -> dict:
    target = sum(row["fit_available"] and row["diagnostic"]["target_detected"] for row in rows)
    raw = sum(row["diagnostic"]["raw_triggered"] is True for row in rows)
    columns = Counter(kind for row in rows for kind in row["diagnostic"]["rejected_columns"])
    return {
        "fit_failed": sum(not row["fit_available"] for row in rows),
        "raw_triggered": raw,
        "raw_trigger_rate": raw / count,
        "raw_unavailable": sum(row["diagnostic"]["raw_triggered"] is None for row in rows),
        "target_detected": target,
        "target_detection_rate": target / count,
        "rejected_columns": dict(columns),
        "elapsed_seconds": sum(row.get("elapsed_seconds", 0.0) for row in rows),
        "fit_seconds": sum(row.get("fit_seconds", 0.0) for row in rows),
    }


def summarize(rows: list[dict], *, required_count: int) -> dict:
    if required_count < 1 or len(rows) > required_count:
        raise ValueError("invalid fixed seed count")
    states = Counter(_state(row) for row in rows)
    rejected, unavailable = states["rejected"], states["unavailable"]
    profile = _interval_counts(rows, "profile", required_count)
    return {
        "attempted": len(rows),
        "required": required_count,
        "missing": required_count - len(rows),
        "protocol_complete": len(rows) == required_count,
        "rejected": rejected,
        "unavailable": unavailable,
        "failure_or_rejection": rejected + unavailable,
        "rejection_rate": rejected / required_count,
        "unavailable_rate": unavailable / required_count,
        "failure_or_rejection_rate": (rejected + unavailable) / required_count,
        "calibration_available": rejected + states["not_rejected"],
        "not_triggered": states["not_triggered"],
        "unavailable_reasons": _reason_counts(rows),
        "profile": profile,
        "bootstrap": _interval_counts(rows, "bootstrap", required_count),
        "all_seed_coverage": profile["all_seed_coverage"],
        "conditional_coverage": profile["conditional_coverage"],
    } | _descriptive_counts(rows, required_count)


def acceptance(summary: dict, kind: str) -> list[dict] | None:
    if not summary["protocol_complete"]:
        return None
    if kind == "null":
        return [
            endpoint(summary["failure_or_rejection"], summary["required"], "F"),
            endpoint(summary["unavailable"], summary["required"], "U"),
        ]
    if kind == "power":
        return [endpoint(summary["target_detected"], summary["required"], "power")]
    return []


def rejection_interval(summary: dict) -> dict | None:
    """The registered F upper endpoint also bounds R without an extra claim."""
    if not summary["protocol_complete"]:
        return None
    upper = cp_bound(summary["failure_or_rejection"], summary["required"], "upper")
    return {
        "count": summary["rejected"],
        "trials": summary["required"],
        "bounds": [0.0, upper],
        "gamma": GAMMA,
        "basis": "R <= F = R + U; same predeclared F endpoint, no additional claim",
    }
