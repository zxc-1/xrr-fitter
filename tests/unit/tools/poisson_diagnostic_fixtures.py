"""Toy observations/identities for tool tests; never real preregistration."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def tool(load_tool_module):
    return _LazyTool(load_tool_module)


class _LazyTool:
    def __init__(self, loader):
        self.loader = loader
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = self.loader("check_poisson_diagnostics")
        return getattr(self.module, name)


def _row(state="not_triggered", *, covered=True, fit=True, target=False):
    available = covered is not None
    interval = {"available": available, "bounds": [97.0, 103.0] if covered else [101.0, 103.0]}
    diagnostic = {
        "status": state,
        "raw_triggered": state != "not_triggered",
        "rejected": state == "rejected",
        "unavailable_reason": "injected_failure" if state == "unavailable" else None,
        "target_detected": target,
        "rejected_columns": ["background"] if target else [],
    }
    return {
        "fit_available": fit,
        "failure_reason": None if fit else "product_failure",
        "diagnostic": diagnostic,
        "profile": interval,
        "bootstrap": {"available": False, "bounds": None},
        "elapsed_seconds": 0.2,
        "fit_seconds": 0.1,
    }


def _test_registration(tool, monkeypatch, tmp_path, *, shards=3):
    review = tmp_path / "toy-review.md"
    review.write_bytes(b"unit-test-only review; no scientific preregistration")
    monkeypatch.setattr(tool.protocol, "REVIEW_SHA256", hashlib.sha256(review.read_bytes()).hexdigest())
    identity = {"sha256": "a" * 64, "head": "test-only", "files": []}
    monkeypatch.setattr(tool.protocol, "source_identity", lambda *_args: identity)
    monkeypatch.setattr(tool.protocol, "version_identity", lambda: dict(tool.protocol.EXPECTED_VERSIONS))
    output = tmp_path / "experiment"
    manifest = tool.register(output, review, shard_count=shards)
    return output, manifest, identity


def _residual(*, calibration=None, executed=True, raw=False, reason=None):
    return SimpleNamespace(
        dataset_id="member",
        executed=executed,
        systematic=False if executed else None,
        autocorrelation=False if executed else None,
        raw_systematic=raw,
        raw_autocorrelation=False,
        advisories=(),
        diagnostics=(),
        unavailable_reason=reason,
        calibration=calibration,
    )


def _calibration(*, p=0.5, target_p=0.5, status="available"):
    statistic = SimpleNamespace(
        dataset_id="member", kind="background", adjusted_p_value=target_p, observed=1.0, center=0.0, scale=1.0
    )
    return SimpleNamespace(
        status=status,
        p_value=p,
        rejected=p <= 0.01,
        statistics=(statistic,),
        alpha=0.01,
        sample_count=999,
        attempted_count=999,
        successful_count=999,
        child_seed=1,
        method="poisson_refit_null_rms_v4",
        refit_policy="declared_sobol4_lbfgsb_trf_v3",
        owner_sha256="a" * 64,
        tail_count=int(p * 1000),
        tie_count=1,
        observed_score=1.0,
        resolution=0.001,
        null_statistics_sha256="b" * 64,
        refit_nfev=15,
        refit_discrepancy=(0.0, 0.0, 0.0),
        unavailable_reason=None if status == "available" else "fit_failed",
        failure_reasons=(),
        provenance_sha256="c" * 64,
    )
