"""RMS preregistration boundaries; no new holdout observations are drawn."""

import hashlib

import pytest
from tests.unit.tools.poisson_diagnostic_fixtures import _test_registration
from tests.unit.tools.poisson_diagnostic_fixtures import tool as tool

STARTS = {
    "null_single": (3000000, 2000),
    "null_joint": (3010000, 2000),
    "background": (4000000, 100),
    "footprint": (4010000, 100),
    "surface": (4020000, 100),
    "acf": (4030000, 100),
}


def test_protocol_binds_rms_method_and_unchanged_two_stage_refitter(tool):
    definition = tool.protocol.definition()
    assert definition.get("diagnostic_method") == "poisson_refit_null_rms_v4"
    assert definition["refit_policy"] == "declared_sobol4_lbfgsb_trf_v3"
    assert definition.get("family_standardization") == {
        "center": "all-row median",
        "scale": "all-row median-centered RMS; ddof=0",
        "score": "peak-factorized z; S_i=max(0,max_j(z_ij))",
        "reduction": "per-column contiguous float64; sorted squared relatives; math.fsum",
        "constant": "exact all-row equality; scale=0; z=0",
        "tails": "inclusive >=; no jitter",
        "numeric_failure": "unavailable; no epsilon or fallback",
    }
    assert definition["diagnostic_samples"] == 999
    assert definition["diagnostic_alpha"] == 0.01
    assert definition["simultaneous_one_sided_claim_count"] == 8


def test_old_review_approval_is_not_reused_for_changed_statistics(tool):
    assert tool.protocol.REVIEW_SHA256 != "6eea27709eb71995da52c6521c855e2af4a5644c84ae46b515c59ac5334fc002"


def test_all_new_holdout_seeds_are_disjoint_from_old_holdout_and_replay(tool):
    groups = tool.protocol.definition()["scenarios"]
    new = [seed for name in STARTS for seed in groups[name]["seeds"]]
    old = set(range(1000000, 1002000)) | set(range(1010000, 1012000))
    for start in (2000000, 2010000, 2020000, 2030000):
        old.update(range(start, start + 100))
    assert len(new) == len(set(new)) == 4400
    assert set(new).isdisjoint(old | set(range(200)))
    assert groups["replay_low"]["seeds"] == list(range(200))


@pytest.mark.parametrize("group", STARTS)
def test_new_first_and_last_seed_route_without_drawing_counts(tool, monkeypatch, tmp_path, group):
    calls, sentinel = [], object()

    def recorded(root, scenario, seed):
        calls.append((root, scenario, seed))
        return sentinel

    monkeypatch.setattr(tool.cases, "build_observations", recorded)
    start, count = STARTS[group]
    for seed in (start, start + count - 1):
        assert tool.cases.build_case(tmp_path, group, seed) is sentinel
    assert calls == [(tmp_path, group, start), (tmp_path, group, start + count - 1)]


@pytest.mark.parametrize("group", STARTS)
def test_previous_and_out_of_range_seeds_are_rejected_before_generation(tool, monkeypatch, tmp_path, group):
    monkeypatch.setattr(tool.cases, "build_observations", lambda *_args: pytest.fail("undeclared observation drawn"))
    start, count = STARTS[group]
    for seed in (start - 2000000, start - 1, start + count):
        with pytest.raises(ValueError, match="seed"):
            tool.cases.build_case(tmp_path, group, seed)


@pytest.mark.parametrize("version", (1, 2, 3))
def test_superseded_versions_cannot_register_or_create_an_output(tool, monkeypatch, tmp_path, version):
    review = tmp_path / "toy-review.md"
    review.write_text("test fixture, not a scientific approval")
    monkeypatch.setattr(tool.protocol, "REVIEW_SHA256", hashlib.sha256(review.read_bytes()).hexdigest())
    monkeypatch.setattr(
        tool.protocol,
        "version_identity",
        lambda: {
            "schema_version": 4,
            "algorithm_version": f"xrr-fit-v2-poisson-{version}",
            "objective_version": "2",
            "diagnostic_version": f"poisson-refit-null-v{version}",
        },
    )
    with pytest.raises(ValueError, match="integrated version"):
        tool.register(tmp_path / "forbidden", review, shard_count=1)
    assert not (tmp_path / "forbidden").exists()


def test_rehashed_manifest_cannot_switch_back_to_exposed_seeds(tool, monkeypatch, tmp_path):
    output, manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)
    manifest["protocol"]["scenarios"]["null_single"]["seeds"] = list(range(1000000, 1002000))
    encoded = tool.protocol.canonical(manifest)
    (output / "preregistration.json").write_bytes(encoded)
    (output / "preregistration.sha256").write_text(hashlib.sha256(encoded).hexdigest() + "\n")
    with pytest.raises(ValueError, match="protocol"):
        tool.load_manifest(output)
