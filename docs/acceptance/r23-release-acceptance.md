# R23 release acceptance

## Verdict

| Scope | Verdict |
|---|---|
| Source-candidate automated gates | `PASS` |
| Synthetic statistical corpus | `PASS: 220/220; failed_case_ids=[]` |
| Frozen R22 equivalence | `PASS: 8/8 registered groups` |
| Isolated clean-snapshot rehearsal | `PASS: manifest + ledger + distribution + identity` |
| Real-data acceptance | `NOT_RUN: owner post-delivery acceptance` |
| Distribution | `GATED: clean exact-SHA branch and tag verification` |
| Release identity | `GATED: clean exact-SHA artifact and source binding` |
| Release | `DRAFT: five bound assets; owner publication remains separate` |

The canonical distribution and identity are produced only by the clean exact-SHA
release gates. The source-candidate verdict does not claim an owner real-data
verdict, and the GitHub Release remains a draft for owner review.

## Machine evidence

- Synthetic node:
  `tests/acceptance/test_synthetic_recovery_corpus.py::test_synthetic_recovery_corpus_meets_approved_thresholds`.
  Report schema `xrr-r23-synthetic-recovery-v1`; the test requires exactly 220
  fitted cases, `status=PASS`, and an empty failed-case list.
- Frozen-reference node:
  `tests/acceptance/test_r22_reference_equivalence.py::test_frozen_r22_reference_matches_all_registered_r23_groups`;
  records: `verification/r22/reference/manifest.json` and
  `verification/release-spec.json`.
- Runnable source modes are registered in `tools/verify_registry.py` and
  executed through `tools/verify.py`.
- The committed `verification/r23/tests.json` binds the final collected nodes
  to their source commit; the final migration ledger covers all 1563 frozen R22
  source nodes.
- Owner-data nodes:
  `tests/acceptance/test_real_data_workflows.py::test_real_data_workflows_produce_four_run_candidate_records`
  and
  `tests/acceptance/test_gui_real_data_workflows.py::test_gui_real_data_workflows_round_trip_owner_projects`.
  No `verification/approved-data/**` record exists before owner acceptance.
- Clean-HEAD artifact gates: `tools/verify.py distribution`,
  `tools/verify.py identity`, and `tools/verify.py release`.
