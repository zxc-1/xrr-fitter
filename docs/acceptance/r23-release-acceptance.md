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

## 2026-08-22 audit delta

- Candidate classification now chooses the largest support cluster inside the
  equivalent-objective band before applying the objective tie-break. This
  prevents floating-point round-off from making a supported result
  `UNTRUSTED`; a materially better singleton remains untrusted.
- SLD reporting validates finite depth extents and adapts the reporting step to
  a bounded point budget. Long periodic stacks therefore fail explicitly or
  produce a finite, bounded diagnostic profile instead of an unbounded
  allocation.
- Project replacement now follows `fsync(file) -> replace -> fsync(directory)`
  where the platform supports directory synchronization. Windows resolver
  failures retain the resolver's stderr for diagnosis.
- Fresh locked-environment evidence for this delta: `quality` 186 passed,
  `tools` 448 passed, `unit` 1688 passed, `integration` 14 passed, `gui` 520
  passed, `spawn` 4 passed, `regression` 50 passed, and the statistical corpus
  `2 passed in 8510.99s (2:21:50)`. The corpus report retained its approved
  `220/220` result with no failed case IDs.

The delta does not change the owner-data, distribution, or release-identity
claims above. Those claims still require a clean exact-SHA run after the final
source and test manifest commit.

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
