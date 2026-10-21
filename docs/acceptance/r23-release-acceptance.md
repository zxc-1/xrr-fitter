# R23 release acceptance

## Current release preparation — 2026-10-21

This section supersedes the undated interpretation of the historical verdict
below. The source now prepares **0.2.5**; this document is not publication approval.

| Scope | Evidence and boundary |
|---|---|
| Integrated source baseline | `6be4d94fd65e254deffc4941901f7b3fe7a8b80f`; PRs #32, #34 and #35 merged |
| Baseline branch verification | [verify 39535230978](https://github.com/zxc-1/xrr-fitter/actions/runs/39535230978) and [audit 39535231032](https://github.com/zxc-1/xrr-fitter/actions/runs/39535231032) completed successfully |
| Published release | `v0.2.3`, published 2026-08-22 |
| Existing draft | `v0.2.4`, tagged 2026-09-16 at `7c93d06119d38ab169a2e698c8b7612aea5810fb`; draft identity and assets retain that commit |
| Next package | `0.2.5` preparation; no new tag or public release is claimed |
| Real-data acceptance | `NOT_RUN`; owner-approved data is unavailable. Synthetic fixtures do not replace it |
| macOS desktop input | Command-Q, application-menu quit/cancel, save/discard exit and unfocused-editor scrolling verified; see the [desktop record](macos-desktop-2026-10-21.md) |
| Windows | Earlier native-platform Qt rendering/event/save-reopen acceptance passed at `a479cac3db86dbe6e93e6d51930546e9ac5c8686` in [run 39411965629](https://github.com/zxc-1/xrr-fitter/actions/runs/39411965629); current-candidate and frozen-GUI acceptance deferred without a Windows host |
| Scientific identity | Project schema `5`, algorithm `xrr-fit-v2-poisson-5`, objective `2`; approved cases and limits are in [V2 acceptance](fitting-algorithm-v2.md) |
| Native dependency coverage | Ordinary locked-pin advisory audit passed on the baseline; OS, bootstrap, VCS and full runtime-loader coverage remain incomplete |

The baseline main run executed the ordinary engineering and distribution gates.
Tag-only statistical, candidate-readiness, identity and release jobs were skipped
by their configured conditions. A successful branch run does not satisfy those
release jobs. The baseline test manifest was bound to source
`199b7ea58bee51380f52e9293e102f8050b56d3a`; the baseline has the same test tree and
matching current lock digest. The current manifest is bound to
`919ef67d0b9043bddb673048342c4c02c3773d96` after the candidate's online-help
regression was updated; fresh collection still contains 7,263 nodes. Final
release preparation must recheck this binding.

The old 0.2.4 assets predate the GUI/MCMC follow-ups and Pillow update. Preserve
their tag and identity. Build and verify the next version from one clean final
commit, then bind any compatible statistical evidence through the existing tools.
New statistical computation remains an explicit operation. Publishing the release
is separate from local package and identity verification.

Two older GUI limitations have follow-up evidence: the MCMC initialization fix
replayed the same 1,201-point case, seed and 128,000-by-8 sampling budget with
split-Rhat 1.0275–1.0749, minimum ESS about 718.16 and no diagnostic warnings; the two
recorded instrument boundary hits remain. The GUI menu lifetime fix also removed
the observed widget accumulation. These bounded results supersede the older
"Rhat 41.13" and menu-leak status; they do not certify every real-data case.

The retained external evidence directory `gui-followups-20261020-s44_1cbt`
contains `mcmc/full-replay.json`, `mcmc/final-initialization-equivalence.json`,
the before/after `gui/widgets-{baseline,fixed}/widgets.json` records, and
`windows/remote-39411965629/reports/gui-native/acceptance.json`. These are
historical follow-up results, not new candidate-wide acceptance.

Historical source snapshots and scientific experiment outputs remain recovery
evidence. They are not pending branches to merge wholesale. Move them only after
a separate backup destination and a successful restore check are available.

## Historical verdict — 2026-08-22

The following records the original R23 candidate. Its frozen R22 subsystem and
migration ledger were subsequently removed from the active source and retained on
`archive/r22-reference`. Its test counts and release wording describe that dated
candidate, not the current checkout.

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
