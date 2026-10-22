# R23 release acceptance

## Current release preparation — 2026-10-22

This section supersedes the undated interpretation of the historical verdict
below. The source now prepares **0.2.5**; this document is not publication approval.

| Scope | Evidence and boundary |
|---|---|
| Integrated source baseline | `07b2bdb9d8ef5342c079c1f33e8f0de403e37744`; PRs #32, #34, #35 and #36 merged |
| Baseline branch verification | [verify 39610101995](https://github.com/zxc-1/xrr-fitter/actions/runs/39610101995) and [audit 39610101749](https://github.com/zxc-1/xrr-fitter/actions/runs/39610101749) completed successfully |
| Published release | `v0.2.3`, published 2026-08-22 |
| Existing draft | `v0.2.4`, tagged 2026-09-16 at `7c93d06119d38ab169a2e698c8b7612aea5810fb`; draft identity and assets retain that commit |
| Next package | `0.2.5` preparation; no new tag or public release is claimed |
| Real-data acceptance | `NOT_RUN`; owner-approved data is unavailable. Synthetic fixtures do not replace it |
| macOS desktop input | Command-Q, application-menu quit/cancel, save/discard exit and unfocused-editor scrolling verified; see the [desktop record](macos-desktop-2026-10-21.md) |
| Windows | At the integrated baseline, [run 39663019105](https://github.com/zxc-1/xrr-fitter/actions/runs/39663019105) passed 117 headless tests, native-platform QtTest rendering/events/save-reopen and CLI EXE checks; [run 39663031877](https://github.com/zxc-1/xrr-fitter/actions/runs/39663031877) built the frozen GUI and passed its offscreen startup smoke. Physical desktop input and native frozen-GUI interaction remain unverified |
| Scientific identity | Project schema `5`, algorithm `xrr-fit-v2-poisson-5`, objective `2`; approved cases and limits are in [V2 acceptance](fitting-algorithm-v2.md) |
| Native dependency coverage | Ordinary locked-pin advisory audit passed on the baseline; reviewed pip 26.2.2 fixes the subsequently identified CVE-2026-13346. Exact-pin advisories, verified build inputs and installed bytes cover different scopes; the ensurepip seed, OS and full runtime-loader closure remain incomplete |
| Pip security candidate | PR #37 merged at `4946de513a9760aa7b52f7646738f5dbbc87e108`; main verify/audit and Windows runs `39675427347`, `39675427212`, `39675549133`, `39675693903` succeeded in their recorded scopes |
| Latest synthetic attempt | [run 39675537530](https://github.com/zxc-1/xrr-fitter/actions/runs/39675537530), attempt 1 at `4946de5`, **FAIL**: `oxide-cap-14012` returned UNTRUSTED. Seven complete shards retained 193 outcomes; failed shard 4 retained four outcomes and no complete manifest. The other 23 outcomes were not retained. Aggregate failed on the missing manifest; release was skipped |

The baseline main run executed the ordinary engineering and distribution gates.
Tag-only statistical, candidate-readiness, identity and release jobs were skipped
by their configured conditions. A successful branch run does not satisfy those
release jobs. The integrated baseline test manifest was bound to source
`919ef67d0b9043bddb673048342c4c02c3773d96` with 7,263 nodes. The pip security
regressions added three nodes, collected from
`8277c58ac5aa51488ce3c2026c137ed6622b3959` with 7,266 nodes. Ten failure-diagnostic
regressions produced 7,276 nodes at `65765b579ea5db9177858276788a9c2351e65060`.
The current manifest adds six bounded-proposal regressions, freshly collected
from `bfc980bac3eb2f1d020b0b1e7f7833319c5a6cbc` with 7,282 nodes and the
unchanged application lock digest. The release identity gate checks the committed test
tree, collection hash and source ancestry. The updated installer/build inputs
require fresh final-candidate evidence under the existing statistical policy;
the baseline runs above do not certify the updated candidate.

The failed attempt preserved the confidence assertion and warnings, but not its
classification reasons, candidates or bootstrap failure details. Static review
cannot distinguish the three possible UNTRUSTED triggers from those warnings.
The PR #38 diagnostic follow-up retains these already-computed inputs in a separate
failure sidecar, with controlled tests for spawn transport and persistence errors;
it does not change fitting or scientific thresholds, recover the lost result, or
turn the failed attempt into PASS. No additional scientific computation was
started for that diagnostic-retention change. The 197 retained outcomes are not an executed-fit
count: running workers can finish after another worker fails without their result
being consumed. Complete statistical and release acceptance remain unmet.

A subsequent controlled local replay on Python 3.12.13 reproduced
`oxide-cap-14012` with `insufficient_cluster_support`: only one of the four
final seeds found the materially better solution. The first three declared-start
paths had zero resolution width after clipping; its analytic width derivative
vanishes at zero. Reflecting the same sampled displacement inward recovered the
better solution within the original local budget. The bounded-proposal repair
now reflects excursions while preserving interior draws, seed streams, sigma,
counts and solver budgets. The unchanged case replay produced three agreeing
final seeds and passed output and recovery assertions with confidence
`CORRELATED`; bootstrap failures remained zero. This local diagnostic differs
from the historical runner's Python patch and concurrency. It does not replace
full 220-case acceptance or relabel the historical failed run. Existing completed
checkpoints retain their historical candidates; applying the repair requires a
new affected search, and a completed Stage-E checkpoint has no remaining search.

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
