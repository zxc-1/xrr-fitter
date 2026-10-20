# Declared wavelength import repair

Goal: keep a clean-home import from silently fitting PETRA synchrotron data with the Cu default. Continue the already-authorized repair in place; preserve existing changes and dependencies.

Evidence: both GUI projects have source SHA-256 `d4497440f2f91f6bb66b3d6de0529f1f46f23ea2722b074b57cc80281621ac65`, but the failed default run stores 1.5406 Å rather than the header's 0.6889 Å. A changed random seed is not a controlled comparison.

- [x] RED: run `PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen ../venvs/repo-qt6112/bin/python -m pytest -o addopts= -p no:cacheprovider tests/unit/io/test_xy_wavelength.py tests/gui/test_import_wavelength.py -q`. Assert wrong metadata is rejected and the home dialog persists the explicit header wavelength.
- [x] GREEN: add one header parser in `io/xy.py`, expose it via services/API, and validate the same bytes used to prepare qz. Recognize explicit scalar wavelengths with A/Å/angstrom or nm units; do not guess from names, intensity, or angle range. Reject invalid/conflicting declarations and a mismatching beam rather than modifying measured values.
- [x] GUI: prefill the monochromatic editor from a consistent explicit header when no preset exists; keep the required beam choice. Do not replace a saved preset. Refresh validation after wavelength edits and prevent accepting an invalid preview. Preserve partial-batch imports and display conflict reasons.
- [x] Verify: run focused tests, quality, unit, GUI and clean-home GUI real-data fit. Check saved wavelength/source digest and record elapsed time/objective/quality warnings. Recheck 1200-point performance without downsampling the final solve.
- [x] Document results and cleanup only this verification's transient processes/drafts. Preserve raw source, derived evidence and all unrelated modifications.

Verification and remaining limits: `../../../../.codex-artifacts/root-fix-20260925/verification-report.md`. The 1200-point sample is synthetic; the user's own file was not available.

## 2026-10-11 follow-up verification

The historical checkmarks above did not establish full GUI acceptance: a fresh
native audit reproduced ordinary-header wavelength loss, recursive and partial
batch prefill failures, column-mapping reset, and PNGs that were only 100 dpi.
The follow-up repairs these paths and mapping-correction refresh, then adds
regressions for native-discovered pause/resume accessibility labels and stale
import feedback after successful new/open operations. Saved presets, failed
project replacement, and unrelated workspace changes must remain intact.

Current repair evidence, commands, native captures, export integrity and explicit
remaining limits: `../../../../.codex-artifacts/gui-repair-20261011-UMWGYh0Y/verification-report.md`.
This follow-up is not a Windows, owner-data, statistical-corpus or formal release
sign-off; the user's own 1200-point measurement is still outside this evidence.
