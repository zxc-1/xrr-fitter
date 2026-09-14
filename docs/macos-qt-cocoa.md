# macOS Qt Cocoa build contract

The supported macOS arm64 setup is `tools/macos_environment.py`, invoked by the
README installation command and the existing shared CI setup action. It keeps
PySide6, Shiboken and Qt at 6.11.2. Windows installation is unchanged.

## Why a source derivative is required

Qt Cocoa synthesizes native accessibility rows, columns and placeholders whose
Qt accessibility IDs are borrowed from the owning table. Deleting the borrowed
C++ interface while releasing those native snapshots invalidates live model
state. `tools/qt-cocoa/ownership.patch` fixes ownership and explicitly invalidates
retained snapshots when the table model changes. Disabling accessibility or
removing model display data is not a substitute for fixing that lifetime.

## Inputs and output identity

- `tools/package-manifests/qt-cocoa-macos-arm64-py312.json` binds the official
  Qt 6.11.2 Cocoa source files, Qt SDK archive, patch, qmake recipes, native
  regression harness and original PySide6-Essentials wheel by SHA-256.
- The cache stores only original source/SDK bytes, scoped by the manifest and
  trust domain. A corrupt cache fails validation; it is not silently replaced.
  No compiled Cocoa plugin or derived wheel is restored from this cache.
- Each setup compiles offline in a job-private temporary stage with a sanitized
  environment. It checks the Qt version, arm64 Mach-O target, minimum macOS 13.0,
  loader-relative Qt rpath, and code signature, then runs six native regression
  cases. The repeated-lifecycle case exercises 100 rounds of five operations.
  Each case checks the path of the plugin actually loaded by the native runtime.
- The derived filename is
  `pyside6_essentials-6.11.2-1xrrcocoa-cp310-abi3-macosx_13_0_arm64.whl`.
  It is **not universal2**. Only the Cocoa plugin, `WHEEL`, and `RECORD` change;
  declared provenance files are added under
  `pyside6_essentials-6.11.2.dist-info/licenses/xrr-qt-cocoa/`.
  All other upstream members, package metadata and existing licenses retain
  their original bytes. Large unchanged members are copied and hashed as streams.
- The provenance subtree contains original Cocoa source files, the patch and
  build/test recipes, the source manifest and a notice. SDK and Apple toolchain
  identities are recorded separately. Preserve the original Qt license notices
  when distributing a derived wheel.

The README's first run requires network access for the pinned originals and a
working Apple compiler/SDK. Subsequent source builds still take time and temporary
disk space. The ad-hoc signature check is **not** a Developer ID signature or
Apple notarization, and this is not a claim of cross-host reproducible binaries.

## Installation and evidence

Setup builds Qt **before** ordinary package installation. Hash-required offline
pip installation selects the derivative directly; it never overwrites an
installed vendor dylib. Missing, changed or unsuccessful build receipts fail
installation. No application runtime hook, `QT_PLUGIN_PATH` or `DYLD_*` override
is needed for normal GUI startup.

Each owned job retains:

- `reports/qt-cocoa-inputs/`: original input/cache evidence;
- `reports/qt-cocoa-build/build.json`: recipe and builder hashes, toolchain,
  native results, derived archive/member inventory and plugin identity;
- `reports/installation/`: exact local wheel requirements and pip logs;
- `reports/installed/`: installed dependency-byte inventory and CycloneDX SBOM
  when the README command or existing advisory audit is used.

The installed audit requires `--qt-build` for macOS and binds the receipt,
original and derived archives, recipes and builder helpers. Its Essentials
component uses `qt-cocoa-source-build` provenance rather than an unmodified
PyPI-release purl. The aggregate remains `incomplete`: the host interpreter,
macOS loader and whole operating-system dependency closure are outside this
byte audit. An ad-hoc signed Qt build is not a release/readiness authorization.

The advisory bundle also takes `--qt-build` and the original `--wheel-dir`,
rechecks the source derivative, and matches its build-input hashes to the
installed SBOM. Ordinary installed wheels still require exact manifest bytes.
The original Essentials version remains in the pinned advisory query but is
reported separately as `upstream_advisory_only`; the derived wheel remains
`not_scanned`. An upstream version advisory result is not a vulnerability scan
of the modified native library.

The README audits the dependency environment and canonical application artifact
**before** installing the application wheel with `--no-deps`; this avoids falsely
claiming the application's additional installed files belong to dependency
wheels. Subsequent user-added packages likewise require their own audit inputs.

Build stages are automatically removed. Reports and the installed environment
remain in the owned job directory. Do not copy a successful receipt to a changed
checkout: source, recipe or builder changes require rebuilding and revalidating.

## Maintaining the patch

A Qt upgrade must update the original wheel, source/SDK and recipe pins together,
review private-ABI assumptions, and reproduce the ownership regression on the
unpatched version before deciding whether to remove this patch. Never update
hashes merely to accept an unexplained download or alter the native assertions
so a failing upstream build appears successful. Keep the relative resource
collection name `qcocoaresources`, which the Cocoa integration initializes.

Run the focused Qt input, build, wheel, evidence and installation tests through
the repository's Python 3.12 environment, then `python tools/verify.py tools`
and the complete `python tools/check_radon.py` policy. A final native acceptance
must launch installed `python -m xrr_fitter` without plugin overrides and verify
project open/edit/theme/reload, accessible model data and normal shutdown from
fresh state. Synthetic unit fixtures are not native GUI acceptance evidence.
