"""PyInstaller recipe for the Windows x86-64 standalone executable."""

from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


source_root = Path(os.environ["XRR_SOURCE_ROOT"]).resolve()
entrypoint = source_root / "src/xrr_fitter/__main__.py"
if not entrypoint.is_file():
    raise ValueError(f"XRR source entry point is missing: {entrypoint}")

analysis = Analysis(
    [str(entrypoint)],
    pathex=[str(source_root / "src")],
    binaries=[],
    datas=collect_data_files("periodictable"),
    hiddenimports=["xlsxwriter"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
archive = PYZ(analysis.pure)

executable = EXE(
    archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="xrr-fitter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="x86_64",
    codesign_identity=None,
    entitlements_file=None,
)
