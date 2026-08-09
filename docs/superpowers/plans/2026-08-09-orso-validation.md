# ORSO Validation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor the ORSO community reflectivity validation suite into the repository and prove `parratt_reflectivity` reproduces every unpolarised test case at the suite's own tolerances, without changing any production numerical behavior.

**Architecture:** A new offline sync tool freezes three tiers of suite files (manifest, layer table, data) plus a sha256 index under `tests/fixtures/orso/`. A new regression module parses the frozen manifests, converts each 4-column layer table into the existing `SlabStack`, and dispatches on the data file's column count: fewer than 4 columns compares bare `parratt_reflectivity` at `rtol=8e-5`, exactly 4 columns feeds the `dQ` column into `gaussian_smear` as pointwise 1-sigma widths and compares at `rtol=0.03`.

**Tech Stack:** Python 3.12, numpy, pytest. No new dependency — all three tiers are plain text.

**Design source:** `docs/superpowers/specs/2026-08-09-orso-validation-design.md`

## Global Constraints

- Do not modify any file under `src/`. This plan adds tests, one tool, and frozen data only.
- Do not make the test suite reach the network. `tools/sync_orso_suite.py` is run by hand; the tests read only frozen bytes.
- Do not use `pytest.skip`, `xfail`, or conditional collection. `tests/outcome_gate.py` fails the whole run on `skipped`/`xfailed`/`xpassed`/`deselected`.
- Do not relax the existing `tests/regression/test_numerical_reference.py` tolerances. The suite is a looser, additional oracle.
- Pin the suite to commit `6a01b4a4febfc52cd3881d2147c732dd1701bc8e`. Do not track `master`.
- Derive the frozen file list from manifest references, never from a directory listing — upstream `data/` contains an unrelated `Untitled.ipynb`.
- New code under `tools/` and `tests/` must pass `tools/check_radon.py` (per-block CC ≤ 10, file average CC ≤ 5.0, MI rank A).
- Do not stage or modify `.claude/` or root-level probe files.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `tools/sync_orso_suite.py` | Fetch the pinned suite by hand, resolve manifest references, write frozen files plus a sha256 index. |
| `tests/fixtures/orso/index.json` | Frozen sha256 index binding every vendored file to the pinned suite commit. |
| `tests/fixtures/orso/unpolarised/` | Frozen manifests, `layers/*.layers`, and `data/*.dat`. |
| `tests/regression/test_orso_validation.py` | Parse the frozen suite and compare reflectivity per case at the suite's two tolerances. |
| `tests/unit/tools/test_sync_orso_suite.py` | Prove manifest resolution, sha256 binding, and the directory-scan refusal. |
| `tools/verify_registry.py` | Register the new regression module in the `regression` mode. |
| `tests/unit/tools/test_verify_registry.py` | Keep the exact-registry assertion in sync. |
| `docs/algorithm.md` | Document suite coverage, the two tolerances, and the neutron-SLD boundary. |

---

### Task 1: Freeze the suite offline

**Files:**
- Create: `tools/sync_orso_suite.py`
- Create: `tests/unit/tools/test_sync_orso_suite.py`
- Create: `tests/fixtures/orso/index.json` (generated)
- Create: `tests/fixtures/orso/unpolarised/**` (generated)

**Interfaces:**
- Consumes: nothing from `src/`. Standard library only (`argparse`, `hashlib`, `json`, `pathlib`, `urllib.request`).
- Produces: `SUITE_COMMIT`, `INDEX_SCHEMA`, `manifest_references(text)`, `frozen_index(root)`, `verify_index(root)`.
- Preserves: the `tools/freeze_approved_data.py` convention — canonical JSON, sorted keys, sha256 per file.
- Removes: nothing.

- [x] **Step 1: Write the failing tool contract**

Create `tests/unit/tools/test_sync_orso_suite.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest


MANIFEST = """\
# Ti-Ni multilayer, from BornAgain FitSpecularBasics.py
# dQ/Q = 0.05 FWHM or 0.0212 1-sigma
layers/test1.layers
data/test5.dat
"""


def _suite(root: Path) -> Path:
    unpolarised = root / "unpolarised"
    (unpolarised / "layers").mkdir(parents=True)
    (unpolarised / "data").mkdir(parents=True)
    (unpolarised / "test5.txt").write_text(MANIFEST, encoding="utf-8")
    (unpolarised / "layers/test1.layers").write_text("0 0 0 0\n30 -1.9493 0 0\n", encoding="utf-8")
    (unpolarised / "data/test5.dat").write_text("0.005 0.9995 0.0 1.0617e-04\n", encoding="utf-8")
    (unpolarised / "data/Untitled.ipynb").write_text("{}\n", encoding="utf-8")
    return root


def test_suite_commit_is_pinned_to_forty_hex_characters(load_tool_module) -> None:
    module = load_tool_module("sync_orso_suite")
    assert module.SUITE_COMMIT == "6a01b4a4febfc52cd3881d2147c732dd1701bc8e"
    assert len(module.SUITE_COMMIT) == 40
    assert set(module.SUITE_COMMIT) <= set("0123456789abcdef")


def test_manifest_references_skip_comments_and_keep_order(load_tool_module) -> None:
    module = load_tool_module("sync_orso_suite")
    assert module.manifest_references(MANIFEST) == ("layers/test1.layers", "data/test5.dat")


def test_manifest_without_exactly_two_references_is_rejected(load_tool_module) -> None:
    module = load_tool_module("sync_orso_suite")
    with pytest.raises(ValueError, match="two references"):
        module.manifest_references("# only a comment\nlayers/test1.layers\n")


def test_frozen_index_excludes_files_no_manifest_references(load_tool_module, tmp_path: Path) -> None:
    module = load_tool_module("sync_orso_suite")
    index = module.frozen_index(_suite(tmp_path))
    paths = {record["path"] for record in index["files"]}
    assert paths == {
        "unpolarised/test5.txt",
        "unpolarised/layers/test1.layers",
        "unpolarised/data/test5.dat",
    }
    assert not any("Untitled" in path for path in paths)


def test_frozen_index_records_schema_commit_and_sha256(load_tool_module, tmp_path: Path) -> None:
    module = load_tool_module("sync_orso_suite")
    index = module.frozen_index(_suite(tmp_path))
    assert index["schema"] == module.INDEX_SCHEMA
    assert index["suite_commit"] == module.SUITE_COMMIT
    assert all(len(record["sha256"]) == 64 and record["size"] > 0 for record in index["files"])
    assert [record["path"] for record in index["files"]] == sorted(
        record["path"] for record in index["files"]
    )


def test_verify_index_rejects_mutated_content(load_tool_module, tmp_path: Path) -> None:
    module = load_tool_module("sync_orso_suite")
    root = _suite(tmp_path)
    (root / "index.json").write_bytes(module.canonical_json_bytes(module.frozen_index(root)))
    module.verify_index(root)
    (root / "unpolarised/data/test5.dat").write_text("0.005 0.5 0.0 1e-04\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        module.verify_index(root)


def test_verify_index_rejects_a_missing_referenced_tier(load_tool_module, tmp_path: Path) -> None:
    module = load_tool_module("sync_orso_suite")
    root = _suite(tmp_path)
    (root / "index.json").write_bytes(module.canonical_json_bytes(module.frozen_index(root)))
    (root / "unpolarised/layers/test1.layers").unlink()
    with pytest.raises(ValueError, match="missing"):
        module.verify_index(root)


def test_index_json_in_the_repository_matches_its_recorded_hashes(load_tool_module) -> None:
    module = load_tool_module("sync_orso_suite")
    root = Path(__file__).resolve().parents[2] / "fixtures/orso"
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert index["suite_commit"] == module.SUITE_COMMIT
    assert len(index["files"]) == 22
    module.verify_index(root)
```

- [x] **Step 2: Confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider tests/unit/tools/test_sync_orso_suite.py -q
```

Expected: 8 failures, each `missing implementation: tools/sync_orso_suite.py` from the `load_tool_module` fixture.

- [x] **Step 3: Implement the sync tool**

Create `tools/sync_orso_suite.py`. Keep every function at CC ≤ 10 by splitting fetch, freeze, and verify.

```python
#!/usr/bin/env python3
"""Freeze the pinned ORSO validation suite for offline regression testing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence
import urllib.request


SUITE_COMMIT = "6a01b4a4febfc52cd3881d2147c732dd1701bc8e"
INDEX_SCHEMA = "xrr-r23-orso-suite-index-v1"
RAW_ROOT = "https://raw.githubusercontent.com/reflectivity/analysis"
SUITE_ROOT = "validation/test"
MANIFEST_COUNT = 8


def canonical_json_bytes(value: object) -> bytes:
    text = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return (text + "\n").encode("utf-8")


def manifest_references(text: str) -> tuple[str, ...]:
    lines = tuple(
        stripped
        for line in text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    )
    if len(lines) != 2:
        return _reject(f"manifest must name exactly two references, found {len(lines)}")
    return lines


def _reject(detail: str) -> tuple[str, ...]:
    raise ValueError(detail)


def _manifests(unpolarised: Path) -> tuple[Path, ...]:
    return tuple(sorted(unpolarised.glob("test?.txt")))


def _referenced_paths(unpolarised: Path) -> tuple[Path, ...]:
    collected: list[Path] = []
    for manifest in _manifests(unpolarised):
        collected.append(manifest)
        text = manifest.read_text(encoding="utf-8")
        collected.extend(unpolarised / reference for reference in manifest_references(text))
    return tuple(sorted(set(collected)))


def _record(root: Path, path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing referenced suite file: {path.relative_to(root).as_posix()}")
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def frozen_index(root: Path) -> dict[str, object]:
    unpolarised = Path(root) / "unpolarised"
    records = [_record(Path(root), path) for path in _referenced_paths(unpolarised)]
    records.sort(key=lambda record: str(record["path"]))
    return {"schema": INDEX_SCHEMA, "suite_commit": SUITE_COMMIT, "files": records}


def verify_index(root: Path) -> None:
    root = Path(root)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if index.get("schema") != INDEX_SCHEMA or index.get("suite_commit") != SUITE_COMMIT:
        raise ValueError("frozen ORSO index schema or suite commit does not match this tool")
    for record in index["files"]:
        path = root / str(record["path"])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing frozen suite file: {record['path']}")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"] or len(content) != record["size"]:
            raise ValueError(f"sha256 mismatch for {record['path']}; rerun tools/sync_orso_suite.py")


def _download(relative: str) -> bytes:
    url = f"{RAW_ROOT}/{SUITE_COMMIT}/{SUITE_ROOT}/{relative}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - pinned https literal
        return response.read()


def _write(root: Path, relative: str, content: bytes) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _fetch_suite(root: Path) -> None:
    for ordinal in range(MANIFEST_COUNT):
        relative = f"unpolarised/test{ordinal}.txt"
        content = _download(relative)
        _write(root, relative, content)
        for reference in manifest_references(content.decode("utf-8")):
            _write(root, f"unpolarised/{reference}", _download(f"unpolarised/{reference}"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        verify_index(args.root)
        print(json.dumps({"status": "PASS", "suite_commit": SUITE_COMMIT}, sort_keys=True))
        return 0
    _fetch_suite(args.root)
    index = frozen_index(args.root)
    (args.root / "index.json").write_bytes(canonical_json_bytes(index))
    print(json.dumps({"status": "PASS", "file_count": len(index["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 4: Fetch the suite once, by hand**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/sync_orso_suite.py --root tests/fixtures/orso --fetch
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/sync_orso_suite.py --root tests/fixtures/orso --check
```

Expected: `--fetch` reports `"file_count": 22` (8 manifests + 6 layer tables + 8 data files) and `--check` reports `PASS`. If the count differs, stop and reconcile against the pinned commit before continuing — do not adjust the assertion to match.

- [x] **Step 5: Confirm GREEN**

Run the exact command from Step 2.

Expected: `8 passed`. `Untitled.ipynb` is absent from `tests/fixtures/orso/`; confirm with `rg --files tests/fixtures/orso | wc -l` returning `23` (22 suite files + `index.json`).

---

### Task 2: Compare reflectivity against every unpolarised case

**Files:**
- Create: `tests/regression/test_orso_validation.py`
- Modify: `tools/verify_registry.py` (`regression` mode, one added line)
- Modify: `tests/unit/tools/test_verify_registry.py` (`_expected_registry`, one added line)

**Interfaces:**
- Consumes: `xrr_fitter.model.structure.SlabStack`, `xrr_fitter.physics.parratt.parratt_reflectivity`, `xrr_fitter.physics.resolution.gaussian_smear`.
- Preserves: every existing tolerance and every entry already in the `regression` mode.
- Produces: one parametrized regression test per frozen manifest.
- Removes: nothing.

- [x] **Step 1: Write the failing regression module**

Create `tests/regression/test_orso_validation.py`:

```python
"""Community ORSO validation suite parity for the unpolarised specular path.

Layer tables are always four columns: thickness, real SLD, imaginary SLD, and
the roughness of that row's top interface. Data files carry two to four
columns; a fourth column is a pointwise 1-sigma dQ and selects the smeared
comparison at the suite's looser tolerance. The suite is an additional, looser
oracle: tests/regression/test_numerical_reference.py remains the strict gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xrr_fitter.model.structure import SlabStack
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.resolution import gaussian_smear


SUITE = Path(__file__).resolve().parents[1] / "fixtures/orso"
UNPOLARISED = SUITE / "unpolarised"
SLD_SCALE = 1e-6
KERNEL_RTOL = 8e-5
SMEARED_RTOL = 0.03
ROUGHNESS_COLUMN = 3
# Row N's roughness belongs to interface [N-1, N]; the fronting row's value is
# unused. This mirrors the suite's own reference construction and the existing
# layers[1:, 3] slice in test_numerical_reference.py.
FIRST_INTERFACE_ROW = 1


def _frozen_index() -> dict[str, object]:
    return json.loads((SUITE / "index.json").read_text(encoding="utf-8"))


def _manifest_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            Path(str(record["path"])).name
            for record in _frozen_index()["files"]
            if Path(str(record["path"])).suffix == ".txt"
        )
    )


def _references(manifest: Path) -> tuple[str, str]:
    lines = tuple(
        stripped
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    )
    assert len(lines) == 2, f"{manifest.name} must name a layer table and a data file"
    return lines[0], lines[1]


def _load_case(name: str) -> tuple[np.ndarray, np.ndarray]:
    layers_reference, data_reference = _references(UNPOLARISED / name)
    layers_path = UNPOLARISED / layers_reference
    data_path = UNPOLARISED / data_reference
    assert layers_path.is_file(), f"{name} references a missing layer table: {layers_reference}"
    assert data_path.is_file(), f"{name} references a missing data file: {data_reference}"
    layers = np.atleast_2d(np.loadtxt(layers_path))
    data = np.atleast_2d(np.loadtxt(data_path))
    assert layers.shape[1] == 4, f"{name} layer table must be four columns, got {layers.shape[1]}"
    assert data.shape[1] in (2, 3, 4), f"{name} data file has unregistered column count {data.shape[1]}"
    return layers, data


def _stack(layers: np.ndarray) -> SlabStack:
    return SlabStack(
        layers[:, 0],
        (layers[:, 1] + 1j * layers[:, 2]) * SLD_SCALE,
        layers[FIRST_INTERFACE_ROW:, ROUGHNESS_COLUMN],
    )


def _actual(stack: SlabStack, data: np.ndarray) -> tuple[np.ndarray, float]:
    qz = data[:, 0]
    if data.shape[1] < 4:
        return parratt_reflectivity(qz, stack), KERNEL_RTOL
    smeared = gaussian_smear(
        qz,
        lambda query: parratt_reflectivity(query, stack),
        sigma_q_a_inv=data[:, 3],
        emit_warning=False,
    )
    return smeared, SMEARED_RTOL


@pytest.mark.parametrize("name", _manifest_names())
def test_unpolarised_case_matches_the_orso_suite(name: str) -> None:
    layers, data = _load_case(name)
    expected = data[:, 1]
    actual, rtol = _actual(_stack(layers), data)

    deviation = np.abs(actual - expected) / np.abs(expected)
    worst = int(np.argmax(deviation))
    assert np.all(deviation <= rtol), (
        f"{name}: worst relative deviation {deviation[worst]:.3e} exceeds rtol {rtol:g} "
        f"at q={data[worst, 0]:.6g} (columns={data.shape[1]})"
    )


def test_every_frozen_case_is_covered_and_both_tolerances_are_exercised() -> None:
    names = _manifest_names()
    assert len(names) == 8
    widths = {_load_case(name)[1].shape[1] for name in names}
    assert any(width < 4 for width in widths), "no case exercises the strict kernel tolerance"
    assert any(width == 4 for width in widths), "no case exercises the smeared tolerance"


def test_reversed_roughness_attribution_breaks_at_least_one_case() -> None:
    broken: list[str] = []
    for name in _manifest_names():
        layers, data = _load_case(name)
        if not np.any(layers[:, ROUGHNESS_COLUMN]):
            continue
        reversed_stack = SlabStack(
            layers[:, 0],
            (layers[:, 1] + 1j * layers[:, 2]) * SLD_SCALE,
            layers[:-1, ROUGHNESS_COLUMN],
        )
        actual, rtol = _actual(reversed_stack, data)
        if not np.all(np.abs(actual - data[:, 1]) / np.abs(data[:, 1]) <= rtol):
            broken.append(name)
    assert broken, "roughness attribution is untested: no case has a discriminating roughness column"


def test_frozen_suite_content_is_hash_bound() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "tools/sync_orso_suite.py"
    spec = importlib.util.spec_from_file_location("orso_suite_sync_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_index(SUITE)
```

- [x] **Step 2: Confirm RED and read the real deviations**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider tests/regression/test_orso_validation.py -q
```

Expected before Task 1 lands: collection error on the missing fixture directory. After Task 1: the run should be green. If a case fails, the assertion message names the manifest, the worst deviation, the q value, and the column count — use it to decide between a real numerical defect and a reading error. Do not widen `KERNEL_RTOL` or `SMEARED_RTOL`; they are the suite's own published values.

- [x] **Step 3: Register the module in the regression mode**

In `tools/verify_registry.py`, add one entry to the `regression` mode tuple, after `test_numerical_reference.py`:

```python
                "tests/regression/test_numerical_reference.py",
                "tests/regression/test_orso_validation.py",
```

In `tests/unit/tools/test_verify_registry.py`, make the same insertion inside `_expected_registry`'s `"regression"` tuple. That test asserts exact registry equality, so both files must change together.

- [x] **Step 4: Record the measured baseline**

Once green, replace the deviation comment in the module docstring with the observed worst deviation per tolerance class, taken from the Step 2 output. This is the baseline for future regressions.

- [x] **Step 5: Document coverage and its boundary**

In `docs/algorithm.md`, insert a new section immediately after `## Pinned Refnx Benchmark`:

```markdown
## ORSO Community Validation Suite

The suite is vendored at commit
`6a01b4a4febfc52cd3881d2147c732dd1701bc8e` under `tests/fixtures/orso`, hash
bound by `index.json` and refreshed only through
`tools/sync_orso_suite.py --fetch`. Tests never reach the network.

Eight unpolarised cases are compared. Layer tables are four columns
(thickness, real SLD, imaginary SLD, top-interface roughness). Data files with
fewer than four columns compare the bare Parratt kernel at `rtol=8e-5`; a
fourth column is a pointwise 1-sigma `dQ` fed to Gauss-Hermite smearing and
compared at `rtol=0.03`. Both values are the suite's own published tolerances.

The suite covers neutron SLD magnitudes near `1e-6 Å⁻²`. X-ray work runs near
`1e-5 Å⁻²` with a different absorption balance, so passing the suite does not
by itself validate the XRR path. The pinned refnx benchmark above, two orders
of magnitude tighter, remains the primary gate.
```

- [x] **Step 6: Run affected repository gates**

Run each command separately:

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py tools
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py regression
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py quality
PYTHONDONTWRITEBYTECODE=1 python tools/check_radon.py
git diff --check
```

Expected: every command exits 0. `tools` covers the new sync tests and the registry equality assertion; `regression` runs the new module through the outcome gate, which proves no case was skipped or deselected.

- [x] **Step 7: Commit the vendored suite and its gate**

```bash
git add tools/sync_orso_suite.py tests/unit/tools/test_sync_orso_suite.py tests/regression/test_orso_validation.py tests/fixtures/orso tools/verify_registry.py tests/unit/tools/test_verify_registry.py docs/algorithm.md
git commit -m "test: validate reflectivity against the pinned ORSO suite"
```

The suite is CC0 1.0, so vendoring carries no attribution obligation; the provenance comment in `index.json`'s `suite_commit` is the audit trail.

---

## 最终验收记录

| 项 | 命令 | 结果 |
| --- | --- | --- |
| 工具契约 | `pytest tests/unit/tools`（= `verify.py tools` 的 pytest 段） | 388 passed |
| 回归对标 | `pytest` regression 五个模块（= `verify.py regression` 的 pytest 段） | 47 passed，outcome gate 无 skip/xfail/deselect |
| 架构与复杂度 | `pytest tests/architecture`（9 模块）+ `lock_windows_environment.py --check` + `check_radon.py` | 156 passed；lock exit 0；radon exit 0 |
| 冻结绑定 | `python tools/sync_orso_suite.py --root tests/fixtures/orso --check` | `{"status": "PASS", "suite_commit": "6a01b4a4..."}` exit 0 |
| 最差偏差（严格档） | 记入 `test_orso_validation.py` docstring | 6.3854e-05（test1.txt @ q=0.57205，6 个 case，rtol 8e-5） |
| 最差偏差（宽松档） | 记入 `test_orso_validation.py` docstring | 1.3366e-04（test4.txt @ q=0.013916，2 个 case，rtol 0.03） |

本地未直接执行 `tools/verify.py <mode>`：它在每个 mode 前后硬编码 `check_hygiene.py`，而仓内 `.venv/`、`.pytest_cache/`、`.ruff_cache/` 会被判为 "generated directory inside repository"，产生数万条本地失败；CI 把 venv 建在 `$RUNNER_TEMP/venv`（仓外）故不受影响。上表按 `verify_registry.py` 打印出的各 mode 真实命令逐条执行。

计划外的必要修正：`.gitattributes` 增加 `tests/fixtures/orso/** -text -whitespace`。上游 `layers/test1.layers` 为 CRLF，而 `index.json` 按磁盘字节记 sha256；原有的 `* text=auto eol=lf` 会把入库 blob 规范化为 LF（668 → 646 字节，哈希 `52e31afe…` → `cf708188…`），新克隆下 `--check` 必然失败。已用 `git checkout-index` 导出「新克隆等价树」复验：CRLF 保留，`--check` PASS。`-whitespace` 使 `git diff --check` 不把上游 CR 判为行尾空白。

计划外的必要修正 2：`tests/unit/tools/test_verify.py::test_registry_commands_are_exact_for_completed_suites` 还有第二份 regression 期望副本，计划只提到 `test_verify_registry.py`。两处必须同改。

宽松档实现说明：`gaussian_smear` 无截断参数且本计划禁止改 `src/`，故宽松档由测试内 `_suite_smeared` 复刻套件生成约定（±3.5σ 截断 Gauss-Legendre + 解析归一化，401 点）。该约定为套件四个参考实现共有（refnx `_INTLIMIT`、refl1d `linspace(q±3.5dQ)`、BornAgain `DistributionGaussian(0,1,21,3.5)`、GenX `resintrange=3.5`）。生产的无截断路径另由 `test_untruncated_production_smearing_agrees_away_from_minima` 钉住。两档容差均未放宽。
