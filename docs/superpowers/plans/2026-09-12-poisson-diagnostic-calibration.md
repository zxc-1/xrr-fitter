# Poisson Diagnostic Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复低计数 Poisson 启发式误拒，同时保留真正失配、不可用和物理硬门禁。

**Architecture:** 同一有界 nuisance refit 映射作用于 observed/null；纯统计层生成不可变证据，服务层注入 fit callable。所有区间、automatic 和导出消费同一份 owner-bound 证据。

**Tech Stack:** Python 3.12、现有 NumPy/SciPy、pytest、Radon；不新增依赖。

## 当前总体状态（2026-09-14）：已批准范围的修复与验收闭环

RMS v4 已完成原低计数误拒修复和八个 diagnostics 端点；后继 `2026-09-14-joint-bootstrap-provenance.md` / `2026-09-14-joint-bootstrap-coverage.md` 又补齐联合 owner/content 双 seal，并完成 joint Poisson 有限 B 区间修订。当前为 schema5 / `xrr-fit-v2-poisson-5`，diagnostic v4、refit policy v3不变。

新固定2000例、seeds7010000..7011999：formal1995、正式覆盖 **1954/2000=97.7%**，单侧95% CP下界约 **.9706866254≥.95**；R5/U0，无缺例或补样，满足预登记K≥1917。完整独立审计002 PASS：2000病例观测重放/公共回读、无逐例缺口、末态身份复核；工程同快照 **4483 passed / Radon611零问题**。这些证据关闭 RMS 父计划遗留的**已批准低计数共享厚度场景**覆盖项，不声称普适95%保证；新区间更宽。

当前证据根 J=`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/joint-bootstrap-20260914`；汇总 `joint-bootstrap-outcome-001.json` SHA `35d54da704b9b02ae506f5c7c591de46e79f6ec16ae2b57fdc15273f57e23856`，真实审计 `scientific-audit-002.json` SHA `01f1ab1c5bf94ee2c775bd2be8796b9a3872bd1a4c3d381f091b1c0b72d12249`。详情见修复 progress 与 `docs/acceptance/fitting-algorithm-v2.md`。

旧P5/P6模板已由后继RMS与joint验收替代，不再执行；旧未完成审计不追认为PASS，丢失原件不声称找回。原81%、旧4/8 FAIL、旧joint93.4%与所有失败保留。所有实验/summary/audit均已结束，不重启；没有新增旧格式兼容、依赖或CI门禁。收尾只同步文档和执行账，保留原件、分支与未提交改动，不提交/发布。下方“当前/运行中/未关闭”均为历史快照。

## 历史总体状态（2026-09-14）：RMS v4 诊断验收通过，联合区间覆盖当时待关闭

按用户已批准 RMS 方案完成实现、真实 RED→GREEN、独立代码审查、七 MODE **4420 passed**、Radon602零问题和五例实际B999。预登记 **4400新holdout+200开发重放**全部完成，完整独立 audit003 PASS（4600病例/6600观测流、0逐例证据缺口、最终hash复核）；八个 diagnostics 端点全部 PASS。原200种子profile覆盖 **188/200=94%**，新single **1908/2000=95.4%**，不是只完成工程子计划。
**总体覆盖条件仍不关闭：** joint bootstrap **1868/2000=93.4%**、条件1868/1990=93.8693%；事后分位数敏感性不等于已校准的修复。若修改 bootstrap 方法，需另行设计与新独立验证，不在已暴露 holdout 上调法。joint 也没有独立 bootstrap context/content seal；审计只验证当前共享report关联和工件完整性，增强该保证需另立产品契约。
当前主计划为 `2026-09-13-poisson-rms-calibration.md`，结果/证据入口为 `2026-09-12-poisson-diagnostic-calibration-progress.md` 与 `docs/acceptance/fitting-algorithm-v2.md`。其 R4 实验/审计/文档步骤已完成，覆盖总体关闭项保留；下方 P1–P6 的旧版本检查点和未勾选模板属于历史过程，不作为重启已完成实验的指令。原81%、旧4/8 FAIL、旧审计器FAIL和证据丢失事故不改写。

## 历史总体状态（2026-09-13）：未完成，继续原 81% 目标

用户要求的是低计数整体误拒/区间产出修复，而非只完成数值子计划。原 200 例有 32 个区间被诊断阻断，162/200=81%；不能以软件测试通过替代科学验证。当前运行时已是两阶段 v3，T1–T4 的工程闭环保持有效，但本计划 P5/P6 的总体科学关闭仍未完成。

当前 v3 已新鲜复现剩余 ACF 问题：`family-calibration-001/acf-v3-baseline-001/report.json` 为 FAIL，仅失败项 `original injected target detection`；999/999 null 成功、4000 paths、最大单路径 44/80，但 ACF adjusted p=.03>.01。该结果不能因为拟合成功而改写为修复通过。

继续处理同族尺度竞争，并完成原 200 例开发重放与另行冻结的新 seed 科学验收。**此前 minP 推荐已撤回，旧提案原件保留，不得直接实施。** 当前实际支持 3-member/12-column joint 和 B99；确定性反例验证 minP 在这些合法范围可能无可拒绝的 p。修订推荐为全行 median-centered RMS + 连续 max，见外部 `family-calibration-001/design-proposal-rms-002.md`；范围证据为 `family-calibration-001/family-scope-003.json`。修订设计已提交用户确认，尚未修改生产统计规则或启动新 holdout。旧 FAIL、旧协议/证据与两阶段成果都保留；不得再次把数值子计划完成当作本总目标完成。

本节外部路径基准：`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913`。当前身份是 schema4 / `xrr-fit-v2-poisson-3` / objective 字符串2 / `poisson-refit-null-v3`，详细工程账在 `2026-09-12-poisson-diagnostic-calibration-progress.md`。

## 历史：2026-09-13 Poisson 精度 v2（精度子计划已闭环）

相对总成本 `ftol` 过早停止的窄修复已实施；低计数 GN 慢收敛与 ACF 尺度竞争尚未修完，不能宣称整体修复或科学验收通过。
当前唯一身份为 schema `4`、algorithm `xrr-fit-v2-poisson-2`、objective 字符串 `"2"`、diagnostic `poisson-refit-null-v2`；无旧格式转换。
新计划为 `docs/superpowers/plans/2026-09-13-poisson-local-precision.md`，完整最新记录见 `docs/superpowers/plans/2026-09-12-poisson-diagnostic-calibration-progress.md`。
持久证据根为 `/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913`；runtime SHA256 `9e3cbbef49d6105c1924cd7ffb437dd35a03ef66205a96906bf019163156fecb`。
seed37 footprint/surface 的当前 API 开发重放均完成 `999/999`，目标 `p=.001`；119/148 仍 `diagnostic_null_refit_failed`，不是新的覆盖率或独立验收。
`validation-precision-003` 在同一个 678-file 快照上 fresh quality/tools/unit/integration/gui/spawn/regression 全部通过，共 **4189 passed**，完整 Radon PASS；两份独立窄审查 Approved、无未解决项。001 的 unit FAIL、002 的 tools FAIL 均保留，测试期望修正有独立推导及 RED→GREEN。最终汇总为 `precision-engineering-final.json`；没有重跑 220-case corpus 或新 Poisson holdout。
旧 `4400 + 200` 实验的 `4/8` 端点 FAIL 保留；旧协议主动拒绝新运行时，未注册新 holdout，未重用已暴露 seeds 或旧审查。以下原“当前／进行中”表述均为各自时间点的历史记录。

## 2026-09-13 continuation: completed FAIL and evidence retention incident

The fixed experiment was observed completing before the later environment/time gap: **4400 holdout + 200 replay complete, 0 incomplete records; final scientific status FAIL, 4 / 8 endpoints failed**. Tool session 7968 ended with exit 1 and `FINAL FAIL complete= True`; all 200 driver jobs exited 0. Do not restart that finished driver.

| Scenario | Observed completed outcome | Gate |
| --- | --- | --- |
| null_single | R=3, U=18, F=21; profile 1979 / covered 1884 | F PASS; U FAIL |
| null_joint | R=8, U=0; formal Bootstrap 1992 / covered 1878 | F PASS; U PASS |
| background | target 100/100 | PASS |
| footprint | target 0/100; mismatch unavailable 100 | FAIL |
| surface | target 2/100; mismatch unavailable 98 | FAIL |
| acf | target 0/100; all 100 calibrated, omnibus 23 from other detectors | FAIL |

At UTC 2026-09-13 03:28 the old `/tmp/xrr-v2-poisson-fix-fww0fik3` evidence root and its `/private/tmp` alias were found absent. The cause is unknown; this agent did not delete it. Prior completed tool observations remain historical evidence, but missing raw files/reports cannot now be independently rechecked. Outstanding N2/power audits must not be relabeled complete. The old preservation/RUNNING statements below describe an earlier state, not current file availability.

New persistent evidence root: `/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913`. `evidence-loss-incident.json` distinguishes historical observations from new file-verifiable evidence. A fresh 676-file `baseline-source/` snapshot is not a recreation of the lost original backup. The current 258-file runtime identity was independently recomputed and still exactly equals `7685c53a0c700fa15b8b4765f54837d5a3cd4318b7336d59984229cb293058c0`.

Current work: reproduce the three failure mechanisms using original development seeds 119/148 and existing non-holdout development seed 37; then make the smallest approved numerical/statistical revision with genuine RED -> GREEN and new unexposed-seed preregistration. Never overwrite this version's failed result or reuse its exposed holdout as independent acceptance. No legacy compatibility work, automatic commit or release.

Fresh development evidence is under `development/`: seed 37 footprint and surface mismatch reproduce exactly. Disabling only relative-cost `ftol` in an isolated process lets footprint complete B=999 and correctly reject, while all match limits, starts and budgets remain unchanged. These are diagnostic probes, not production changes or scientific PASS.

## Global Constraints

- 规格：`docs/superpowers/specs/2026-09-12-poisson-diagnostic-calibration-design.md`。
- 工作树：`/Users/dala/Desktop/XRR-Fitter/fit-algorithm-v2`；解释器：`/Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python`。
- 没有旧格式兼容需求；当前 schema_version=5、algorithm_version=xrr-fit-v2-poisson-5、objective_version="2"（既有字符串类型），diagnostic_version=poisson-refit-null-v4；refit_policy仍为declared_sobol4_lbfgsb_trf_v3。P1–P6 下方代码与命令保留其 v1 实施历史，不再执行；后继RMS/joint成果与适用范围以上方总体状态和最新进度账为准。
- 不改 Poisson 似然、signed deviance、零计数和物理模型；不调宽原阈值。
- B=999 默认、alpha=.01、固定同 estimator/随机流；不足/失败/取消/错配不发布通过。
- fit 与 analysis 不互相导入；callback 不进入保存对象，构造与 load 不执行物理。
- 原 V2 Task 1–8 已完成，不重做；所有既有未提交改动、原失败和封存证据保留。
- 不自动 commit、merge、push、tag、release；每个任务 review 后仍保留未提交。
- 子代理继承主模型和推理强度，不降级；实现写集不重叠。
- 当前联合验收报告根 `/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/joint-bootstrap-20260914`；`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913` 仅为历史诊断/RMS证据根。下方旧 `/tmp` 命令仅为历史步骤，不可用作当前证据。

### Task P1: 不可变校准证据、预算与数值 owner

**Files:**
- Create: `src/xrr_fitter/model/diagnostic_calibration.py`
- Modify: `src/xrr_fitter/model/inference.py`, `src/xrr_fitter/model/fitting.py`, `src/xrr_fitter/model/provenance.py`
- Test: `tests/unit/model/test_diagnostic_calibration.py`, `tests/unit/model/test_diagnostic_provenance.py`

**Interfaces:**
Consumes: 现有 FitEvaluationContext/ModelEvaluation/PhysicsDiagnostic 和 canonical hash。
Produces: 规格 §6 三个 dataclass、三个 provenance 函数；SearchBudget.diagnostic_samples=999、FitConfig.diagnostic_version。

- [x] **Step 1: Write failing behavior tests**

```python
def test_diagnostic_budget_is_separate_from_interval_bootstrap():
    config = FitConfig.fast(7)
    assert config.budget.bootstrap_samples == 8
    assert config.budget.diagnostic_samples == 999
    assert config.diagnostic_version == "poisson-refit-null-v1"
```

- [x] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/model/test_diagnostic_calibration.py tests/unit/model/test_diagnostic_provenance.py`
Expected: FAIL for the missing behavior above; import typos or unrelated failures are not RED evidence.

- [x] **Step 3: Implement the single scoped change**

```python
@property
def p_value(self):
    return None if self.status != "available" else self.tail_count / (self.sample_count + 1)

@property
def rejected(self):
    return None if self.p_value is None else self.p_value <= self.alpha
```
逐项实现规格 §6 状态不变量、冻结/pickle 与 §6 provenance；available 的 min(p_j)==p，unavailable 不含尾概率。
把新字段追加到旧 ResidualEvidence 字段之后；校准与 effective 布尔必须按 dataset_id 一致。
不修改 codec/GUI/服务，不删除现有 model 行为。

- [x] **Step 4: Verify GREEN and related regressions**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/model/test_diagnostic_calibration.py tests/unit/model/test_diagnostic_provenance.py`
Expected: all collected tests PASS; save actual output in the external task report and add existing directly affected test files.

- [x] **Step 5: Independent Spec/Quality review; retain uncommitted**

Generate a diff against the pre-task snapshot, review this task and its stated interfaces, fix all Critical/Important findings and re-run the covering test paths. Record actual results; do not create a commit.

### Task P2: 固定声明起点的单/联合数值 refit

**Files:**
- Create: `src/xrr_fitter/fit/diagnostic_refit.py`
- Test: `tests/unit/fit/test_diagnostic_refit.py`

**Interfaces:**
Consumes: P1 DiagnosticRefit、现有 least_squares callbacks、joint compiler/evaluation。
Produces: `refit_diagnostic_single(problem, *, cancelled=None)`、`refit_diagnostic_joint(template, members, *, cancelled=None)`。

- [x] **Step 1: Write failing behavior tests**

```python
def test_declared_starts_are_independent_of_observed_optimum():
    first = diagnostic_starts(np.array([.4, .6]))
    second = diagnostic_starts(np.array([.4, .6]))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], [.4, .6])
    assert len(first) == 4
```

- [x] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/fit/test_diagnostic_refit.py`
Expected: FAIL for the missing behavior above; import typos or unrelated failures are not RED evidence.

- [x] **Step 3: Implement the single scoped change**

```python
points = qmc.Sobol(initial.size, scramble=False).random_base2(2)[1:]
starts = [np.array(initial, copy=True)]
for point in points:
    if not any(np.array_equal(point, previous) for previous in starts):
        starts.append(point)
```
按规格 §5 用同一 bounded least_squares 与原 objective；所有路径成功后取最低值，否则返回带已用 nfev 的 DiagnosticRefit failure。
Joint 每次完整 recompile，返回 global unit 及所有成员 evaluation；禁止调用 analysis/report/API。
验证有限差异、固定变量、0维、共享参数、每路径预算、非收敛和取消。

- [x] **Step 4: Verify GREEN and related regressions**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/fit/test_diagnostic_refit.py`
Expected: all collected tests PASS; save actual output in the external task report and add existing directly affected test files.

- [x] **Step 5: Independent Spec/Quality review; retain uncommitted**

Generate a diff against the pre-task snapshot, review this task and its stated interfaces, fix all Critical/Important findings and re-run the covering test paths. Record actual results; do not create a commit.

### Task P3: 连续统计、MC 引擎与全部推断入口

**Files:**
- Create: `src/xrr_fitter/analysis/residual_statistics.py`, `src/xrr_fitter/analysis/residual_calibration.py`
- Modify: `src/xrr_fitter/analysis/diagnostics.py`, `report.py`, `profile_calibration.py`, `profile_tasks.py`, `profiles.py`, `binary_profiles.py`, `automatic.py`, `joint.py`, `classification.py`
- Modify: `src/xrr_fitter/services/fitting.py`, `services/fitting_phases/automatic_dataset.py`, `joint_analysis.py`, `joint_execution.py`, `joint_selection.py`
- Test: `tests/unit/analysis/test_residual_calibration.py`, `test_residual_calibration_flow.py`, `tests/unit/services/test_diagnostic_calibration_flow.py`
- Clarified integration: `model/bootstrap.py` and its focused model tests preserve samples/quantiles while sharing Poisson diagnostic publication eligibility; `fit/problem.py` uses NaN-aware equality for unchanged masked q coordinates (both have RED→GREEN evidence).

**Interfaces:**
Consumes: P1 evidence/owner、P2 refit，通过 service callable 注入。
Produces: `residual_statistics(problem,residuals,dataset_id=None)`、`calibrate_residuals(...)`；AnalysisRequest.residual_evidence；一次证据消费的 report/profile/automatic/joint。

- [x] **Step 1: Write failing behavior tests**

```python
def test_continuous_max_does_not_tie_unrelated_column_maxima():
    values = np.zeros((200, 4))
    values[0, 0], values[1, 1], values[2, 2], values[3, 3] = 100, 4, 5, 6
    result = symmetric_max(values)
    assert result.tail_count == 1
    assert result.p_value == .005
```

- [x] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/analysis/test_residual_calibration.py tests/unit/analysis/test_residual_calibration_flow.py tests/unit/services/test_diagnostic_calibration_flow.py`
Expected: FAIL for the missing behavior above; import typos or unrelated failures are not RED evidence.

- [x] **Step 3: Implement the single scoped change**

```python
centers = np.median(values, axis=0)
scales = np.maximum(1.0, 1.4826 * np.median(np.abs(values - centers), axis=0))
standardized = (values - centers) / scales
scores = np.maximum(0.0, np.max(standardized, axis=1))
tail_count = int(np.count_nonzero(scores >= scores[0]))
adjusted = tuple(float(np.mean(scores >= max(0., item))) for item in standardized[0])
```
实现 §4 全统计链（有限性拒绝、固定几何检测族、无 jitter）、§5 16 个样本有界批次和 observed-match 三重保护。
由 service 组合 callback，校准在 profile tasks 外执行；参数化报告复用，不将 derived advisory 带回物理输入。
所有有效推断资格检查统一消费 executed/effective diagnostics，unknown 不得自动通过。

- [x] **Step 4: Verify GREEN and related regressions**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/analysis/test_residual_calibration.py tests/unit/analysis/test_residual_calibration_flow.py tests/unit/services/test_diagnostic_calibration_flow.py`
Expected: all collected tests PASS; save actual output in the external task report and add existing directly affected test files.

- [x] **Step 5: Independent Spec/Quality review; retain uncommitted**

Generate a diff against the pre-task snapshot, review this task and its stated interfaces, fix all Critical/Important findings and re-run the covering test paths. Record actual results; do not create a commit.

### Task P4: 当前文件契约、导出与 GUI 校准展示

**Files:**
- Create: `src/xrr_fitter/io/codec_diagnostic_calibration.py`
- Modify: `src/xrr_fitter/io/codec_inference.py`, `codec_common.py`, `codec_declarations.py`, `export_evidence.py`
- Modify: `src/xrr_fitter/model/project.py`, `src/xrr_fitter/gui/results/inference_text.py`
- Test: `tests/unit/io/test_diagnostic_calibration_codec.py`, `tests/gui/test_diagnostic_calibration_text.py`
- Update: 受当前 schema/字段严格断言影响的现有 model/io/export 测试

**Interfaces:**
Consumes: P1 dataclasses/provenance 与 P3 状态；Produces: strict current JSON + shared export projection + Chinese UI text。

- [x] **Step 1: Write failing behavior tests**

```python
def test_calibration_payload_never_recomputes_physics(calibrated_evidence, monkeypatch):
    payload = calibration_to_dict(calibrated_evidence)
    restored = calibration_from_dict(payload)
    assert restored == calibrated_evidence
    payload["tail_count"] += 1
    with pytest.raises((ValueError, ProjectSchemaError)):
        calibration_from_dict(payload)
```

- [x] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/io/test_diagnostic_calibration_codec.py tests/gui/test_diagnostic_calibration_text.py`
Expected: FAIL for the missing behavior above; import typos or unrelated failures are not RED evidence.

- [x] **Step 3: Implement the single scoped change**

```python
SCHEMA_VERSION = 4
ALGORITHM_VERSION = "xrr-fit-v2-poisson-1"
```
严格编码所有新字段并重算 evidence seal；optional null 白名单只增加规格允许的 null，字段缺失仍拒绝。
导出保持 raw advisory / effective / unavailable 区分，GUI 采用“校准未拒绝”，不写“模型正确”。
不新增任何旧格式读取、迁移、兼容 fixture 或双实现。

- [x] **Step 4: Verify GREEN and related regressions**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/io/test_diagnostic_calibration_codec.py tests/gui/test_diagnostic_calibration_text.py`
Expected: all collected tests PASS; save actual output in the external task report and add existing directly affected test files.

- [x] **Step 5: Independent Spec/Quality review; retain uncommitted**

Generate a diff against the pre-task snapshot, review this task and its stated interfaces, fix all Critical/Important findings and re-run the covering test paths. Record actual results; do not create a commit.

### Task P5: 预登记统计与原 200 seeds 重放（历史；由后继验收替代）

**Files:**
- Create: `tools/check_poisson_diagnostics.py`、`tools/poisson_diagnostic_cases.py`
- Test: `tests/unit/tools/test_poisson_diagnostics.py`
- Evidence: 外部报告目录内 preregistration.json、逐 seed 原始输入/结果与汇总

**Interfaces:**
Consumes: 当前 public API 与新持久化 evidence；Produces: 误拒/不可用/功效/覆盖/时间分离的报告，不改 inference 定义。

旧工具实现及复审、原200重放审计在当时记录为完成（197正式区间、187覆盖，93.5%；0/0/0）。后续4400已全部结束，最终 **4/8端点FAIL**，不是仍在运行；N1的U=18超过上限9等失败及原件丢失事故见上方历史记录。旧N2/阳性未完成的独立核验不追认通过。下列原步骤3/5停用，由后继RMS R3/R4与joint独立验收替代，不重启旧driver或覆盖原FAIL。

- [x] **Step 1: Write failing behavior tests**

```python
def test_summary_keeps_unavailable_in_denominator():
    rows = [{"status": "available", "covered": True}, {"status": "unavailable", "covered": False}]
    summary = summarize(rows)
    assert summary["attempted"] == 2
    assert summary["unavailable"] == 1
    assert summary["all_seed_coverage"] == .5
```

- [x] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/tools/test_poisson_diagnostics.py`
Expected: FAIL for the missing behavior above; import typos or unrelated failures are not RED evidence.

- **Step 3（历史停用；后继验收替代，未追认旧版 PASS）: Implement the single scoped change**

```python
all_seed_coverage = covered_count / attempted_count
conditional_coverage = None if available_count == 0 else covered_count / available_count
```
把固定 seed、物理生成参数、null/power 样本量和阈值先写入 manifest 并 hash，再执行；结果不得反写阈值。
原 200 cases 全部用当前 API 重建与 save/load，另跑预登记独立 null/四类失配，失败病例逐项保留。
统计结论只有对应新报告 PASS 才可声称；不能把纯矩阵检查或旧 3699 passed 冒充该验证。

- [x] **Step 4: Verify GREEN and related regressions**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/tools/test_poisson_diagnostics.py`
Expected: all collected tests PASS; save actual output in the external task report and add existing directly affected test files.

- **Step 5（历史停用；旧缺失审计不补称完成）: Independent Spec/Quality review; retain uncommitted**

Generate a diff against the pre-task snapshot, review this task and its stated interfaces, fix all Critical/Important findings and re-run the covering test paths. Record actual results; do not create a commit.

### Task P6: 整合审查、回归和清理（历史；由后继验收替代）

**Files:**
- Modify: 当前修复计划勾选、修复 progress 与 acceptance 文档
- Evidence: 外部每个 verify MODE、Radon、独立审查、cleanup 和最终 snapshot

**Interfaces:**
Consumes: P1–P5 新鲜证据和 source diff；Produces: 独立 Spec/Quality 审查及可重放最终清单。

旧工程记录为八类MODE合计4172 passes（保留三个实际快照归属）、Radon586零违规及工程复核Approved；旧完整科学协议已结束并FAIL，不是仍在运行。下列历史模板不再作为待办，也不追认为旧整合验收PASS。后继同快照4483工程、Radon611零问题和新固定N2000完整独立审计已通过，当前闭环见本页顶部；旧证据丢失状态保留。

- **Step 1（历史停用；后继回归替代）: Write failing behavior tests**

```python
def test_invalid_calibration_cannot_publish_formal_interval():
    result = fit_case_with_exhausted_diagnostic_budget()
    assert result.uncertainty.profiles[0].confidence_level is None
    assert not result.uncertainty.member_residuals[0].executed
```

- **Step 2（历史停用；不补造旧 RED）: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/analysis/test_residual_calibration_flow.py`
Expected: FAIL for the missing behavior above; import typos or unrelated failures are not RED evidence.

- **Step 3（历史停用；后继完整验收替代）: Implement the single scoped change**

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python tools/verify.py quality --report-dir /tmp/xrr-v2-poisson-fix-fww0fik3/implementation/final-quality
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python tools/check_radon.py --output /tmp/xrr-v2-poisson-fix-fww0fik3/implementation/radon.json
```
依次运行 tools/unit/integration/gui/spawn/regression，报告目录唯一且外置；按生产影响重新执行 statistical 验收，不复用旧 PASS。
独立 reviewer 检查本修复增量与旧变更边界；Critical/Important 全部修复并补测后才勾完成。
只清理本轮明确创建的缓存，保留报告、失败输入、基线备份和 .superpowers；不提交/发布。

- **Step 4（历史停用；不补造旧 GREEN）: Verify GREEN and related regressions**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/xrr-v2-poisson-fix-fww0fik3/implementation/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/analysis/test_residual_calibration_flow.py`
Expected: all collected tests PASS; save actual output in the external task report and add existing directly affected test files.

- **Step 5（历史停用；旧缺失审计不补称完成）: Independent Spec/Quality review; retain uncommitted**

Generate a diff against the pre-task snapshot, review this task and its stated interfaces, fix all Critical/Important findings and re-run the covering test paths. Record actual results; do not create a commit.
