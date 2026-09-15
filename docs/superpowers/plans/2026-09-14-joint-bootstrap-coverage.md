# Joint Bootstrap Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans; independent statistical and code reviews precede new scientific data generation.

**Goal:** 修复 joint Poisson 的有限 B 区间欠校准并取得新独立全 seed 正式覆盖的正面证据。

**Architecture:** 仅 joint_poisson_parametric 显式选择有限 MC content-tolerance；生成的策略/秩进入独立来源+内容seal。非 Poisson/custom 方法不静默套用诊断预算。

**Tech Stack:** Python 3.12，现有 NumPy/SciPy/pytest/Radon，无新依赖。

**验收状态（2026-09-14）：** 实现/工程/预登记唯一科学主端点/完整独立审计均PASS。固定新2000例正式覆盖1954/2000=97.7%，单侧95%CP下界约.9706866254≥.95；无补样或重跑，结论边界不变。

## Global Constraints

- 继承同日 provenance 计划所有保留/验证/不提交约束，E 相同。
- 不改 RMS diagnostics、B999、alpha=.01、MATCH_LIMITS、四起点、每路径80或失败规则。
- 不重新试旧2000/4600，不增加 bootstrap B，不扩大默认 CI。
- 新 schema5 / xrr-fit-v2-poisson-5；diagnostic v4 / refit policy v3 保持不变；无旧格式兼容。

## Statistical specification — fixed before new data

For m successful draws, choose the largest integer k>=1 with
`BetaCDF(.96; m+1-2*k, 2*k) <= .05` and `m+1-2*k > 0`.
Publish the closed interval `[X_(k), X_(m+1-k)]` (one-based order statistics).
At m=200, k=2; BetaCDF=.039529337898, k=3 fails at .185649694837.

Separate quantities: published nominal level=.95; bootstrap content target=.96; MC assurance=.95; diagnostic error-budget reference=.01. The content tolerance statement is exact for continuous iid draws from the bootstrap distribution, not a theorem of fixed-parameter plug-in bootstrap coverage and not an equal-tailed guarantee. MC assurance plus nominal diagnostic alpha alone does not prove overall coverage. Actual diagnostics/unavailable/fit/persistence loss remain measured in the all-seed primary endpoint. The minimum successful samples=200 and failure-rate gate<=.20 remain unchanged.

This is the narrowest symmetric nested order-statistic interval satisfying the fixed content-assurance rule. It deliberately pays for finite B tail uncertainty with wider intervals; no endpoint is tuned on prior exposed samples. Type6 alone would not handle diagnostic selection loss. Simply increasing B would not correct the inferential error budget. BCa requires a separate jackknife/acceleration and boundary design and is not this minimal repair.

Files: `analysis/bootstrap_samples.py`, `model/bootstrap.py`, `io/codec_inference.py`, `io/codec_common.py`, new `tests/unit/analysis/test_joint_bootstrap_intervals.py`; exact identity and protocol tests.

- [x] RED: B200 rows0..199 must give (1,198), ranks(2,199), explicit policy/levels; non-Poisson linear behavior unchanged. Deterministic n250/n999 formula, ties, monotonic transformations, finite extremes and gates are required.
- [x] Implement binary search for largest admissible k and direct sorted selection; no interpolation, jitter, tail clipping or numerical fallback. Numeric failure raises rather than fake bounds.
- [x] Add dataclass fields `interval_method` and `interval_ranks`; policy `poisson_joint_content_tolerance_v1` only for joint_poisson_parametric, otherwise percentile_linear_v1. Codec persists and validates derived `bootstrap_content_target`, `monte_carlo_assurance`, `diagnostic_error_budget` separately. Full content seal includes policy and ranks.
- [x] GREEN targeted model/analysis/codec and explicit new schema/algorithm identity tests; review and related regression. Preserve old failures/raw evidence.

## Frozen independent validation

- [x] External create-once tool: register/run-shard/summarize, no source or protocol monkeypatch. Source/runtime/tool/exact approved review SHA and full recipe bound before generation; immutable raw/input/fitted/sample evidence; each success/failure/interruption counted once, never retried.
- [x] One development-only case seed37 for end-to-end instrumentation (not a coverage claim), then fixed independent N2000, seeds7010000..7011999, 8 shards × 250, existing observation RNG domain with new seed namespace, local_workers=1.
- [x] Existing physical scene: amplitudes400/1600,80points,truth100, shared thickness with two local scales. One project is one Bernoulli trial; member count and bootstrap replicates are not extra trials.
- [x] Sole primary endpoint: formal coverage count K over all2000, single one-sided CP gamma=.05 lower bound>=.95; K>=1917. Every missing/rejected/unavailable/failed/persistence-invalid case is not covered. No stopping early, appending seeds or selecting methods after results.
- [x] Secondary descriptive output: formal availability, conditional coverage, exploratory raw bounds, R/U, widths and lower/upper misses, never substitutes for primary. Planning alternative true coverage=.965 gives power~94.67%; not a promise of PASS.
- [x] New ordinary clone after independent implementation review; run all seven engineering MODEs and complete Radon on one frozen source. No old experiment restart.
- [x] Independent audit checks fresh RNG inputs, source binding, all sample matrices/order-statistic bounds/metadata, candidate/layout/context/content seals and saved/public-load state. Only after engineering+primary scientific+audit PASS close parent coverage item.

## 实际执行与结果

- `interval-red.log` / `interval-green.log`与身份/codec回归保留；实现及发布合同独立审查PASS。709文件普通clone同快照七MODE **4483 passed**、完整Radon **611 files/0 issues/PASS**，原失败记录不覆盖。
- `science-tool-003`经104 toy及独立复审后使用`execution-bindings-003.json`，旧002的absolute输入路径误拒有真实RED→GREEN。开发001失败在拟合前，原件保留；全新开发002的seed37完成200/200 bootstrap及公共保存/加载，ranks(2,199)，区间[98.34785222347752,102.16544455125005]。diagnostic为not_triggered，并未实际生成B999 null；开发不计独立验收。
- `scientific-joint-001`按固定8×250运行，全部2000 success、missing0、fit_failed0；formal1995、K=1954，**全seed97.7%**，单侧95%CP下界约 **.9706866254**，通过K≥1917。R5/U0；formal两侧miss为22/19，无正式区间5例全部留在分母。Raw1959/2000、探索性线性1863/2000只描述；正式平均宽度3.8960645379618364（1995例），探索性3.0830675744978846（2000例）。
- 独立`scientific-audit-002.json`实际session31061 exit0/PASS，2000病例全部观测重放/公共回读、无unverifiable partial、完整树与源末态复核。原生`observations_replayed=2000`按病例计数；重放保存raw_mean，不独立重算forward。独立CP=.9706866253773252，与producer .9706866253773265差约1.33e-15，其余汇总精确一致。
- 审计001在runtime hash预检因producer规范混淆而失败；原件保留。A2显式分开science ASCII+LF、product/auditor UTF8无LF及raw hash，105 toy GREEN、独立复审PASS后新授权；没有重跑任何科学shard/summary，未修改原2000病例。

证据根E下：`joint-bootstrap-outcome-001.json` SHA `35d54da704b9b02ae506f5c7c591de46e79f6ec16ae2b57fdc15273f57e23856`；`scientific-audit-002.json` SHA `01f1ab1c5bf94ee2c775bd2be8796b9a3872bd1a4c3d381f091b1c0b72d12249`；实验树SHA `04bd3371ec7808d39e1ff6a435bd6a202abf9804a048247e5deb0d96dba89a98`；`validation-joint-002/snapshot.json` SHA `f49762be2a6069ceaa99a7d72687e5b92330aed49bf1f6fdf651d2bd62638f8f`。状态文档与冻结源的差异保存在`status-joint-001/`，旧证据/失败/clone均保留，无工作树临时探针需删，不提交或发布。

## Scope of conclusion

A PASS supports this registered low-count shared-thickness scenario, not arbitrary models, boundary parameters, all nuisance coordinates jointly, or real user measurements. The old81%, old FAIL and old joint93.4% remain unchanged historical records.
