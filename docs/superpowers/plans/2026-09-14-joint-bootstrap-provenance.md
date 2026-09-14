# Joint Bootstrap Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task; independent read-only task and final reviews. Steps use checkbox syntax.

**Goal:** 完整封印并验证联合 bootstrap 的来源与内容，不以本项代替覆盖率验收。

**Architecture:** pure model 的 owner/content 双 seal；analysis 生成与独立消费；pure codec 内容检查；public service 使用原编译路径重建数值 owner。

**Tech Stack:** Python 3.12、现有 NumPy/SciPy/pytest/Radon。

**验收状态（2026-09-14）：** owner/content合同、公共保存/加载、当前身份、同快照工程及新N2000独立审计均已通过；不以来源校验单独代替覆盖率验收。新科学结果与适用边界见配套coverage计划及验收文档；文档终核单独留证据。

## Global Constraints

- 已有 isolated worktree `fit-algorithm-v2`；不 commit/merge/push/release，不覆盖既有改动。
- 不新增依赖、旧格式兼容或 CI 门禁，不重启旧4600/220实验，不删证据。
- `xrr_fitter.api` 是唯一公开 API，model 不依赖 fit/analysis/services。
- 保持 diagnostics B999、alpha=.01、MATCH_LIMITS=(1e-4,1e-6,1e-6)、四起点和每路径预算80。
- E=/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/joint-bootstrap-20260914，before 已备份697文件和 Git 索引hash。
- P=/Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python；测试显式 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.，禁用 cacheprovider，basetemp 唯一且在 E。
- 主代理执行共享抽象写入；子代理只读研究/审查，推理程度一致。

## J1 — Pure ownership contract

Files: `model/bootstrap.py`, new `model/joint_bootstrap_provenance.py`, new `tests/unit/model/test_joint_bootstrap_provenance.py`.

Interfaces: `joint_bootstrap_owner_sha256(problem, candidates, unit_vector, child_seed) -> str`; `seal_joint_bootstrap(result, candidate_id, owner_sha256) -> BootstrapResult`; `validate_joint_bootstrap(result, candidate_id, owner_sha256, parameter_names) -> None`; `validate_joint_bootstrap_content(result) -> None`.

- [x] RED: assert new functions exist, then determinism/pickle/moved source and mutation matrix. `owner(problem, candidates, vector, seed) != owner(changed_problem, candidates, vector, seed)`.
- [x] Implement `joint_owner_sha256: str | None = None`; paired syntax validation; owner over all context/layout/numerical fields; content excludes only its own provenance field. Use canonical model serialization, no physics call.
- [x] GREEN focused pure tests; keep original failures and exact diff.

## J2 — Generation, consumption and persistence

Files: `analysis/joint_bootstrap.py`, `analysis/joint.py`, `services/fitting.py`, `services/fitting_phases/joint_analysis.py`, `model/project.py`, `io/codec_inference.py`, new `services/bootstrap_ownership.py`; focused analysis/service/io tests.

Interfaces: `bootstrap_joint_local(problem, candidates, unit_vector, *, sample_count, child_seed, recompile, refit, ...)`; `analyze_joint_ensemble(..., bootstrap_owner=callable)`; `validate_project_bootstrap_ownership(project) -> None`.

- [x] RED: current callback sample with same axes/ID but other context must be rejected; generated evidence must carry owner; unchanged summaries with swapped samples must fail content seal; joint root rejects null owner/member fork.
- [x] Implement callback(candidate_id, vector), independent owner callback, seal generation and post-qualification reseal. Require report.parameter_members to match compiled layout when publishing/validating.
- [x] Public save/load: after existing source checks, if joint bootstrap exists, `seed=service_seed_branches(project)[1]`; `prepare_dataset_fit(project,id,seed)` in root order, `compile_joint_problem`, `joint_candidate_vectors`, saved candidates and validate owner/content. No forward evaluation, optimization or random draw. Pure codec validates content on encode/decode.
- [x] RED public roundtrip/tamper/relocation tests then GREEN; no silent source fallback for valid-source provenance mismatch.
- [x] Independent Spec/Quality review and fixes, preserving all RED logs.

## J3 — Current identity, complete integration and coverage handoff

- [x] Bump current schema/algorithm identity for changed ownership payload together with interval revision; exact current-format codec/identity tests RED -> GREEN; no legacy branch.
- [x] New normal validation clone after both repairs; fresh `tools/verify.py quality|tools|unit|integration|gui|spawn|regression` and `tools/check_radon.py`.
- [x] New frozen independent coverage protocol and data from the companion statistics design; verify both seals on every saved case; never count rejected/exploratory bounds as formal coverage.
- [x] Update parent plans only with actual evidence; final index hash and diff check; preserve all external evidence, no commit.

## Evidence

Baseline focused tests: 42 passed in 376.15s; E/baseline-pytest.log. Initial snapshot manifest sha256 ed73b08e8f2a82d3821e46b5a3667360ca709c7579481150f0c37d7a19699150. No production file changed at plan creation.

本轮E下新增证据：

- `j1-red.log` / `j1-green-003.log`、`j2-red.log` / `j2-green.log`、`j2-public-red*.log` / `j2-public-green*.log`、`identity-red.log` / `identity-green.log`保留真实RED→GREEN。`implementation-review-final-001.md`及测试合同补审`publish-contract-review-final-001.md`均Spec/Quality PASS。
- `validation-joint-002/source`普通clone709文件；七MODE **4483 passed**，完整Radon **611 files/0 issues/PASS**。快照SHA `f49762be2a6069ceaa99a7d72687e5b92330aed49bf1f6fdf651d2bd62638f8f`；`engineering-recheck-closeout-001.json` SHA `f4bdcac19160a7b94c287956cd9a440236819b0755a0af779f20e13f495fda7e`逐原始回执和源码重新核验，未重跑门禁。
- `scientific-audit-002.json` SHA `01f1ab1c5bf94ee2c775bd2be8796b9a3872bd1a4c3d381f091b1c0b72d12249`，实际exit0/PASS，2000病例全部公共回读/观测重放与双seal、sample/bounds检查，无逐例证据缺口；不独立重算forward。审计器001哈希域FAIL保留，002经105 toy GREEN及独立复审后新授权；科学数据未重跑。
- `joint-bootstrap-outcome-001.json` SHA `35d54da704b9b02ae506f5c7c591de46e79f6ec16ae2b57fdc15273f57e23856`：批准场景正式覆盖1954/2000=97.7%、单侧95%CP下界约.9706866254≥.95，工程与科学验收PASS。旧81%/93.4%及失败不改写；状态文档收尾证据独立保存在`status-joint-001/`，不把新文档冒称原冻结源。
