# Poisson RMS Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans; independent review follows each bounded implementation task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 实施已批准 RMS 校准，完成原 81% 总体修复的剩余统计实现与科学验收，不以局部 PASS 收尾。

**Architecture:** 原 continuous detector 和 refitter 不变；纯统计层做规范 RMS，既有不可变证据/codec 接线，版本和独立协议同步。

**Tech Stack:** Python 3.12、已有 NumPy/SciPy/pytest/Radon、标准库 math；无新增依赖。

## 当前收尾：2026-09-14 后继 Joint Bootstrap v5 验收通过

RMS v4 的实现与4600例诊断验收保持原结果；其遗留的joint区间与独立来源/内容校验已由同日provenance/coverage两份后继计划完成。新固定N2000为正式1954/2000=97.7%，单侧95% CP下界约.9706866254≥.95，完整独立审计002 PASS；同快照工程4483 passed、Radon611零问题。**本计划总体覆盖关闭项仅就已批准低计数共享厚度场景关闭**，不扩张为普适95%保证；旧93.4%、81%、4/8 FAIL和原件丢失均保留。

当前产品身份为schema5 / `xrr-fit-v2-poisson-5`，诊断v4和refit policy v3不变；不重启任何旧实验或审计。后继证据J=`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/joint-bootstrap-20260914`：`joint-bootstrap-outcome-001.json` SHA `35d54da704b9b02ae506f5c7c591de46e79f6ec16ae2b57fdc15273f57e23856`；`scientific-audit-002.json` SHA `01f1ab1c5bf94ee2c775bd2be8796b9a3872bd1a4c3d381f091b1c0b72d12249`。新区间更宽，验收适用范围见`docs/acceptance/fitting-algorithm-v2.md`。下方E与执行账保持RMS阶段的历史归属，不把原outcome=false改写为当时通过。

## Global Constraints

- 规格：`../specs/2026-09-13-poisson-rms-calibration-design.md`，用户已批准“修”。
- B=999、alpha=.01、MATCH_LIMITS=(1e-4,1e-6,1e-6)、原有全部路径成功/预算与错误规则不变。
- RMS阶段固定身份 schema4 / xrr-fit-v2-poisson-4 / objective 字符串2 / poisson-refit-null-v4 / poisson_refit_null_rms_v4；refit_policy 为 declared_sobol4_lbfgsb_trf_v3。当前v5身份及后继验收以上方收尾说明为准。
- 已有 isolated worktree `fit-algorithm-v2`；关键实现由主代理处理，独立只读审查并行；代理推理强度一致。
- 不提交、合并、发布、删旧证据、加旧格式兼容、新依赖或扩大默认 CI。
- E=`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913/family-calibration-001/rms-implementation-001`；初始全文件备份/manifest 已在 E/before。
- P=`/Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python`；每条验证命令显式带 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.`，日志/缓存写 E。

## R1：稳定 RMS 统计内核（已实现并通过独立审查）

Files: `src/xrr_fitter/analysis/residual_statistics.py`、`tests/unit/analysis/test_residual_calibration.py`、新 `tests/unit/analysis/test_residual_rms.py`。

Interface: 保留 `symmetric_max(values)` 及 centers/scales/scores/adjusted_p_values/tail_count/tie_count，原 continuous detector 完全不改。

- [x] 替换明确被批准改变的三项 MAD-floor 测试期望，新增单位竞争、规范 ties、常量、极端值和置换测试。
- [x] 运行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/unit/analysis/test_residual_calibration.py tests/unit/analysis/test_residual_rms.py`；保存真实 RED（不得仅导入/语法失败）。
- [x] 实现每列连续副本、安全 median、peak-relative/fixed sorted fsum RMS 与显式数值失败，主函数只组合各列：`center,scale,z=_rms_column(matrix[:,index])`；`scores=np.maximum(0.,np.max(standardized,axis=1))`，p 计数原样。
- [x] 同命令 GREEN；从旧备份生成 R1 精确 diff，独立 Spec/Quality 审查。

核心数值断言：`values=[[1.,0.],[0.,2.],[0.,-2.],[0.,0.]]` 的一列单位缩放不能改变 family p；`[100,0,0,0]` 与 `[0,1,0,0]` 单峰 z 必须同幅；`[0,0,1e±200]` 的正尺度有限；非恒定正 scale 下溢为零必须失败。

## R2：证据边界和当前身份

Files: `src/xrr_fitter/model/diagnostic_calibration.py`、`model/fitting.py`、`model/project.py`、两个 `examples/*.xrrproj.json`；对应 `tests/unit/model/test_diagnostic_calibration.py`、`test_poisson_two_stage_identity.py`、新 `test_poisson_rms_identity.py`、`tests/unit/io/test_diagnostic_calibration_codec.py`、`tests/unit/analysis/test_residual_calibration_flow.py` 及固定身份断言。

Interface: DiagnosticStatistic 字段不变，scale=0 表示真常量；codec 使用当前字段和既有 seal；数值 refit policy 不变。

- [x] 先测试 `DiagnosticStatistic(None,'acf',1.,1.,0.,1.)` 和正 fractional scale 成功，负 scale/零尺度不等中心/零尺度 p<1 拒绝；伪造嵌套值、pickle、codec round-trip 与 summary/seal 负控均单独验证。
- [x] 在 v3 生产基线上取得 R2 RED；再实现非负尺度/真常量约束和嵌套重建校验；全常量 family 检查 `observed_score==0 and tail_count==tie_count==sample_count+1`。
- [x] 当前身份测试先 RED：断言 algorithm/diagnostic/method v4 而 policy v3；随后只更新当前身份及例子的声明。旧方法/版本必须拒绝，无别名/兼容。
- [x] 对冻结 fingerprint 测试独立推导配置只差 diagnostic_version，更新断言而非数值基线；必要精确消费断言逐个 RED→GREEN，不宽泛替换历史失败文本。
- [x] 运行 model/analysis/io/services/GUI 相关测试及身份测试，保存 GREEN；独立审查后关闭 R2。

## R3：开发公共 API 重放和验收协议冻结

Files: `tools/poisson_diagnostic_protocol.py`、对应 `tests/unit/tools/test_poisson_diagnostics.py`/fixtures/协议测试；外部 recorder/driver 派生物只放 E。

- [x] 先为 v4 版本、method/RMS 规则、同一 refit_policy 和新 seed 表编写协议 RED，再更新唯一定义。历史协议原件和 FAIL 不改；新 REVIEW_SHA256 必须绑定实际新审查。
- [x] 从已审 recorder 派生当前五例重放，冻结输入/版本及原记录器差分；真实执行公共 API + B999 + strict save/load，不复用旧输出冒充新执行。
- [x] 保存完整矩阵/work/input/project/report；用独立固定公式复算尺度、得分、tail/tie、定位 p 和全部失败/预算。119/148 的任何拒绝必须保留。
- [x] 完成 source/tests/tools 独立审查后创建新的普通 Git validation clone；记录文件/runtime hash。
- [x] 同快照运行 `python tools/verify.py quality|tools|unit|integration|gui|spawn|regression --report-dir <E/mode>` 和完整 `python tools/check_radon.py --output <E/radon.json>`。每次真正新运行，不借旧 v3 4340 passed。

## R4：原 200 重放及完整独立科学验收

Files: 原外部证据脚本的新受审副本/新 manifest；总体计划/进度/验收状态文档。

- [x] 预登记全部新 4400 + 开发 200 seeds、固定源快照/driver/runtime、8 端点与失败规则。预检须拒绝错误版本/审查/源码。
- [x] 按固定 shard 顺序运行完整实验，所有结果先保存 seal 后汇总；不因结果差删列、换 seed、补样或调阈值。已有完成记录只能核验后 skip，不重复已结束 driver。
- [x] 分别核算正式区间数、覆盖数、全 seed 覆盖、条件覆盖、误拒/不可用与 4 个目标 power；独立核对所有输入 RNG 与样本分母。
- [x] 必需端点全完成且通过、没有未解释的明显欠覆盖，才关闭总体目标。否则保留完整 FAIL，返回最早不确定环节继续根因定位。
  - RMS阶段八个diagnostics端点PASS、原200重放188/200=94%，但joint1868/2000=93.4%，因此当时未勾选，旧outcome=false不变。后继计划按预先固定的有限B规则修订，使用未暴露seeds7010000..7011999取得1954/2000=97.7%、单侧95%CP下界约.9706866254≥.95，且独立审计002 PASS，现仅关闭批准场景的总体项。不是用事后type6替换旧结果，也不声称任意模型或真实测量的一般95%校准。
- [x] 最终文档同步、diff-check、源身份核对；只清理本轮非证据临时项，旧输入/日志/失败不删；不 commit/merge/push/release。

## 执行账

- R1: complete，真实 RED 20 failed / 17 passed → GREEN 37 passed；复杂度拆分后完整 Radon PASS；`r1-r2-review-001.md` 未发现生产阻断项。补丁 EOF 工件另存 002，`git apply --check` exit 0，P3 勘误已独立关闭，旧原件保留。
- R2: complete，scale RED 19 failed / 6 passed → 292 passed；identity RED 5 failed / 4 passed → 334 passed。独立审查通过。另补 frozen config identity RED 1 failed / 15 passed → 16 passed，仅 diagnostic_version 改变的独立推导见 `r2-config-comparison.json`，纳入 R3 补审。
- R3: complete（2026-09-14）。协议 RED 19 failed / 57 passed → GREEN 109 passed，实际 `protocol-v4-final-review-001.md` 已绑定；批准 hash 绑定另有 RED→GREEN 和独立复核。整体审查的 checkpoint hash / freezer 文件表绑定缺口已修复并关闭。第一轮 unit 暴露两份 examples 多出的尾部 LF，保留失败记录，仅移除各一字节，RED 1 → 完整 codec 63 passed，限定补审通过。
- 最终 `validation-rms-002/source` 同快照七 MODE：189 / 609 / 2798 / 39 / 711 / 4 / 70，共 **4420 passed**；完整 Radon **602 files、0 issues、PASS**，仅一条既有 GUI constrained_layout warning。五例 002 均完成实际 B999，119/148 family p=.124/.020，三个注入目标 p=.001；`rms-development-binding-audit-003.json` AUDIT_OK，精确衔接已独立复算的 001 矩阵/work 与 20000 条路径，不增加独立样本量。
- R4（2026-09-14）：固定 8 × 575 = **4600** 已全部完成，register / 8 runs / summarize 均真实 exit 0，禁止重启。manifest SHA 仍为 `42fe7536bebfa49726c156fb2dc52135b036908075a73d092c59ab016bd5c628`；真实命令、时间、session 与日志保留在 `E/science-operations-001/`。
- 完整独立审计 `E/rms-scientific-audit-full-003.json` **PASS**，SHA `19a5d5b7d6d52d0eaf266f669d136caef3b960e2634dddc610789f8463bb551a`：4600 cases、6600 observation streams、30327 files、697 source files，0 个逐例证据缺口，final hash recheck=true。八个固定端点 PASS；single R8/U0、joint R10/U0，四阳性目标各 100/100。single profile 1992 正式/1908 覆盖（全 seed 95.4%）；replay 198/188（94%）；joint bootstrap 1990/1868（93.4%，条件 93.8693%），不捏造 joint profile。
- audit001 的 `/dev/null` 误拦、audit002 的 joint 内层 owner 误判均保留 FAIL；两个外部审计器窄修复分别有真实 RED→GREEN 和独立补审，最终绑定 completion003，未修改冻结产品或科学原件。joint 当前通过共享 winner report 关联 bootstrap，没有独立 bootstrap context/content seal；审计 PASS 不补称这种保证。
- 汇总 `E/rms-r4-outcome-001.json` SHA `a0543cc4ebc99b4e775a577639d58a12e92fd041892e958118177cf290b3cccc` 明确 `overall_coverage_goal_closed=false`。具体数字与保留限制见 `docs/acceptance/fitting-algorithm-v2.md`；原 81%、旧 4/8 FAIL 和证据丢失事故不改写。
- 冻结 snapshot SHA `78d681b6bba4eef2afe5c73d2331728d0fc9ac448f5242c4c5c4264bbfa697e0`；runtime SHA `7702a73a4013982919c7db21e12f39d068b14168ed663dbdd7a8a42e6381c42c`。冻结后工作树只有状态文档同步；原两份文档更新见 `E/status-rms-r3-001/`，本次收尾见 `E/status-rms-r4-001/`。不把文档更新后的整树冒称原冻结文件表；所有外置输入、日志、克隆、缓存及派生脚本作为证据保留，无工作树临时探针需删除。
