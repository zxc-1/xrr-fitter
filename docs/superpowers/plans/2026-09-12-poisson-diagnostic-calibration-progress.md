# Poisson diagnostic repair progress

## 当前：2026-09-14 Joint Bootstrap v5 已通过完整验收

原低计数误拒/RMS诊断修复之外的剩余joint问题已继续修复：独立owner/content双seal，以及仅joint Poisson使用的有限B content-tolerance区间。当前schema5 / `xrr-fit-v2-poisson-5`，diagnostic v4 / refit policy v3不变；B200选第2/199秩，不增加B、不放宽原门禁、无旧格式兼容。实现和保存/加载合同均经过真实RED→GREEN与独立审查。

- 冻结709文件的普通clone七MODE **4483 passed**，完整Radon **611 files / 0 issues / PASS**；root逐原始回执/log与源码复核，未重跑已完成门禁。
- 开发37的002完成真实public fit/save/load、bootstrap200/200及双seal；diagnostic为not_triggered，配置B999不等于该例实际生成null。开发001路径工具误拒发生在拟合前，失败原件保留，002不增加独立样本量。
- 新独立seeds7010000..7011999，固定2000全部成功；正式1995、覆盖 **1954/2000=97.7%**，单侧95% CP下界约 **.9706866254≥.95**，满足唯一主端点K≥1917；R5/U0、missing0，拒绝仍在分母。Raw1959/2000和探索性1863/2000仅描述。
- 独立审计002 **PASS**，session31061 exit0，stderr空；2000病例完成观测重放/公共回读，逐例缺口0，末态科学树/source/tool/runtime复核通过。独立CP=.9706866253773252，与producer相差约1.33e-15，其余汇总字段完全一致；未重拟合、生成null/bootstrap或独立重算forward。
- 审计001的producer hash域混淆FAIL保留；A2显式区分science ASCII+LF、product/auditor UTF8无LF与raw文件hash，真实RED→105 toy GREEN且独立复审PASS后才新授权。没有try-both/fallback，也没有重跑2000例。

证据根J=`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/joint-bootstrap-20260914`。`joint-bootstrap-outcome-001.json` SHA `35d54da704b9b02ae506f5c7c591de46e79f6ec16ae2b57fdc15273f57e23856`，`scientific-audit-002.json` SHA `01f1ab1c5bf94ee2c775bd2be8796b9a3872bd1a4c3d381f091b1c0b72d12249`，`engineering-recheck-closeout-001.json` SHA `f4bdcac19160a7b94c287956cd9a440236819b0755a0af779f20e13f495fda7e`。

同日provenance/coverage计划与RMS父计划的已批准场景覆盖项据此关闭。结论仅支持预登记低计数共享厚度场景，不是任意模型/边界/真实测量的普适95%保证；正式平均宽度3.8961，对比探索性线性3.0831，更宽是代价。旧81%/旧4/8 FAIL/证据丢失/旧joint93.4%全部保留，旧P5/P6未完成审计只标后继替代，不追认旧PASS。

最终仅同步7份状态文档与ignored账，前后证据在`J/status-joint-001/`；冻结实现不变，无工作树临时探针需删除，外置失败/输入/日志/缓存/clone保留；不提交/合并/发布。以下均为历史检查点，旧“当前/未关闭”不替代本节，不重启任何已结束实验或审计。

### 历史摘要：RMS v4 当时状态

**RMS v4 的诊断修复、固定全量实验与独立审计已通过；总体 95% 区间校准仍未关闭。** 原 200 种子开发重放已从历史 81% 改善到 94%，新单曲线 holdout 为 95.4%；剩余明确问题是 joint bootstrap 的 93.4%，不是未执行原 81% 修复。当前入口仍为 `2026-09-13-poisson-rms-calibration.md`。原 81% 和旧 4400 + 200 的 4/8 FAIL 保留，不能写成当前 v4 的新结果。

## 历史：2026-09-14 RMS v4 修复

证据 I = `/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913/family-calibration-001/rms-implementation-001`。原始 690 文件备份保留在 I/before。

- R1 已落地：全行 median-centered RMS、规范 sorted fsum、peak-factorized z、精确 ties 和真常量零尺度。原 detector/refitter 不变；RED 20 failed / 17 passed → GREEN 37 passed；复杂度拆分后完整 Radon PASS。
- R2 已落地：零/fractional scale 与嵌套不可变证据重验，当前 algorithm/diagnostic/method v4，数值 policy 仍 v3。scale RED 19/6 → 292 passed，身份 RED 5/4 → 334 passed。独立 `r1-r2-review-001.md` 未发现生产阻断项；P3 补丁 EOF 工件已另存有效 002，原件不改。非 Poisson frozen config 仅 diagnostic_version 推导 hash 改变，RED 1/15 → 16 passed，数值基线不改。
- R3 已完成：协议 RED 19/57 → 109 passed，实际最终批准 hash 已绑定并独立复核。整体审查的 checkpoint 身份/冻结文件表绑定缺口已关闭；两份 examples 尾部 LF 修复有真实 RED 1 → 完整 codec 63 passed 及限定补审。最终普通 clone `validation-rms-002/source` 同快照七 MODE 共 **4420 passed**（189/609/2798/39/711/4/70），完整 Radon **602 files、0 issues、PASS**；仅一条既有 GUI 小画布 warning。
- 五例 B999 的 002 新执行均通过：119/148 family p=.124/.020，footprint37/surface37/acf37 的指定目标 adjusted p 均为 .001。新执行绑定审计 `rms-development-binding-audit-003.json` AUDIT_OK，与已独立复算的 001 输入/U/JSONL/校准精确对账，衔接 20000 条原生成功路径。固定开发病例不充当新独立样本。
- R4 固定实验与审计已完成：新 4400 + 开发 200、8 shards × 575 全部封存，10 项操作均 exit 0；manifest SHA `42fe7536bebfa49726c156fb2dc52135b036908075a73d092c59ab016bd5c628` 不变，不重启任何 shard。完整审计 `rms-scientific-audit-full-003.json` PASS，4600 cases / 6600 observation streams、30327 files、697 source files，0 个逐例证据缺口，末尾 hash 复核通过。
- 八个预登记 diagnostics 端点全部 PASS：single R8/U0、joint R10/U0；四种阳性指定目标各 100/100；4600 个病例均 fit_failed=0。single 正式 profile 1992/覆盖1908，全 seed **95.4%**、条件 **95.7831%**；原 200 重放正式198/覆盖188，全 seed **94%**、条件 **94.9495%**；joint 正式 bootstrap 1990/覆盖1868，全 seed **93.4%**、条件 **93.8693%**。joint 不提供 profile，不将 profile=0 当失败或伪造覆盖。
- 外部 audit001 的空字符设备误拦与 audit002 的 joint 归属投影误判均保留真实 FAIL。窄修复分别经 guard RED→GREEN60、owner RED64/7/0→GREEN64 及独立补审；最终 audit003 session33083 exit0，未新 fit/null/bootstrap 或改写实验。003 SHA `19a5d5b7d6d52d0eaf266f669d136caef3b960e2634dddc610789f8463bb551a`，completion003 SHA `87320a1766e8fafbcd58e0424e7ce7e7cce2051e29078900e854ba0dbf715812`。

本轮汇总为 I/`rms-r4-outcome-001.json`（SHA `a0543cc4ebc99b4e775a577639d58a12e92fd041892e958118177cf290b3cccc`），显式保留 `overall_coverage_goal_closed=false`。最终 snapshot SHA `78d681b6bba4eef2afe5c73d2331728d0fc9ac448f5242c4c5c4264bbfa697e0`，runtime SHA `7702a73a4013982919c7db21e12f39d068b14168ed663dbdd7a8a42e6381c42c`；工程回执 `rms-engineering-completion-002.json`、实际命令及 session 回执 `science-operations-001/`。旧失败及原件全部保留；冻结后仅同步状态文档，差异分列 `status-rms-r3-001/`、`status-rms-r4-001/`，冻结源未变，不将当前整树冒称原快照。

### 当前保留项，不以端点 PASS 掩盖

- **Joint bootstrap 一般 95% 校准仍打开。** B200/type7 与 RMS 前源码相同，2000 例各成功200/200；10 个正式资格缺失恰为诊断拒绝。仅在已保存样本上改 type6 的事后反事实：保持 gate 为 1890/2000=94.5%，包含被拒绝组探索区间才是1899/2000=94.95%。两者都不是新验收；`32=22+9+1` 只是反事实算术，不是唯一根因证明。解释证据为 `rms-joint-coverage-interpretation-review-001.md` / `002.md`、`rms-joint-quantile-sensitivity-001.json`。如改区间方法，需另行设计及新独立验证，不在已暴露 holdout 上调参。
- **Joint bootstrap 当前没有独立 context/content provenance seal。** 审计显式核全部 winner、共享 uncertainty 全内容及工件 seals；这不等于上述独立 seal。合同与限定补审见 `rms-auditor-bootstrap-owner-contract-review-001.md`、`rms-auditor-bootstrap-owner-review-003.md`。若增强这一保证，应另立产品契约，不回填本轮原件。
- 未验真实用户测量数据；B99/更大联合族的确定性回归不升级为独立统计验收。无工作树临时探针需清理；外置报告、失败、克隆、缓存及输入作为证据保留。不 commit/merge/push/release。

此前 minP 推荐已撤回，改用已批准 RMS 连续同族校准；B99/12 列单测不是更大联合配置的独立功效验收。全部固定预算、四起点、原生 success、MATCH_LIMITS、B999、alpha=.01 与 8 端点保持不变。

以下为历史检查点，不替代上面的 v4 当前状态。

## 历史：2026-09-13 两阶段 v3 的 T1–T4 已闭环（仅数值工程子修复）

- 执行计划：`2026-09-13-poisson-two-stage-refit.md`；设计：`../specs/2026-09-13-poisson-two-stage-refit-design.md`。固定每起点 **L-BFGS-B 定位 → 计费的完整交接 → TRF 精化**，不是失败后 fallback。119/148 暴露的低计数坏路径已修复并完成对应 B999 重放；不据此声称全域收敛。
- 四起点、每路径预算、真实 optimizer success、全路径成功、MATCH_LIMITS、alpha、B 与统计量均未放宽。新增不可变分阶段 work 直方图和严格保存/加载校验，不改 ACF/RMS，不处理旧格式，不新增依赖。
- 当前唯一身份：schema `4`；algorithm `xrr-fit-v2-poisson-3`；objective 字符串 `"2"`；diagnostic `poisson-refit-null-v3`；method `poisson_refit_null_v3`；policy `declared_sobol4_lbfgsb_trf_v3`。

### 最终 003 验证与证据

本节 E = `/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913/two-stage-001`。
`E/validation-two-stage-003/source` 是普通 Git clone，690 文件快照 SHA256 为 `807292b7470be0069fd7c707300f83abeac1d5f9b827d3e64b60a41230c0396b`；260 文件 runtime SHA256 为 `d1c6151c0c2f678c46934193da2a5653f0e70d941d9cc84c622c246012ff51f0`。

七个正式 MODE 全部在此同一快照实际执行：quality **189**、tools **587**、unit **2740**、integration **39**、GUI **711**、spawn **4**、regression **70**，合计 **4340 passed、全部 exit 0**。完整 Radon **597 文件、0 issues、PASS**；GUI 保留 1 条既有小画布 `constrained_layout` warning。Ruff 不在现有环境，未安装，也不计为通过；本轮没有重跑 220-case corpus。

四份 `E/development-v3-*-003/report.json` 均为公共 API 实际新执行及 strict save/load，每组 **999/999 null、1000 refits、4000 paths、每路径预算 80**：

| 开发病例 | 状态 | 总实际 nfev | 最大单路径 nfev | 矩阵 | family p | 注入目标 adjusted p |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| replay119 | available | 51812 | 48 | 1000×3 | .573 | — |
| replay148 | available | 52718 | 46 | 1000×3 | .544 | — |
| footprint37 | available | 68051 | 42 | 1000×4 | .001 | .001 |
| surface37 | available | 62325 | 41 | 1000×4 | .001 | .001 |

119/148 的适用统计族确为三列；记录器按保存 calibration 的完整有序 family axis 和 seed 验证，不增删 detector。四组 003 的 12 个原始 JSONL/NPY 与 002 逐字一致，保存 calibration 一致；输入与 precision-v2 除当前身份/输出路径外相同。固定暴露病例的重复运行不增加独立样本量。

最终整体代码审查在 002 找到一个成功 unit/work 维度错配漏洞 R6，开发重放未出现该错配。最小交叉校验有真实 **RED 4 failed / 28 passed → GREEN 430 passed**，构造/pickle 的 32 个独立值场景通过，`E/refit-dimension-review-003.md` 已关闭 R6。原 `E/two-stage-final-code-review-002.md` 的 Changes required 结论保留；最终 003 七门禁覆盖修复后的实现。

证据链：`E/two-stage-offline-audit-003.json` 完整复算四矩阵、16000 条原始路径及 20 个拒绝负控；`E/development-evidence-review-002.md` 提供独立统计/工作审查，必须连同 `E/development-seed-erratum-002.md` 引用，原报告被 JS Number 舍入的 seed 表不得用于重放；`E/final-validation-review-003.md` 独立确认 003 正式收据与 002 数值原件的精确衔接，0 新发现。
最终工程汇总为 `E/two-stage-engineering-final.json`，SHA256 `b03f674f1db4a263041400ce967b5ad7127a14c7d8e9f8f29a2f816f4f98cbbb`；其中保留全部审查范围、原失败与更正链，不把后续关闭倒写为早期通过。

正式验证绑定冻结 003；随后只同步 6 份状态文档（5 份非忽略文档及忽略的 `.superpowers/sdd/progress.md`），前后 manifest 和独立 diff 见 `E/status-update-two-stage-003/`。生产/测试/工具与 runtime 不随文档收尾变化。日志、缓存、验证 clone 和必要输入外置保留作复现证据；未删除用户缓存或旧产物，无工作树临时探针待清理。保留分支与未提交改动，不 commit/merge/push/release；不声称跨会话 index 二进制 hash 从未变化，已记录的历史变化原因仍未知。

### 仍未完成的科学范围

**ACF 的 MAD 尺度下限竞争未修复，v3 总体覆盖率/FWER/power 尚无新独立验收。** 原 81% 是历史低计数覆盖率，不是 v3 的新结果；原 `4400 holdout + 200 replay` 的 `4/8` 端点 FAIL 与原件丢失事故保持原记录。后续需另行批准 ACF 尺度设计及未暴露 seed 的新科学协议，不复用旧 holdout、旧审查或放宽阈值。本次只关闭两阶段工程子计划，不将原 Poisson 总体计划标为科学 PASS。

以下各节均为历史检查点，其中“当前”“待验证”只指记录当时，不替代上面的 v3 状态。

## 历史：2026-09-13 精度 v2 的 N1–N4 已闭环（仅精度子修复）

- 计划：`2026-09-13-poisson-local-precision.md`，N1–N4 的实现、严格身份绑定、独立审查与相关工程验证已完成；原 Poisson 整体科学修复仍未完成。
- 仅对含 Poisson 成员的 single/joint/diagnostic TRF 设置 `ftol=None`；原四起点、边界、预算、`xtol=gtol=1e-10`、全路径成功、三重匹配阈值与失败语义不变。非 Poisson-only 停止规则不变，mixed diagnostic 仍不支持。
- 唯一身份：schema `4`；algorithm `xrr-fit-v2-poisson-2`；objective 字符串 `"2"`；diagnostic `poisson-refit-null-v2`；method `poisson_refit_null_v2`；policy `declared_sobol4_xtol_v2`。旧身份被明确拒绝，无兼容/转换。两个示例没有 saved fit/checkpoint，仅更新声明身份。
- 持久证据根：`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913`。当前 258-file runtime SHA256 `9e3cbbef49d6105c1924cd7ffb437dd35a03ef66205a96906bf019163156fecb`；它不再等于下文的历史 v1 快照。

### 新鲜工程证据（不等于科学验收）

- N1 `precision-n1/`：真实 RED 4，含选项控制 RED 7/PASS 3 → GREEN 54。
- N2 `precision-n2/`：真实 RED 4/PASS 10 → GREEN 142；现有 regression MODE 归属 RED 1 → GREEN 6；scoped Radon 7 文件通过。
- 身份 `precision-version/`：RED 8/PASS 156；错误异常类型的第一次测试尝试保留为失败，纠正测试后在 v1 基线上确实 RED；最终 `green-002.log` 471 passed。
- 首个普通 Git 验证 clone `validation-precision-001/`：quality 189 passed（含完整 Radon）、regression 65 passed；unit **2 failed / 2592 passed**，失败报告未覆盖。
- `precision-frozen-identity-001/identity-comparison.json`：两配置规范载荷只差 `diagnostic_version`。独立重算旧 hash，再只替换该字段即可导出新 hash；数据/结构/仪器/参数身份不变。v1/v2 完整 A–E 数值与进度规范字节一致；仅 config fingerprint 和绑定配置的 provenance 改变。仅更新两项测试身份断言，真实 RED 2 → 两完整测试模块 GREEN 14。记录脚本初两次序列化失败保留，比较仅引用完整 `*-frozen-003.json`。
- 审查补强 `precision-review-tests-001/`：四个产品 starts 记录原 solver success，恰好 4 个且全部成功；六个无重复解对检查原 unit/Q/KL 门槛。旧 v1 副本上的更强回归真实 RED 4，当前完整回归 GREEN 14。没有额外求解或放宽门槛。
- `validation-precision-002/`：quality 189、unit 2594 passed；tools **2 failed / 585 passed**。根因是完整 MODE 期望漏列新 regression、旧正向测试仍要求 v1 协议接受 v2。仅修两处测试，并让 int objective 拒绝测试隔离到单个类型差异；fresh RED 2 → GREEN 4，独立补审通过。旧失败不覆盖。

### N4 最终同快照工程闭环

`validation-precision-003/source-manifest.json` 的 678 文件快照 SHA256 为 `e903df38072364cbf1b18d0aec84207891ad5a764ff72060689eb27e8370cec1`。下面七个 MODE **全部实际在该同一普通 Git clone 上运行**，每个 start/result receipt 与 log 均绑定该快照；运行前后源码未变，不借用 001/002 的结果补齐。

| MODE | fresh passed |
| --- | ---: |
| quality | 189 |
| tools | 587 |
| unit | 2594 |
| integration | 39 |
| gui | 711 |
| spawn | 4 |
| regression | 65 |
| 合计 | 4189 |

完整 Radon PASS、0 issues；GUI 仅有既有隐藏画布 `constrained_layout` warning。`precision-engineering-final.json` 从七份正式 receipt/log、678 个源文件、两个独立审查的所有受审文件 hash 与开发结果只读复核重新装配通过。`precision-numeric-v2-review.*` 和 `precision-tools-contract-review.*` 均为 Spec/Quality Approved，未解决 C/I/M=0/0/0；它们的结论不被扩展为科学验收。

`precision-development-verification.json` 使用当前 strict API 重新只读加载四份已保存 fitted project，核对 calibration、winner unit、可用病例的 1000×4（observed + 999 null）矩阵、原报告和 runtime hash；没有重拟合、替换输入或重跑 holdout。此后只同步本组状态文档，变更及与验证快照的文档-only 差异记录在 `status-update-003/`、`status-update-004/` 和 `precision-final-handoff.json`。

所有测试运行产物/缓存隔离在持久证据根，普通验证 clone、原失败日志和必要输入作为复现证据保留；工作树没有临时探针文件。保留 `feat/fitting-algorithm-v2` 与未提交改动，HEAD/索引不变；没有删除用户缓存/旧证据或执行提交/发布。

### 历史 v2 公共 API 开发重放

`development/*-precision-v2/report.json` 均通过新输入构建、实际拟合、strict save/load；包装器只记录/断言原 solver 参数及结果，无数值替换。运行前后 runtime SHA 相同。

| 开发病例 | 当前结果 | 已尝试 / 成功 | 目标检测 |
| --- | --- | --- | --- |
| footprint37 | available | 999 / 999 | footprint p=.001 |
| surface37 | available | 999 / 999 | surface p=.001 |
| replay119 | diagnostic_null_refit_failed | 336 / 335 | 不发布 p |
| replay148 | diagnostic_null_refit_failed | 496 / 495 | 不发布 p |

119/148 的原坏路径仍在 80 nfev 用尽；16 样本批次的 attempted 计数不是坏 replicate 的索引。上述四病例不是独立新 holdout，不能给出新的总体覆盖率或科学 PASS。

### v2 当时未完成与边界

低计数 GN 慢收敛和 ACF 的 MAD 尺度下限竞争仍未修复。直接 L-BFGS-B、TNC、trust-constr 及 log1p 单点探针失败均保留；不把失败优化器改成 success，不追加预算，不实施事后 fallback。审查提出的统一两阶段定位→TRF 精化、全行对称 RMS 仍需明确设计决策及相应 RED→GREEN；本轮已询问是否先做固定两阶段数值设计，尚未收到答复，未把继续收尾等同于批准新 estimator 或 RMS。纯数组 RMS 结果不是独立科学验收。

原 `4400 holdout + 200 replay` 最终 `4/8` 端点 FAIL 与原件丢失事故继续保留。`tools/poisson_diagnostic_protocol.py` 的旧版本/审查标识刻意不改，新运行时被预检拒绝；没有新注册、旧 holdout 重用、重复 corpus、commit、merge、push、release、依赖新增或用户缓存/证据删除。以下为历史记录。

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

## Earlier implementation ledger (historical; original external files now unavailable)


User: 没有旧格式文件，不用管旧的；继续。 No compatibility work, no automatic commit/release.
Plan: `2026-09-12-poisson-diagnostic-calibration.md`.
Baseline: 647-file snapshot a9a6e921ffbe4007a6ed09bb4148506377518f6b685c730af57438c84defa2b0; `42 passed in 1.37s`.
Evidence: `/tmp/xrr-v2-poisson-fix-fww0fik3/implementation`.

- P1: complete (223 passed; scoped Radon PASS; independent Spec compliant / Quality Approved, 0 findings; `P1-review.md`). Architecture registration: exhaustive RED → 43 passed.
- P2: complete (84 module tests; 253 related regression; scoped Radon PASS; independent review I1 numerical overflow fixed with real compiler RED→GREEN, final Spec compliant / Quality Approved, 0 findings; `P2-review-final.md`)
- P3: complete (independent Spec compliant / Quality Approved, 0 findings; `P3-review.md`; final 117 focused + 108 report/profile + 43 architecture passed, scoped Radon PASS; original seed10 current-API smoke p=.178 is not statistical acceptance)
- P4: complete (independent Spec compliant / Quality Approved, 0 findings; `P4-review.md`; new 87 / related 503 / architecture 43 passed; scoped Radon PASS; 4 outside-write-set current-contract test updates additionally 58 passed, reserved for whole-change review)
- P5: tool implementation independently Approved (`P5-tool-review-final.md`); strict objective version remains string `"2"`. Final preregistration created with 8 fixed shards, B=999, 4400 new seeds plus 200 replay seeds. Manifest SHA256 `90c26e10c4115762c2ff21503c50aee0f7aab405a52c8e9254b3638188598d95`; runtime source SHA256 `7685c53a0c700fa15b8b4765f54837d5a3cd4318b7336d59984229cb293058c0` (258 files). Original 200 replay COMPLETE: 197 formal profiles, 187 covered, all-seed coverage .935, conditional 187/197; 1 rejected, 2 diagnostic-null-refit unavailable, 0 product fit failures. Independent replay audit Approved (0/0/0). Full holdout is RUNNING with the reviewed 8-worker driver after 100 fixed prefix cases and successful corpus completion. N1 null_single is complete: R=3, U=18, F=21; F CP upper .01765747 passes .02, U CP upper .01575269 fails .01 (predeclared U<=9). Full protocol status is still UNVERIFIED while N2/power continue, but a required endpoint has already FAILED; do not claim overall acceptance.
- P6: implementation review Approved after I1 writer close-before-unlink and I2 identifier-only fixes (`P6-whole-change-review-final.md`, 0/0/0). Final normal clone `validation-003`: quality 189 passed, tools 586 passed, complete Radon 586 files / 0 issues. `validation-002`: unit 2590, integration 39, GUI 711 (1 existing layout warning), spawn 4, regression 51 passed. First full 220-case corpus PASSED in `validation-001`: 2 passed in 10469.32s. Eight MODEs total 4172 passes with explicit cross-snapshot source-equivalence binding (`P6-engineering-gates.json`); original failed modes and the stopped duplicate are never counted as PASS. Source rebind is proven by 153-file import closure / 3 identifier-only AST-equivalent changes, not misrepresented as same-snapshot execution. Final bounded engineering evidence audit Approved (0/0/0), `P6-engineering-final-review.md` / `.json`; engineering gates can close. P6 remains open for full Poisson statistical completion and final evidence/cleanup closeout.

## Earlier continuation point (historical; superseded by the current section)

- Frozen 676-file implementation snapshot: `validation-003/source-manifest.json`, source content SHA256 `8599bd5912e56ef04501bd4af7890a5655bb1744cf516c426fc82ae0d5eaa113`.
- Exact repair delta against original 647-file backup: `P6-final-source.diff` / `P6-final-source-handoff.json` (66 files, diff SHA256 `f36dc2606571d5dbd286c0f71392b638093b105ae8178f996a83a070b220fe2b`). Later status-only document edits are not retroactively part of that snapshot.
- External execution driver D1 independently Closed: `execution-driver-review-closed.md`; added 3 RED, final 7 GREEN including real TERM-ignoring descendants and original-exception preservation. Driver SHA256 `0c96e78a5346564ea43a0722e7c826e88f091c99d8961afc6e567b45b844ced7`.
- Replay execution record: `poisson-replay-operation-001/execution-plan.json`, 4 completed-pair receipts and `replay-summary.json` (200/200 complete; summary CLI exit 2 because holdout is incomplete, not a PASS). All replay record/artifact seals validated; compression preserves logical bytes.
- Current execution: full reviewed driver `poisson-full-operation-001`, tool session **7968**; recover its existing session/`jobs.json`, do not launch another driver. The earlier 2-worker prefix is finished and its cutover is recorded in `poisson-holdout-prefix-001/cutover-to-full-driver.json`. Previously completed cases are verified/skipped, never refit or replaced. No N/B/seed/threshold changes; no stopping based on statistical outcomes. Do not start another 220-case corpus.
- Replay independent audit Approved, 0/0/0 (`P5-replay-independent-review.md` / `.json`): 200 seals, 1000 artifact hashes, 200 RNG reconstructions, and all saved profile boundaries independently match. Seeds 119/148 are preserved null-refit unavailable; seed 142 has p=.006 and remains rejected.
- Prefix cutover COMPLETE after the corpus passed: 100 null_single cases already sealed (shards 0..3, 25 each), remaining prefix pairs delegated to the unchanged full driver. `poisson-full-operation-001` is RUNNING with 8 workers; tool session receipt `poisson-full-driver-tool-session.json`. Completed replay/prefix cases are verified and skipped. Scientific status still UNVERIFIED until all frozen scenarios and their independent audit finish.
- N1 completed summary: `poisson-full-operation-001/summary-null_single.json`, SHA256 `934e762f419c956ba4617f28c18afca76047b1f1b25d30342f620496460cddc0`. 2000/2000 completed; raw 349, not triggered 1651, calibration available 331, all 18 unavailable are `diagnostic_null_refit_failed`, product fit_failed 0. Formal profiles 1979, covered 1884; all-seed .942, conditional 1884/1979. U endpoint FAILED; no early stopping or threshold relaxation. Independent N1 result audit is in progress.
- Root-cause investigation is restricted to original development replay seeds (119/null replicate 322, 148/493) in a separate process. Do not modify frozen runtime source; preserve current holdout failure. Any numerical revision requires a separate freeze, new preregistration and unexposed seeds.
- Final engineering audit JSON SHA256 `1fd68cf65c2dc57f315f1dbafccefb147e4c197bb494c786b0a1d659dafdf023`; it is not Poisson scientific acceptance.
- All prior failure logs, backups, review evidence, `.superpowers`, and pre-existing `.ruff_cache` are preserved. Only owned toy processes were stopped/reaped; no commit/release or legacy compatibility work.

Original V2 Tasks 1–8 remain complete. This ledger never relabels old coverage as new verification.
