# 拟合算法 V2 验收

## 当前验收：2026-09-14 Joint Bootstrap v5

**已完成批准范围内的剩余修复，工程、新独立覆盖率主端点及完整独立审计均 PASS。** 原 81% 对应单曲线旧实验，其诊断误拒修复已在 RMS v4 验证；本轮继续解决旧 joint 93.4% 暴露的有限 B 区间问题，并补齐联合 bootstrap 的独立来源与内容校验。不同场景、版本和种子的数值分开报告，不把 81% 与本轮 97.7% 当作同一批样本的直接比较。

### 修复内容与当前合同

- 联合 bootstrap 增加 owner/content 双 seal：owner 绑定完整有序 context、联合布局/约束/scatter、全部 winner、global vector 和真实 uint64 seed；仅排除可合法搬迁的 `source_path`。生成、独立消费和公共保存/加载均验证；不能用各成员 single bootstrap 拼成 joint。公共加载通过原 prepare/compile/vector 路径重建 owner，不执行 forward/refit/RNG。
- 仅 `joint_poisson_parametric` 使用 `poisson_joint_content_tolerance_v1`：名义水平 .95、bootstrap content target .96、Monte Carlo assurance .95、diagnostic budget reference .01 分别保存。B200 选第 2、199 秩；成功样本至少 200、失败率不超过 20% 的门禁不变。其他方法保留原线性分位数。
- 这是满足固定 content-assurance 规则的对称嵌套秩区间族中最窄的区间，不是 plug-in bootstrap 对任意参数的普适覆盖定理。实际诊断拒绝/不可用/失败全部计入全 seed 分母；不以调整阈值、增加 B、补样或事后选法过关。
- 当前唯一身份为 schema `5`、algorithm `xrr-fit-v2-poisson-5`、objective 字符串 `"2"`；diagnostic `poisson-refit-null-v4` / method `poisson_refit_null_rms_v4`、refit policy `declared_sobol4_lbfgsb_trf_v3` 保持不变。B999、alpha=.01、四起点、每路径预算80及 MATCH_LIMITS 均未放宽；不处理旧格式、不新增依赖或扩大默认 CI。

### 新固定科学验收

固定 seeds `7010000..7011999`，8 shards × 250；两成员幅度400/1600、各80点、真厚度100、共享厚度和两自由 scale。一个项目才是一个 Bernoulli trial，成员和 bootstrap 重采样不增加独立样本量。全部 2000 成功、missing/fit_failed 均为0，无重试、换 seed、补样或早停。

| 指标 | 本轮结果 |
| --- | ---: |
| 正式区间 / 固定分母 | 1995 / 2000 |
| 正式覆盖 K / 全 seed 覆盖 | **1954 / 2000 = 97.7%** |
| 单侧 95% Clopper–Pearson 下界 | **约 0.9706866254 ≥ .95，PASS** |
| 预登记覆盖门槛 | K ≥ 1917 |
| 诊断拒绝 R / 不可用 U | 5 / 0 |
| 正式区间两侧 miss / 无正式区间 | 22 + 19 / 5 |

Raw 覆盖1959/2000、探索性线性分位数覆盖1863/2000，以及条件覆盖，均只作描述，不替代唯一主端点。正式区间平均宽度3.8961，探索性线性区间平均宽度3.0831；**区间更宽是明确代价**，两种宽度的可用样本数分别为1995和2000。

### 工程与独立审计证据

- 冻结普通 clone `validation-joint-002/source`：quality189 / tools609 / unit2861 / integration39 / gui711 / spawn4 / regression70，共 **4483 passed**；完整 Radon **611 files、0 issues、PASS**。本轮逐原始 receipt/log 和709源文件重新核验，不重复已完成门禁；仅有既有小画布 warning。
- 开发 seed37 的 `development-joint-002` 完成真实 public fit/save/load、200/200 bootstrap、ranks(2,199)。diagnostic 为 `not_triggered`，虽配置 B999，但该例未生成 diagnostic null；开发病例不计独立验收。001 的输入路径工具误拒原件保留。
- `scientific-audit-002.json` **PASS**，实际 session31061 exit0、stderr空：2000病例全部完成保存均值的观测重放、公共 API 回读、样本矩阵/三轴秩区间/声明/owner/content 检查，无 unverifiable partial，末态树/source/tool/runtime 复核通过。审计独立 CP 为 .9706866253773252，与 producer .9706866253773265 相差约1.33e-15，其他汇总字段完全一致。
- 审计器001误混 science ASCII+LF 与 product/auditor UTF8无LF，真实审计在 runtime hash 处失败；002 显式分域，不 try-both/fallback，原89测试172断言保留，真实 RED→最终105 toy GREEN，并经独立 Spec/Quality PASS 后新授权。旧审计 FAIL 不改写，2000病例与 summary 未重跑。

证据根 J：`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/joint-bootstrap-20260914`。

```text
joint-bootstrap-outcome-001.json
  35d54da704b9b02ae506f5c7c591de46e79f6ec16ae2b57fdc15273f57e23856
scientific-audit-002.json
  01f1ab1c5bf94ee2c775bd2be8796b9a3872bd1a4c3d381f091b1c0b72d12249
audit-execution-completed-002.json
  ca6225d666759b4ed4defef9f1e274db1f6eea3410685d69c4565e4b9c0d951e
validation-joint-002/snapshot.json
  f49762be2a6069ceaa99a7d72687e5b92330aed49bf1f6fdf651d2bd62638f8f
engineering-recheck-closeout-001.json
  f4bdcac19160a7b94c287956cd9a440236819b0755a0af779f20e13f495fda7e
```

### 结论边界与保留

同日 provenance/coverage 计划及 RMS 父计划的**已批准场景**覆盖目标据此关闭；不推论任意模型、边界参数、全部 nuisance 同时覆盖或真实测量的普适95%保证。独立审计重放的是保存 raw_mean，并未独立重算 forward physics。原81%、旧4/8 FAIL/原件丢失事故、旧joint93.4%及所有失败记录保留，不追认丢失证据已恢复。

收尾只同步7份计划/验收文档与 ignored 执行账；`J/status-joint-001/` 保存原件、差异及验证，文档更新后的整树不冒称原冻结快照。没有工作树临时探针需删除；外置输入、失败、日志、缓存和验证 clone 保留。保留分支及全部未提交改动，不 commit/merge/push/release。下方“当前/未关闭”等表述均为相应历史快照，不是重启旧任务的指令。

## 历史验收：2026-09-14 Poisson RMS v4（当时状态）

**低计数诊断修复与八项预登记科学端点已通过完整独立核验；整体 95% 区间校准仍未关闭。** 原 200 种子开发重放的全 seed 覆盖从历史 **162/200=81%** 改善为 **188/200=94%**，新单曲线 holdout 为 **1908/2000=95.4%**。剩余 joint bootstrap **93.4%** 单列，不能以 diagnostics PASS 掩盖。

原缺陷是把低计数 Poisson 残差的正常非交换性当成趋势/白噪声失配：原 200 例虽均有效拟合，却有 32 个正式区间被启发式诊断阻断。已按同噪声模型、同一有界重拟合映射校准，而不是删样本、放宽阈值或隐藏 advisory；后续数值精度/两阶段 refitter 修复及全行 RMS 标准化分别解决了重拟合失败与 detector 尺度竞争。RMS 保留连续间距，实际得分用 `relative/relative_rms`，不从舍入后的 scale 反推 ties。

当前唯一身份：schema `4`、algorithm `xrr-fit-v2-poisson-4`、objective 字符串 `"2"`、diagnostic `poisson-refit-null-v4`、method `poisson_refit_null_rms_v4`；refit policy 保持 `declared_sobol4_lbfgsb_trf_v3`。B999、alpha=.01、四起点、每路径预算80、MATCH_LIMITS 和全路径原生成功条件未放宽；无旧格式兼容或新增依赖。

### 固定科学结果

新 holdout 4400 例与原 200 种子开发重放分别报告；8 shards ×575 一次运行、全部封存，0 incomplete、0 fit_failed。原 200 不计作新独立样本。

| 分组 | 固定分母 | 诊断拒绝 / 不可用 | 正式区间 | 覆盖 | 全 seed 覆盖 | 条件覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 新 single，profile | 2000 | 8 / 0 | 1992 | 1908 | **95.4%** | 95.7831% |
| 新 joint，bootstrap | 2000 | 10 / 0 | 1990 | 1868 | **93.4%** | 93.8693% |
| 原低计数种子重放，profile | 200 | 2 / 0 | 198 | 188 | **94%** | 94.9495% |

八个预登记端点全部 PASS（固定 gamma=.00625）：single/joint 的 F=R+U 上界约 .00907924/.01047128，均 ≤.02；两个 U=0 上界约 .00253437，均 ≤.01；background、footprint、surface、ACF 指定目标各 100/100，功效下界约 .95051462 ≥.8。四阳性均拒绝100/不可用0；不把 omnibus 拒绝替代指定目标命中。joint 未提供 profile，不能捏造其 profile 覆盖。

### 工程、开发与独立审计

- 冻结普通 clone `validation-rms-002/source`：quality189 / tools609 / unit2798 / integration39 / gui711 / spawn4 / regression70，共 **4420 passed**；完整 Radon **602 files、0 issues、PASS**。GUI 保留一条既有 `constrained_layout` warning。
- 五例公共 API 开发重放均完成实际 B999 与严格保存/加载，每例999/999 null、4000原生成功路径；119/148 family p=.124/.020，三个注入目标 p=.001。绑定审计衔接独立复算的矩阵和20000条路径，不增加独立样本量。
- `rms-scientific-audit-full-003.json` **audit PASS / 预登记科学端点 PASS**：4600 cases、6600 observation streams、30327 files、697 source files，逐例证据缺口0，末尾全局 hash 复核通过；实际 session33083 exit0。观测 RNG 只重放已封存 raw_mean，没有重新拟合、生成 diagnostic null/bootstrap 或补样；没有独立重算 forward physics。
- audit001 的 `/dev/null` 误拦与 audit002 的 joint owner 规则误套均保留真实 FAIL。两个外部审计器窄修复有真实 RED→GREEN、独立限定复审和新 completion/hash 绑定，不改科学原件，不将前两次失败改成通过。

证据根：`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913/family-calibration-001/rms-implementation-001`。

```text
rms-r4-outcome-001.json
  a0543cc4ebc99b4e775a577639d58a12e92fd041892e958118177cf290b3cccc
rms-scientific-audit-full-003.json
  19a5d5b7d6d52d0eaf266f669d136caef3b960e2634dddc610789f8463bb551a
validation-rms-002/snapshot.json
  78d681b6bba4eef2afe5c73d2331728d0fc9ac448f5242c4c5c4264bbfa697e0
runtime source identity
  7702a73a4013982919c7db21e12f39d068b14168ed663dbdd7a8a42e6381c42c
```

### 尚未关闭的限制

1. **Joint bootstrap 的一般 95% 校准未通过。** 当前 B200/type7 与 RMS 前相同，本批每例200/200成功；10例恰因诊断拒绝而无正式资格。仅对已保存样本改 type6 的事后反事实，保持 gate 是1890/2000=94.5%；1899/2000=94.95% 还包含探索性区间，不是正式覆盖。该机械敏感性不是唯一根因证明或新验收；不能将反事实替换原93.4%，也不能把 Uniform 理想公式当作 plug-in bootstrap 普适定理。方法修订需新设计及独立验证，不能在已暴露 holdout 上调法凑95%。详见 `rms-joint-coverage-interpretation-review-001.md` / `002.md`。
2. **Joint bootstrap 没有独立 context/content provenance seal。** 当前通过外层共享 winner report 关联；审计新增全部 winner 和完整 report 一致性检查，并保留工件 seals，但不冒称已验证不存在的独立 seal。增强该保证是另一个产品契约任务，不回填本轮原件。详见 `rms-auditor-bootstrap-owner-contract-review-001.md`、`rms-auditor-bootstrap-owner-review-003.md`。
3. 本次不构成真实用户测量数据、B99 或更大联合族的一般统计验收；没有重跑旧 holdout 或220-case corpus。原81%、旧4/8 FAIL与原件丢失事故全部保留。

当前计划 `docs/superpowers/plans/2026-09-13-poisson-rms-calibration.md` 因此保留总体覆盖关闭项未勾选，汇总显式 `overall_coverage_goal_closed=false`。冻结后只更新工作树状态文档，`status-rms-r4-001/` 保存前后备份/diff/身份核对；不把当前整树冒称原冻结快照。没有工作树临时探针需清理，外置原件、失败、日志、缓存和克隆作为证据保留；保留分支与所有未提交改动，不 commit/merge/push/release。

## 历史：2026-09-13 RMS 批准前的待修状态

**原 81% 低计数整体修复：尚未完成科学验收。** 两阶段软件工程通过不能代替该目标。当前 v3 新复现 ACF 目标漏检（B999 全成功，目标 p=.03>.01）；统计规则修订与完整覆盖率/误拒/不可用/功效验收继续由 `docs/superpowers/plans/2026-09-12-poisson-diagnostic-calibration.md` 的总体计划推进。

当前候选为全行 RMS 连续同族校准，设计待确认；此前 minP 推荐因实际支持的联合规模/低预算分辨率风险已撤回。范围反例和修订提案见外部 `family-calibration-001/family-scope-003.json`、`design-proposal-rms-002.md`；它们不是新产品实现或科学 PASS。

## 历史验收：2026-09-13 Poisson 两阶段 v3（数值工程子计划已闭环）

`docs/superpowers/plans/2026-09-13-poisson-two-stage-refit.md` 的 T1–T4 已完成：固定每起点 L-BFGS-B → 完整计费交接 → TRF，保留起点、预算、真实成功门槛及统计量；补齐不可变 work、严格 codec 和成功 unit/work 维度校验。
当前身份为 schema `4`、algorithm `xrr-fit-v2-poisson-3`、objective 字符串 `"2"`、diagnostic `poisson-refit-null-v3`；没有旧格式转换或新增依赖。
同一普通 clone `validation-two-stage-003/source` 的七个正式 MODE **4340 passed、全部 exit 0**，完整 Radon **597 文件、0 issues、PASS**；保留 1 条既有 GUI 小画布 warning。四组固定开发 B999 均 available、999/999 null，最大单路径 nfev 为 48/46/42/41，预算仍为 80；119/148 的 family p 为 .573/.544，footprint37/surface37 的目标 adjusted p 均为 .001。
R6 已取得 RED→GREEN 及独立关闭；最终运行绑定审查 Approved、0 新发现。工程汇总是 `/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913/two-stage-001/two-stage-engineering-final.json`；完整路径、hash、审查链与 seed 勘误见 `docs/superpowers/plans/2026-09-12-poisson-diagnostic-calibration-progress.md`。
验证后只更新状态文档，差异另记 `status-update-two-stage-003/`，不冒称文档更新后的整树仍是冻结快照。所有测试缓存、原始输入、日志和失败记录外置保留；未删除旧证据，不提交/合并/发布。
**这里只验收数值工程修复，不是 Poisson 总体科学 PASS。** ACF 尺度竞争未修复，v3 新种子覆盖率/FWER/power 未验；旧 81% 及 `4400 + 200` 的 `4/8` 端点 FAIL 保留。本轮没有重跑旧 holdout 或 220-case corpus，下一科学协议仍需另行批准。以下“当前／进行中”属于各历史快照。

## 历史：2026-09-13 Poisson 精度 v2（精度子计划已闭环）

相对总成本 `ftol` 过早停止的窄修复已实施；低计数 GN 慢收敛与 ACF 尺度竞争尚未修完，不能宣称整体修复或科学验收通过。
当前唯一身份为 schema `4`、algorithm `xrr-fit-v2-poisson-2`、objective 字符串 `"2"`、diagnostic `poisson-refit-null-v2`；无旧格式转换。
新计划为 `docs/superpowers/plans/2026-09-13-poisson-local-precision.md`，完整最新记录见 `docs/superpowers/plans/2026-09-12-poisson-diagnostic-calibration-progress.md`。
持久证据根为 `/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913`；runtime SHA256 `9e3cbbef49d6105c1924cd7ffb437dd35a03ef66205a96906bf019163156fecb`。
seed37 footprint/surface 的当前 API 开发重放均完成 `999/999`，目标 `p=.001`；119/148 仍 `diagnostic_null_refit_failed`，不是新的覆盖率或独立验收。
`validation-precision-003` 在同一个 678-file 快照上 fresh quality/tools/unit/integration/gui/spawn/regression 全部通过，共 **4189 passed**，完整 Radon PASS；两份独立窄审查 Approved、无未解决项。001 的 unit FAIL、002 的 tools FAIL 均保留，测试期望修正有独立推导及 RED→GREEN。最终汇总为 `precision-engineering-final.json`；没有重跑 220-case corpus 或新 Poisson holdout。
旧 `4400 + 200` 实验的 `4/8` 端点 FAIL 保留；旧协议主动拒绝新运行时，未注册新 holdout，未重用已暴露 seeds 或旧审查。以下原“当前／进行中”表述均为各自时间点的历史记录。

## 原 Task 1–8 验收状态（历史）

本计划约定的软件与合成实验验收已完成。Task 1–7 已有实施与验证证据；Task 7 的入口、GUI 多参数布局/证据入口、
联合导出参数轴映射均通过独立审查。Task 8 真实 pilot 暴露的重复起点、联合排名验证、
全锁定零维分类及联合 bootstrap 失败后的置信门禁均已补回归。修复后 4×200 固定种子
覆盖率、五次重复 micro/automatic 性能和补充集成测试已完成；全分支静态 Spec/Quality
审查无新阻塞项。上一轮 statistical 暴露 Stage A/B 丢失声明起点的连续回归。
修复后 `double-12019` 两次完整重放通过，但下一轮 statistical 在
`oxide-cap-14017` 的粗糙度支持区间断言失败。2026-09-12 已将 profile basin
恢复接入既有解析残差/Jacobian：原病例两次完整重放通过且结果字节一致，扩大回归
347 项通过，独立窄复审通过。新快照的科学实验及独立补核已完成；官方八门禁
共 3699 passed，其中 statistical 完整 220-case corpus 通过。运行后源码/索引与
临时产物清理已核对，最终独立整合复审 Spec / Quality PASS，Task 8 第 4、5 项已关闭。
这不等于通用 95% 区间校准或真实用户测量数据验收；下文统计限制继续保留。

工作树为 `/Users/dala/Desktop/XRR-Fitter/fit-algorithm-v2`，分支
`feat/fitting-algorithm-v2`，基准 HEAD 为 `1a8c71d1c5e7fc9e3e2435914195f7b1c856bec4`。
Task 6/7/8 改动未提交；验收须绑定完整文件快照，不能仅用 HEAD 代表当前源码。
当前非文档源码 SHA 为
`91d2a711cfebf479a2061401ab354d64664529aa5de18cfca66a07c744c66e1b`；
生产源码集合 SHA 为
`592fa0600aefe3a0b1edb7fd34dc14090e2c861ecc4dc845bdbc5956187f87fc`。
旧报告根为
`/var/folders/4b/3j6m_kld2klgm7dxq___yd2m0000gn/T/xrr-v2-resume-final-m12xh1q8/`，
上一轮科学证据与失败门禁保留于其 `final-verification/`、`final-verification-ab/` 和
`final-verification-ab-final/`。当前修复、重放和新验收证据根为
`/tmp/xrr-v2-profile-resume-qms2ukkz/`；科学实验与官方 clone 位于其中
`final-verification/`。不会将修复前快照的通过记录直接视为本轮通过。
不自动 commit、merge、push、tag 或 release。

## 范围与证据入口

| 项目 | 实现与对应验证 |
| --- | --- |
| C1：联合成员残差诊断 | 每成员保存真实 residual evidence；未执行与 False 分开；系统误差影响可信分类 |
| C2：协方差尺度 | Gaussian 总信息、Poisson 期望信息、robust sandwich；SVD 秩与不可用原因 |
| C3：搜索散布不冒充误差 | `search_parameter_spread` 与统计 covariance/sigma 分开；联合推断使用全局坐标 |
| C4：约束尺度求导 | 完整物理/共享/多级约束 Jacobian 进入尺度 prior，中心差分回归 |
| C5：拟合与 MCMC 目标 | 同一总量 Q、展示量 J=Q/N；无额外参数先验时 log probability=-Q/2 |
| C6：粗网格先验 | 冻结完整 N、平台 prior 和区域语义，仅积分采样质量随网格变化 |
| O1：拟合点内环 | 仅 fit_mask 行进入物理与导数内环，发布保留完整曲线；点数与数值等价检查 |
| O2：联合系统缓存 | 同一点 residual/Jacobian 共用系统，线程隔离、取消边界与数组所有权回归 |
| O3：显式测量噪声 | robust/Gaussian/Poisson 同一目标边界，负观测/零 counts/非法输入及入口传递 |
| O4：区间口径 | profile_steps 独立；保存 interval kind/confidence/method/reason；正式 bootstrap 至少 200 成功样本 |
| O5：网格与预算 | 128/256/512/full 渐进网格、逐条纹采样与完整复核、确定性有界重启及恢复证据 |

逐任务 RED→GREEN、精确接口和已执行命令记录在
`docs/superpowers/plans/2026-09-08-fitting-algorithm-v2-progress.md`。
设计为 `docs/superpowers/specs/2026-09-08-fitting-algorithm-v2-design.md`，
实施计划为 `docs/superpowers/plans/2026-09-08-fitting-algorithm-v2.md`。

## 验收要求

- 普通 Git clone 的完整未提交快照及 SHA-256。
- 已登记 quality/tools/unit/integration/gui/spawn/regression/statistical 门禁和完整 Radon。
  注册表没有 `r22-reference` mode；物理参照使用已有 regression，不新增 CI mode。
- 固定 seeds 0..199 的 Gaussian、低/常规 counts Poisson 和共享参数分组实验；保留每例
  estimate、interval、覆盖、失败与 unavailable，分别报告可用区间条件覆盖和完整种子产出。
- 名义 95% 的经验覆盖率和双侧精确二项检验；robust moving-block bootstrap 代表实验单列，
  不将 loss_support 纳入正式 95% 统计，也不以单个代表实验宣称覆盖率已校准。
- 无并行验收负载时至少 5 次墙钟中位数、实际物理工作点数/系统次数、阶段 nfev，伴随数值一致性证据。
- 独立审查、临时产物清理记录和尚未解决的风险。

## 已执行的真实 pilot

固定原 seed 0，未替换数据、预算或失败记录。四个正式分组的单个 pilot 均有可用区间；
shared Gaussian 与 robust 代表实验各完成 200/200 次 bootstrap，无失败。
这只验证工具链及证据保存，不代表 200 个独立数据集的经验覆盖率。
报告 `/tmp/xrr-v2-task8-t718qpds/coverage-pilot-after-core.json`，
`protocol_complete=false`、原因为 `pilot_only`。

核心回归证据位于 `/tmp/xrr-v2-task8-t718qpds/core-runtime-fixes/`：原 Poisson seed-0、
60/30 点三模式联合、400/200 点真实 active-prior 联合和原始全锁定输入均做无插桩
fit/save/load 重放，成对完整保存 JSON 精确一致。原始失败输入、日志和后续成功证据
分开保留；不会以改变 fixture 或去掉验证来制造成功。

## 固定种子覆盖率复验

当前报告为 `/tmp/xrr-v2-profile-resume-qms2ukkz/final-verification/coverage.json`，
四组各包含完整 seeds 0..199，`protocol_complete=true`，耗时 193.19 s。
800 次正式分组 public fit 均有有效结果；1000 份成员观测的固定 RNG、原始文件字节、
normalization、锁定参数及所有分母/二项检验经独立补核通过。
报告、审计与复审入口分别为同目录 `coverage-audit.json`、`scientific-review.md`。

| 分组 | 可用区间 | 覆盖真值 | 条件覆盖 | 全 seed 覆盖 | 条件二项 p | 全 seed 二项 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 198/200 | 191 | 96.46% | 95.5% | 0.41647 | 0.87164 |
| Poisson low | 168/200 | 162 | 96.43% | 81.0% | 0.48142 | 1.40209e-12 |
| Poisson regular | 200/200 | 191 | 95.5% | 95.5% | 0.87164 | 0.87164 |
| Shared Gaussian | 200/200 | 184 | 92.0% | 92.0% | 0.07080 | 0.07080 |

Gaussian 2 例和低 counts 32 例因 `residual_diagnostics_failed` 没有正式区间，保留
`covered=null`，没有删去失败或 unavailable 病例。低 counts 全 seed 二项检验拒绝
名义 95%；条件覆盖不能替代完整产出率。其余未拒绝也不证明一般校准。

独立复审从保存轨迹重建 566 个可用 LR 区间、从样本重建 200 个共享 bootstrap 区间，
并以 65 位 Decimal 二项 PMF 枚举补核两分母 p 值。另一个 robust 代表实验完成
200/200 moving-block bootstrap、零失败，区间为
`[99.7779655572968, 100.16343307360218] Å`；不把单例称为覆盖率校准。
这些实验仅对应其他参数已知的合成单层膜厚度，不代表真实用户数据。

运行前后 HEAD、源码和索引不变，绑定本文当前源码 SHA。与旧报告根
`final-verification-ab/coverage.json` 相比，800 例完整 interval、全部非时间字段及
1000 份 observations 精确一致；此前 18 个区间详情变化是更早轮的历史比较，
不是本轮差异。历史报告仍保留，完整软件验收仍以当前全部官方门禁为准。

## 性能复验

Apple M4、10 logical CPUs、macOS arm64。每项取五次墙钟中位数；原始阶段回执证明
micro → automatic → coverage → official 顺序执行，未重叠本任务验收负载。
硬件在审计时采集，没有监测整机外部负载，不能宣称运行期间整机空闲。
micro baseline 为 `691f609`，对比本文当前冻结源码；1200 个源点中仅 120 个拟合点，
包含混合波长和逐点角度分辨率。

| micro 工作量 | baseline / s | 当前 / s | 倍率 |
| --- | ---: | ---: | ---: |
| single_residual | 0.12000 | 0.01605 | 7.48× |
| single_system | 0.18483 | 0.02276 | 8.12× |
| joint_callback_pair | 0.61518 | 0.04905 | 12.54× |
| joint_local_fit | 0.54129 | 0.08855 | 6.11× |

保存的 residual/Jacobian 数组及五份 local fit 记录前后精确一致；局部拟合均为
nfev=8，五次工作计数也各自一致。每次重复的累计 primal/tangent 点调用量分别由
1,224,000 / 1,200,000 / 4,848,000 / 4,368,000 降至
122,400 / 120,000 / 240,000 / 681,600。前三项每次包含 10 个探针，local 每次为
1 次拟合；这些是累计点调用量，不是独立观测数。joint callback 从 10 次 residual +
10 次 Jacobian 变为 10 次共享 system。夹具 scale 解位于边界，不把 micro 倍率
推广为一般拟合或整体应用加速保证。

| 完整 automatic 场景 | 中位秒数 | 总 nfev | profile 数 | 结果 |
| --- | ---: | ---: | ---: | --- |
| direct-sld | 0.6532 | 1028 | 0 | 全部 passed |
| shared-local | 4.3108 | 3597 | 0 | 全部 passed |
| isolated-outlier | 3.4838 | 3916 | 2 | P4 review，其余 passed |
| roughness-release | 5.3956 | 4622 | 0 | 全部 passed |

四类各五次，全部 bootstrap_count=0。所有非时间证据逐次精确一致，且与上一 A+B 轮
非时间字段一致；异常样本保留预期复核状态。完整 stage nfev、profile 数及
recovery_error 保存在当前证据根 `final-verification/automatic.json` 和
`automatic-audit.json`；micro 原始次数和数值在 `performance-micro/`，运行区间与
负载声明在 `performance-context.json`，独立补核记录在 `scientific-review.md`。

## 原病例修复验证

失败链为 Stage A 新复核集提前丢掉声明起点，继而 Stage B 把其作为审计记录保留却
按 ID 排除于 C/D。现在声明起点参加每级真实完整复核，并进入既有几何代表与归档
规则；重复几何可合并，无效或高成本候选仍归档，不改物理、seed、置信阈值或预算。

原 `double-12019` 两次无插桩重放分别耗时 107.92/106.95 s，原始字节与 Task 5
对照一致，输出与恢复指标原断言通过。声明支路 B→C→D 与 Task 5 精确一致；四个
不同最终 seed 的目标均约 `2.521114435166e-06`，分类为“可用但相关”，保留边界、
系统残差和协方差不可用证据。8/8 bootstrap 成功仍标探索性，未伪装正式 95% 区间。
完整保存 JSON 逐字节一致、strict codec 往返通过；报告在旧报告根 `statistical-failure/`，
独立运行证据复核在其 `statistical-failure-ab-runtime-review.json`。本轮又以当前源码
重放并核对完整结果字节一致，见当前证据根 `double-regression/` 与 `replay-audit.json`。

## 上一轮官方失败证据

普通临时 clone 的完整文件快照经逐文件 SHA 验证，所有命令均为既有
`python tools/verify.py MODE --report-dir ...`，启用原 outcome gate，没有 skip/deselect。
本节为历史记录，报告位于旧报告根 `final-verification/official/`，不是当前验收目录。

| Mode | 结果 |
| --- | ---: |
| quality | 189 passed |
| tools | 512 passed |
| unit | 2175 passed |
| integration | 39 passed |
| gui | 704 passed |
| spawn | 4 passed |
| regression | 50 passed |
| statistical | 1 passed / 1 failed；7150.46 s |

quality 包含全仓 Radon：557 文件 PASS，平均 CC 3.1721，零违规；GUI 仅有既有隐藏
画布 `constrained_layout` warning。integration 39 项包含该历史轮补齐注册的联合导出和
全锁定模型测试。statistical 的语料定义检查通过；实际拟合检查在 `double-12019`
的“不可信”分类断言失败，原始日志和失败快照完整保留。该病例后续已由 A/B 修复及
完整重放闭合；历史失败记录不改写，整体验收仍须当前完整八门禁通过。
该历史普通临时 clone 已按 runner 的 finally 清理。

第一次 A/B 修复快照的 quality 189、tools 512、unit 2190 通过；integration 为
38 passed / 1 failed。失败是合成观测生成器固定查找已被正常合并的 `B-declared-start`。
仅将 design 参数全部锁在声明初值，再消费真实公开 API 返回的前向曲线；80 点预测、
三份原始 xy 字节及实际自动拟合参数设置均与旧源码快照精确一致。原3项集成测试通过，
物理输入、随机 seeds、自动联合拟合及导出断言未变。

## 上一轮官方整合门禁（profile 修复前）

证据目录为旧报告根 `final-verification-ab-final/official/`，使用独立普通 clone、原命令和完整模式。
已完成七项合计 **3688 passed**；statistical 返回 **1 passed / 1 failed**，并非八门禁通过。

| Mode | 结果 |
| --- | ---: |
| quality | 189 passed |
| tools | 512 passed |
| unit | 2190 passed |
| integration | 39 passed |
| gui | 704 passed；1 条既有布局 warning |
| spawn | 4 passed |
| regression | 50 passed |
| statistical | 1 passed / 1 failed；7868.12 s；oxide-cap-14017 |

该历史轮 quality 的完整 Radon 为 559 文件 PASS，平均 CC 3.1733，零违规。
生成夹具独立窄复审 PASS，记录为 `sigma-generator-review.md` / `.json`；12 份科学
报告 hash 与 `rebind-evidence.json` 一致，两个冻结快照的生产源码完全相同。
这些历史报告不能代替当前 profile 修复后的整合门禁。

## 2026-09-12 profile basin 恢复修复

旧恢复路径对真实模型仍使用有限差分标量求解，冻结失败中心的 16 次优化中有 12 次
`ABNORMAL`，未发现 basin。相同网格改用现有解析系统后得到更优目标 1.430287814，
真实四路 continuation 将最终最优目标由 3.893017607 降为 3.05284e-26。

- RED：5 failed / 4 passed；GREEN：45 passed；扩大回归：347 passed。
- 原始 xy、全部配置、seeds、A–D 候选及摘要未变；四路 E 均保留历史工作并追加实际预算。
- 两次完整重放 72.93 / 72.02 s，结果 SHA 为
  `38e2d364c6a482d7bfe02375df588fd7ef2b7dc984663c9238b07125d04ecac6`；
  strict codec 和原恢复断言通过。分类保留“可用但相关”，不强行提升为“可信”。
- `double-12019` 新重放与上一轮完整结果逐字节一致。
- 全仓 Ruff lint/format、完整 Radon（560 文件、零违规）通过；继承主模型与推理强度的
  独立审查未发现新增阻断，并另跑 9 项 solver/约束测试和 1 项真实冻结中心回归。
- 精确输入、源码、工作量及重放审计见新证据根的 `replay-audit.json`；
  本轮只改 1 个生产文件及 3 个测试文件，未改物理定义、预算配置、网格或验收阈值。
- 新快照的 micro/automatic 性能、4×200 固定种子覆盖率及独立科学补核已通过；
  当前官方门禁状态见下一节，不以单病例成功替代完整语料。

## Poisson 诊断修复前的普通 clone 官方门禁

当前目录为 `/tmp/xrr-v2-profile-resume-qms2ukkz/final-verification/official/`，
完整 tracked/untracked 快照 647 文件逐一核 SHA 后在普通 clone 执行原注册命令；
原 outcome gate、物理参照与验收阈值均未改变。

| Mode | 当前结果 |
| --- | ---: |
| quality | 189 passed |
| tools | 512 passed |
| unit | 2198 passed |
| integration | 39 passed |
| gui | 704 passed；1 条既有布局 warning |
| spawn | 4 passed |
| regression | 51 passed |
| statistical | 2 passed；完整 220-case corpus；8382.91 s |

八门禁合计 3699 passed，完整 Radon 为 560 文件 PASS、平均 CC 3.17184、零违规。
statistical 的两项原测试验证完整且确定的 220-case 定义、实际 220 次拟合、所有原
恢复/降级阈值及空失败集合；没有用单病例或子集替代。次数由未改的真实调用链与
测试完成状态确认，不声称存在额外逐例计数插桩。GUI 的 warning 来自隐藏画布
`constrained_layout`，所有模式没有 skip/deselect，原 outcome gate 保留。

整个官方阶段耗时 8911.32 s，运行前后 HEAD、非文档源码、生产源码与索引全部一致。
普通 clone 在 runner 的 finally 清理；七个 pytest-tmp 和十六个 mpl/xdg 缓存目录
已按归属删除，statistical 未产生 pytest-tmp。另一个 micro baseline clone 已自动清理，
只读审查遗留的空目录已移除。失败输入、报告、日志、源码差分及 `.superpowers` 保留。
回执为 `official/cleanup.json`、`completed-scratch-cleanup.json`、`auxiliary-cleanup.json`。
Task 8 第 4、5 项已关闭。最终独立 Spec / Quality 复核通过 108 项只读检查，
原整分支 85 个生产变更路径的审查覆盖链完整；原整分支审查、本轮窄修复复审与本次
运行证据补核各自独立说明，没有把旧静态 PASS 当作新验收结果。
独立报告为当前证据根 `whole-branch-final-review.md` / `.json`。

最终源码/索引、12 份科学证据和三份文档的绑定在
`final-verification/rebind-evidence.json`；完整门禁、Radon、逐文档 hash 和实际缓存
不存在状态的最终审计回执为 `final-verification/final-audit.json`。
独立封存窄核记录为 `final-verification/final-seal-review.json`。只清理本轮有归属的
临时产物，历史失败与全部审计材料继续保留；分支和所有未提交改动保留，不自动提交、
合并、推送、打 tag 或发布。

## 计划清单同步补记（2026-09-12）

Task 1–8 的进度账、计划勾选和 `.superpowers/sdd/progress.md` 恢复入口已统一。
此补记只修正 25 个遗漏勾选及过期的 Task 1 进行中状态；不新增算法实现或统计校准。
早期批次自审与最终独立整合复核分别记录，Task 6–8 仍未提交。

以上封存报告保持原样。同步后的文档快照与原验收源码的绑定，以及本次窄回归记录，
见 `/tmp/xrr-v2-plan-closeout-z49kl5k9/verification.json`。
不能将文档同步说成重新运行了八门禁；Poisson low 81% 和真实测量数据未验限制不变。
本轮针对原计划关键节点的 10 个测试文件另获 206 passed（13.67 s）；
这是源码未变条件下的窄回归，不替代上文已有的完整验收记录。

## 2026-09-12 Poisson 诊断校准修复（统计进行中）

原低计数结果仍是 200 个 seed 中 168 个正式区间、162 个覆盖：全 seed 产出 81%，条件覆盖 96.43%。该旧结果不能当作新代码的验证结果。

根因修复采用固定声明起点的 Poisson refit-null 校准；raw 提示保留，正式推断消费校准后的 effective 诊断。B=999、alpha=.01、原阈值和物理模型未调整；当前 schema=4、algorithm=`xrr-fit-v2-poisson-1`，不提供旧格式兼容。

本轮证据根：`/tmp/xrr-v2-poisson-fix-fww0fik3/implementation`。

| 检查 | 本轮已完成结果 | 实际运行归属 |
| --- | --- | --- |
| quality / tools | 189 / 586 passed | `validation-003` |
| unit / integration | 2590 / 39 passed | `validation-002` |
| GUI / spawn / regression | 711 / 4 / 51 passed；GUI 1 条既有布局 warning | `validation-002` |
| 完整 Radon | 586 文件，PASS，0 issues | `validation-003` |
| 整体实现定向复审 | Approved，0 Critical / 0 Important / 0 Minor | `P6-whole-change-review-final.md` |
| 完整 220-case corpus | 2 passed，10469.32 s | `validation-001` |

三快照之间的适用性由 `P6-final-verification-report.md` 与 `validation-003/corpus-import-audit.json` 单独解释：corpus 的 153 文件依赖闭包中，150 文件逐字节相同，余下 3 文件仅五个常量的等价改名。该说明不是一次新运行，也不把 `validation-001/002` 冒充 `validation-003`。八类 MODE 汇总 4172 passes，逐项日志与运行归属封存在 `P6-engineering-gates.json`。早期失败 MODE 和已停止的重复 corpus 不计入通过结果；220-case corpus 也不是新的 4400-case Poisson 校准验收。

正式统计登记已冻结（`poisson-acceptance-final/preregistration.json`）：两个 null 各 2000 个新 seed，四个阳性各 100 个新 seed，另以当前 API 重建原 0..199。manifest SHA256 为 `90c26e10c4115762c2ff21503c50aee0f7aab405a52c8e9254b3638188598d95`，生产/工具运行身份为 `7685c53a0c700fa15b8b4765f54837d5a3cd4318b7336d59984229cb293058c0`。

原 200 重放已全部完成，无替换或重试：197 个正式区间、187 个覆盖，全 seed 覆盖 **93.5%**、区间可用率 **98.5%**、条件覆盖 **187/197 = 94.9239%**。raw 提示仍为 32 个，其中 29 个校准后未拒绝、1 个校准拒绝、2 个 null refit 失败而 unavailable；产品拟合失败为 0。全部 3 个不可发布病例仍计入分母，不把 unavailable 静默删除。`poisson-replay-operation-001/replay-summary.json` 是该完整重放的不可变汇总；独立审计 `P5-replay-independent-review.md` 为 Approved、0/0/0，并从 200 份实际保存的 fitted JSON 重算全部区间，边界差值为 0。119/148 因 null refit 超过声明 nfev 而不可用，142 的校准 p=.006，保留拒绝。

4,400 个新 seed 使用固定方案运行：前 100 个 N1 病例以最多 2 个工作进程完成，完整 corpus 通过后切换到 `poisson-full-operation-001` 的 8-worker reviewed driver；已封存病例只验证并跳过，不重做，不改变预算、seeds、阈值或总数。

**N1 完整结果：不可用率端点未通过。** 单成员 null 的 2000 个 seed 均完成：R=3、U=18、F=21，产品拟合失败为 0；18 个不可用均为 `diagnostic_null_refit_failed`。F 的预登记单侧 CP 上界为 0.01765747，满足 <=.02；U 上界为 0.01575269，不满足 <=.01（固定 N=2000 时须 U<=9）。不能因点估计 18/2000=.9% 小于 1% 就宣称通过。正式 profile 为 1979 个，覆盖 1884 个：全 seed 94.2%，条件覆盖 1884/1979=95.1996%；这两项描述性覆盖不能抵消 U 门槛失败。完整汇总为 `poisson-full-operation-001/summary-null_single.json`，SHA256 `934e762f419c956ba4617f28c18afca76047b1f1b25d30342f620496460cddc0`；独立 N1 审计进行中。

N2 与四组阳性仍按原协议继续。总体汇总暂为 **UNVERIFIED**（CLI exit 2，协议未完整），但已有必需端点失败，当前版本不能声明科学验收 PASS。不早停，不调宽阈值，不修改冻结源码；数值根因只用原开发重放样本在独立进程调查。后续算法修订必须另行冻结、新预登记并使用未暴露 seed，不能将本次 holdout 用于追调后再称独立验证。93.5% 仅为原 200 个重放 seed 的实际结果，不是一般 95% 校准保证。

最终工程证据独立审计 `P6-engineering-final-review.md` / `.json` 为 Approved、0/0/0（JSON SHA256 `1fd68cf65c2dc57f315f1dbafccefb147e4c197bb494c786b0a1d659dafdf023`）；只关闭工程门禁，不改变上述统计失败。全部原失败证据和既有缓存保留，未提交或发布；原生 Windows 与真实测量数据仍未验证。

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

## 不包含的验收

没有收到并运行真实用户测量数据；合成实验不能替代真实数据验收。
不改变物理内核或参照容差，不新增生产依赖、默认部署方式或 CI mode。
性能结果只适用于记录的工作量与硬件条件，不承诺通用加速倍率。
