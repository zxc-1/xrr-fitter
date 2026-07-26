# XRR 真实数据验收记录模板

> 本记录用于一个经 domain owner 批准的真实 dataset。曲线视觉重合本身不得作为
> pass；必须同时证明参数与 SLD/profile 物理合理、残差证据可接受、候选行为可复现、
> confidence 分类诚实，并取得 domain owner 批准。

## 1. 输入与 provenance

- 验收日期：
- 记录人 / domain owner：
- 数据类别：已知单层 / 当前可工作的多层膜 / 当前失败或不稳定的多层膜
- 只读 source path：
- immutable copy path（与运行输出分离）：
- 文件大小：
- SHA-256：
- 采集 provenance（仪器、采集日期、样品/批次、提供者、批准用途）：
- 输入角度约定：`two_theta_deg`
- `import_angle_offset_deg`：
- 列映射（two-theta/intensity/sigma/resolution 及 resolution kind）：
- 光路：单色 / 混合 Kα
- `λ` 或 `λ1/λ2/r21`：
- `InstrumentSpec` / `instrument_id`：
- footprint mode：`geometry|fit|none`
- 样品长度、束宽、`θ_fp`：
- background model / 固定值或参数化配置：
- resolution model / kind / 固定值或参数化配置：
- 原始文件保持不变：是 / 否

验证命令：

```bash
shasum -a 256 '<source-path>' && stat -f '%N %z bytes' '<source-path>'
```

## 2. 声明模型

- fronting / backing：
- 普通层（材料、公式、密度、厚度、粗糙度）：
- 周期块（有序子层、repeat、top roughness）：
- 氧化层建议：接受 / 拒绝 / 无；对应 `OxideDecision`：
- `StructureEvidence`（`m_data`、`m_model`、峰位/警告）：
- 拟合范围与 mask/excluded indices：
- 参数初值、bounds、locks、sharing：
- batch mode：independent / joint

## 3. 软件与可复现配置

- commit/artifact identity：
- Python / OS / architecture：
- NumPy / SciPy / PySide6 / Matplotlib 版本：
- 锁定依赖 identity（lockfile/requirements path、SHA-256、环境导出）：
- schema / algorithm / objective / seed-tree 版本：
- `FitConfig`：
- master seed：
- service seed(s)：
- optimizer child seed(s)：
- bootstrap/profile seed(s)：
- MCMC seed(s)：
- checkpoint path/hash：
- 输出 project/export 目录（与 source 分离）：

## 4. 运行结果

- runtime：
- cancelled / warnings：
- 所有 candidate ID、seed、objective、valid、archived、stop reason：
- candidate clusters / weights：
- 自动推荐 candidate：
- 显式选择 candidate：
- confidence：`可信|可用但相关|多解|不可信`
- confidence 原因：
- fitted 参数（每个长度同时记录 stored Å 与 displayed nm）：
- profile likelihood（左右 closed/open）：
- bootstrap intervals / failure rate：
- strong correlations / boundary hits：
- residual ACF / systematic residual：
- structured physics diagnostics：
- 可选 MCMC owner、acceptance、split-Rhat、ESS、boundary hits：
- SLD/profile 物理合理性判断：
- domain owner 判断：
- observed/model plot path/hash：
- residual/ACF plot path/hash：
- SLD/profile plot path/hash：
- runtime resource notes（CPU、峰值内存、worker 数、异常终止/重试）：
- export manifest path/hash：
- export 文件 SHA-256 清单 path/hash：

## 5. 重跑一致性

使用相同 config/seed 至少运行三次，再使用一个新 master seed 运行一次。

| Run | Master seed | Candidate order | Selected cluster | Confidence | Runtime | Notes |
|---|---:|---|---|---|---:|---|
| same-1 | | | | | | |
| same-2 | | | | | | |
| same-3 | | | | | | |
| fresh | | | | | | |

- 同 seed candidate ordering/confidence 完全一致：是 / 否
- fresh seed 进入同一 accepted cluster：是 / 否
- 若否，是否诚实分类为 `多解/不可信`：

## 6. 验收结论

- 自动化 gate：pass / fail
- 真实数据结论：pass / fail / blocked
- pass/fail/block 原因：
- 遗留风险：
- 是否批准替换 legacy scripts：是 / 否
- reviewer：
- review date：
- domain-owner sign-off（姓名/角色/日期/结论）：

只有三类批准数据全部 pass、自动化 gate 通过、SLD/profile 物理合理且 confidence
诚实时，才能批准替换 legacy scripts。
