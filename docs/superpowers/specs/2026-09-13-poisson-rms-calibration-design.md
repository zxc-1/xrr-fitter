# Poisson RMS continuous-family calibration

用户已以“修”批准修订方案。此规格替换 `2026-09-12-poisson-diagnostic-calibration-design.md` 的 MAD-floor 标准化和当前身份部分；其他检测器、拟合、门禁和科学验收规则不变。此前 minP 推荐撤回，不实施秩字段迁移。

## 目标和边界

修复原 81% 全 seed 区间覆盖产出所暴露的诊断误拒，并保留真正失配检出。当前 v3 ACF 开发例完成 B999 却 adjusted p=.03，仍是失败。工程 PASS 不等于总体科学完成。

- 默认 B=999、alpha=.01；B<99 不可用。一个完整 single/joint fit 的所有声明适用 detector/member 列同族。
- 保留原 raw trigger、detector 数值/符号/窗口、生成器、随机流、MATCH_LIMITS=(1e-4,1e-6,1e-6)。
- 保留声明四起点、固定 L-BFGS-B → 计费交接 → TRF、预算及全部路径原生成功要求，不增加求解或 fallback。
- 不改非 Poisson 数值行为；不安装依赖、不扩大默认 CI、不兼容旧文件、不提交/合并/发布。

## 固定统计规则

U 为 observed 第 0 行加 B 个完整 null 的有限 float64 矩阵。每列 center=全行 median、scale=全行相对 center 的 RMS（ddof=0），非恒定列 z=(U-center)/RMS，真常量列 z=0、scale=0。S=max(0,max_j z_ij)，family p=count(S>=S0)/N，定位 p_j=count(S>=max(0,z_0j))/N；精确 inclusive ties，不加 jitter 或 isclose。

固定实际计算路径：

1. 每列复制为独立连续一维 float64。全行精确相等才是常量；常量 center 为该值、scale/z 为正零，数值零统一正零。原始 observed 不改写。
2. 非恒定列排序。奇数 N 取中间值；偶数中间 a<=b，相等取 a；异号或 abs(a)、abs(b) 均 <=float64_max/2 时 `(a+b)/2`，否则 `a/2+b/2`。计算 center 后规范零。
3. d=column-center，peak=max(abs(d))；要求 d 有限、peak>0。
4. relative=d/peak；非零 d 若下溢为 relative=0，明确 FloatingPointError。允许 squared relative 的 IEEE 舍入下溢，但原 relative 保留，且至少一个绝对值为 1。
5. `relative_rms=sqrt(math.fsum(sorted(relative*relative))/N)`，要求有限且 0<relative_rms<=1。
6. `scale=peak*relative_rms` 必须有限且 >0；非恒定 scale 舍入为 0 明确失败，不冒充常量。
7. `z=relative/relative_rms` 必须有限；随后计算 S 和 p。输入不修改，返回数组只读。不能改回 d/rounded_scale。

数值失败沿现有 `diagnostic_statistics_unavailable` 边界传播，不发布部分 p。极端有限值但 centered 差值不可表示可以明确失败，不引入高精度/epsilon/截顶。

## 不变量与诚实限制

精确算术下行等变、列不变、正向仿射单位不变，min(p_j)=p；固定环境下规范列复制和排序归约使行/列/布局排列可逐位验证。浮点变换可能在输入前合并数值，不承诺任意浮点变换或跨平台逐位不变。

RMS 保留连续间距，避免 minP 在一般连续 12 列族的强制最高秩 ties，但不是一般最优检验。稀疏单峰各列仍标准化为 sqrt(N) 并列；旧全零四单峰示例的 p 从 .005 变为 .020，这是明确的新契约，不是删除回归。RMS 不对齐尾形，observed 会膨胀自身分母，近常量真实差异可能放大。plug-in 的全局误拒/覆盖仍需实际独立验收。

## 不可变证据和身份

保留 DiagnosticStatistic 的 observed/center/scale/adjusted_p_value；校准字段 all-present/all-absent。scale 有限且 >=0；scale=0 仅允许 observed=center、adjusted_p_value=1。嵌套 statistic 的构造/重建校验不能被伪造实例绕过。全零尺度 family 的 observed_score=0、tail=tie=N、p=1。

scale 是原单位浮点 RMS 描述量；实际 observed_score/p 使用 factorized z。精确重放用完整 U 和上述版本固定算法，不能从 rounded scale 单独重建临界 ties；无需新增 scale_factors 或 standardized_observed。

唯一当前身份：schema 4；algorithm `xrr-fit-v2-poisson-4`；objective 字符串 `"2"`；diagnostic `poisson-refit-null-v4`；method `poisson_refit_null_rms_v4`；refit_policy 仍为 `declared_sobol4_lbfgsb_trf_v3`。严格拒绝旧身份，不迁移旧文件。

## 验证与科学完成条件

所有实现先真实 RED 再 GREEN；覆盖 RMS 数值、原缺陷的单位竞争、常量、稀疏 ties、B99/12 列、非法轴、极端数值、精确排列、codec/重建和不可用传播。开发重放 119/148/footprint37/surface37/acf37 使用当前公共 API、实际 B999 与 strict save/load。后续同一冻结 clone 执行 quality/tools/unit/integration/gui/spawn/regression 和完整 Radon。

新独立协议保留原生成规则和 8 个端点：gamma=.00625；null F upper<=.02/count<=24、U upper<=.01/count<=9；各目标 power lower>=.8/count>=90。新种子：null_single 3000000..3001999；null_joint 3010000..3011999；四阳性从 4000000/4010000/4020000/4030000 起各 100。原 0..199 仅单列开发重放，不计作新 holdout。全部预登记后执行，不预探、补样或中途挑结果。

完整报告正式区间产出、全 seed 覆盖、条件覆盖、误拒、不可用及目标功效。B99/更大联合族的确定性回归不冒称这些配置已通过独立科学验收。旧 4/8 FAIL 与原件丢失事故保留，任何必需端点失败/未解释欠覆盖都不能关闭总体目标。
