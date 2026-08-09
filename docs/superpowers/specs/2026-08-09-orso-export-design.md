# ORSO 导出设计

## 目标

产出一个自足的 `.ort` 文件，携带模型、参数、不确定度、协方差与数据，使外部人员能够
独立复现本项目的拟合结论。参照 GenX 的做法：导出的不只是曲线，而是复现所需的最小完整
包。读取器优先级低于写入器——导出建立可信度，导入只是便利性。

## 前置改动：留住协方差标度

`analysis/report.py:166` 已算出 `physical_covariance`，但 `:167` 归一化成 correlation 后
只有 `correlation_names` 与 `correlation_matrix` 进入 `UncertaintyReport`
（`model/analysis.py:464`）与持久化（`io/codec_results.py:132`）。协方差的标度信息在此
丢失，无法从 correlation 反推。

因此本设计的第一步是让标度可复原，二选一：

- **方案 A**：`UncertaintyReport` 增加 `parameter_sigma: np.ndarray` 字段（对角标准差），
  协方差由 `sigma[:,None] * correlation * sigma[None,:]` 重建。
- **方案 B**：直接持久化 `covariance_matrix`，correlation 改为派生。

**选方案 A。** 理由：correlation 已被 GUI、`export_tables.py` 与既有测试消费，改成派生会
扩大改动面；而 `parameter_sigma` 是纯增字段，旧工程文件缺失时可判定为"该结果无协方差
导出能力"并在 `.ort` 中省略该段，不破坏向后兼容。

## 文件分段

`.ort` 头是 YAML，体是列数据。写入器分四段：

| 段 | 内容 | 数据来源 |
| --- | --- | --- |
| 数据段 | q、测量强度、误差、模型强度、残差 | `io/xy.py` 的 `DataRecord` 加模型列 |
| `orso.data_source` | 测量元数据与源文件 sha256 provenance | `io/source.py` |
| `orso.reduction` | 软件版本、`SERVICE_SEED_TREE_VERSION`、完整 `FitConfig` | `model/fitting.py` |
| 自定义扩展段 | 参数表、误差棒、协方差、`ConfidenceClass` 与 reason codes | `UncertaintyReport` |

第四段是关键决策。ORSO schema 没有"置信度分级"的位置，而
`ConfidenceClass.TRUSTED/CORRELATED/MULTIPLE/UNTRUSTED`（`model/analysis.py:237`，值为
中文字面量）加 reason codes 是本项目相对所有竞品最独特的产物。**不为了塞进标准而阉割
它**：放进 ORSO 允许的自定义键，前缀命名空间化为 `xrr_fitter.confidence`。标准部分严格
合规，独有部分显式标注为扩展。

`ConfidenceClass` 的值是中文，导出时同时写 enum 名（`TRUSTED`）与显示值（`可信`），
让非中文环境的消费者也能机器解析。

## 代码边界

- 新增 `src/xrr_fitter/io/orso.py`，只依赖 `{io, model}`，符合 `ALLOWED["io"]`。协方差
  重建这类数值组合放 `analysis/`，不放 `io/`。schema 校验的调用集中在此文件一处。
- 改 `pyproject.toml`：`orsopy` 与 `jsonschema` 同为生产依赖（后者是校验路径的运行时
  必需项，见下文「失败与状态」），并按 `tools/lock_environment.py` 与
  `tools/lock_windows_environment.py` 重新生成两个平台的 lock。
- 改 `src/xrr_fitter/model/analysis.py` 增加 `parameter_sigma` 字段（可选，默认空）。
- 改 `src/xrr_fitter/analysis/report.py` 填充该字段。
- 改 `src/xrr_fitter/io/codec_results.py` 序列化该字段，缺失时按空处理。
- 扩 `src/xrr_fitter/services/exports.py` 与 `io/export_run.py` 的 artifact 列表。
- CLI 增加 `--format ort`。
- 落盘走既有 `atomic_replace_bytes`，与 `project_codec.py:save_project` 同一路径。
- 改 `src/xrr_fitter/gui/export/dialog.py`。

## 界面

`.ort` 是随导出批次一起产出的 artifact，不是独立的导出动作。`ExportWorkflow.export_results`
（`dialog.py:81`）调 `api.export_result` 得到 `ExportManifest`，`.ort` 文件加进
`manifest.files` 后，`export_summary`（`dialog.py:25`）会自动把它列进完成摘要——这一段
无需改动，因为它遍历的是 manifest 记录而非硬编码清单。

需要改的是选项：导出前让用户选是否产出 `.ort`。用 `QFileDialog.getExistingDirectory`
（`dialog.py:94`）拿不到额外选项，所以在它之前插一个小对话框（或把目录选择改为自建对话框
带一个复选框）。默认勾选，理由是 ORSO 兼容输出对用户几乎零成本，而"忘了勾"的代价是要重跑
一次导出。

**扩展段的存在要在摘要里说清。** `.ort` 里 `ConfidenceClass` 与被排除的 NaN 行数走的是
`xrr_fitter.confidence` 命名空间的自定义键。完成摘要在 `.ort` 那一行后追加一句，说明该文件
含本项目扩展字段，标准消费者会忽略它们。这避免用户以为扩展内容是标准的一部分并据此对外分发。

`parameter_sigma` 缺失（旧结果）时协方差段缺席，摘要要写出原因，不能让用户看到一个静默变
瘦的文件。

## 失败与状态

- JSON schema 是 ORSO 的**规范来源**，YAML 头只是一种序列化。写入后必须用 schema 校验，
  不能只靠 `orsopy` 能读回就算通过。

  校验入口的三个约束已在 orsopy 1.2.3 上实测确认，实施时按此接线：

  1. **入口是私有函数 `orsopy.fileio.base._validate_header_data(dct_list)`**，接受 dict
     列表。orsopy 未导出公开的校验 API，因此调用点要集中在 `io/orso.py` 内一处并注释
     说明依赖私有名，orsopy 升级时这里是首个检查点。
  2. **`jsonschema` 必须显式声明为生产依赖。** 它在 `_validate_header_data` 函数体内
     被 import，orsopy 自身不强制要求它。缺失时抛 `ModuleNotFoundError`，而非降级跳过
     校验——这与"校验失败则导出整体失败"一致，不需额外处理，但依赖必须在 lock 里。
  3. **schema 文件必须随冻结包落到 `orsopy/fileio/schema`。** 查找路径是
     `os.path.join(os.path.dirname(base.__file__), "schema", "refl_header.schema.json")`，
     纯目录相对，因此 PyInstaller 的 `datas` 目标路径必须逐字匹配。
     `packaging/windows/xrr-fitter.spec` 已用
     `collect_data_files("orsopy", subdir="fileio/schema")` 满足这一点，并在 macOS 冻结
     产物上验证过校验可真实执行。
- NaN 表示是最可能出问题的地方。`evaluation.py:788` 的 `qz` 在非正角度处存 NaN，而
  `project_codec.py` 用 `allow_nan=False`。ORSO 对缺失值的表示若与之冲突，处理方式是：
  非正角度的行**不写入** `.ort` 数据段，并在扩展段记录被排除的行数与原因，不写 NaN。
- `parameter_sigma` 缺失（旧结果）时省略协方差段并在扩展段标注原因，不写零矩阵。
- schema 校验失败时导出整体失败，不产出部分文件。原子发布保证不留半成品。

## 验证

三层，缺一不可：

1. `orsopy` 读回自洽。
2. JSON schema 校验通过。
3. 往返测例：导出后导入，数据与参数逐位相等。

外加：

- NaN 行排除的显式测例：构造含非正角度的数据集，断言 `.ort` 行数等于有效行数，且扩展段
  记录的排除计数正确。
- 协方差重建测例：`sigma[:,None] * correlation * sigma[None,:]` 与 `report.py:166` 的
  `physical_covariance` 逐位相等。
- `parameter_sigma` 为空时导出仍成功且协方差段缺席。
- 全量测试证明 `parameter_sigma` 的引入未改变任何既有 correlation 数值。
- GUI 测例：勾选 `.ort` 后完成摘要含该文件路径与扩展字段说明；不勾选时摘要不含该行且
  磁盘上无 `.ort`。
- GUI 测例：`parameter_sigma` 为空时摘要写出协方差段缺席的原因。

## 非目标

- 不做 ORSO 导入器本体（读取 `.ort` 作为输入数据集）。留作后续。
- 不改变 `correlation_matrix` 的既有语义或消费方。
- 不把 `.ort` 变成本项目的工程格式；工程文件仍是现有 JSON。
- 不导出 MCMC 原始链（体积不可控）；只导出派生的区间与协方差。
