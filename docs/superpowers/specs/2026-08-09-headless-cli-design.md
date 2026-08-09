# 无头 CLI 设计

## 目标

让完整的拟合、采样与导出流程可以无人值守运行，不启动 Qt 事件循环。这是本项目相对
Multifitting 的结构性差异的载体：Multifitting 的 `main.cpp` 把 `argc/argv` 直接交给
`QApplication` 从不解析，其手册明确承认一般情形下拟合"requires manual intervention"，
方法在设计上无法自动化。

## 设计

`api.py` 已经零 GUI import，`__all__` 导出了 `FitConfig`、`FitResult`、`FitProgress`、
`FitReadiness`、`McmcConfig`、`McmcReport`、`ExportManifest`、`OperationJob` 等完整契约。
因此 CLI 是纯外壳，不新增业务逻辑。

架构表扩两行：

```python
"cli": {"api"},
"__main__": {"gui", "cli"},
```

`cli` 只允许 `{"api"}` 而非 `{"api", "services"}`，与 `gui` 平级。这条约束的价值是：
任何 CLI 需要而 `api.py` 未导出的能力，都会暴露成 `api` 层的真实缺口，而不是被 CLI
私自绕过。

子命令（第三列是唯一允许调用的 `api.py` 导出，已逐个对照 `tests/architecture/test_public_api.py`
的 `SIGNATURES` 核实存在）：

| 命令 | 作用 | api 入口 |
| --- | --- | --- |
| `fit <project.json>` | 跑五阶段流水线并写出结果 | `load_project` + `preflight_fit` + `fit_project` |
| `fit --auto <project.json>` | 自动批次拟合 | `fit_automatically` + `summarize_automatic_results` |
| `mcmc <project.json>` | 跑采样并产出 `McmcReport` | `run_mcmc` |
| `export <project.json>` | 复用 `io/export_run.py` 的原子发布 | `export_result` |
| `validate <project.json>` | 只读检查，含源文件 TOCTOU 新鲜度 | `inspect_sources` + `preflight_fit` |

**没有独立的 `resume` 子命令，也不需要。** 续跑不是一条平行路径，而是 `fit` 的自动行为：
`services/workers.py` 的 `checkpoint_path` 落盘用的是 `save_project`，写出的是**工程文件**而非
另一种 checkpoint 格式；`io/project_codec.py:283`/`:342` 把 `dataset.checkpoint` 完整序列化与
反序列化；`services/fitting_phases/base.py:176` 从 `prepared.updated_dataset.checkpoint` 取值
喂给 `FitSearchRequest.resume_checkpoint`；`fit/pipeline.py:181` 判非 `None` 后经
`validate_resume_checkpoint` 得到 `plan.remaining_stages` 续跑。所以 CLI 只要
`load_project(那个工程文件)` 再 `fit_project`，续跑就自动发生。
早先版本写的"复用 `fit/resume.py` 的续跑路径"是错的：`fit/resume.py` 在 `fit` 层，`cli → {api}`
根本触达不到，`api.py` 的 42 个导出里也没有任何 resume 入口。`--checkpoint <path>` 因此是
`fit` 的一个可选**输出**参数（透传给 `start_fit_job` 的 `checkpoint_path`），不是输入参数。

**`cli → {api}` 已经抓出一个真实的 api 缺口，照约定补 api 而不是绕过。** 退出码 `1`
的判据是 `ProjectFitResult.datasets[i].fit_result.confidence`，其类型
`ConfidenceClass`（`model/analysis.py:237`，`StrEnum`）**不在 `api.__all__` 的 91 项里**
——`FitResult` 被导出了，它的字段类型没有。因为是 `StrEnum`，CLI 技术上可以写
`confidence == "不可信"` 绕过，但那把一个用户可见的中文标签硬编码进退出码逻辑，标签
一改 CLI 静默失效。决策是给 `api.py` 加一行 re-export，`__all__` 按字母序插在
`BeamSpec` 与 `DataColumnMapping` 之间。这连带 `tests/architecture/test_public_api.py:9`
的 `PUBLIC_NAMES` 与 `:196` 的 `tuple(api.__all__) == PUBLIC_NAMES` 精确相等断言，
两处同一次提交改掉。这是本设计里唯一对 `api.py` 的改动，且只是导出既有类型，不新增
签名、不改行为。

入口用 `[project.scripts] xrr-fitter-cli` 与既有 `[project.gui-scripts] xrr-fitter` 并列。
**不让同一个 `xrr-fitter` 靠有无参数切换模式**：Windows 上 `gui-scripts` 走
`pythonw.exe`，无控制台，stdout 会被丢弃。

进度输出默认写 stderr 的人类可读行，`--json-progress` 切成 stdout 的 JSON Lines，字段
直接映射 `FitProgress`，便于外部编排消费。

## 代码边界

- 新增 `src/xrr_fitter/cli/`：`__init__.py`、`main.py`，每个子命令一个模块。
- 改 `src/xrr_fitter/__main__.py`：保留 GUI 启动为默认，`_parser()` 仍不接收参数。
- 改 `tests/architecture/test_dependency_rules.py` 的 `ALLOWED` 表（两行）。
- 改 `pyproject.toml` 增加 `[project.scripts]`。**这一条会连带三处发布链改动**：
  `tools/build_release_spec.py:217` 只读 `project.get("gui-scripts")`，新增的
  `[project.scripts]` 会被静默丢弃，生成的 sdist 里没有 CLI 入口；同文件 `:295`
  的 `_expected_generated_metadata` 也只按 `gui-scripts` 判断要不要带
  `entry_points.txt`；`tests/architecture/test_distribution.py:35` 对 `gui-scripts`
  是精确相等断言，新增 `scripts` 键会让 release-spec fixture 与真实
  `pyproject.toml` 漂移。渲染器与断言必须同一次提交改掉，否则 `distribution`
  模式转红或（更糟）发布物默默缺入口。
- 改 `tools/verify_registry.py` 的 `unit` 模式加 `tests/unit/cli`。该模式是**显式枚举
  目录**的，不是 `tests/unit` 整目录，新目录不加进去就永远不被执行；且
  `tests/unit/tools/test_verify_registry.py::test_registry_is_exact_for_completed_suites`
  用 `observed == _expected_registry(module)` 精确相等，两处必须同一次提交改掉。
- 改 `src/xrr_fitter/api.py` 与 `tests/architecture/test_public_api.py`：只为 `ConfidenceClass`
  加一处 re-export 与 `PUBLIC_NAMES` 一项（见上文）。不新增函数、不改任何既有签名。
- 不改 `services/`、`fit/`、`analysis/` 的任何代码。

## 失败与状态

退出码约定：

- `0` 成功
- `1` 拟合完成但未收敛（`ConfidenceClass` 为 `UNTRUSTED` 时也归此类）
- `2` 输入校验失败（工程文件不合法、参数越界、共享规则冲突）
- `3` 源文件不新鲜（TOCTOU 检查失败）

`freeze_support()` 必须在 CLI 路径同样先调用，否则 Windows 冻结包的并行 worker 会递归
启动。当前 `__main__.py` 在 `main()` 内调用它，CLI 入口需保持同一顺序。

错误信息沿用 `io/source.py` 的中文风格，不因为是 CLI 就改成英文。

## 验证

- `tests/unit/cli/` 每个子命令一个 `subprocess` 冒烟测例，断言退出码与 stdout 结构。
- 一个等价性测例：同一 project 经 CLI 与经 GUI 服务层运行，结果逐位相等。这把
  `SERVICE_SEED_TREE_VERSION = 1` 的确定性从声明变成被测事实，是本项的核心价值。
- 一个架构测例：`cli` 包内出现 `import xrr_fitter.services` 时必须报 `package-edge`。
- `--json-progress` 输出逐行可被 `json.loads` 解析。
- 更新 `docs/user-guide.md` 增加 CLI 一节。

## 非目标

- 不做交互式 CLI、TUI 或 shell 补全。
- 不做新的配置文件格式；一切输入仍是现有的工程 JSON。
- 不让 CLI 具备 GUI 没有的能力，避免两条路径的行为分叉。
