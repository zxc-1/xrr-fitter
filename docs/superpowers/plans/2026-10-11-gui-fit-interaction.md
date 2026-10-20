# GUI 操作与拟合交互修补

**目标：** 按实际控件操作验收既有 GUI，用 1201 点物理合成观测测量各模式，修复会误导使用者的拟合反馈。

**边界：** 不降低默认科学预算、不减少最终求解点数、不添加依赖、不改项目 schema；保留已有未提交改动。后续交付授权要求完成验收后提交、推送并创建 PR，直至既有 PR 门禁全部通过；不合并、不强推。

## 设计与验收

- 保持现有 Qt / public API / worker 分层，只修改已复现的交互链路。
- 同阶段进度仅在同一数据集内合并；独立批次按成员汇总，第一成员完成不能让整批显示 100%。
- 实时预览必须属于当前数据集；无明确归属的联合预览不叠到任意曲线上。
- 左侧实时读数按成员更新原有行，不清空重建数据树，不把一条曲线的 J 写到所有曲线上。
- 暂停请求立即可见；后端没有全线程 ACK 时不能宣称“已暂停”。停止请求期间禁止再次暂停、跳过或重复停止，仍保留强停出口。
- 自动开始与手动开始统一清理上一轮预览、检查点反馈和进度状态；运行中的检查点不重复执行启动预检，退出时重新检查。
- 不确定度对话框显示 MCMC 进度和校验/运行错误；关闭重开不丢失状态；结束后时钟停止。
- 共享页提供新增与删除入口，复用既有校验 API；取消与失败不修改项目。

## 独立写集与步骤

1. 主线程：`gui/fitting/{controller,panel,progress}.py`、`gui/data/panel.py`；必要时 `gui/main_window.py`。先在 `tests/gui/test_fit_batch_interaction.py` 和 `tests/gui/test_fit_request_feedback.py` 记录 RED，再最小修复与 GREEN。
2. MCMC：`gui/results/panel.py`、`gui/results/mcmc*.py`、`tests/gui/test_mcmc_interaction.py`。覆盖按钮点击、对话框可见性、后台帧投影与取消。
3. 共享：`gui/parameters/sharing.py`、`gui/parameters/sharing_dialog.py`、`tests/gui/test_sharing_operations.py`。覆盖键鼠输入、确认、取消、删除、无效输入和已有规则冲突。
4. 基准只写仓库外证据：物理前向生成真值；独立噪声抽样；单层 3 噪声 × 手动/自动、双层、双数据集独立/联合；串行运行，保留默认 fast 预算和每阶段时间。超时与小预算诊断单独标记。
5. 控件清单分别记原生操作、QTest 操作、方法直调/状态断言、未验证，不把构造点数或总测试数当完成率。

## 验证顺序

- 定向测试：`PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen ../venvs/repo-qt6112/bin/python -m pytest -p no:cacheprovider --basetemp=<仓库外独立目录> <本组测试> -q`。
- 变更后按影响面执行 `python tools/verify.py gui`、`python tools/verify.py integration`、`python tools/verify.py quality`，并运行 `python tools/check_radon.py`；产物目录均在仓库外。
- 原生运行与 offscreen Qt 测试分时，避免进程识别冲突；性能基准与其它重拟合分时。
- 最终核对本轮基线哈希与文件白名单，停止本轮启动的进程；保留可重跑脚本、原始测量和失败证据。

初始证据：`.codex-artifacts/gui-full-ux-20261011-5xUXEuqL/interaction-trace/counterexamples-red.log` 与 `control-inventory/hidden-mcmc-status-probe.json`（路径相对外层工作区）。
