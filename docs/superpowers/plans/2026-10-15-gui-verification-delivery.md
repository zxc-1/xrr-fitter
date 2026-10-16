# GUI 验证与 PR 交付执行计划

> **For agentic workers:** 使用 `executing-plans` 按任务推进；新增行为修补先取得 RED，再作最小修改并验证 GREEN。

**Goal:** 完成 GUI 控件操作、1201 点拟合矩阵和当前代码验证，提交、推送 PR 并确认其准确 head SHA 的全部既有检查通过。

**Architecture:** 延续现有 Qt / public API / worker 架构。复用操作驱动器和真实对话框，操作证据与数值拟合、原生验收证据分开记录；不以源码存在、方法直调或测试总数代替操作验收。

**Tech Stack:** Python 3.12、PySide6、pytest-qt、Matplotlib、pyqtgraph。

## Global Constraints

- 继续使用 `feat/gui-redesign-checklist` 现有工作区，保留既有 staged / unstaged 内容，不复制到新工作区或回滚。
- 不改变科学预算、最终求解点数、资格阈值、模型、依赖、项目 schema 或 CI 门禁。
- 证据和临时文件放外层工作区 `.codex-artifacts/goal-pr-green-20261015-9kfxuajb`；旧失败日志保留。
- 使用 `/Users/dala/Desktop/XRR-Fitter/venvs/repo-qt6112/bin/python`，设置 `PYTHONDONTWRITEBYTECODE=1`；Qt 自动测试使用 `QT_QPA_PLATFORM=offscreen`。
- 原生 GUI 与 offscreen GUI 测试分时；完整数值基准与重拟合测试分时。
- PR 创建后附加到当前聊天，检查准确提交的全部实际 checks。当前配置预计 12 项：7 个 standard job、checkpoint、3 个 Audit job 和 windows-headless；用户未授权合并。

## Task 1: 核对控件身份与操作证据

**Files:** `tests/gui/control_operation_driver.py`、`tests/gui/test_control_operations_*.py`；仓库外 `control-reconciliation.json`。

- [x] 对照旧 321 条清单逐项检查当前控件：存在、改名、同一控件别名、固定输出（非选项）分别记录源码和运行证据，不为凑数量新增控件。
- [x] 复核已有真实 QTest 输入及断言；未满足的拟合、绘图、源恢复、布局和弹窗路径补充可重放的操作用例。
- [x] 将鼠标/键盘、Qt context/wheel event、方法直调、原生验收分别记账；参数化状态单独列明。
- [x] 运行完整控件用例并保存逐操作记录，只将通过用例中的有效证据纳入最终清单。

对账完成不等于全状态验收：318 项有通过用例证据，范围拖动和原生树重排各有一项失败，另 1 项为无生产调用的 helper 按钮，予以排除。仓库外 `control-reconciliation-reviewed.json` 保留来源、失败和证据类型，不将夹具或几何检查计作原生输入。

## Task 2: 拟合基准矩阵

**Files:** 仓库外 `benchmark/benchmark_1201.py`、输入项目、逐阶段进度、结果与哈希。

- [x] 核对前向生成真值、独立噪声、1201 点、默认 fast 预算和 4 workers。
- [x] 串行运行单层 robust_log / gaussian / poisson 各手动与自动、双层手动与自动、双集独立及联合各手动与自动。
- [x] 分别报告运行耗时、逐阶段耗时、发布状态、质量标记和真值恢复；失败或未合格不得记为成功联合拟合。
- [x] 对联合自动拒绝路径从成员资格开始定位；保留失败输入，若需物理正常对照则另建输入并明确区别，不降低门槛。

仓库外 `benchmark/README.md`、`truth-recovery.json` 和 `phase-summary.csv` 记录矩阵结果及 42 次文件哈希核对。默认自动联合因成员边界命中拒绝发布；固定已知仪器参数的成功对照单列，不用于默认速度比较。

## Task 3: 原生补验与最小修补

**Files:** 仅已复现问题对应的 GUI 模块及回归测试；仓库外原生时间线。

- [ ] 补验原生先验菜单和 splitter 拖动，区分坐标路由/控件门禁与应用错误。
- [ ] 复用已完成默认 MCMC 的保存项目确认全参数纵向滚动；不重复采样，不掩盖未收敛诊断。
- [ ] 每个行为缺陷记录单一复现测试 RED，最小改动后运行同一测试 GREEN 和相关回归。

## Task 4: 最终验证与差异审查

- [ ] 以 `python tools/verify.py MODE --report-dir=<仓库外独立目录>` 依次验证 quality、tools、unit、gui、integration、spawn、regression。
- [x] 补验既有 typing、advisories 和 coverage 审计；审计工具按现有 hash-pinned lock 装入仓库外独立环境，不改变 GUI 环境或依赖锁。coverage 与 GUI/数值任务分时执行。
- [ ] PR 上核对 Audit advisories 的 distribution / installed SBOM / byte binding，以及 Windows CLI / 导出 / executable / byte binding；本机锁定包漏洞查询不能替代这些检查。
- [ ] 确认完整 `python tools/check_radon.py` 通过，不删除断言或降低复杂度阈值。
- [ ] 对完整 staged / unstaged / untracked 差异按原任务核对，保留预存改动；检查失败发布、取消、运行门禁与显示改动不触及科学模型。
- [ ] 核对验证前后源码哈希、停止自建进程，将工具缓存和临时产物移出仓库。

## Task 5: 提交、推送、PR 全绿

- [ ] 将经过审查的相关文件显式暂存并提交；GitHub 本机认证不可用时由用户重新登录，不读取或传递凭据。
- [ ] 正常推送 `feat/gui-redesign-checklist`，创建目标为 `main` 的 PR，并附加到当前聊天。
- [ ] 检查 PR head SHA 与本地提交一致；跟踪全部既有 checks，失败则保留日志、修复、验证、提交后重新等待。
- [ ] 所有需求证据充分且 PR 全绿后完成目标；汇报改动、验证、清理项和残余科学/平台限制。

## 本地检查点（2026-10-15）

- 用户选择先完成本地工作；暂不重试 GitHub 登录或推送。
- `local-verification-round2` 在不变源码上完成 7 个模式：quality 224、tools 1417、unit 3170、integration 41、spawn 4、regression 70 个用例通过；GUI 为 1889 通过、1 失败，唯一失败是范围拖动，未标记全绿。
- 后验图固定 4 位有效数字造成窄分布刻度重复。新增厚度、scale、background × 两种窗口宽度的 6 个 RED，用标准自动精度、独立倍率间距和每格 180 逻辑像素修补后，16 个布局用例及 138 个相关用例 GREEN；原滚动、字号、直方图与分位线断言保留。
- 后验精度修补后的正式 GUI 复验完成：1895 通过、1 失败，仍为范围拖动（目标 0.2 ± 0.03，实际 0.28）；验证前后 1002 个工作区文件哈希未变，未跳过该失败。7 条 warnings 来自现有 pyqtgraph 的 NumPy shape 弃用提示。
- 正式 typing 和 advisories 审计已通过，源码哈希未变；macOS / Windows 锁中的 44 / 38 个包完整查询且未报已知漏洞。VCS refnx 和原生库不在漏洞服务扫描范围，Windows 实机与字节绑定审计仍未验证。报告位于仓库外 `pr-audits-20261015`。
- 正式 coverage 审计在 GUI 结束后串行完成：3285 个测试通过，合并 25 个进程数据文件并生成 JSON / XML / HTML，输入哈希与 typing / advisories 一致。该审计排除 GUI，语句覆盖率 93.22%、分支覆盖率 80.48%，不能替代原生验收。
- coverage 结束后的原生工具复查仍返回 Mac locked；未启动新 GUI，也未修改系统锁屏设置，最新后验图的原生显示继续待验收。
- 只重绘既有默认 MCMC 保存结果，没有重跑采样或修改科学参数。样本为 128000×8，最大 split Rhat 为 41.13029142662623，仍未收敛，显示验收不能当作科学收敛验收。
- 范围手柄松开跳动、原生拖放后树丢行的最小修补均已单独提出，等待用户范围确认。保留失败截图、事件时间线和输入项目；不靠重载、放宽容差或跳过测试掩盖。
- 独立审查代理不可用；当前审查为主代理直接审查完整 `origin/main` 到工作树的差异，不能声称完成独立 review。最终 GUI/quality 复验与交付仍以最新仓库外报告和准确提交 checks 为准。
