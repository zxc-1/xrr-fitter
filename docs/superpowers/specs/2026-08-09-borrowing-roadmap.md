# 竞品借鉴总体路线

## 目标

把六项外部借鉴与 Multifitting 前四项能力落地为可分阶段实施的设计集合。本文件只记录
跨设计共享的架构约束、实施顺序与依赖决策；单项细节见同目录下对应的 `-design.md`。

## 跨设计的架构约束

`tests/architecture/test_dependency_rules.py:74` 的 `ALLOWED` 表由穷举文件系统检查强制
执行，`:565` 的 `assert owner in ALLOWED` 让未注册的包直接失败。三条后果决定了所有
设计的落点：

- `io` 只能 import `{io, model}`。不确定度带、先验密度、ORSO 头部的**计算**都不能写在
  `io/`；计算结果必须先成为 `model` 类型，再传进渲染器与编解码器。
- `__main__` 只能 import `{gui}`。CLI 必须是新包并同步扩表，不能塞进 `__main__.py`。
- `MODEL_ALLOWED["parameters"]` 与 `["structure"]` 都是 `set()`。约束节点、先验声明、
  界面过渡种类只能是自洽的数据类型，数学实现放 `physics/` 或 `evaluation.py`。
- `ALLOWED["gui"] = {"gui", "api"}`。GUI 碰不到 `services` 与 `model` 内部，所以**每个新
  模型类型都要在 `api.py` 配一对 `validate_*` / `set_*` 函数**，否则界面无处落地。这条是
  下面"界面同批交付"的技术根据，不是风格偏好。

新增 `PACKAGE_EDGE_EXCEPTIONS` 条目不在任何一项设计的范围内。该表注释要求每条例外
配一个三向 fixture，成本高于把计算移动到正确的包。

## 界面同批交付

七项功能各自的界面与后端**同批交付**，不拆成"后端先行、UI 后补"。理由：这是 PySide6 桌面
应用，表达式约束和先验如果只能靠手写工程 JSON 使用，对真实用户等于没做；而先落后端再补 UI
会让 `api.py` 的 validate/set 契约缺席，等到做界面时才发现要回头改模型层。

GUI 侧有四种既有提交模式，新界面照最贴近的那种复制，不发明第五种：

| 模式 | 范本 | 适用 |
| --- | --- | --- |
| api 校验 + 原子提交 + 错误标签 | `gui/parameters/sharing.py` | 约束规则、先验 |
| detached 编辑 + panel 发布 | `gui/structure/editor.py` + `panel.py` | 结构类改动 |
| 单元格改动即提交 | `gui/parameters/panel.py:235` | 参数表新列 |
| 表单构造完整不可变对象 | `gui/structure/dialogs.py` | 各类新对话框 |

统一约定：错误消息写进带 `objectName` 的 `QLabel` 后重新抛出（照
`SharingEditor.apply_rules`）；`document.project_changed` 驱动重渲染，界面不持有可变副本；
提交前后比较 `updated is current` 以避免无变化时误发信号。

## 阶段与顺序

| 阶段 | 设计文档 | 依赖 |
| --- | --- | --- |
| 一 | `2026-08-09-orso-validation-design.md` | 无 |
| 一 | `2026-08-09-headless-cli-design.md` | 无 |
| 二 | `2026-08-09-orso-export-design.md` | 阶段一的 ORSO 语义映射 |
| 三 | `2026-08-09-sld-uncertainty-bands-design.md` | 无 |
| 三 | `2026-08-09-parameter-priors-design.md` | 无 |
| 三 | `2026-08-09-interface-transitions-design.md` | 无 |
| 四 | `2026-08-09-expression-constraints-design.md` | 阶段一至三的接口冻结 |
| 四 | `2026-08-09-stack-drift-design.md` | 表达式约束 |
| 四 | `2026-08-09-nested-periodic-stacks-design.md` | 无（可提前） |

阶段一两项都不触碰 `src/` 的核心数据结构。阶段三三项写集互不重叠，可并行。表达式约束
触及 17 个消费 `SharingRule` 的文件，放最后让前序接口先稳定。

每一项的界面工作计入该项自身，不单列阶段。触及的 GUI 文件按项分布如下，写集互不重叠是
阶段三可并行的前提：

| 设计 | GUI 落点 |
| --- | --- |
| headless-cli | 无（CLI 与 GUI 并列入口） |
| orso-validation | 无（不触碰 `src/`） |
| orso-export | `gui/export/dialog.py` |
| sld-uncertainty-bands | `gui/plots/sld.py` |
| parameter-priors | `gui/parameters/{table,panel}.py` |
| interface-transitions | `gui/structure/{dialogs,editor}.py` |
| expression-constraints | 新增 `gui/parameters/constraints.py`，改 `panel.py`、`table.py` |
| stack-drift | `gui/structure/dialogs.py` |
| nested-periodic-stacks | `gui/structure/{dialogs,editor}.py` |

阶段四三项都改 `gui/structure/dialogs.py` 或 `gui/parameters/`，必须串行；这与它们在后端的
依赖顺序一致，不额外增加约束。

## 依赖决策

- `orsopy` 进生产依赖。取舍：打包体积与一个新的供应链面，换取导出功能对用户可用。
  放 `test` extra 等于功能做了但用不上。`orsopy` 由 ORSO 官方维护、纯 Python、依赖面
  小。实施时按 `tools/lock_environment.py` 的既有姿势 pin 到确定版本。
- `jsonschema` 与 `orsopy` 一同进生产依赖。orsopy 不把它列为强依赖，但
  `orsopy.fileio.base._validate_header_data` 在函数体内 `import jsonschema`，缺失时
  schema 校验静默不可用——而校验是本项目导出路径的验收条件之一，不能可选。
- `refnx` 保持 `test` extra 不变。它是数值对标基准而非运行时依赖，`pyproject.toml:35`
  的 commit pin 与 `tests/unit/tools/test_lock_environment.py` 的守卫都不动。
- 所有新增模块必须通过 `tools/check_radon.py` 的既有复杂度门禁。

## 非目标

- 不做 PSD 粗糙度族、off-specular 扫描、二维 GISAS。这是 BornAgain 的护城河，且服务的
  是 EUV 多层镜用户群而非薄膜计量用户群。
- 不做样品弯曲的仪器模型项。本轮 Multifitting 借鉴只取前四项。
- 不做 nested sampling 本体。先验设计只负责把单位立方体映射准备好。
- 不改动 Parratt、分辨率卷积、五阶段流水线与确定性种子树的既有数值行为。
