"""一条 MCMC 链能不能当后验读：判据与那句「为什么不能」。

右栏分位卡、专家对话框的散文与中栏「MCMC 后验」页读的是同一条链，必须是同一个判定：一处不给
分位、另一处照念 P16/P50/P84，读者会把后者当成可信区间抄走。所以判据单独放在这里，只用标准库——
``results.uncertainty`` 借 ``plots.posterior`` 的控件，``plots.posterior`` 又要这个判定，放进任何
一边都会成环。
"""

from __future__ import annotations

from math import isfinite

# 两个阈值照 ``analysis.mcmc.problem_mcmc_warnings`` 的判据抄：split-R̂ 到了 1.10 出一条警告，
# ESS 掉到 100 以下出另一条。GUI 层不许 import 那个模块（它带 numpy），所以这里镜像一份并注明
# 出处——判据要改，改的是那边，这里跟着走。
SPLIT_RHAT_LIMIT = 1.10
EFFECTIVE_SAMPLE_FLOOR = 100.0


def _finite_values(values: object) -> tuple[float, ...] | None:
    try:
        converted = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if not converted or not all(isfinite(value) for value in converted):
        return None
    return converted


def unmixed_chain_reason(mcmc: object) -> str | None:
    """链没混合时那句「为什么不能当后验读」，混合了就是 ``None``。

    只看 split-R̂：越过 1.10 说明各 walker 还停在各自的地方，采样点不是后验的样本，分位无从
    谈起。ESS 低是另一回事——样本来自后验，只是少，分位仍有意义而噪声大，那一条归收敛段的
    ESS 行去报。R̂ 不是有限数（链太短算不出）时同样拦下：收敛无从确认。
    """
    rhats = _finite_values(mcmc.split_rhat)
    if rhats is None:
        return "split-R̂ 含非有限值，无法确认链已收敛"
    worst = max(rhats)
    if worst < SPLIT_RHAT_LIMIT:
        return None
    return f"链未收敛（split-R̂ {worst:.3f} ≥ {SPLIT_RHAT_LIMIT:.2f}）"
