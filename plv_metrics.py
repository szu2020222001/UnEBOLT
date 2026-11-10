"""单扫描、单频段的 PLV 评估。

矩阵已由调用方展平成严格上三角边，不含对角线及重复对称元素。
常量向量等导致的无定义相关保留 NaN，同时报告有效比例。
"""
import numpy as np

def corr_rows(a, b, eps=1e-8):
    """沿最后一维求 Pearson r，其余维度保持不变。

    eps 是两向量中心化后范数乘积的阈值，不是逐向量方差阈值。
    分母太小时返回 NaN；不通过给分母加常数将无定义相关变成零。
    """
    # 使用 float64 中心化，避免 float32 常量序列的均值舍入产生虚假方差。
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    # 中心化后，内积除以范数乘积就是 Pearson 相关系数。
    a, b = a-a.mean(-1, keepdims=True), b-b.mean(-1, keepdims=True)
    den = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    return np.divide((a*b).sum(-1), den, out=np.full(den.shape, np.nan), where=den>eps)


def fisher_mean(r):
    """排除非有限值，先 atanh 转到 Fisher-z 空间平均，再 tanh 返回 r。

    裁剪避免精确 ±1 导致无穷大；若没有有效值则返回 NaN。
    """
    r = np.asarray(r)
    return float(np.tanh(np.arctanh(np.clip(r[np.isfinite(r)], -1+1e-7, 1-1e-7)).mean())) if np.isfinite(r).any() else float('nan')


def metrics(y, pred, template):
    """计算一份扫描、一个频段的指标字典。

    y/pred: [N, E]，N 是测试窗口数，26 通道时 E=325。
    template: [E]，仅由训练集估计，广播为每个窗口的固定预测。
    当前数值误差对窗口和边等权平均；相关先逐窗口/边计算再 Fisher 汇总。
    """
    # MSE/MAE 衡量 PLV 绝对数值误差；模板为静态先验对照。
    error = pred-y
    mse = float(np.mean(error**2))
    base = float(np.mean((template-y)**2))
    # 每个窗口跨 E 条边计算相关，得到 [N]：连接空间分布是否一致。
    spatial = corr_rows(y, pred)
    # 转置后沿 N 个窗口计算相关，得到 [E]：每条边动态是否跟得上。
    temporal = corr_rows(y.T, pred.T)
    # delta_mse > 0 / skill > 0 表示优于模板；模板误差近零时 skill 无定义。
    # 有效比例必须随相关分数一起检查，避免忽略退化预测导致分数偏高。
    # 平均模板沿时间为常量，因此其 temporal_r 通常为 NaN。
    out = dict(mae=float(np.abs(error).mean()), mse=mse, template_mse=base,
               delta_mse=base-mse, skill=1-mse/base if base>1e-12 else np.nan,
               spatial_r=fisher_mean(spatial), temporal_r=fisher_mean(temporal),
               spatial_valid=float(np.isfinite(spatial).mean()),
               temporal_valid=float(np.isfinite(temporal).mean()))
    # 各窗口分别保留最强 ceil(q*E) 条边，再比较真实与预测集合。
    # 两集合等大，precision=recall=F1=交集大小/k；扫描分数取窗口算术平均。
    # 独立随机选择的期望 F1 为 k/E，所以不同 q 的值不能直接比较高低。
    for q in (.25, .5):
        k = int(np.ceil(y.shape[-1]*q))
        # 稳定降序排序：PLV 并列时优先原始索引较小的边，保证恰好取 k 条。
        gi = np.argsort(-y, axis=-1, kind='stable')[:, :k]
        hi = np.argsort(-pred, axis=-1, kind='stable')[:, :k]
        out[f'f1_{q}'] = float(np.mean([len(np.intersect1d(g,h))/k for g,h in zip(gi,hi)]))
    return out
