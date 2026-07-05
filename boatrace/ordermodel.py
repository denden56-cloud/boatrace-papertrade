"""着順確率モデル(Benter型のHarville修正)。

素朴なHarville式は「1着確率が高い艇は2着・3着にも入りやすい」を
そのまま外挿するが、実際には過大評価になることが知られている
(Benter 1994)。勝率に減衰指数λをかけた

  P(jが2着 | iが1着) = p_j^λ2 / Σ_{k≠i} p_k^λ2
  P(kが3着 | i,j)    = p_k^λ3 / Σ_{m≠i,j} p_m^λ3

の形にし、λを実データの対数尤度最大化でフィットする。
λ=1がHarville。λ<1なら「2着以降は勝率ほど差がつかない」。
"""

import numpy as np
import pandas as pd

LAMBDA_GRID = np.arange(0.05, 1.55, 0.05)


def _fit_lambda(p_mat: np.ndarray, mask_prev: np.ndarray,
                target_idx: np.ndarray) -> float:
    """1つの着順段のλをグリッドサーチでフィットする。

    p_mat: (races, 6) 正規化済み勝率
    mask_prev: (races, 6) 前の着に入った艇のマスク(分母から除外)
    target_idx: (races,) その着に実際に入った艇の列番号
    """
    best_lam, best_ll = 1.0, -np.inf
    rows = np.arange(len(p_mat))
    for lam in LAMBDA_GRID:
        w = np.where(mask_prev, 0.0, np.power(p_mat, lam))
        denom = w.sum(axis=1)
        num = w[rows, target_idx]
        ok = (num > 0) & (denom > 0)
        ll = np.log(num[ok] / denom[ok]).mean()
        if ll > best_ll:
            best_ll, best_lam = ll, float(lam)
    return best_lam


def fit_lambdas(df: pd.DataFrame) -> tuple[float, float]:
    """検証データ(p列=正規化勝率, rank列=実着順)からλ2, λ3を推定する。"""
    complete = df.groupby("race_id").filter(
        lambda g: len(g) == 6 and set(g["rank"].dropna()) >= {1, 2, 3})
    wide_p = complete.pivot_table(index="race_id", columns="lane", values="p")
    wide_r = complete.pivot_table(index="race_id", columns="lane", values="rank")
    wide_p, wide_r = wide_p.dropna(), wide_r.reindex(wide_p.index)
    p = wide_p.to_numpy()
    r = wide_r.to_numpy()

    first = np.argmax(r == 1, axis=1)
    second = np.argmax(r == 2, axis=1)
    third = np.argmax(r == 3, axis=1)
    n = len(p)
    m1 = np.zeros_like(p, dtype=bool)
    m1[np.arange(n), first] = True
    lam2 = _fit_lambda(p, m1, second)
    m2 = m1.copy()
    m2[np.arange(n), second] = True
    lam3 = _fit_lambda(p, m2, third)
    return lam2, lam3


def trifecta_probs(p: dict[int, float], lam2: float = 1.0,
                   lam3: float = 1.0) -> dict[tuple[int, int, int], float]:
    """勝率から3連単の各並びの確率を組み立てる(λ=1でHarville相当)。"""
    import itertools
    w2 = {i: v ** lam2 for i, v in p.items() if v > 0}
    w3 = {i: v ** lam3 for i, v in p.items() if v > 0}
    out = {}
    s2_all = sum(w2.values())
    s3_all = sum(w3.values())
    for i in p:
        if i not in w2:
            continue
        s2 = s2_all - w2[i]
        for j in p:
            if j == i or j not in w2 or s2 <= 0:
                continue
            s3 = s3_all - w3[i] - w3[j]
            for k in p:
                if k in (i, j) or k not in w3 or s3 <= 0:
                    continue
                out[(i, j, k)] = p[i] * (w2[j] / s2) * (w3[k] / s3)
    return out
