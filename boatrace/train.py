"""LightGBMで各艇の1着確率を学習する。

時系列で分割(直近を検証・テストに)し、リークを避ける。
確率はレース内で合計1になるよう正規化して評価する。
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .dataset import CATEGORICAL, FEATURES, build_dataset

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"


def normalize_by_race(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] / df.groupby("race_id")[col].transform("sum")


def time_split(df: pd.DataFrame, test_months: int = 3, valid_months: int = 2):
    dates = np.sort(df["date"].unique())
    last = pd.Timestamp(dates[-1])
    test_from = (last - pd.DateOffset(months=test_months)).strftime("%Y-%m-%d")
    valid_from = (last - pd.DateOffset(months=test_months + valid_months)).strftime("%Y-%m-%d")
    return (df[df["date"] < valid_from],
            df[(df["date"] >= valid_from) & (df["date"] < test_from)],
            df[df["date"] >= test_from])


def train(df: pd.DataFrame | None = None, save: bool = True):
    if df is None:
        df = build_dataset()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    tr, va, te = time_split(df)
    print(f"train {tr['date'].min()}..{tr['date'].max()} ({len(tr)})")
    print(f"valid {va['date'].min()}..{va['date'].max()} ({len(va)})")
    print(f"test  {te['date'].min()}..{te['date'].max()} ({len(te)})")

    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
        verbose=-1,
    )
    dtr = lgb.Dataset(tr[FEATURES], tr["win"], categorical_feature=CATEGORICAL)
    dva = lgb.Dataset(va[FEATURES], va["win"], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])

    te = te.copy()
    te["p_raw"] = model.predict(te[FEATURES])
    te["p"] = normalize_by_race(te, "p_raw")

    # 検証データで着順モデルのλをフィット(テストには触れない)
    from .ordermodel import fit_lambdas
    va = va.copy()
    va["p_raw"] = model.predict(va[FEATURES])
    va["p"] = normalize_by_race(va, "p_raw")
    lam2, lam3 = fit_lambdas(va)
    print(f"\n着順モデル: λ2={lam2:.2f} λ3={lam3:.2f} (1.0=Harville)")
    _eval_trifecta(te, lam2, lam3)

    print(f"\nbest_iteration: {model.best_iteration}")
    print(f"test logloss (raw): {log_loss(te['win'], te['p_raw']):.4f}")
    print(f"test logloss (normalized): {log_loss(te['win'], te['p']):.4f}")
    print(f"test AUC: {roc_auc_score(te['win'], te['p']):.4f}")

    top = te.loc[te.groupby("race_id")["p"].idxmax()]
    lane1 = te[te["lane"] == 1]
    print(f"top-1的中率(モデル): {top['win'].mean():.3f}")
    print(f"top-1的中率(常に1号艇): {lane1['win'].mean():.3f}")

    imp = pd.Series(model.feature_importance("gain"), index=FEATURES)
    print("\nfeature importance (gain):")
    print(imp.sort_values(ascending=False).round(0).to_string())

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "features": FEATURES,
                         "lambda2": lam2, "lambda3": lam3}, f)
        print(f"\nsaved: {MODEL_PATH}")
    return model, te


def _eval_trifecta(te: pd.DataFrame, lam2: float, lam3: float) -> None:
    """テスト期間の実際の3連単に対する平均対数尤度でHarvilleと比較する。"""
    from .ordermodel import fit_lambdas, trifecta_probs
    _ = fit_lambdas  # 明示: λはvalidでフィット済み
    lls = {"harville": [], "benter": []}
    for _, g in te.groupby("race_id"):
        ranks = g.set_index("lane")["rank"]
        p = g.set_index("lane")["p"].to_dict()
        try:
            combo = (int(ranks[ranks == 1].index[0]),
                     int(ranks[ranks == 2].index[0]),
                     int(ranks[ranks == 3].index[0]))
        except IndexError:
            continue
        h = trifecta_probs(p, 1.0, 1.0).get(combo)
        b = trifecta_probs(p, lam2, lam3).get(combo)
        if h and b:
            lls["harville"].append(np.log(h))
            lls["benter"].append(np.log(b))
    print(f"3連単 平均LL: Harville {np.mean(lls['harville']):.4f} → "
          f"Benter {np.mean(lls['benter']):.4f} (高いほど良い, n={len(lls['benter'])})")


if __name__ == "__main__":
    train()
