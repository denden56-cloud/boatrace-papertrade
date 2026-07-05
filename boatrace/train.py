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
            pickle.dump({"model": model, "features": FEATURES}, f)
        print(f"\nsaved: {MODEL_PATH}")
    return model, te


if __name__ == "__main__":
    train()
