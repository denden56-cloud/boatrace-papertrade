"""テスト期間の実際の払戻金で投票戦略のROIを検証する。

歴史的オッズは配布データに含まれないため、EVフィルタではなく
「モデルの予測確率がしきい値以上のときだけ買う」戦略を検証する。
払戻は的中時のみ判明すればROI計算には十分。
賭け金は1点100円固定。
"""

import sqlite3

import pandas as pd

from .dataset import build_dataset
from .parse import DB_PATH
from .train import normalize_by_race, time_split, train


def load_payouts(race_ids: set[str]) -> dict[tuple[str, str], dict[str, int]]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT race_id, bet_type, combo, amount FROM payouts").fetchall()
    con.close()
    table: dict[tuple[str, str], dict[str, int]] = {}
    for rid, bt, combo, amount in rows:
        if rid in race_ids:
            table.setdefault((rid, bt), {})[combo] = amount
    return table


def simulate(picks: pd.DataFrame, payouts, bet_type: str, combo_fn, thresholds):
    """picks: 1レース1行、p1(本命の確率)と買い目情報を持つDataFrame。"""
    out = []
    for th in thresholds:
        sel = picks[picks["p1"] >= th]
        if len(sel) == 0:
            out.append((th, 0, float("nan"), float("nan")))
            continue
        returns = [
            payouts.get((r.race_id, bet_type), {}).get(combo_fn(r), 0)
            for r in sel.itertuples()
        ]
        n = len(sel)
        hit = sum(1 for x in returns if x > 0) / n
        roi = sum(returns) / (100 * n)
        out.append((th, n, hit, roi))
    return pd.DataFrame(out, columns=["threshold", "bets", "hit_rate", "roi"])


def run():
    df = build_dataset()
    model, te = train(df, save=True)

    # レースごとに確率順で上位3艇を並べる
    te = te.sort_values(["race_id", "p"], ascending=[True, False])
    top3 = te.groupby("race_id").head(3).copy()
    top3["pos"] = top3.groupby("race_id").cumcount()
    lanes = top3.pivot(index="race_id", columns="pos", values="lane").astype(int)
    picks = pd.DataFrame({
        "p1": top3.groupby("race_id")["p"].first(),
        "lane1": lanes[0], "lane2": lanes[1], "lane3": lanes[2],
    }).reset_index()

    payouts = load_payouts(set(picks["race_id"]))
    ths = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print("\n=== 単勝(本命1点) ===")
    print(simulate(picks, payouts, "単勝",
                   lambda r: str(r.lane1), ths).to_string(index=False))
    print("\n=== ２連単(本命→対抗 1点) ===")
    print(simulate(picks, payouts, "２連単",
                   lambda r: f"{r.lane1}-{r.lane2}", ths).to_string(index=False))
    print("\n=== ３連単(上位3艇の並び 1点) ===")
    print(simulate(picks, payouts, "３連単",
                   lambda r: f"{r.lane1}-{r.lane2}-{r.lane3}", ths).to_string(index=False))

    # 予測確率の帯ごとの実測1着率(キャリブレーション確認)
    top = te.loc[te.groupby("race_id")["p"].idxmax()].copy()
    top["bucket"] = pd.cut(top["p"], [0, .3, .4, .5, .6, .7, .8, 1.0])
    cal = top.groupby("bucket", observed=True).agg(
        n=("win", "size"), pred=("p", "mean"), actual=("win", "mean"))
    print("\n=== キャリブレーション(本命の予測確率 vs 実測1着率) ===")
    print(cal.round(3).to_string())


if __name__ == "__main__":
    run()
