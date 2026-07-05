"""当日レースの予測と買い目提示。

当日の番組表を取得→過去データと結合して特徴量を作成→学習済みモデルで
1着確率を予測。--odds を付けると締切前レースの単勝オッズを公式サイトから
取得し、期待値(EV = 確率×オッズ)がプラスの買い目だけを提示する。
"""

import argparse
import itertools
import pickle
from datetime import date as date_cls
from datetime import datetime, timedelta

import pandas as pd

from .dataset import CATEGORICAL, FEATURES, KLASS_MAP, add_history, load_raw
from .download import DATA_DIR, fetch_day
from .odds import fetch_3tan_odds, fetch_tansho_odds
from .parse import parse_b_file
from .train import MODEL_PATH, normalize_by_race

VENUES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}

ENTRY_COLS = ["race_id", "lane", "regno", "name", "age", "branch", "weight",
              "klass", "nat_win", "nat_2in", "loc_win", "loc_2in",
              "motor_no", "motor_2in", "boat_no", "boat_2in"]
RACE_COLS = ["race_id", "date", "jcd", "rno", "title", "distance", "deadline"]


def build_today(target: str) -> pd.DataFrame:
    """当日分の特徴量付きDataFrameを返す。"""
    d = date_cls.fromisoformat(target)
    path = fetch_day("B", d, DATA_DIR / "B")
    if path is None:
        raise SystemExit(f"{target} の番組表がまだ公開されていない(または開催なし)")
    races, entries = parse_b_file(path)
    today = pd.DataFrame(entries, columns=ENTRY_COLS).merge(
        pd.DataFrame(races, columns=RACE_COLS), on="race_id")

    hist = load_raw()
    hist = hist[hist["date"] < target]
    for col in ("rank", "course", "st", "flying"):
        today[col] = pd.NA
    combined = pd.concat([hist, today], ignore_index=True)
    combined["klass_num"] = combined["klass"].map(KLASS_MAP)
    combined = add_history(combined)
    out = combined[combined["date"] == target].copy()
    return out


def harville(p: dict[int, float]) -> dict[tuple[int, int, int], float]:
    """勝率から3連単の各並びの確率を組み立てる(Harville近似)。

    競艇はイン有利の構造があるため厳密には成り立たないが、
    1次近似としては実用になる。過信しないこと。
    """
    out = {}
    for i, j, k in itertools.permutations(p, 3):
        d1 = 1 - p[i]
        d2 = 1 - p[i] - p[j]
        if d1 <= 0 or d2 <= 0:
            continue
        out[(i, j, k)] = p[i] * p[j] / d1 * p[k] / d2
    return out


def predict_day(target: str, jcd: int | None, with_odds: bool,
                ev_min: float, kelly_frac: float, bankroll: int) -> None:
    df = build_today(target)
    if jcd:
        df = df[df["jcd"] == jcd]
    if df.empty:
        raise SystemExit("対象レースがない")

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    df["p_raw"] = model.predict(df[FEATURES])
    df["p"] = normalize_by_race(df, "p_raw")

    now = datetime.now()
    recs = []
    for rid, race in df.sort_values(["jcd", "rno"]).groupby("race_id", sort=False):
        race = race.sort_values("p", ascending=False)
        top = race.iloc[0]
        jname = VENUES.get(int(top["jcd"]), "?")
        line = " ".join(f"{int(r.lane)}号艇{r.p:.0%}" for r in race.itertuples())
        deadline = top["deadline"] or "?"
        print(f"{jname}{int(top['rno']):>2}R 締切{deadline}  {line}")

        if not with_odds:
            continue
        # 締切済み・直前レースはオッズを取らない
        if target == now.strftime("%Y-%m-%d") and deadline != "?":
            dl = datetime.strptime(f"{target} {deadline}", "%Y-%m-%d %H:%M")
            if dl < now + timedelta(minutes=3):
                continue
        jn, rn = int(top["jcd"]), int(top["rno"])
        probs = {int(r.lane): r.p for r in race.itertuples()}

        t_odds = fetch_tansho_odds(jn, rn, target)
        for lane, p_ in probs.items():
            o = t_odds.get(lane)
            if o and o > 1 and p_ * o >= ev_min:
                recs.append((jname, rn, deadline, f"単勝 {lane}", p_, o, p_ * o))

        # Harville近似は人気薄の並びを過大評価するので、3連単は
        # 確率5%以上の並びに限定し、1レースあたり上位3点まで
        s_odds = fetch_3tan_odds(jn, rn, target)
        cands = []
        for combo, p_ in harville(probs).items():
            o = s_odds.get(combo)
            if o and o > 1 and p_ >= 0.05 and p_ * o >= ev_min:
                cands.append((jname, rn, deadline,
                              f"3連単 {combo[0]}-{combo[1]}-{combo[2]}", p_, o, p_ * o))
        recs.extend(sorted(cands, key=lambda x: -x[6])[:3])

    if with_odds:
        print("\n=== EVプラスの買い目 (EV = 予測確率×オッズ) ===")
        print("※オッズは締切直前まで動く。締切10分前より早い時点の結果は参考値。")
        shown = 0
        for jname, rno, dl, bet, p, o, ev in sorted(recs, key=lambda x: -x[6]):
            kelly = (p * o - 1) / (o - 1)
            stake = int(max(kelly * kelly_frac, 0) * bankroll // 100 * 100)
            if stake < 100:
                continue
            shown += 1
            print(f"{jname}{rno:>2}R 締切{dl}  {bet:<14} "
                  f"予測{p:5.1%} オッズ{o:6.1f} EV={ev:.2f}  推奨{stake}円")
        if not shown:
            print("(該当なし。オッズに歪みが出ていない)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=date_cls.today().isoformat())
    p.add_argument("--jcd", type=int, help="場コード(1-24)で絞り込み")
    p.add_argument("--odds", action="store_true", help="公式サイトから単勝オッズを取得しEV計算")
    p.add_argument("--ev-min", type=float, default=1.2)
    p.add_argument("--kelly", type=float, default=0.25, help="ケリー基準の掛け率")
    p.add_argument("--bankroll", type=int, default=10000)
    a = p.parse_args()
    predict_day(a.date, a.jcd, a.odds, a.ev_min, a.kelly, a.bankroll)
