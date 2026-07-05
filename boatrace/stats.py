"""選手履歴の集計スナップショット。

クラウド実行では2年分のDBを持ち歩けないため、履歴特徴量
(通算勝率・平均ST・枠別勝率)の計算に必要な合計値だけを
CSVに保存し、毎晩の採点時に前日分を加算して更新する。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .parse import DB_PATH

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STATS_PATH = STATE_DIR / "racer_stats.csv"

BASE_COLS = ["n", "ran", "wins", "st_sum", "st_n"]
LANE_COLS = [f"{c}{i}" for i in range(1, 7) for c in ("ran", "win")]


def build_stats(db_path: Path = DB_PATH) -> pd.DataFrame:
    """DB全体からスナップショットを作る(初期化用)。"""
    import sqlite3
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT regno, lane, rank, st FROM results", con)
    con.close()

    g = df.groupby("regno")
    out = pd.DataFrame({
        "n": g.size(),
        "ran": g["rank"].count(),
        "wins": g["rank"].apply(lambda s: (s == 1).sum()),
        "st_sum": g["st"].apply(lambda s: s[s > 0].sum()),
        "st_n": g["st"].apply(lambda s: (s > 0).sum()),
    })
    for i in range(1, 7):
        sub = df[df["lane"] == i].groupby("regno")
        out[f"ran{i}"] = sub["rank"].count()
        out[f"win{i}"] = sub["rank"].apply(lambda s: (s == 1).sum())
    out = out.fillna(0).astype({c: int for c in ["n", "ran", "wins", "st_n"] + LANE_COLS})
    return out.reset_index()


def load_stats() -> pd.DataFrame:
    return pd.read_csv(STATS_PATH).set_index("regno")


def save_stats(stats: pd.DataFrame) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    stats.reset_index().to_csv(STATS_PATH, index=False)


def update_stats(stats: pd.DataFrame, results: list[tuple]) -> pd.DataFrame:
    """1日分の結果 (race_id, lane, regno, rank_raw, rank, course, st, flying)
    をスナップショットに加算する。冪等ではないので同じ日を二度足さないこと。
    """
    for _, lane, regno, _, rank, _, st, _ in results:
        if regno not in stats.index:
            stats.loc[regno] = 0
        stats.loc[regno, "n"] += 1
        if rank is not None:
            stats.loc[regno, "ran"] += 1
            stats.loc[regno, f"ran{lane}"] += 1
            if rank == 1:
                stats.loc[regno, "wins"] += 1
                stats.loc[regno, f"win{lane}"] += 1
        if st is not None and st > 0:
            stats.loc[regno, "st_sum"] += st
            stats.loc[regno, "st_n"] += 1
    return stats


def attach_features(today: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """当日出走表に履歴特徴量を付ける。初出走の選手はNaNのまま。"""
    s = stats.reindex(today["regno"])
    today = today.copy()
    today["racer_n"] = s["n"].to_numpy()
    today["racer_wr"] = (s["wins"] / s["ran"].replace(0, np.nan)).to_numpy()
    today["racer_st_mean"] = (s["st_sum"] / s["st_n"].replace(0, np.nan)).to_numpy()
    lane_ran = np.array([s[f"ran{int(l)}"].to_numpy()[i]
                         for i, l in enumerate(today["lane"])], dtype=float)
    lane_win = np.array([s[f"win{int(l)}"].to_numpy()[i]
                         for i, l in enumerate(today["lane"])], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        today["racer_lane_wr"] = np.where(lane_ran > 0, lane_win / lane_ran, np.nan)
    return today


if __name__ == "__main__":
    stats = build_stats()
    save_stats(stats.set_index("regno"))
    print(f"saved {STATS_PATH} ({len(stats)} racers)")
