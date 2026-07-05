"""学習用データセットの構築。

1行 = 1レースの1艇。目的変数 win は「1着になったか」。
特徴量は番組表に載っている情報(=レース前に確実に手に入る情報)と、
過去の結果から計算するリーク無しの履歴集計のみを使う。
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .parse import DB_PATH

KLASS_MAP = {"B2": 0, "B1": 1, "A2": 2, "A1": 3}

# レース内で相対化する列。Bファイルの事前情報+履歴集計+展示タイム
# (展示は過去分がKファイル、当日分が公式の直前情報ページから取れる)
REL_COLS = ["nat_win", "loc_win", "motor_2in", "boat_2in", "weight",
            "klass_num", "racer_wr", "racer_st_mean", "racer_lane_wr", "tenji"]

FEATURES = [
    "lane", "jcd", "klass_num", "age", "weight",
    "nat_win", "nat_2in", "loc_win", "loc_2in", "motor_2in", "boat_2in",
    "racer_st_mean", "racer_lane_wr", "racer_wr", "racer_n", "tenji",
    *[f"{c}_rank" for c in REL_COLS],
    *[f"{c}_z" for c in REL_COLS],
]
CATEGORICAL = ["lane", "jcd"]


def load_raw(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        """
        SELECT e.*, r.date, r.jcd, r.rno,
               res.rank, res.tenji, res.course, res.st, res.flying
        FROM entries e
        JOIN races r USING (race_id)
        LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
        ORDER BY r.date, r.jcd, r.rno, e.lane
        """,
        con,
    )
    con.close()
    return df


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """選手ごとの履歴集計(当該レースより前の情報のみ)。

    同日内のレース順は正確な発走時刻が無いためレース番号順とみなす。
    shift(1)で必ず「過去のレースまで」に限定する。
    """
    df = df.sort_values(["regno", "date", "jcd", "rno"]).reset_index(drop=True)

    won = (df["rank"] == 1).astype(float)
    ran = df["rank"].notna().astype(float)

    def past_sum(s: pd.Series, keys) -> pd.Series:
        return s.groupby(keys).transform(lambda x: x.shift(1).cumsum())

    df["racer_n"] = df.groupby("regno", sort=False).cumcount()
    df["racer_wr"] = past_sum(won, df["regno"]) / past_sum(ran, df["regno"])

    st = df["st"].where(df["st"] > 0)  # F(マイナス値)は平均から除外
    df["racer_st_mean"] = st.groupby(df["regno"]).transform(
        lambda x: x.shift(1).expanding().mean())

    lane_keys = [df["regno"], df["lane"]]
    df["racer_lane_wr"] = past_sum(won, lane_keys) / past_sum(ran, lane_keys)

    return df.sort_values(["date", "jcd", "rno", "lane"]).reset_index(drop=True)


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内での相対位置(6艇中の強さ)。

    同じ勝率でも他艇の顔ぶれで意味が変わるので、絶対値に加えて
    レース内順位(rank, 1=最上位)と標準化値(z)を持たせる。
    Bファイルの事前情報だけで計算でき、学習時も予測時も同一に得られる。
    """
    g = df.groupby("race_id")
    for col in REL_COLS:
        df[f"{col}_rank"] = g[col].rank(ascending=False, method="min")
        mean, std = g[col].transform("mean"), g[col].transform("std")
        df[f"{col}_z"] = ((df[col] - mean) / std).fillna(0.0)
    return df


def build_dataset(db_path: Path = DB_PATH) -> pd.DataFrame:
    df = load_raw(db_path)
    df["klass_num"] = df["klass"].map(KLASS_MAP)
    df = add_history(df)
    df = add_relative_features(df)
    df["win"] = (df["rank"] == 1).astype(int)
    # 結果ファイルが無い(未開催/取得漏れ)レースは学習から外す
    has_result = df.groupby("race_id")["rank"].transform(lambda x: x.notna().any())
    return df[has_result].reset_index(drop=True)


if __name__ == "__main__":
    d = build_dataset()
    print(d.shape)
    print(d[FEATURES + ["win"]].describe())
