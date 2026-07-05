"""全国全場での自動ペーパートレード検証。

実際の投票は一切行わない。公開ページの読み取りだけで
「締切直前のオッズでEVプラスの買い目を買ったつもり」を記録し、
翌日の公式結果で採点して戦略の実測ROIを積み上げる。

サブコマンド:
  run     1日分の常駐実行(朝に起動→最終レース締切まで巡回→終了)
  settle  未採点の仮想投票を公式結果で採点
  report  累積成績レポート
"""

import argparse
import pickle
import sqlite3
import time
from datetime import date, datetime, timedelta

from .dataset import CATEGORICAL, FEATURES
from .download import DATA_DIR, fetch_day
from .odds import fetch_3tan_odds, fetch_tansho_odds
from .parse import DB_PATH, build_db
from .predict import VENUES, build_today
from .train import MODEL_PATH, normalize_by_race

# 戦略パラメータ(バックテスト・predict と揃える)
EV_MIN = 1.2
P_MIN_3TAN = 0.05
MAX_BETS_PER_RACE_3TAN = 3
VIRTUAL_BANKROLL = 10000  # 仮想資金。実際の金は一切動かない
KELLY_FRAC = 0.25
ODDS_WINDOW_MIN = 3   # 締切何分前まで取得するか
ODDS_WINDOW_MAX = 12  # 締切何分前から取得するか

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_bets (
    race_id TEXT, bet_type TEXT, combo TEXT,
    p REAL, odds REAL, ev REAL, stake INTEGER,
    placed_at TEXT, ret INTEGER, settled_at TEXT,
    PRIMARY KEY (race_id, bet_type, combo)
);
CREATE TABLE IF NOT EXISTS paper_checked (
    race_id TEXT PRIMARY KEY, checked_at TEXT, n_bets INTEGER
);
"""


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.executescript(SCHEMA)
    return con


def _stake(p: float, o: float) -> int:
    kelly = (p * o - 1) / (o - 1)
    return int(max(kelly * KELLY_FRAC, 0) * VIRTUAL_BANKROLL // 100 * 100)


# 高オッズ帯のEVはモデル確率のノイズが増幅された幻が多いので上限を切る
MAX_ODDS_TANSHO = 15.0
MAX_ODDS_3TAN = 60.0


def select_from_odds(probs: dict[int, float], t_odds: dict, s_odds: dict,
                     lam2: float = 1.0, lam3: float = 1.0):
    """取得済みオッズからEVプラスの買い目を選ぶ(通信しない純粋関数)。"""
    from .ordermodel import trifecta_probs

    bets = []
    for lane, p in probs.items():
        o = t_odds.get(lane)
        if (o and 1 < o <= MAX_ODDS_TANSHO and p * o >= EV_MIN
                and _stake(p, o) >= 100):
            bets.append(("単勝", str(lane), p, o))

    cands = []
    for (a, b, c), p in trifecta_probs(probs, lam2, lam3).items():
        o = s_odds.get((a, b, c))
        if (o and 1 < o <= MAX_ODDS_3TAN and p >= P_MIN_3TAN
                and p * o >= EV_MIN and _stake(p, o) >= 100):
            cands.append(("３連単", f"{a}-{b}-{c}", p, o))
    cands.sort(key=lambda x: -(x[2] * x[3]))
    bets.extend(cands[:MAX_BETS_PER_RACE_3TAN])
    return bets


def _select_bets(probs: dict[int, float], jcd: int, rno: int, target: str,
                 lam2: float = 1.0, lam3: float = 1.0):
    """締切直前オッズを取得してEVプラスの買い目を返す。"""
    t_odds = fetch_tansho_odds(jcd, rno, target)
    s_odds = fetch_3tan_odds(jcd, rno, target)
    return select_from_odds(probs, t_odds, s_odds, lam2, lam3)


def run(target: str | None = None) -> None:
    target = target or date.today().isoformat()
    settle()  # 前日までの分を先に採点

    # モデルが1週間より古ければ最新データで再学習
    if (not MODEL_PATH.exists()
            or MODEL_PATH.stat().st_mtime < time.time() - 7 * 86400):
        print("モデルを再学習...", flush=True)
        from .train import train
        train()

    df = build_today(target)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    lam2 = bundle.get("lambda2", 1.0)
    lam3 = bundle.get("lambda3", 1.0)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    df["p_raw"] = model.predict(df[FEATURES])
    df["p"] = normalize_by_race(df, "p_raw")

    races = {}
    for rid, race in df.groupby("race_id"):
        top = race.iloc[0]
        if not top["deadline"]:
            continue
        races[rid] = {
            "jcd": int(top["jcd"]), "rno": int(top["rno"]),
            "deadline": datetime.strptime(f"{target} {top['deadline']}",
                                          "%Y-%m-%d %H:%M"),
            "probs": {int(r.lane): r.p for r in race.itertuples()},
        }

    con = _con()
    checked = {r[0] for r in con.execute("SELECT race_id FROM paper_checked")}
    pending = {rid: v for rid, v in races.items() if rid not in checked}
    print(f"{target}: {len(races)}レース (未処理 {len(pending)})", flush=True)
    if not pending:
        con.close()
        return
    last_deadline = max(v["deadline"] for v in pending.values())

    while pending and datetime.now() < last_deadline + timedelta(minutes=5):
        now = datetime.now()
        due = [rid for rid, v in pending.items()
               if now + timedelta(minutes=ODDS_WINDOW_MIN) <= v["deadline"]
               <= now + timedelta(minutes=ODDS_WINDOW_MAX)]
        expired = [rid for rid, v in pending.items()
                   if v["deadline"] < now + timedelta(minutes=ODDS_WINDOW_MIN)]
        for rid in expired:  # 取り逃したレースは記録せず終了扱い
            con.execute("INSERT OR IGNORE INTO paper_checked VALUES (?,?,?)",
                        (rid, now.isoformat(timespec="seconds"), -1))
            del pending[rid]
        for rid in due:
            v = pending.pop(rid)
            try:
                bets = _select_bets(v["probs"], v["jcd"], v["rno"], target,
                                    lam2, lam3)
            except Exception as e:
                print(f"NG {rid}: {e}", flush=True)
                continue
            ts = datetime.now().isoformat(timespec="seconds")
            for bt, combo, p, o in bets:
                con.execute(
                    "INSERT OR IGNORE INTO paper_bets"
                    " (race_id,bet_type,combo,p,odds,ev,stake,placed_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (rid, bt, combo, p, o, p * o, _stake(p, o), ts))
            con.execute("INSERT OR IGNORE INTO paper_checked VALUES (?,?,?)",
                        (rid, ts, len(bets)))
            if bets:
                jname = VENUES.get(v["jcd"], "?")
                for bt, combo, p, o in bets:
                    print(f"[仮想] {jname}{v['rno']:>2}R {bt} {combo} "
                          f"p={p:.1%} odds={o:.1f} EV={p*o:.2f}", flush=True)
        con.commit()
        time.sleep(45)
    con.commit()
    con.close()
    print("本日分終了", flush=True)
    report()


def settle() -> None:
    """未採点の仮想投票を公式結果(払戻)で採点する。"""
    con = _con()
    days = [r[0] for r in con.execute(
        "SELECT DISTINCT substr(race_id,1,10) FROM paper_bets WHERE ret IS NULL")]
    if not days:
        con.close()
        return
    for d in days:
        try:
            fetch_day("K", date.fromisoformat(d), DATA_DIR / "K")
            fetch_day("B", date.fromisoformat(d), DATA_DIR / "B")
        except Exception as e:
            print(f"結果取得NG {d}: {e}", flush=True)
    con.close()
    build_db()  # 増分取り込み

    con = _con()
    ts = datetime.now().isoformat(timespec="seconds")
    n = 0
    for rid, bt, combo, stake in con.execute(
            "SELECT race_id, bet_type, combo, stake FROM paper_bets"
            " WHERE ret IS NULL").fetchall():
        # 結果がまだ無ければ次回に持ち越し
        has = con.execute("SELECT 1 FROM results WHERE race_id=? LIMIT 1",
                          (rid,)).fetchone()
        if not has:
            continue
        row = con.execute(
            "SELECT amount FROM payouts WHERE race_id=? AND bet_type=? AND combo=?",
            (rid, bt, combo)).fetchone()
        ret = stake * (row[0] if row else 0) // 100
        con.execute("UPDATE paper_bets SET ret=?, settled_at=?"
                    " WHERE race_id=? AND bet_type=? AND combo=?",
                    (ret, ts, rid, bt, combo))
        n += 1
    con.commit()
    con.close()
    if n:
        print(f"{n}件を採点", flush=True)


def report() -> None:
    import pandas as pd
    con = _con()
    df = pd.read_sql("SELECT * FROM paper_bets WHERE ret IS NOT NULL", con)
    unsettled = con.execute(
        "SELECT COUNT(*) FROM paper_bets WHERE ret IS NULL").fetchone()[0]
    con.close()
    print("\n=== ペーパートレード累積成績 (実際の金銭は一切動いていない) ===")
    if df.empty:
        print(f"採点済みなし (未採点 {unsettled}件)")
        return
    df["date"] = df["race_id"].str[:10]
    df["hit"] = df["ret"] > 0
    # 均一100円換算は投票時オッズではなく実際の確定払戻で計算する
    df["unit_ret"] = (df["ret"] / df["stake"] * 100).fillna(0)

    def agg(g):
        return pd.Series({
            "bets": len(g), "hit_rate": g["hit"].mean(),
            "staked": g["stake"].sum(), "returned": g["ret"].sum(),
            "roi_kelly": g["ret"].sum() / max(g["stake"].sum(), 1),
            "roi_flat100": g["unit_ret"].sum() / (100 * len(g)),
        })

    print(f"未採点 {unsettled}件")
    print("\n[式別]")
    print(df.groupby("bet_type").apply(agg, include_groups=False).round(3).to_string())
    print("\n[EV帯別]")
    df["ev_bucket"] = pd.cut(df["ev"], [1.2, 1.5, 2.0, 3.0, 100])
    print(df.groupby("ev_bucket", observed=True).apply(
        agg, include_groups=False).round(3).to_string())
    print("\n[日別]")
    print(df.groupby("date").apply(agg, include_groups=False).round(3).to_string())
    total = agg(df)
    print(f"\n合計: {int(total['bets'])}点 的中率{total['hit_rate']:.1%} "
          f"ROI(ケリー配分){total['roi_kelly']:.3f} ROI(均一100円){total['roi_flat100']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "settle", "report"])
    p.add_argument("--date", default=None)
    a = p.parse_args()
    {"run": lambda: run(a.date), "settle": settle, "report": report}[a.cmd]()
