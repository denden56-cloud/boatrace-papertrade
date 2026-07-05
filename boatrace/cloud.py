"""GitHub Actions上で動くペーパートレード(PC不要)。

状態はすべてリポジトリ内の小さなファイルで持つ:
  state/model.pkl        学習済みモデル(ローカルで学習してコミット)
  state/racer_stats.csv  選手履歴の集計スナップショット
  state/bets.csv         仮想投票の記録
  state/checked.csv      処理済みレースID
  docs/data.json         ダッシュボード用データ(GitHub Pagesで公開)

実際の投票機能は存在しない。読み取りと記録のみ。

  python -m boatrace.cloud sweep    締切直前レースの仮想投票(10分おき)
  python -m boatrace.cloud settle   前日分の採点と選手統計の更新(毎朝)
"""

import argparse
import csv
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .dataset import CATEGORICAL, FEATURES, KLASS_MAP
from .download import DATA_DIR, fetch_day
from .papertrade import _select_bets, _stake
from .parse import parse_b_file, parse_k_file
from .predict import ENTRY_COLS, RACE_COLS, VENUES
from .stats import STATE_DIR, attach_features, load_stats, save_stats, update_stats

JST = ZoneInfo("Asia/Tokyo")
MODEL_PATH = STATE_DIR / "model.pkl"
BETS_PATH = STATE_DIR / "bets.csv"
CHECKED_PATH = STATE_DIR / "checked.csv"
DOCS = Path(__file__).resolve().parent.parent / "docs"

BET_FIELDS = ["race_id", "bet_type", "combo", "p", "odds", "ev", "stake",
              "placed_at", "ret", "settled_at"]

# 締切がこの範囲内のレースを処理する(cron遅延があるため広めに取り、
# checked.csv で二重処理を防ぐ)
WINDOW_MIN, WINDOW_MAX = 3, 16


def _now() -> datetime:
    return datetime.now(JST)


def _load_bets() -> pd.DataFrame:
    if BETS_PATH.exists():
        return pd.read_csv(BETS_PATH, dtype={"combo": str})
    return pd.DataFrame(columns=BET_FIELDS)


def _load_checked() -> set[str]:
    if CHECKED_PATH.exists():
        return {r.split(",")[0] for r in
                CHECKED_PATH.read_text().splitlines()[1:] if r}
    return set()


def _append_checked(race_id: str, n_bets: int) -> None:
    new = not CHECKED_PATH.exists()
    with open(CHECKED_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["race_id", "checked_at", "n_bets"])
        w.writerow([race_id, _now().isoformat(timespec="seconds"), n_bets])


def _predict_today(target: str) -> pd.DataFrame | None:
    d = datetime.strptime(target, "%Y-%m-%d").date()
    path = fetch_day("B", d, DATA_DIR / "B")
    if path is None:
        return None
    races, entries = parse_b_file(path)
    df = pd.DataFrame(entries, columns=ENTRY_COLS).merge(
        pd.DataFrame(races, columns=RACE_COLS), on="race_id")
    df["klass_num"] = df["klass"].map(KLASS_MAP)
    df = attach_features(df, load_stats())

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)["model"]
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    df["p_raw"] = model.predict(df[FEATURES])
    df["p"] = df["p_raw"] / df.groupby("race_id")["p_raw"].transform("sum")
    return df


def sweep() -> None:
    now = _now()
    target = now.strftime("%Y-%m-%d")
    df = _predict_today(target)
    if df is None:
        print("番組表なし")
        return
    checked = _load_checked()

    due = []
    for rid, race in df.groupby("race_id"):
        top = race.iloc[0]
        if rid in checked or not top["deadline"]:
            continue
        dl = datetime.strptime(f"{target} {top['deadline']}",
                               "%Y-%m-%d %H:%M").replace(tzinfo=JST)
        if now + timedelta(minutes=WINDOW_MIN) <= dl <= now + timedelta(minutes=WINDOW_MAX):
            due.append((rid, race, top))
    print(f"{target} {now:%H:%M} 対象 {len(due)}レース")

    for rid, race, top in due:
        probs = {int(r.lane): r.p for r in race.itertuples()}
        try:
            bets = _select_bets(probs, int(top["jcd"]), int(top["rno"]), target)
        except Exception as e:
            print(f"NG {rid}: {e}")
            continue
        ts = _now().isoformat(timespec="seconds")
        if bets:
            new = not BETS_PATH.exists()
            with open(BETS_PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(BET_FIELDS)
                for bt, combo, p, o in bets:
                    w.writerow([rid, bt, combo, round(p, 5), o,
                                round(p * o, 4), _stake(p, o), ts, "", ""])
                    print(f"[仮想] {VENUES.get(int(top['jcd']))}{top['rno']}R "
                          f"{bt} {combo} p={p:.1%} odds={o} EV={p*o:.2f}")
        _append_checked(rid, len(bets))
    dashboard()


def settle() -> None:
    bets = _load_bets()
    if bets.empty:
        dashboard()
        return
    unsettled_days = sorted({rid[:10] for rid in
                             bets.loc[bets["ret"].isna(), "race_id"]})
    today = _now().strftime("%Y-%m-%d")
    stats = load_stats()
    stats_updated = False
    for day in unsettled_days:
        if day >= today:  # 当日分はまだ結果が出ていない
            continue
        d = datetime.strptime(day, "%Y-%m-%d").date()
        path = fetch_day("K", d, DATA_DIR / "K")
        if path is None:
            print(f"{day} の結果が未公開")
            continue
        _, results, payouts = parse_k_file(path)
        paytable = {(rid, bt, combo): amount for rid, bt, combo, amount, _ in payouts}
        resolved = {r[0] for r in results}
        ts = _now().isoformat(timespec="seconds")
        mask = bets["race_id"].str.startswith(day) & bets["ret"].isna()
        for i in bets.index[mask]:
            b = bets.loc[i]
            if b["race_id"] not in resolved:
                continue  # レース中止等。次回また見る
            amount = paytable.get((b["race_id"], b["bet_type"], b["combo"]), 0)
            bets.loc[i, "ret"] = int(b["stake"]) * amount // 100
            bets.loc[i, "settled_at"] = ts
        stats = update_stats(stats, results)
        stats_updated = True
        print(f"{day}: 採点 {int(mask.sum())}件")
    bets.to_csv(BETS_PATH, index=False)
    if stats_updated:
        save_stats(stats)
    dashboard()


def _agg(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"bets": 0, "hit_rate": None, "staked": 0, "returned": 0,
                "roi_kelly": None, "roi_flat100": None}
    hit = df["ret"] > 0
    unit = (df["odds"] * 100).where(hit, 0)
    return {"bets": int(len(df)), "hit_rate": round(float(hit.mean()), 3),
            "staked": int(df["stake"].sum()), "returned": int(df["ret"].sum()),
            "roi_kelly": round(float(df["ret"].sum() / max(df["stake"].sum(), 1)), 3),
            "roi_flat100": round(float(unit.sum() / (100 * len(df))), 3)}


def dashboard() -> None:
    bets = _load_bets()
    settled = bets[bets["ret"].notna()].copy()
    out = {"updated_at": _now().isoformat(timespec="seconds"),
           "unsettled": int(bets["ret"].isna().sum()),
           "total": _agg(settled), "by_type": {}, "by_ev": {},
           "daily": [], "equity": [], "recent": []}
    if not settled.empty:
        settled["date"] = settled["race_id"].str[:10]
        for bt, g in settled.groupby("bet_type"):
            out["by_type"][bt] = _agg(g)
        settled["ev_bucket"] = pd.cut(
            settled["ev"], [1.2, 1.5, 2.0, 3.0, 100]).astype(str)
        for b, g in settled.groupby("ev_bucket"):
            out["by_ev"][b] = _agg(g)
        cum_k = cum_f = 0.0
        for d, g in sorted(settled.groupby("date"), key=lambda x: x[0]):
            a = _agg(g)
            out["daily"].append({"date": d, **a})
            cum_k += a["returned"] - a["staked"]
            hit = g["ret"] > 0
            cum_f += float(((g["odds"] * 100).where(hit, 0) - 100).sum())
            out["equity"].append({"date": d, "kelly": round(cum_k),
                                  "flat": round(cum_f)})
    recent = bets.tail(200).iloc[::-1]
    for b in recent.itertuples():
        out["recent"].append({
            "race_id": b.race_id, "venue": VENUES.get(int(b.race_id[11:13]), "?"),
            "rno": int(b.race_id[14:16]), "bet_type": b.bet_type,
            "combo": b.combo, "p": b.p, "odds": b.odds, "ev": b.ev,
            "stake": int(b.stake),
            "ret": None if pd.isna(b.ret) else int(b.ret)})
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    t = out["total"]
    print(f"dashboard: {t['bets']}点 ROI(均一100円)={t['roi_flat100']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["sweep", "settle", "dashboard"])
    a = p.parse_args()
    {"sweep": sweep, "settle": settle, "dashboard": dashboard}[a.cmd]()
