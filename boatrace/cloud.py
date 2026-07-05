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

from .dataset import CATEGORICAL, FEATURES, KLASS_MAP, add_relative_features
from .download import DATA_DIR, fetch_day
from .odds import fetch_3tan_odds, fetch_beforeinfo, fetch_tansho_odds
from .papertrade import _stake, select_from_odds
from .parse import parse_b_file, parse_k_file
from .predict import ENTRY_COLS, RACE_COLS, VENUES
from .stats import STATE_DIR, attach_features, load_stats, save_stats, update_stats

JST = ZoneInfo("Asia/Tokyo")
MODEL_PATH = STATE_DIR / "model.pkl"
BETS_PATH = STATE_DIR / "bets.csv"
CHECKED_PATH = STATE_DIR / "checked.csv"
DOCS = Path(__file__).resolve().parent.parent / "docs"

BET_FIELDS = ["race_id", "bet_type", "combo", "p", "odds", "ev", "stake",
              "placed_at", "ret", "settled_at", "final_odds"]
# スタート展示の記録(履歴データが貯まれば将来の特徴量にする)
BEFOREINFO_PATH = STATE_DIR / "beforeinfo.csv"
# オッズ履歴(月別)。live=締切直前(投票判断時)、final=確定
# 賭けの有無に関わらず全対象レースを記録し、EV戦略のバックテストを可能にする
ODDS_DIR = STATE_DIR / "odds"

# 締切がこの範囲内のレースを処理する(cron遅延があるため広めに取り、
# checked.csv で二重処理を防ぐ)
WINDOW_MIN, WINDOW_MAX = 3, 16


def _now() -> datetime:
    return datetime.now(JST)


def _load_bets() -> pd.DataFrame:
    """bets.csv を読む。列追加前の古い行が混ざっていても耐える。"""
    if not BETS_PATH.exists():
        return pd.DataFrame(columns=BET_FIELDS)
    with open(BETS_PATH, newline="", encoding="utf-8") as f:
        rows = [r + [""] * (len(BET_FIELDS) - len(r))
                for r in csv.reader(f)][1:]  # ヘッダは読み飛ばす
    df = pd.DataFrame(rows, columns=BET_FIELDS)
    for c in ("p", "odds", "ev", "stake", "ret", "final_odds"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


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


def _base_today(target: str) -> pd.DataFrame | None:
    """当日の出走表+特徴量(展示タイム以外)。"""
    d = datetime.strptime(target, "%Y-%m-%d").date()
    path = fetch_day("B", d, DATA_DIR / "B")
    if path is None:
        return None
    races, entries = parse_b_file(path)
    df = pd.DataFrame(entries, columns=ENTRY_COLS).merge(
        pd.DataFrame(races, columns=RACE_COLS), on="race_id")
    df["klass_num"] = df["klass"].map(KLASS_MAP)
    df["tenji"] = float("nan")
    return attach_features(df, load_stats())


import itertools

PERMS = list(itertools.permutations(range(1, 7), 3))
ODDS_FIELDS = (["phase", "race_id", "fetched_at"]
               + [f"t{i}" for i in range(1, 7)]
               + [f"s{a}{b}{c}" for a, b, c in PERMS])


def archive_odds(phase: str, rid: str, t_odds: dict, s_odds: dict) -> None:
    """1レース分のオッズを月別CSVに1行で追記する。"""
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    path = ODDS_DIR / f"{rid[:7]}.csv"
    new = not path.exists()
    row = [phase, rid, _now().isoformat(timespec="seconds")]
    row += [t_odds.get(i, "") for i in range(1, 7)]
    row += [s_odds.get(p, "") for p in PERMS]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(ODDS_FIELDS)
        w.writerow(row)


def _log_beforeinfo(rid: str, info: dict[int, dict]) -> None:
    new = not BEFOREINFO_PATH.exists()
    with open(BEFOREINFO_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["race_id", "lane", "tenji", "ex_course", "ex_st"])
        for lane in sorted(info):
            d = info[lane]
            w.writerow([rid, lane, d.get("tenji", ""),
                        d.get("ex_course", ""), d.get("ex_st", "")])


def sweep() -> None:
    now = _now()
    target = now.strftime("%Y-%m-%d")
    df = _base_today(target)
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
            due.append((rid, top))
    print(f"{target} {now:%H:%M} 対象 {len(due)}レース")
    if not due:
        dashboard()
        return

    # 対象レースの展示タイムを直前情報ページから取得して特徴量に反映
    for rid, top in due:
        try:
            info = fetch_beforeinfo(int(top["jcd"]), int(top["rno"]), target)
        except Exception as e:
            print(f"beforeinfo NG {rid}: {e}")
            continue
        _log_beforeinfo(rid, info)
        for lane, d in info.items():
            if "tenji" in d:
                df.loc[(df["race_id"] == rid) & (df["lane"] == lane),
                       "tenji"] = d["tenji"]

    df["tenji"] = pd.to_numeric(df["tenji"], errors="coerce")
    df = add_relative_features(df)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    lam2 = bundle.get("lambda2", 1.0)
    lam3 = bundle.get("lambda3", 1.0)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    df["p_raw"] = model.predict(df[FEATURES])
    df["p"] = df["p_raw"] / df.groupby("race_id")["p_raw"].transform("sum")

    for rid, top in due:
        race = df[df["race_id"] == rid]
        probs = {int(r.lane): r.p for r in race.itertuples()}
        try:
            t_odds = fetch_tansho_odds(int(top["jcd"]), int(top["rno"]), target)
            s_odds = fetch_3tan_odds(int(top["jcd"]), int(top["rno"]), target)
            archive_odds("live", rid, t_odds, s_odds)
            bets = select_from_odds(probs, t_odds, s_odds, lam2, lam3)
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
                                round(p * o, 4), _stake(p, o), ts, "", "", ""])
                    print(f"[仮想] {VENUES.get(int(top['jcd']))}{top['rno']}R "
                          f"{bt} {combo} p={p:.1%} odds={o} EV={p*o:.2f}")
        _append_checked(rid, len(bets))
    dashboard()


def _archive_final_odds(bets: pd.DataFrame, day: str,
                        race_ids: set[str]) -> None:
    """全レースの確定オッズをアーカイブし、仮想投票分のfinal_odds列も埋める。

    公式サイトは過去レースのオッズページも確定値で残している。
    賭けの有無に関わらず全レースを記録するのは、EV戦略そのものを
    後からバックテストできるようにするため。
    """
    done = set()
    path = ODDS_DIR / f"{day[:7]}.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            done = {r[1] for r in csv.reader(f) if r and r[0] == "final"}
    for rid in sorted(race_ids - done):
        jcd, rno = int(rid[11:13]), int(rid[14:16])
        try:
            t = fetch_tansho_odds(jcd, rno, day)
            s = fetch_3tan_odds(jcd, rno, day)
        except Exception as e:
            print(f"final odds NG {rid}: {e}")
            continue
        archive_odds("final", rid, t, s)
        rmask = (bets["race_id"] == rid) & bets["final_odds"].isna()
        for i in bets.index[rmask]:
            if bets.loc[i, "bet_type"] == "単勝":
                o = t.get(int(bets.loc[i, "combo"]))
            else:
                o = s.get(tuple(int(x) for x in bets.loc[i, "combo"].split("-")))
            if o:
                bets.loc[i, "final_odds"] = o


STATS_DATE_PATH = STATE_DIR / "stats_date.txt"


def settle() -> None:
    """前回処理日の翌日から昨日までを日次で処理する。

    投票の有無に関わらず毎日、(1)仮想投票の採点 (2)選手統計の更新
    (3)確定オッズの全レースアーカイブ を行う。統計の欠落を防ぐため
    処理済み日付は stats_date.txt で管理する。
    """
    bets = _load_bets()
    today = _now().strftime("%Y-%m-%d")
    stats = load_stats()
    last = (STATS_DATE_PATH.read_text().strip() if STATS_DATE_PATH.exists()
            else (_now() - timedelta(days=2)).strftime("%Y-%m-%d"))
    d = datetime.strptime(last, "%Y-%m-%d").date() + timedelta(days=1)
    changed = False
    while d.isoformat() < today:
        day = d.isoformat()
        path = fetch_day("K", d, DATA_DIR / "K")
        if path is None:
            print(f"{day} の結果が未公開。次回に持ち越し")
            break  # 日付順を保つためここで打ち切る
        _, results, payouts = parse_k_file(path)
        paytable = {(rid, bt, combo): amount for rid, bt, combo, amount, _ in payouts}
        resolved = {r[0] for r in results}
        ts = _now().isoformat(timespec="seconds")
        mask = bets["race_id"].str.startswith(day) & bets["ret"].isna()
        for i in bets.index[mask]:
            b = bets.loc[i]
            if b["race_id"] not in resolved:
                bets.loc[i, "ret"] = int(b["stake"])  # レース中止=返還扱い
                bets.loc[i, "settled_at"] = ts
                continue
            amount = paytable.get((b["race_id"], b["bet_type"], b["combo"]), 0)
            bets.loc[i, "ret"] = int(b["stake"]) * amount // 100
            bets.loc[i, "settled_at"] = ts
        stats = update_stats(stats, results)
        _archive_final_odds(bets, day, resolved)
        STATS_DATE_PATH.write_text(day)
        changed = True
        print(f"{day}: 採点 {int(mask.sum())}件 / 確定オッズ {len(resolved)}レース")
        d += timedelta(days=1)
    bets.to_csv(BETS_PATH, index=False)
    if changed:
        save_stats(stats)
    dashboard()


def _agg(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"bets": 0, "hit_rate": None, "staked": 0, "returned": 0,
                "roi_kelly": None, "roi_flat100": None, "clv": None}
    hit = df["ret"] > 0
    # 均一100円換算は投票時オッズではなく実際の確定払戻で計算する
    unit = (df["ret"] / df["stake"] * 100).fillna(0)
    # CLV: 投票時オッズが確定オッズよりどれだけ有利だったか。
    # 系統的にプラスなら「締切までに賢い金が同じ方向に動いている」=エッジの先行指標
    wf = df[df["final_odds"].notna() & (df["final_odds"] > 0)]
    clv = (round(float((wf["odds"] / wf["final_odds"]).mean() - 1), 4)
           if len(wf) else None)
    return {"bets": int(len(df)), "hit_rate": round(float(hit.mean()), 3),
            "staked": int(df["stake"].sum()), "returned": int(df["ret"].sum()),
            "roi_kelly": round(float(df["ret"].sum() / max(df["stake"].sum(), 1)), 3),
            "roi_flat100": round(float(unit.sum() / (100 * len(df))), 3),
            "clv": clv}


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
            cum_f += float((g["ret"] / g["stake"] * 100 - 100).sum())
            out["equity"].append({"date": d, "kelly": round(cum_k),
                                  "flat": round(cum_f)})
    recent = bets.tail(200).iloc[::-1]
    for b in recent.itertuples():
        out["recent"].append({
            "race_id": b.race_id, "venue": VENUES.get(int(b.race_id[11:13]), "?"),
            "rno": int(b.race_id[14:16]), "bet_type": b.bet_type,
            "combo": b.combo, "p": b.p, "odds": b.odds, "ev": b.ev,
            "stake": int(b.stake),
            "final_odds": None if pd.isna(b.final_odds) else float(b.final_odds),
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
