"""ペーパートレード検証のブラウザダッシュボード。

127.0.0.1限定のローカルサーバー。表示と仮想投票の操作のみで、
実際の投票機能は存在しない。

  .venv\\Scripts\\python.exe -m boatrace.webapp   → http://127.0.0.1:8500
"""

import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string

from .parse import DB_PATH
from .predict import VENUES

app = Flask(__name__)
_proc: subprocess.Popen | None = None

ROOT = Path(__file__).resolve().parent.parent


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def _venue(race_id: str) -> str:
    return VENUES.get(int(race_id[11:13]), "?")


def _agg(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"bets": 0, "hit_rate": None, "staked": 0, "returned": 0,
                "roi_kelly": None, "roi_flat100": None}
    hit = df["ret"] > 0
    unit_ret = (df["ret"] / df["stake"] * 100).fillna(0)
    return {
        "bets": int(len(df)),
        "hit_rate": round(float(hit.mean()), 3),
        "staked": int(df["stake"].sum()),
        "returned": int(df["ret"].sum()),
        "roi_kelly": round(float(df["ret"].sum() / max(df["stake"].sum(), 1)), 3),
        "roi_flat100": round(float(unit_ret.sum() / (100 * len(df))), 3),
    }


@app.get("/api/summary")
def api_summary():
    con = _db()
    df = pd.read_sql("SELECT * FROM paper_bets WHERE ret IS NOT NULL", con)
    unsettled = con.execute(
        "SELECT COUNT(*) FROM paper_bets WHERE ret IS NULL").fetchone()[0]
    con.close()

    out = {"total": _agg(df), "unsettled": int(unsettled),
           "by_type": {}, "by_ev": {}, "daily": [], "equity": []}
    if not df.empty:
        df["date"] = df["race_id"].str[:10]
        for bt, g in df.groupby("bet_type"):
            out["by_type"][bt] = _agg(g)
        df["ev_bucket"] = pd.cut(df["ev"], [1.2, 1.5, 2.0, 3.0, 100]).astype(str)
        for b, g in df.groupby("ev_bucket"):
            out["by_ev"][b] = _agg(g)
        cum_kelly = cum_flat = 0.0
        for d, g in sorted(df.groupby("date"), key=lambda x: x[0]):
            a = _agg(g)
            out["daily"].append({"date": d, **a})
            cum_kelly += a["returned"] - a["staked"]
            cum_flat += float((g["ret"] / g["stake"] * 100 - 100).sum())
            out["equity"].append(
                {"date": d, "kelly": cum_kelly, "flat": cum_flat})
    return jsonify(out)


@app.get("/api/today")
def api_today():
    today = date.today().isoformat()
    con = _db()
    bets = [dict(r) for r in con.execute(
        "SELECT * FROM paper_bets WHERE race_id LIKE ? ORDER BY placed_at DESC",
        (f"{today}%",))]
    checked = con.execute(
        "SELECT COUNT(*), SUM(n_bets = -1), MAX(checked_at)"
        " FROM paper_checked WHERE race_id LIKE ?", (f"{today}%",)).fetchone()
    total_races = con.execute(
        "SELECT COUNT(*) FROM races WHERE date = ?", (today,)).fetchone()[0]
    con.close()
    for b in bets:
        b["venue"] = _venue(b["race_id"])
        b["rno"] = int(b["race_id"][14:16])
    # このプロセスが起動したものに加え、外部起動の巡回も直近の処理記録から推定
    running = _proc is not None and _proc.poll() is None
    if not running and checked[2]:
        last = datetime.fromisoformat(checked[2])
        running = datetime.now() - last < timedelta(minutes=15)
    return jsonify({"date": today, "bets": bets, "running": running,
                    "checked": checked[0] or 0, "missed": int(checked[1] or 0),
                    "total_races": int(total_races)})


@app.post("/api/run")
def api_run():
    global _proc
    if _proc is not None and _proc.poll() is None:
        return jsonify({"status": "already_running"})
    log = open(ROOT / "data" / "papertrade.log", "a")
    _proc = subprocess.Popen(
        [sys.executable, "-m", "boatrace.papertrade", "run"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return jsonify({"status": "started"})


@app.post("/api/settle")
def api_settle():
    r = subprocess.run([sys.executable, "-m", "boatrace.papertrade", "settle"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    return jsonify({"status": "done", "output": (r.stdout or "")[-2000:]})


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>競艇ペーパートレード検証</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{font-family:"Yu Gothic UI",sans-serif;margin:0;background:#f4f6f8;color:#1a2733}
 header{background:#12385e;color:#fff;padding:14px 24px;display:flex;align-items:center;gap:16px}
 header h1{font-size:18px;margin:0}
 .safe{background:#1d7a4f;font-size:12px;padding:4px 10px;border-radius:12px}
 main{max-width:1100px;margin:20px auto;padding:0 16px;display:grid;gap:16px}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
 .card{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
 .card .k{font-size:12px;color:#61707d}.card .v{font-size:22px;font-weight:600;margin-top:4px}
 .card .v.pos{color:#1d7a4f}.card .v.neg{color:#b3362a}
 section{background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
 h2{font-size:14px;margin:0 0 10px;color:#12385e}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{padding:6px 8px;text-align:right;border-bottom:1px solid #e5eaee}
 th:first-child,td:first-child{text-align:left}
 thead th{color:#61707d;font-weight:600}
 .btn{background:#12385e;color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer;font-size:13px}
 .btn:disabled{background:#9db2c4}
 .muted{color:#61707d;font-size:12px}
 #status{font-size:13px}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
 .on{background:#1d7a4f}.off{background:#b3362a}
</style></head><body>
<header><h1>競艇ペーパートレード検証</h1>
 <span class="safe">仮想投票のみ — 実際の金銭は動きません</span>
 <span id="status" style="margin-left:auto"></span></header>
<main>
 <div class="cards" id="cards"></div>
 <section><h2>累積損益(円)</h2><canvas id="equity" height="80"></canvas></section>
 <section><h2>本日の仮想投票 <span class="muted" id="todayinfo"></span>
  <button class="btn" style="float:right" id="runbtn" onclick="startRun()">本日の巡回を開始</button>
  <button class="btn" style="float:right;margin-right:8px;background:#4a6076" onclick="settle()">採点する</button></h2>
  <table id="today"><thead><tr><th>場</th><th>R</th><th>式別</th><th>買い目</th>
   <th>予測確率</th><th>オッズ</th><th>EV</th><th>仮想賭金</th><th>払戻</th></tr></thead><tbody></tbody></table>
 </section>
 <section><h2>式別成績</h2><table id="bytype"><thead><tr><th>式別</th><th>点数</th><th>的中率</th>
  <th>ROI(ケリー)</th><th>ROI(均一100円)</th></tr></thead><tbody></tbody></table></section>
 <section><h2>EV帯別成績</h2><table id="byev"><thead><tr><th>EV帯</th><th>点数</th><th>的中率</th>
  <th>ROI(ケリー)</th><th>ROI(均一100円)</th></tr></thead><tbody></tbody></table></section>
 <section><h2>日別成績</h2><table id="daily"><thead><tr><th>日付</th><th>点数</th><th>的中率</th>
  <th>賭金計</th><th>払戻計</th><th>ROI(ケリー)</th><th>ROI(均一100円)</th></tr></thead><tbody></tbody></table></section>
 <p class="muted">判断基準: 均一100円換算ROIが数百点以上のサンプルで安定して1.0を超えるまで実投票には移行しない。</p>
</main>
<script>
const pct = v => v==null ? "-" : (v*100).toFixed(1)+"%";
const num = v => v==null ? "-" : v.toLocaleString();
const roi = v => v==null ? "-" : v.toFixed(3);
let chart;
async function refresh(){
  const s = await (await fetch("/api/summary")).json();
  const t = await (await fetch("/api/today")).json();
  document.getElementById("status").innerHTML =
    `<span class="dot ${t.running?"on":"off"}"></span>` +
    (t.running?"巡回中":"停止中") +
    ` — 本日 ${t.checked}/${t.total_races} レース処理済み`;
  document.getElementById("runbtn").disabled = t.running;
  const tot = s.total;
  document.getElementById("cards").innerHTML = [
    ["採点済み点数", num(tot.bets)],
    ["的中率", pct(tot.hit_rate)],
    ["ROI(ケリー配分)", roi(tot.roi_kelly), tot.roi_kelly],
    ["ROI(均一100円)", roi(tot.roi_flat100), tot.roi_flat100],
    ["損益(仮想円)", num(tot.returned - tot.staked), tot.returned-tot.staked],
    ["未採点", num(s.unsettled)],
  ].map(([k,v,c])=>`<div class="card"><div class="k">${k}</div>
    <div class="v ${c==null?"":(c>=(k.startsWith("ROI")?1:0)?"pos":"neg")}">${v}</div></div>`).join("");
  fill("bytype", Object.entries(s.by_type).map(([k,a])=>
    [k, a.bets, pct(a.hit_rate), roi(a.roi_kelly), roi(a.roi_flat100)]));
  fill("byev", Object.entries(s.by_ev).map(([k,a])=>
    [k, a.bets, pct(a.hit_rate), roi(a.roi_kelly), roi(a.roi_flat100)]));
  fill("daily", s.daily.map(d=>
    [d.date, d.bets, pct(d.hit_rate), num(d.staked), num(d.returned),
     roi(d.roi_kelly), roi(d.roi_flat100)]));
  document.getElementById("todayinfo").textContent =
    `${t.bets.length}点 (取り逃し ${t.missed}レース)`;
  fill("today", t.bets.map(b=>
    [b.venue, b.rno, b.bet_type, b.combo, pct(b.p), b.odds.toFixed(1),
     b.ev.toFixed(2), num(b.stake), b.ret==null?"未確定":num(b.ret)]));
  const labels = s.equity.map(e=>e.date);
  const data = {labels, datasets:[
    {label:"ケリー配分", data:s.equity.map(e=>e.kelly), borderColor:"#12385e", tension:.2},
    {label:"均一100円", data:s.equity.map(e=>e.flat), borderColor:"#1d7a4f", tension:.2}]};
  if(chart){chart.data=data; chart.update();}
  else chart = new Chart(document.getElementById("equity"),
    {type:"line", data, options:{animation:false, plugins:{legend:{position:"bottom"}}}});
}
function fill(id, rows){
  document.querySelector(`#${id} tbody`).innerHTML =
    rows.length ? rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join("")}</tr>`).join("")
    : `<tr><td colspan="9" class="muted">データなし</td></tr>`;
}
async function startRun(){ await fetch("/api/run",{method:"POST"}); refresh(); }
async function settle(){
  const b=event.target; b.disabled=true; b.textContent="採点中...";
  await fetch("/api/settle",{method:"POST"});
  b.disabled=false; b.textContent="採点する"; refresh();
}
refresh(); setInterval(refresh, 30000);
</script></body></html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8500)
