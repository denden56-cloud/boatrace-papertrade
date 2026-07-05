"""番組表(B)・競走成績(K)テキストをSQLiteに取り込む。

どちらもShift-JIS固定長。会場セクションは `NNBBGN`/`NNKBGN` で始まる
(NN=場コード01〜24)。数値フィールドはバイト位置で切り出す。
"""

import re
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "boatrace.db"

ZEN2HAN = str.maketrans("０１２３４５６７８９：ＲＨｍ", "0123456789:RHm")

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY, date TEXT, jcd INTEGER, rno INTEGER,
    title TEXT, distance INTEGER, deadline TEXT,
    weather TEXT, wind_dir TEXT, wind_speed INTEGER, wave INTEGER
);
CREATE TABLE IF NOT EXISTS entries (
    race_id TEXT, lane INTEGER, regno INTEGER, name TEXT, age INTEGER,
    branch TEXT, weight INTEGER, klass TEXT,
    nat_win REAL, nat_2in REAL, loc_win REAL, loc_2in REAL,
    motor_no INTEGER, motor_2in REAL, boat_no INTEGER, boat_2in REAL,
    PRIMARY KEY (race_id, lane)
);
CREATE TABLE IF NOT EXISTS results (
    race_id TEXT, lane INTEGER, regno INTEGER,
    rank_raw TEXT, rank INTEGER, tenji REAL, course INTEGER, st REAL,
    flying INTEGER,
    PRIMARY KEY (race_id, lane)
);
CREATE TABLE IF NOT EXISTS payouts (
    race_id TEXT, bet_type TEXT, combo TEXT, amount INTEGER, popularity INTEGER
);
CREATE INDEX IF NOT EXISTS idx_results_regno ON results (regno);
CREATE INDEX IF NOT EXISTS idx_payouts_race ON payouts (race_id);
"""


def _f(s: bytes) -> float | None:
    try:
        return float(s.decode("cp932").strip())
    except ValueError:
        return None


def _i(s: bytes) -> int | None:
    try:
        return int(s.decode("cp932").strip())
    except ValueError:
        return None


def race_id(date: str, jcd: int, rno: int) -> str:
    return f"{date}-{jcd:02d}-{rno:02d}"


# ---------------------------------------------------------------- B(番組表)

B_RACE_RE = re.compile(r"^[　\s]*([0-9]+)R\s*(\S*)")


def parse_b_file(path: Path):
    """(races, entries) を返す。racesは天候なし(K側で埋める)。"""
    date = f"20{path.stem[1:3]}-{path.stem[3:5]}-{path.stem[5:7]}"
    races, entries = [], []
    jcd = rno = None
    for raw in path.read_bytes().splitlines():
        line = raw.decode("cp932", errors="replace")
        m = re.match(r"^(\d{2})BBGN", line)
        if m:
            jcd = int(m.group(1))
            continue
        if jcd is None:
            continue
        if "電話投票締切予定" in line:
            han = line.translate(ZEN2HAN)
            m = B_RACE_RE.match(han)
            if not m:
                continue
            rno = int(m.group(1))
            dm = re.search(r"H(\d+)m", han)
            dl = re.search(r"電話投票締切予定(\d{1,2}:\d{2})", han)
            races.append((race_id(date, jcd, rno), date, jcd, rno,
                          m.group(2).replace("　", "").strip() or None,
                          int(dm.group(1)) if dm else None,
                          dl.group(1) if dl else None))
            continue
        if rno is not None and re.match(rb"^[1-6] \d{4}", raw):
            entries.append((
                race_id(date, jcd, rno),
                int(raw[0:1]), _i(raw[2:6]),
                raw[6:14].decode("cp932", errors="replace").replace("　", ""),
                _i(raw[14:16]),
                raw[16:20].decode("cp932", errors="replace").strip(),
                _i(raw[20:22]),
                raw[22:24].decode("cp932", errors="replace"),
                _f(raw[24:29]), _f(raw[29:35]), _f(raw[35:40]), _f(raw[40:46]),
                _i(raw[46:49]), _f(raw[49:55]), _i(raw[55:58]), _f(raw[58:64]),
            ))
    return races, entries


# ---------------------------------------------------------------- K(成績)

K_RACE_RE = re.compile(r"^ +(\d{1,2})R ")
K_RESULT_RE = re.compile(r"^  (..)  ([1-6]) (\d{4}) (.{8})(.*)$")
BET_TYPES = ("単勝", "複勝", "２連単", "２連複", "拡連複", "３連単", "３連複")
COMBO_RE = re.compile(r"(\d(?:-\d){0,2})\s+(\d+)(?:\s+人気\s+(\d+))?")


def _parse_result_tail(tail: str):
    """→ (tenji, course, st, flying)。欠場等でデータがない艇は全てNone/0。"""
    t = tail.split()
    if len(t) < 4:
        return None, None, None, 0
    # t = [motor, boat, 展示, 進入, ST, タイム...] 展示が数値でなければ欠場系
    try:
        tenji = float(t[2])
    except ValueError:
        return None, None, None, 0
    course = int(t[3]) if t[3] in "123456" else None
    st = flying = None
    if len(t) >= 5:
        stx = t[4]
        if stx.startswith("F"):
            flying = 1
            try:
                st = -float(stx[1:])
            except ValueError:
                st = None
        else:
            try:
                st = float(stx)
            except ValueError:
                st = None
    return tenji, course, st, flying or 0


def parse_k_file(path: Path):
    """(race_meta, results, payouts) を返す。"""
    date = f"20{path.stem[1:3]}-{path.stem[3:5]}-{path.stem[5:7]}"
    metas, results, payouts = [], [], []
    jcd = rid = None
    bet_type = None
    for raw in path.read_bytes().splitlines():
        line = raw.decode("cp932", errors="replace")
        m = re.match(r"^(\d{2})KBGN", line)
        if m:
            jcd = int(m.group(1))
            rid = None
            continue
        if jcd is None:
            continue
        m = K_RACE_RE.match(line)
        if m and ("H" in line and "m" in line):
            rid = race_id(date, jcd, int(m.group(1)))
            bet_type = None
            wm = re.search(r"(晴|曇|雨|雪|霧)", line)
            wind = re.search(r"風\s+(\S+)\s+(\d+)m", line)
            wave = re.search(r"波\s+(\d+)cm", line)
            metas.append((rid,
                          wm.group(1) if wm else None,
                          wind.group(1).replace("　", "") if wind else None,
                          int(wind.group(2)) if wind else None,
                          int(wave.group(1)) if wave else None))
            continue
        if rid is None:
            continue
        m = K_RESULT_RE.match(line)
        if m:
            rank_raw = m.group(1).strip()
            tenji, course, st, flying = _parse_result_tail(m.group(5))
            results.append((rid, int(m.group(2)), int(m.group(3)), rank_raw,
                            int(rank_raw) if rank_raw.isdigit() else None,
                            tenji, course, st, flying))
            continue
        stripped = line.strip()
        matched_bt = next((bt for bt in BET_TYPES if stripped.startswith(bt)), None)
        if matched_bt:
            bet_type = matched_bt
        # 払戻行(キーワード行 or 拡連複などの継続行)
        if bet_type and (matched_bt or line.startswith(" " * 12)):
            for combo, amount, pop in COMBO_RE.findall(line):
                payouts.append((rid, bet_type, combo, int(amount),
                                int(pop) if pop else None))
    return metas, results, payouts


# ---------------------------------------------------------------- DB構築

def build_db(db_path: Path = DB_PATH) -> None:
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=OFF")

    b_files = sorted((DATA_DIR / "raw" / "B").glob("b*.txt"))
    k_files = sorted((DATA_DIR / "raw" / "K").glob("k*.txt"))
    done = {r[0] for r in con.execute("SELECT DISTINCT date FROM races")}

    for i, bf in enumerate(b_files):
        date = f"20{bf.stem[1:3]}-{bf.stem[3:5]}-{bf.stem[5:7]}"
        if date in done:
            continue
        races, entries = parse_b_file(bf)
        con.executemany(
            "INSERT OR IGNORE INTO races (race_id,date,jcd,rno,title,distance,deadline)"
            " VALUES (?,?,?,?,?,?,?)", races)
        con.executemany(
            "INSERT OR IGNORE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            entries)
        if i % 100 == 0:
            print(f"B {date}", flush=True)
    con.commit()

    done_k = {r[0] for r in con.execute(
        "SELECT DISTINCT substr(race_id,1,10) FROM results")}
    for i, kf in enumerate(k_files):
        date = f"20{kf.stem[1:3]}-{kf.stem[3:5]}-{kf.stem[5:7]}"
        if date in done_k:
            continue
        metas, results, payouts = parse_k_file(kf)
        con.executemany(
            "UPDATE races SET weather=?, wind_dir=?, wind_speed=?, wave=?"
            " WHERE race_id=?",
            [(w, wd, ws, wv, rid) for rid, w, wd, ws, wv in metas])
        con.executemany("INSERT OR IGNORE INTO results VALUES (?,?,?,?,?,?,?,?,?)",
                        results)
        con.executemany("INSERT INTO payouts VALUES (?,?,?,?,?)", payouts)
        if i % 100 == 0:
            print(f"K {date}", flush=True)
    con.commit()

    for t in ("races", "entries", "results", "payouts"):
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t}: {n}")
    con.close()


if __name__ == "__main__":
    build_db()
