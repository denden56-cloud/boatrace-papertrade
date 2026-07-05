"""boatrace.jp公式サイトから直前オッズを取得する。

単勝: oddstf ページ。EV計算(予測確率×オッズ)に使う。
オッズは締切まで変動するため、取得タイミングに注意。
"""

import re
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://www.boatrace.jp/owpc/pc/race"
session = requests.Session()
session.headers["User-Agent"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
)


def fetch_3tan_odds(jcd: int, rno: int, date: str) -> dict[tuple[int, int, int], float]:
    """{(1着,2着,3着): オッズ} 全120通り。未発売・欠場分は含まれない。

    表は6列グループ(グループg=1着艇g+1)。各2着ブロックの先頭行は
    [2着,3着,オッズ]の3セル、続く行は[3着,オッズ]の2セル。
    """
    hd = date.replace("-", "")
    url = f"{BASE}/odds3t?rno={rno}&jcd={jcd:02d}&hd={hd}"
    resp = session.get(url, timeout=30)
    time.sleep(1.0)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = next((t for t in soup.select("table")
                  if len(t.select("td.oddsPoint")) >= 100), None)
    odds: dict[tuple[int, int, int], float] = {}
    if table is None:
        return odds
    second = [0] * 6
    for tr in table.select("tbody tr"):
        tds = [td.get_text(strip=True) for td in tr.select("td")]
        if len(tds) % 6:
            continue
        per = len(tds) // 6
        if per not in (2, 3):
            continue
        for g in range(6):
            chunk = tds[g * per:(g + 1) * per]
            if per == 3:
                second[g] = int(chunk[0]) if chunk[0].isdigit() else 0
            third, val = chunk[-2], chunk[-1]
            if (second[g] and third.isdigit()
                    and re.fullmatch(r"\d+(\.\d+)?", val) and float(val) > 0):
                odds[(g + 1, second[g], int(third))] = float(val)
    return odds


def fetch_tansho_odds(jcd: int, rno: int, date: str) -> dict[int, float]:
    """{艇番: 単勝オッズ}。取得できない艇(欠場等)は含まれない。"""
    hd = date.replace("-", "")
    url = f"{BASE}/oddstf?rno={rno}&jcd={jcd:02d}&hd={hd}"
    resp = session.get(url, timeout=30)
    time.sleep(1.0)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    odds: dict[int, float] = {}
    # 単勝テーブル: 各行が [艇番セル(is-boatColorN), 選手名, オッズ(oddsPoint)]
    for table in soup.select("table"):
        cells = table.select("td.oddsPoint")
        if len(cells) != 6:
            continue
        lanes = table.select("td[class*=is-boatColor]")
        if len(lanes) < 6:
            continue
        for lane_td, odd_td in zip(lanes, cells):
            m = re.search(r"[1-6]", lane_td.get_text(strip=True))
            v = odd_td.get_text(strip=True)
            if m and re.fullmatch(r"\d+(\.\d+)?", v):
                odds[int(m.group())] = float(v)
        break  # 最初に該当したテーブルが単勝
    return odds
