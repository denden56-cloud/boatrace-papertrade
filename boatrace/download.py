"""競艇公式データ(番組表B・競走成績K)のダウンロード。

配布元: http://www1.mbrace.or.jp/od2/
  B/YYYYMM/bYYMMDD.lzh  番組表(出走表)
  K/YYYYMM/kYYMMDD.lzh  競走成績(結果・払戻)

LZH書庫内に同名の .TXT (Shift-JIS) が1つ入っている。
"""

import argparse
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import lhafile
import requests

BASE_URL = "http://www1.mbrace.or.jp/od2"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
REQUEST_INTERVAL = 1.0  # 単発取得時のサーバー負荷対策
# サーバーが1接続あたり約2.4KB/sに帯域制限しているため、並列化しないと
# 2年分で数時間かかる。6並列でほぼ線形にスケールする。
WORKERS = 6

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = (
            "Mozilla/5.0 (personal data analysis)")
    return _local.s

session = _session()


def fetch_day(kind: str, d: date, out_dir: Path) -> Path | None:
    """kind='B'|'K' の1日分を取得してTXTに展開。既存ならスキップ。

    開催がない日・未公開日は404になるので None を返す。
    """
    assert kind in ("B", "K")
    out = out_dir / f"{kind.lower()}{d:%y%m%d}.txt"
    if out.exists():
        return out
    url = f"{BASE_URL}/{kind}/{d:%Y%m}/{kind.lower()}{d:%y%m%d}.lzh"
    resp = _session().get(url, timeout=120)
    time.sleep(REQUEST_INTERVAL)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    archive = lhafile.Lhafile(io.BytesIO(resp.content))
    name = archive.namelist()[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_bytes(archive.read(name))
    return out


def _fetch_one(job: tuple[str, date]) -> bool:
    kind, d = job
    try:
        return fetch_day(kind, d, DATA_DIR / kind) is not None
    except Exception as e:  # 個別の失敗で全体を止めない
        print(f"NG {kind} {d}: {e}", flush=True)
        return False


def fetch_range(start: date, end: date) -> None:
    jobs = []
    d = start
    while d <= end:
        jobs.extend((k, d) for k in ("B", "K"))
        d += timedelta(days=1)
    ok = n = 0
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for got in ex.map(_fetch_one, jobs):
            n += 1
            ok += got
            if n % 100 == 0:
                print(f"... {n}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {ok}/{len(jobs)} files", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("start", type=date.fromisoformat)
    p.add_argument("end", type=date.fromisoformat)
    a = p.parse_args()
    fetch_range(a.start, a.end)
