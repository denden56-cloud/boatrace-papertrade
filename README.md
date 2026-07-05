# gamble — 競艇予測・期待値ベッティング

競艇公式サイトが無料公開している番組表・競走成績データを使い、
各艇の1着確率をLightGBMで予測して、オッズとの歪み(期待値プラス)が
ある買い目だけを提示するシステム。

## 前提の整理

公営競技の控除率は約25%。全レースを漫然と買えばROIは0.75に収束する。
利益の可能性があるのは「モデルの確率がオッズの示す市場の確率より
正確な場面だけに絞って買う」戦略のみで、それでも保証はない。
**バックテストのROIが1.0を超えない設定で実際に金を賭けないこと。**

## 構成

```
boatrace/
  download.py   公式データ取得 (番組表B・競走成績K、LZH展開)
  parse.py      固定長Shift-JISテキスト → SQLite (data/boatrace.db)
  dataset.py    特徴量生成 (番組表情報 + リーク無し履歴集計)
  train.py      LightGBM学習・時系列分割評価
  backtest.py   実払戻金によるROI検証 (単勝/2連単/3連単)
  odds.py       公式サイトから直前単勝オッズ取得
  predict.py    当日レースの予測と買い目提示
  papertrade.py 全国全場での自動ペーパートレード検証
```

## 使い方

```powershell
# 1. 過去データ取得 (2年分で約75分。サーバーが低速なため)
.\.venv\Scripts\python.exe -m boatrace.download 2024-07-01 2026-07-04

# 2. SQLiteに取り込み
.\.venv\Scripts\python.exe -m boatrace.parse

# 3. 学習 + バックテスト (モデルは data/model.pkl に保存)
.\.venv\Scripts\python.exe -m boatrace.backtest

# 4. 当日の予測。締切の近いレースの単勝オッズを取得してEV計算
.\.venv\Scripts\python.exe -m boatrace.predict --odds
.\.venv\Scripts\python.exe -m boatrace.predict --odds --jcd 12   # 住之江のみ
```

`--odds` の注意: オッズは締切直前まで大きく動く。朝の時点の単勝オッズは
プールが空で意味がないので、買いたいレースの締切10〜15分前に実行する。

## オッズ履歴とCLV

- スイープは賭けの有無に関わらず対象レースの直前オッズを、夜間ジョブは
  全レースの確定オッズを `state/odds/YYYY-MM.csv` に蓄積する
  (単勝6+3連単120通り/レース)。これによりEV戦略のバックテストが可能になる
- CLV(投票時オッズ÷確定オッズ-1)をダッシュボードに表示。系統的にプラスなら
  エッジが本物である先行指標、マイナスならEVは幻

## 着順モデル

3連単の確率は素朴なHarville式ではなく、Benter型の減衰指数入り
(`boatrace/ordermodel.py`)。λは検証データの尤度最大化でフィットし
モデルと一緒に保存される(現在 λ2=0.65, λ3=0.45。テスト期間の3連単
対数尤度が -4.00→-3.85 に改善)。

## モデル

- 目的変数: 各艇が1着になるか (二値分類 → レース内で正規化)
- 特徴量: 枠番・級別・全国/当地勝率・モーター/ボート2連率・年齢・体重・場コード
  + 履歴集計 (平均ST・枠別勝率・通算勝率。当該レースより前のデータのみ使用)
- 検証: 時系列分割。直近3ヶ月をテストに固定し、リークを防ぐ

## クラウド自動検証 (PC不要)

GitHub Actionsが開催時間中(JST 8:00-23:00)10分おきに巡回して仮想投票を記録し
([.github/workflows/sweep.yml](.github/workflows/sweep.yml))、
毎朝7:30に前日分を採点する([.github/workflows/settle.yml](.github/workflows/settle.yml))。
状態はリポジトリ内の `state/` (投票記録CSV・選手統計・モデル)だけで完結し、
成績ダッシュボードは `docs/` がGitHub Pagesとして公開される。

- モデルの再学習だけはローカルで行う: `python -m boatrace.train` のあと
  `data/model.pkl` を `state/model.pkl` にコピーしてコミット
- 選手統計スナップショットの初期化: `python -m boatrace.stats`

## ローカル版ペーパートレード検証

実際の金銭を使わずにEV戦略の実測ROIを検証する仕組み。
投票サイトへのログイン・購入機能は一切持たない(公開ページの読み取りのみ)ので、
構造的に金銭損失は起こり得ない。

```powershell
.\.venv\Scripts\python.exe -m boatrace.papertrade run     # 1日分の常駐実行
.\.venv\Scripts\python.exe -m boatrace.papertrade settle  # 未採点分を結果で採点
.\.venv\Scripts\python.exe -m boatrace.papertrade report  # 累積成績
```

`run` は起動時に前日分の採点と週1回の再学習を行い、全国全場の各レースについて
締切3〜12分前にオッズを取得、EVプラスの買い目を仮想記録して最終レース後に終了する。
毎朝の自動起動はタスクスケジューラに `papertrade.cmd` を登録する:

```powershell
schtasks /Create /TN "BoatracePaperTrade" /TR "<絶対パス>\papertrade.cmd" /SC DAILY /ST 08:20
```

判断基準: **均一100円換算ROI (roi_flat100) が数百点以上のサンプルで安定して1.0を
超えるまで、実際の投票に移行しないこと。**

## データソース

- 過去データ: http://www1.mbrace.or.jp/od2/ (公式・無料)
- 直前オッズ: https://www.boatrace.jp (公式)

## 拡張予定

- 2連単/3連単オッズの取得とEV計算 (単勝よりプールが大きく歪みも大きい)
- 競馬 (JRA-VAN契約が前提)・競輪への同型パイプラインの展開
