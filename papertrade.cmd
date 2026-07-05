@echo off
rem 毎朝タスクスケジューラから起動されるペーパートレード常駐スクリプト。
rem 実際の投票は行わない(公開ページの読み取りのみ)。
cd /d C:\Users\hawkt\OneDrive\Documents\Coding\SHUMI\gamble
.venv\Scripts\python.exe -m boatrace.papertrade run >> data\papertrade.log 2>&1
