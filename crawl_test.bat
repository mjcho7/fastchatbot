@echo off
cd /d "%~dp0"
uv run python crawl.py --limit 100
pause
