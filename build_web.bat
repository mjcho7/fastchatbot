@echo off
cd /d "%~dp0"
uv run python build_web.py
pause
