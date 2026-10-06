@echo off
cd /d "%~dp0"
uv run streamlit run app.py
pause
