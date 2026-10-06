@echo off
cd /d "%~dp0"
rem Instructor only: collect 3 categories for the class (BIZ, BIZ_PLANNING, PROGRAMMING)
uv run python crawl.py --categories BIZ,BIZ_PLANNING,PROGRAMMING
pause
