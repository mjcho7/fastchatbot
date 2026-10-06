@echo off
cd /d "%~dp0"
call :main > push_log.txt 2>&1
exit /b
:main
echo ==== START %date% %time%
git --version
git remote -v
git branch --show-current
git add .
git ls-files --error-unmatch .env >nul 2>&1
if not errorlevel 1 (
  echo RESULT: STOPPED - .env would be uploaded
  git rm --cached -q .env
  exit /b 1
)
echo ---- files in this upload:
git status --short
git diff --cached --quiet
if errorlevel 1 git commit -q -m "Add course recommendation chatbot"
echo ---- push:
git push -u origin main
if errorlevel 1 (
  echo RESULT: PUSH FAILED
  exit /b 1
)
echo ---- last commit:
git log --oneline -3
echo RESULT: DONE
exit /b 0
