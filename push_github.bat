@echo off
cd /d "%~dp0"
git add . || goto :err
git ls-files --error-unmatch .env >nul 2>&1
if not errorlevel 1 (
  echo [!] .env would be uploaded. It contains your API key. Stopping.
  git rm --cached -q .env
  goto :err
)
git diff --cached --quiet
if not errorlevel 1 (
  echo No changed files to commit. Pushing existing commits...
  goto :push
)
echo.
echo Files to upload:
git status --short
echo.
set /p OK=Upload these files to GitHub? [y/n]: 
if /i not "%OK%"=="y" (
  echo Cancelled. Nothing was uploaded.
  pause
  goto :eof
)
git commit -q -m "Update course recommendation chatbot" || goto :err
:push
git push -u origin main || goto :err
echo.
echo DONE. Uploaded to GitHub.
pause
goto :eof
:err
echo.
echo FAILED. Copy the messages above and send them to Claude.
pause
