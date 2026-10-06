@echo off
cd /d "%~dp0"
echo [1/3] Installing packages with uv...
uv sync || goto :err
if not exist .env copy .env.example .env >nul
if not exist data\courses.json (
  echo.
  echo [2/3] data\courses.json is missing.
  echo       Students: get the file from your instructor and put it in the data folder.
  echo       Instructor: run crawl_class.bat once, then share the data folder.
  goto :err
)
echo [2/3] Course data found.
findstr /r /c:"^OPENAI_API_KEY=..*" .env >nul || (
  echo.
  echo [!] No OpenAI key yet. Opening .env - paste your key after OPENAI_API_KEY= and save.
  echo     Without a key the chatbot only shows search results by keyword.
  notepad .env
)
echo [3/3] Starting chatbot... (Ctrl+C to stop)
uv run streamlit run app.py
goto :eof
:err
echo.
echo STOPPED. If there is an error above, copy it and send it to your instructor.
pause
