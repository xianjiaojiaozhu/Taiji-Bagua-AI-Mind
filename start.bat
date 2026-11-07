@echo off
cd /d "%~dp0"
rem =====================================
rem   Taiji Bagua Pan - taiji-gua-visual
rem   backend: 127.0.0.1:8901  frontend: 127.0.0.1:8898
rem   使用前请设置环境变量 DEEPSEEK_API_KEY（不设置也可运行，LLM 层自动降级）
rem   例：set DEEPSEEK_API_KEY=sk-xxxxxxxx
rem =====================================
set PY=python
echo [1/3] starting backend ...
start "taiji-backend-8901" cmd /k "cd /d %~dp0 && %PY% -m uvicorn server:app --host 127.0.0.1 --port 8901"
timeout /t 3 /nobreak >nul
echo [2/3] starting frontend ...
start "taiji-frontend-8898" cmd /k "cd /d %~dp0 && %PY% -m http.server 8898"
timeout /t 2 /nobreak >nul
echo [3/3] opening browser ...
start "" "http://127.0.0.1:8898/index.html?api=8901"
echo.
echo Done. Close the two black windows to stop.
pause
