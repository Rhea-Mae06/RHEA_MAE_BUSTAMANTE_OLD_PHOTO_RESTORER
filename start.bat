@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   OLD PHOTO RESTORER - AI RESTORATION
echo ============================================
echo.

if not exist "backend\.venv\Scripts\python.exe" (
  echo Creating Python environment...
  py -3.12 -m venv backend\.venv 2>nul
  if not exist "backend\.venv\Scripts\python.exe" (
    py -3 -m venv backend\.venv 2>nul
  )
  if not exist "backend\.venv\Scripts\python.exe" (
    python -m venv backend\.venv
  )
)

if not exist "backend\.venv\Scripts\python.exe" (
  echo Could not create the Python environment. Install Python 3.10-3.12 and try again.
  pause
  exit /b 1
)

echo Installing backend dependencies...
backend\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r backend\requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

if not exist "backend\models\big-lama.pt" (
  echo Downloading AI models - this is a one-time download...
  backend\.venv\Scripts\python.exe backend\download_models.py
)

echo.
echo Starting the restoration server at http://127.0.0.1:8765
echo Keep this window open while you use the app.
echo.
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:8765/'"
backend\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8765

endlocal
