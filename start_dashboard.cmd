@echo off
setlocal
cd /d "%~dp0"
set "APP_PYTHON=C:\Users\Sarim Khan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%APP_PYTHON%" (
  echo Python runtime not found at: %APP_PYTHON%
  echo Open this file in Codex and ask to update the launcher path.
  pause
  exit /b 1
)
"%APP_PYTHON%" run.py --serve
if errorlevel 1 (
  echo.
  echo The dashboard stopped with an error shown above.
  pause
)
