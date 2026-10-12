@echo off
rem Video Enhancer launcher. Run INSTALL.bat once before the first start.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Environment not found. Run INSTALL.bat first.
    pause & exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0gui.py"
endlocal
