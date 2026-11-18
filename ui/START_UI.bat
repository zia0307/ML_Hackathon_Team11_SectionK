@echo off
REM PathSnap UI - Laptop 2 (Windows). Double-click this file.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.11+ from python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -q -r requirements.txt
echo Open http://localhost:8501 and enter the API address (Laptop 1) in the sidebar.
start "" http://localhost:8501
python -m streamlit run app.py
pause
