@echo off
REM DataViz Launch Script for Windows
echo Starting DataViz - Intelligent EDA Web App...
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install/update requirements
echo Installing/updating packages...
pip install -r requirements.txt

REM Launch the application
echo.
echo Launching DataViz application...
echo Open your browser and navigate to: http://localhost:8501
echo Press Ctrl+C to stop the application.
echo.

streamlit run app.py

pause
