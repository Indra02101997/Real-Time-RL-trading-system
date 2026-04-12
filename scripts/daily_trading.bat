@echo off
REM NSE RL Trader — Daily Trading Batch File
REM Use this with Windows Task Scheduler.
REM Action: Start a program → cmd.exe
REM Arguments: /c "c:\Personal Files\Real Time RL trading system\scripts\daily_trading.bat"

set PROJECT_ROOT=c:\Personal Files\Real Time RL trading system
set OPENALGO_API_KEY=YOUR_API_KEY_HERE

echo [%date% %time%] Starting NSE RL Trader...
cd /d "%PROJECT_ROOT%"

REM Activate venv if exists
if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
)

REM Start trading (paper mode by default)
python main.py --mode trade --api-key %OPENALGO_API_KEY% --capital 500000 --paper >> logs\trading_%date:~-4,4%-%date:~-7,2%-%date:~-10,2%.log 2>&1

echo [%date% %time%] Trading session ended.
