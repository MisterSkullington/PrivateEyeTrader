@echo off
:: PrivateEyeTrader — weekly model retrain
:: Scheduled every Saturday at 5:00 AM via Windows Task Scheduler

set PROJECT=E:\Documents\GitHub\PrivateEyeTrader
set PYTHON=C:\Users\jorda\AppData\Local\Programs\Python\Python313\python.exe
set LOGDIR=%PROJECT%\logs
set LOGFILE=%LOGDIR%\retrain_%DATE:~10,4%-%DATE:~4,2%-%DATE:~7,2%.log

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ============================= >> "%LOGFILE%"
echo Retrain started: %DATE% %TIME% >> "%LOGFILE%"
echo ============================= >> "%LOGFILE%"

cd /d "%PROJECT%"

"%PYTHON%" scripts\train_models.py --symbol BTC/USDT --timeframe 1h >> "%LOGFILE%" 2>&1

if %ERRORLEVEL% == 0 (
    echo [OK] Retrain completed successfully: %DATE% %TIME% >> "%LOGFILE%"
) else (
    echo [FAIL] Retrain exited with code %ERRORLEVEL%: %DATE% %TIME% >> "%LOGFILE%"
)
