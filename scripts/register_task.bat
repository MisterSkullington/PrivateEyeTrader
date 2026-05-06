@echo off
:: Run this ONCE as Administrator to register the weekly retrain task.
:: Right-click this file → "Run as administrator"

schtasks /Create ^
  /TN "PrivateEyeTrader\WeeklyRetrain" ^
  /TR "\"E:\Documents\GitHub\PrivateEyeTrader\scripts\retrain_weekly.bat\"" ^
  /SC WEEKLY ^
  /D SAT ^
  /ST 05:00 ^
  /F ^
  /RL HIGHEST ^
  /RU "%USERNAME%"

if %ERRORLEVEL% == 0 (
    echo.
    echo [OK] Task registered: PrivateEyeTrader\WeeklyRetrain
    echo      Runs every Saturday at 5:00 AM
    echo      Logs written to: E:\Documents\GitHub\PrivateEyeTrader\logs\
) else (
    echo.
    echo [FAIL] Registration failed. Make sure you right-clicked and chose "Run as administrator".
)

pause
