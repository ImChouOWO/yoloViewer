@echo off
setlocal

REM 取得 bat 檔所在資料夾
set "PROJECT_DIR=%~dp0"

REM 進入 bat 檔所在資料夾
cd /d "%PROJECT_DIR%"

REM 啟動 .venv
if exist "%PROJECT_DIR%.venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%.venv\Scripts\activate.bat"
) else (
    echo [ERROR] 找不到虛擬環境 .venv
    pause
    exit /b 1
)

REM 執行 viewer.py
python viewer.py

pause
endlocal
