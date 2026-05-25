@echo off
setlocal


echo [INFO] YOLO Viewer 啟動程序


REM 取得 bat 檔所在資料夾
set "PROJECT_DIR=%~dp0"

echo [INFO] 專案資料夾：
echo %PROJECT_DIR%
echo.

REM 進入 bat 檔所在資料夾
echo [INFO] 切換到專案資料夾...
cd /d "%PROJECT_DIR%"

REM 檢查虛擬環境
echo [INFO] 檢查虛擬環境 .venv...
if exist "%PROJECT_DIR%.venv\Scripts\activate.bat" (
    echo [INFO] 找到虛擬環境，正在啟動 .venv...
    call "%PROJECT_DIR%.venv\Scripts\activate.bat"
    echo [OK] 虛擬環境已啟動
) else (
    echo [ERROR] 找不到虛擬環境 .venv
    echo [ERROR] 請確認 .venv 是否位於：
    echo %PROJECT_DIR%.venv
    pause
    exit /b 1
)

echo.
echo [INFO] 目前使用的 Python：
where python
python --version

echo.
echo [INFO] 開始執行 viewer.py...

echo.

python viewer.py

echo.

echo [INFO] viewer.py 已結束


pause
endlocal