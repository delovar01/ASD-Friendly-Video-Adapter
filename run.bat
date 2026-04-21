@echo off
REM ============================================================
REM  Запуск ASD-Friendly Video Adapter
REM  После старта откройте в браузере: http://localhost:8000
REM  Чтобы остановить — закройте это окно или нажмите Ctrl+C.
REM ============================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Не найдено виртуальное окружение.
    echo Сначала запустите install.bat
    pause
    exit /b 1
)

call .venv\Scripts\activate

echo.
echo Сервис запускается...
echo Откройте в браузере: http://localhost:8000
echo.

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

pause
