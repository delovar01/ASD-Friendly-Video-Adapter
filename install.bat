@echo off
REM ============================================================
REM  Установщик для ASD-Friendly Video Adapter (Windows)
REM  Запустите ОДИН РАЗ при первой настройке.
REM ============================================================

cd /d "%~dp0"

echo.
echo [1/4] Проверяем Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ОШИБКА: Python не найден.
    echo Скачайте и установите Python 3.11 с https://www.python.org/downloads/
    echo При установке ОБЯЗАТЕЛЬНО поставьте галочку "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo.
echo [2/4] Проверяем FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ОШИБКА: FFmpeg не найден.
    echo 1. Скачайте архив с https://www.gyan.dev/ffmpeg/builds/ ^(release full^)
    echo 2. Распакуйте в C:\ffmpeg
    echo 3. Добавьте C:\ffmpeg\bin в системный PATH
    echo    ^(Параметры -^> Система -^> Переменные среды -^> Path -^> Изменить^)
    echo 4. Перезапустите этот скрипт.
    echo.
    pause
    exit /b 1
)

echo.
echo [3/4] Создаём виртуальное окружение в .venv ...
if not exist ".venv" (
    python -m venv .venv
    if errorlevel 1 (
        echo ОШИБКА: не удалось создать .venv
        pause
        exit /b 1
    )
)

echo.
echo [4/4] Устанавливаем зависимости (это займёт несколько минут)...
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ОШИБКА при установке пакетов. См. сообщения выше.
    pause
    exit /b 1
)

echo.
echo Генерируем маски животных...
python scripts\generate_masks.py

echo.
echo ============================================================
echo  Готово! Теперь запускайте run.bat для старта сервиса.
echo ============================================================
pause
