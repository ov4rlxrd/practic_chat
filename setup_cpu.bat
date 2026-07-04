@echo off
echo ============================================
echo   Установка CPU-версии (без GPU-ускорения)
echo ============================================
echo.

pip install -r requirements.txt
pip install llama-cpp-python

echo.
echo ============================================
echo   Готово! Запускай: python app.py
echo ============================================
pause
