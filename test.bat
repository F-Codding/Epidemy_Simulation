@echo off
rem Регрессионные тесты (Windows): не открывают окон, работа headless.
setlocal
set PROGDIR=%~dp0
python "%PROGDIR%tests\test_start_window.py" || goto :fail
python "%PROGDIR%tests\test_smoke_panels.py" || goto :fail
python "%PROGDIR%tests\test_output_files.py" || goto :fail
echo ALL TESTS OK
exit /b 0
:fail
echo TEST FAILED
exit /b 1
