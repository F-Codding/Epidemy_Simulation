@echo off
rem Запуск симуляции эпидемии (Windows). укажите полный путь к python, если его нет в PATH
python "%~dp0epid\epidemic_sim.py" %*
