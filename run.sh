#!/bin/sh
# Запуск симуляции эпидемии (Linux/macOS/BSD). python3 из переменной PATH.
exec python3 "$(dirname "$0")/epid/epidemic_sim.py" "$@"
