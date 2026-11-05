#!/bin/sh
# Регрессионные тесты (Linux/macOS/BSD): не открывают окон, работа headless.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$DIR/tests/test_start_window.py"
python3 "$DIR/tests/test_smoke_panels.py"
python3 "$DIR/tests/test_output_files.py"
echo "ALL TESTS OK"
