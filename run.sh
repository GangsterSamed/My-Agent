#!/usr/bin/env bash
# Запуск агента из venv проекта.
set -e
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  echo "Создаю .venv и ставлю зависимости..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  .venv/bin/playwright install chromium
  echo "Готово."
fi
exec .venv/bin/python3 main.py
