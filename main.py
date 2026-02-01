#!/usr/bin/env python3
"""Точка входа: python3 main.py"""
from __future__ import annotations

import sys

import anyio

from agent import run_agent

if __name__ == "__main__":
    try:
        anyio.run(run_agent, backend="asyncio")
    except KeyboardInterrupt:
        print("\nПрервано (Ctrl+C).")
        sys.exit(130)
