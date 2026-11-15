#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Use Python 3.11 or 3.12 (PYTHON_BIN=python3.11)"'
MAC_ENV="${MAC_ENV:-.venv-mac}"
"$PYTHON_BIN" -m venv "$MAC_ENV"
"$MAC_ENV/bin/python" -m pip install -r requirements-mac.txt
"$MAC_ENV/bin/python" -c 'import nltk; nltk.download("punkt", raise_on_error=True, quiet=True); nltk.download("punkt_tab", raise_on_error=True, quiet=True)'
"$MAC_ENV/bin/python" -m open_instruct.portable doctor
