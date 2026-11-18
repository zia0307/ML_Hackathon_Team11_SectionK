#!/bin/bash
# PathSnap UI - Laptop 2 (macOS / Linux):  bash start_ui.sh
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install -q -r requirements.txt
echo "Open http://localhost:8501 and enter the API address (Laptop 1) in the sidebar."
(sleep 4; open http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501 2>/dev/null) &
.venv/bin/python -m streamlit run app.py
