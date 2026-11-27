@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app_alpr_deep.py --server.address=localhost --server.port=8512