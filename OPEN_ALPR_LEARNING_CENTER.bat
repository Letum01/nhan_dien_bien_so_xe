@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app_learning_center.py --server.address=localhost --server.port=8513