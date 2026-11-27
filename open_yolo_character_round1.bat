@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app_yolo_character_round1.py --server.address=localhost --server.port=8507