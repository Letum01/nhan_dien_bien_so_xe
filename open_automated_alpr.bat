@echo off
cd /d "%~dp0"
start "Development ALPR" ".venv\Scripts\python.exe" -m streamlit run app_development_alpr.py --server.address=localhost --server.port=8509
start "Blind text labeling" ".venv\Scripts\python.exe" -m streamlit run app_quick_ocr_test.py --server.address=localhost --server.port=8510