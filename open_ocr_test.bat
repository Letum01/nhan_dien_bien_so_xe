@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app_ocr_test_dataset.py --server.address=localhost --server.port=8508