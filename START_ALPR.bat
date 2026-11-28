@echo off
setlocal
cd /d "%~dp0"
rem Prefer environments installed for THIS folder over another machine's config.
if exist ".venv\Scripts\python.exe" if exist ".venv-yolo\Scripts\python.exe" goto local
if exist "alpr-local-env.bat" call "alpr-local-env.bat"
goto configured
:local
set "ALPR_OCR_PYTHON=%~dp0.venv\Scripts\python.exe"
set "ALPR_YOLO_PYTHON=%~dp0.venv-yolo\Scripts\python.exe"
:configured
if not defined ALPR_OCR_PYTHON set "ALPR_OCR_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined ALPR_YOLO_PYTHON set "ALPR_YOLO_PYTHON=%~dp0.venv-yolo\Scripts\python.exe"
if not exist "%ALPR_OCR_PYTHON%" goto missing
if not exist "%ALPR_YOLO_PYTHON%" goto missing
"%ALPR_OCR_PYTHON%" -c "import streamlit; from runtime_validation import validate_runtime; validate_runtime()" || goto invalid
"%ALPR_YOLO_PYTHON%" -c "import torch, ultralytics; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))" || goto invalid
powershell -NoProfile -Command "if(Get-NetTCPConnection -LocalPort 8511 -State Listen -ErrorAction SilentlyContinue){exit 1}" || goto busy
echo ALPR demo: http://localhost:8511
"%ALPR_OCR_PYTHON%" -m streamlit run app_alpr_demo.py --server.address=localhost --server.port=8511
if errorlevel 1 goto invalid
exit /b 0
:missing
echo Khong tim thay interpreter. Kiem tra ALPR_OCR_PYTHON va ALPR_YOLO_PYTHON hoac alpr-local-env.bat.
goto failed
:invalid
echo Kiem tra loi ben tren: checkpoint, dependencies, CUDA hoac loi khoi dong.
goto failed
:busy
echo Cong 8511 dang duoc su dung. Neu ALPR dang chay, mo http://localhost:8511.
:failed
pause
exit /b 1