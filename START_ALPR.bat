@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
if not exist ".venv-yolo\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -c "import streamlit; from final_ocr_dataset import check_protocol; check_protocol()" || goto invalid
".venv-yolo\Scripts\python.exe" -c "import torch, ultralytics; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))" || goto invalid
powershell -NoProfile -Command "if(Get-NetTCPConnection -LocalPort 8511 -State Listen -ErrorAction SilentlyContinue){exit 1}" || goto busy
echo ALPR demo: http://localhost:8511
".venv\Scripts\python.exe" -m streamlit run app_alpr_demo.py --server.address=localhost --server.port=8511
if errorlevel 1 goto invalid
exit /b 0
:missing
echo Khong tim thay .venv hoac .venv-yolo trong thu muc du an.
goto failed
:invalid
echo Kiem tra loi ben tren: checkpoint, dependencies, CUDA hoac loi khoi dong.
goto failed
:busy
echo Cong 8511 dang duoc su dung. Neu ALPR dang chay, mo http://localhost:8511.
:failed
pause
exit /b 1