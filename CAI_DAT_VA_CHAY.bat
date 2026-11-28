@echo off
setlocal
cd /d "%~dp0"
title ALPR - Cai dat va chay
echo ============================================================
echo ALPR - TU DONG CAI MOI TRUONG VA KHOI DONG
echo ============================================================
echo Yeu cau: Windows 64-bit, Python 3.11, NVIDIA GPU va driver CUDA.
echo Can Internet va vai GB dung luong. Khong tai model tu Internet.
echo Neu da cai, pip se kiem tra cac goi dung phien ban da pin.
echo Khong dong cua so trong khi cai dat.
echo.
if not exist "%~dp0setup_environment.ps1" goto incomplete
if not exist "%~dp0START_ALPR.bat" goto incomplete
if not exist "%~dp0configs\runtime-inference.json" goto incomplete
where py >nul 2>&1
if errorlevel 1 goto python_missing
py -3.11 -c "import sys,struct; assert sys.version_info[:2] == (3,11) and struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 goto python_missing
where nvidia-smi >nul 2>&1
if errorlevel 1 goto gpu_missing
nvidia-smi --query-gpu=name --format=csv,noheader
if errorlevel 1 goto gpu_missing
echo.
echo [1/2] Kiem tra model, tao venv va cai thu vien...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_environment.ps1" -ProjectLocal
if errorlevel 1 goto install_failed
echo.
echo [2/2] Cai dat thanh cong. Khoi dong ALPR...
echo Truy cap http://localhost:8511 khi server san sang.
call "%~dp0START_ALPR.bat"
exit /b %ERRORLEVEL%
:python_missing
echo LOI: Can cai Python 3.11 64-bit va Python Launcher truoc.
echo Tai bo cai chinh thuc: https://www.python.org/downloads/windows/
echo Sau khi cai, mo lai file nay. Kiem tra bang: py -3.11 --version
goto failed
:gpu_missing
echo LOI: Chua kiem tra duoc GPU NVIDIA/driver qua nvidia-smi.
echo Cai driver NVIDIA phu hop. Ban ALPR nay khong co detector CPU.
goto failed
:incomplete
echo LOI: Thu muc thieu source/config. Giai nen day du goi ban giao.
goto failed
:install_failed
echo LOI: Cai dat chua hoan tat. Xem loi dau tien o ben tren.
echo Neu thieu/sai hash model, can bo model hop le; khong sua manifest.
echo Khong tu doi phien ban thu vien de bo qua loi.
:failed
echo Xem HUONG_DAN_CAI_DAT_TU_DONG.txt de biet cach xu ly.
pause
exit /b 1