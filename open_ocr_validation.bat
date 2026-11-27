@echo off
setlocal
cd /d "%~dp0"
set "ALPR_OCR_QUEUE=validation"
if not exist ".venv\Scripts\python.exe" (
  echo Missing OCR virtual environment .venv
  exit /b 1
)
powershell -NoProfile -Command "$u='http://localhost:8506'; function Healthy {try {(Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 ($u+'/_stcore/health')).Content -eq 'ok'} catch {$false}}; if (-not (Healthy)) {if (Get-NetTCPConnection -LocalPort 8506 -State Listen -ErrorAction SilentlyContinue) {throw 'Port 8506 is already in use'}; $p=Start-Process -FilePath (Resolve-Path '.venv/Scripts/python.exe').Path -ArgumentList '-m streamlit run label_characters.py --server.address=localhost --server.port=8506 --server.headless=true' -WorkingDirectory (Get-Location).Path -RedirectStandardOutput (Join-Path $env:TEMP 'alpr-ocr-validation-out.log') -RedirectStandardError (Join-Path $env:TEMP 'alpr-ocr-validation-err.log') -PassThru; for ($i=0;$i -lt 45 -and -not (Healthy) -and -not $p.HasExited;$i++) {Start-Sleep -Seconds 1}}; if (-not (Healthy)) {throw 'OCR Streamlit failed; see TEMP/alpr-ocr-validation-err.log'}; Start-Process $u; Write-Output $u"
exit /b %errorlevel%