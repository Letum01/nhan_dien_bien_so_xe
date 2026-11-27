@echo off
powershell.exe -NoProfile -File "%~dp0setup_environment.ps1"
exit /b %ERRORLEVEL%