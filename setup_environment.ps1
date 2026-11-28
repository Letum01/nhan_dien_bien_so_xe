$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Checked($Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE)" }
}
if ($env:ALPR_OCR_PYTHON -and $env:ALPR_YOLO_PYTHON) {
    # Explicit existing interpreters are validated, not modified/reinstalled.
    foreach ($python in @($env:ALPR_OCR_PYTHON, $env:ALPR_YOLO_PYTHON)) {
        if (-not (Test-Path -LiteralPath $python)) { throw "Interpreter not found: $python" }
        Invoke-Checked $python @('-m','pip','check')
    }
    Invoke-Checked $env:ALPR_OCR_PYTHON @('-c','import streamlit, tensorflow, sklearn; print(streamlit.__version__)')
    Invoke-Checked $env:ALPR_YOLO_PYTHON @('-c','import torch, ultralytics; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))')
    exit 0
}
foreach ($folder in @('.venv', '.venv-yolo')) {
    if (-not (Test-Path -LiteralPath $folder)) {
        Invoke-Checked 'py' @('-3.11','-m','venv',$folder)
    }
}
$ocr = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$yolo = Join-Path $PSScriptRoot '.venv-yolo\Scripts\python.exe'
Invoke-Checked $ocr @('-m','pip','install','-r','requirements.txt')
Invoke-Checked $yolo @('-m','pip','install','torch==2.5.1+cu121','torchvision==0.20.1+cu121','--index-url','https://download.pytorch.org/whl/cu121')
Invoke-Checked $yolo @('-m','pip','install','-r','requirements-yolo.txt')
foreach ($python in @($ocr, $yolo)) { Invoke-Checked $python @('-m','pip','check') }
Invoke-Checked $yolo @('-c','import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))')