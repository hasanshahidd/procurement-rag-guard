# One-command local setup on Windows: virtual environment, packages, local models, tests.
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1            # API and tests
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -WithML    # also torch, transformers, peft
param([string]$Venv = ".venv", [switch]$WithML)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = if (Get-Command py -ErrorAction SilentlyContinue) { "py", "-3.11" } else { , "python" }
Write-Host "Creating virtual environment in $Venv"
& $python[0] $python[1..9] -m venv $Venv
$venvPy = Join-Path $Venv "Scripts\python.exe"
& $venvPy -m pip install --quiet --upgrade pip
& $venvPy -m pip install --quiet -r requirements-dev.txt
if ($WithML) {
    & $venvPy -m pip install --quiet -r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
}

$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) { $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
if (-not (Test-Path $ollama)) {
    Write-Host "Ollama not found. Install it from https://ollama.com/download (or: winget install Ollama.Ollama), then rerun."
    exit 1
}
& $ollama pull qwen2.5:1.5b
& $ollama pull nomic-embed-text

& $venvPy -m pytest -q
Write-Host ""
Write-Host "Ready. Start the API:  $venvPy -m uvicorn app.main:app --port 8000"
Write-Host "Then open http://127.0.0.1:8000/ (web UI) or http://127.0.0.1:8000/docs (OpenAPI)."
