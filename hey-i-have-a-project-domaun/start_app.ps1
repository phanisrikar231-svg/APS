$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

$BundledPython = "C:\Users\phani\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path -LiteralPath $BundledPython) {
    & $BundledPython "run_app.py" "--port" "8000"
    exit $LASTEXITCODE
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if ($Python) {
    & $Python.Source "run_app.py" "--port" "8000"
    exit $LASTEXITCODE
}

$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PyLauncher) {
    & $PyLauncher.Source "-3" "run_app.py" "--port" "8000"
    exit $LASTEXITCODE
}

Write-Host "Python was not found. Install Python 3.10+ and run: python -m pip install -r requirements.txt"
exit 1
