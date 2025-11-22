# PowerShell script to start the FastAPI server with warnings suppressed
# Usage: .\start_server.ps1

Write-Host "Starting Air Pollution Forecasting API..." -ForegroundColor Green

# Activate virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

# Suppress numpy warnings by setting environment variable
$env:PYTHONWARNINGS = "ignore::RuntimeWarning"

Write-Host "Starting server on http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

