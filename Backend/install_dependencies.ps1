# PowerShell script to install dependencies in correct order
# This ensures numpy is installed before pandas

Write-Host "Installing dependencies..." -ForegroundColor Green

# Activate virtual environment if not already activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

# Install numpy first (required for pandas)
Write-Host "Installing numpy..." -ForegroundColor Cyan
pip install numpy==1.26.2

# Install pandas
Write-Host "Installing pandas..." -ForegroundColor Cyan
pip install pandas==2.1.3

# Install remaining dependencies
Write-Host "Installing remaining dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "All dependencies installed successfully!" -ForegroundColor Green

