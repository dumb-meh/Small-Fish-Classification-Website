# Fish Classification Website - Startup Script

Write-Host "=" -NoNewline; Write-Host ("=" * 59)
Write-Host "Fish Classification Website - Starting Server"
Write-Host "=" -NoNewline; Write-Host ("=" * 59)
Write-Host ""

# Check if we're in the correct directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPath = Join-Path $scriptPath "Backend"

if (-not (Test-Path $backendPath)) {
    Write-Host "ERROR: Backend directory not found!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Yellow
    pause
    exit 1
}

# Check if GROQ_API_KEY is set
if (-not $env:GROQ_API_KEY) {
    Write-Host "WARNING: GROQ_API_KEY environment variable is not set!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The chatbot will not work without this API key." -ForegroundColor Yellow
    Write-Host "To set it, run:" -ForegroundColor Cyan
    Write-Host '  $env:GROQ_API_KEY="your-groq-api-key-here"' -ForegroundColor Green
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 0
    }
    Write-Host ""
}

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher." -ForegroundColor Yellow
    pause
    exit 1
}

# Check if requirements are installed
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Cyan

$requirementsPath = Join-Path $backendPath "requirements.txt"
if (Test-Path $requirementsPath) {
    try {
        python -c "import flask, flask_cors, groq" 2>&1 | Out-Null
        Write-Host "All dependencies are installed." -ForegroundColor Green
    } catch {
        Write-Host "Installing dependencies..." -ForegroundColor Yellow
        Set-Location $backendPath
        pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to install dependencies!" -ForegroundColor Red
            pause
            exit 1
        }
        Set-Location $scriptPath
    }
} else {
    Write-Host "WARNING: requirements.txt not found!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting Flask server..." -ForegroundColor Cyan
Write-Host ""

# Change to Backend directory and run the server
Set-Location $backendPath
python main.py

# Return to original directory when server stops
Set-Location $scriptPath
