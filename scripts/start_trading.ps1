<# 
    NSE RL Trader — Daily Trading Launcher
    Starts OpenAlgo server and the trading system.
    Schedule this via Windows Task Scheduler to run at 8:45 AM Mon-Fri.
#>
param(
    [string]$ApiKey = $env:OPENALGO_API_KEY,
    [string]$Host = "http://127.0.0.1:5000",
    [float]$Capital = 500000,
    [switch]$Live  # Without -Live, runs in paper (analyzer) mode
)

$ErrorActionPreference = "Stop"

# ── Paths (EDIT THESE) ──────────────────────────────────────────
$ProjectRoot  = "c:\Personal Files\Real Time RL trading system"
$OpenAlgoDir  = "c:\Personal Files\openalgo"          # Where OpenAlgo is cloned
$VenvActivate = "$ProjectRoot\venv\Scripts\Activate.ps1"
$LogDir       = "$ProjectRoot\logs"
# ────────────────────────────────────────────────────────────────

$Date = Get-Date -Format "yyyy-MM-dd"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Check if today is a weekday
if ((Get-Date).DayOfWeek -in @('Saturday', 'Sunday')) {
    Write-Host "Weekend — market closed. Exiting."
    exit 0
}

Write-Host "========================================"
Write-Host " NSE RL Trader — Starting for $Date"
Write-Host "========================================"

# 1. Start OpenAlgo server (if not already running)
$openalgoProc = Get-Process -Name "python" -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowTitle -match "openalgo" -or $_.CommandLine -match "app.py" }

if (-not $openalgoProc) {
    Write-Host "[1/3] Starting OpenAlgo server..."
    Start-Process -FilePath "python" `
        -ArgumentList "app.py" `
        -WorkingDirectory $OpenAlgoDir `
        -WindowStyle Minimized `
        -RedirectStandardOutput "$LogDir\openalgo_$Date.log" `
        -RedirectStandardError "$LogDir\openalgo_err_$Date.log"
    
    # Wait for server to be ready
    $retries = 0
    $maxRetries = 30
    do {
        Start-Sleep -Seconds 2
        $retries++
        try {
            $response = Invoke-WebRequest -Uri "$Host/api/v1/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) { break }
        } catch {}
    } while ($retries -lt $maxRetries)

    if ($retries -ge $maxRetries) {
        Write-Error "OpenAlgo server did not start within 60 seconds. Check $LogDir\openalgo_err_$Date.log"
        exit 1
    }
    Write-Host "   OpenAlgo ready."
} else {
    Write-Host "[1/3] OpenAlgo already running."
}

# 2. Activate virtual environment
if (Test-Path $VenvActivate) {
    Write-Host "[2/3] Activating virtual environment..."
    & $VenvActivate
} else {
    Write-Host "[2/3] No venv found, using system Python."
}

# 3. Start the trading system
$paperFlag = if ($Live) { "--no-paper" } else { "--paper" }
$logFile = "$LogDir\trading_$Date.log"

Write-Host "[3/3] Starting trading system ($( if ($Live) {'LIVE'} else {'PAPER'} ) mode)..."
Write-Host "       Capital: INR $Capital"
Write-Host "       Log: $logFile"
Write-Host ""

Set-Location $ProjectRoot
python main.py `
    --mode trade `
    --api-key $ApiKey `
    --host $Host `
    --capital $Capital `
    $paperFlag `
    2>&1 | Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "Trading session complete for $Date."
