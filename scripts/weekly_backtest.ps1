<#
    NSE RL Trader — Weekly Backtest
    Runs every Saturday to validate model performance on recent data.
#>
param(
    [int]$Years = 1,
    [float]$Capital = 500000
)

$ProjectRoot = "c:\Personal Files\Real Time RL trading system"
$LogDir = "$ProjectRoot\logs"
$Date = Get-Date -Format "yyyy-MM-dd"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

Write-Host "========================================"
Write-Host " NSE RL Trader — Weekly Backtest $Date"
Write-Host "========================================"

# Activate venv
$VenvActivate = "$ProjectRoot\venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) { & $VenvActivate }

Set-Location $ProjectRoot

$logFile = "$LogDir\backtest_$Date.log"

Write-Host "Running backtest ($Years year(s), capital INR $Capital)..."
python main.py `
    --mode backtest `
    --years $Years `
    --capital $Capital `
    2>&1 | Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "Backtest complete. Results in $logFile"
