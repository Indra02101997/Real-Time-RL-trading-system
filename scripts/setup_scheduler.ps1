<#
    NSE RL Trader — Windows Task Scheduler Setup
    Run as Administrator to create scheduled tasks for automated trading.
#>
param(
    [string]$ProjectRoot = "c:\Personal Files\Real Time RL trading system"
)

$ErrorActionPreference = "Stop"

# Verify admin privileges
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator. Right-click PowerShell → Run as Administrator."
    exit 1
}

Write-Host "Setting up Windows Task Scheduler for NSE RL Trader..."
Write-Host ""

# ── Task 1: Daily Trading (Mon-Fri at 8:45 AM) ──────────────
$taskName1 = "NSE_RL_DailyTrading"
$action1 = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File `"$ProjectRoot\scripts\start_trading.ps1`"" `
    -WorkingDirectory $ProjectRoot

$trigger1 = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At "08:45AM"

$settings1 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)

Register-ScheduledTask `
    -TaskName $taskName1 `
    -Action $action1 `
    -Trigger $trigger1 `
    -Settings $settings1 `
    -Description "Start NSE RL Trader paper trading every weekday morning." `
    -Force

Write-Host "[OK] Created task: $taskName1 (Mon-Fri 8:45 AM)"

# ── Task 2: Weekly Backtest (Saturday at 10:00 AM) ───────────
$taskName2 = "NSE_RL_WeeklyBacktest"
$action2 = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File `"$ProjectRoot\scripts\weekly_backtest.ps1`"" `
    -WorkingDirectory $ProjectRoot

$trigger2 = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Saturday `
    -At "10:00AM"

$settings2 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask `
    -TaskName $taskName2 `
    -Action $action2 `
    -Trigger $trigger2 `
    -Settings $settings2 `
    -Description "Run weekly backtest to validate model performance." `
    -Force

Write-Host "[OK] Created task: $taskName2 (Saturday 10:00 AM)"

Write-Host ""
Write-Host "=========================================="
Write-Host " Setup Complete!"
Write-Host "=========================================="
Write-Host ""
Write-Host "Created tasks:"
Write-Host "  1. $taskName1 — Mon-Fri 8:45 AM"
Write-Host "  2. $taskName2 — Saturday 10:00 AM"
Write-Host ""
Write-Host "IMPORTANT: Edit start_trading.ps1 to set your OPENALGO_API_KEY"
Write-Host "and verify the paths match your installation."
Write-Host ""
Write-Host "To view tasks: taskschd.msc"
Write-Host "To remove:     Unregister-ScheduledTask -TaskName '$taskName1'"
