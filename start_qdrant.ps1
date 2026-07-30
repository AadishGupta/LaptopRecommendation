# start_qdrant.ps1 - Start Qdrant with proper config

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STARTING QDRANT LOCALLY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

# Create config file
$config = @"
storage:
  storage_path: ./qdrant_storage
"@

$config | Out-File -FilePath "config.yaml" -Encoding utf8

Write-Host "`n[1] Created config.yaml" -ForegroundColor Yellow
Write-Host "[2] Starting Qdrant..." -ForegroundColor Yellow
Write-Host "[3] Dashboard: http://localhost:6333/dashboard" -ForegroundColor Yellow
Write-Host "[4] Press Ctrl+C to stop`n" -ForegroundColor Yellow

