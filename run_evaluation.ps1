$inputDir = ".\eval2"
$outputRoot = ".\eval2_results"
$pipeline = ".\systematic_review_pipeline.py"
$checklist = ".\checklist_example.json"
$python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment Python not found: $python"
}

if (-not (Test-Path $pipeline)) {
    throw "Pipeline not found: $pipeline"
}

if (-not (Test-Path $checklist)) {
    throw "Checklist not found: $checklist"
}

if (-not (Test-Path $inputDir)) {
    throw "Input directory not found: $inputDir"
}

if (-not $env:OPENROUTER_API_KEY) {
    throw "OPENROUTER_API_KEY is not configured."
}

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$failedFiles = @()
$pdfFiles = Get-ChildItem -Path $inputDir -Filter "*.pdf" -File

Write-Host "Found $($pdfFiles.Count) PDF files." -ForegroundColor Cyan

foreach ($pdf in $pdfFiles) {
    $outputDir = Join-Path $outputRoot $pdf.BaseName

    Write-Host "`nProcessing: $($pdf.Name)" -ForegroundColor Cyan

    $pipelineArguments = @(
        $pipeline
        $pdf.FullName
        $checklist
        "--output-dir", $outputDir
        "--parse-mode", "auto"
        "--reasoning-effort", "low"
        "--evaluation-workers", "8"
    )

    & $python @pipelineArguments

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Completed: $($pdf.Name)" -ForegroundColor Green
    }
    else {
        Write-Warning "Failed: $($pdf.Name)"
        $failedFiles += $pdf.Name
    }
}

Write-Host "`nProcessing finished." -ForegroundColor Cyan

if ($failedFiles.Count -gt 0) {
    Write-Host "Failed files:" -ForegroundColor Red

    foreach ($failedFile in $failedFiles) {
        Write-Host " - $failedFile"
    }
}
else {
    Write-Host "All files completed successfully." -ForegroundColor Green
}