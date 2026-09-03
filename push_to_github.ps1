# Automated GitHub Repository Creation and Push Script
# Supply Chain Disruption Early Warning System

$ErrorActionPreference = "Stop"

$repoName = "supply-chain-disruption-warning"
$token = "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"
$username = "satarabdus692-bot"
$remoteUrl = "https://$($username):$($token)@github.com/$($username)/$($repoName).git"

Write-Host "[1/5] Creating private GitHub repository via API..." -ForegroundColor Cyan
$headers = @{
    "Authorization" = "token $token"
    "Accept" = "application/vnd.github.v3+json"
}
$body = @{
    name = $repoName
    description = "Supply Chain Disruption Early Warning System combining UN Comtrade anomaly detection, GDELT NLP signals, and World Bank LPI for predictive supply chain risk scoring."
    private = $true
} | ConvertTo-Json

try {
    $resp = Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $body -ContentType "application/json"
    Write-Host "Repository created successfully: $($resp.html_url)" -ForegroundColor Green
} catch {
    Write-Host "Repository creation note: $_ (it may already exist)" -ForegroundColor Yellow
}

Write-Host "[2/5] Initializing local Git repository..." -ForegroundColor Cyan
if (-not (Test-Path ".git")) {
    git init
}

git config user.name "satarabdus692-bot"
git config user.email "satarabdus692-bot@users.noreply.github.com"

Write-Host "[3/5] Staging files..." -ForegroundColor Cyan
git add .

Write-Host "[4/5] Committing codebase..." -ForegroundColor Cyan
git commit -m "feat: complete Supply Chain Disruption Early Warning System implementation"

Write-Host "[5/5] Pushing to GitHub main branch..." -ForegroundColor Cyan
git branch -M main
try {
    git remote remove origin 2>$null
} catch {}
git remote add origin $remoteUrl
git push -u origin main --force

Write-Host "Push completed successfully! Repo URL: https://github.com/$($username)/$($repoName)" -ForegroundColor Green
