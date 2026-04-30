#requires -version 5
<#
.SYNOPSIS
Run the provider natively on the host with local llama-server subprocesses.

.DESCRIPTION
Loads provider/.env into the current process (if present), then runs
`python -m provider --config provider/models.yaml ...` so chat / embed /
vision backends are launched directly as host subprocesses instead of via
Docker sibling containers.
#>

param(
    [string]$Config = "",
    [Alias("Host")]
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8088,
    [ValidateSet("debug", "info", "warning", "error")]
    [string]$LogLevel = "info",
    [string]$PythonExe = "",
    [string]$PublicBaseUrl = ""
)

$ErrorActionPreference = "Stop"

$providerRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectRoot = (Resolve-Path (Join-Path $providerRoot "..")).Path
$envFile = Join-Path $providerRoot ".env"

function Import-DotEnv([string]$Path) {
    if (-not (Test-Path $Path)) {
        return
    }
    foreach ($raw in Get-Content $Path) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) {
            continue
        }
        $name = $parts[0].Trim()
        if (-not $name) {
            continue
        }
        if (-not (Test-Path "Env:$name")) {
            Set-Item -Path "Env:$name" -Value $parts[1]
        }
    }
}

function Resolve-Python([string]$Override, [string]$ProjectRoot) {
    $candidates = @()
    if ($Override) {
        $candidates += $Override
    }
    if ($env:VIRTUAL_ENV) {
        $candidates += (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
    }
    $candidates += (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    $candidates += "python"

    foreach ($cand in $candidates) {
        if (-not $cand) {
            continue
        }
        if (Test-Path $cand) {
            return (Resolve-Path $cand).Path
        }
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }
    throw "Python interpreter not found. Pass -PythonExe or create .venv in the repo root."
}

Import-DotEnv $envFile

if (-not $Config) {
    $Config = Join-Path $providerRoot "models.yaml"
}
$Config = (Resolve-Path $Config).Path

if ($PublicBaseUrl) {
    $env:PROVIDER_PUBLIC_BASE_URL = $PublicBaseUrl
}

$required = @(
    "PROVIDER_AUTH_PEPPER",
    "PROVIDER_MASTER_KEY",
    "PROVIDER_BOOTSTRAP_ADMIN_USER",
    "PROVIDER_BOOTSTRAP_ADMIN_PASSWORD"
)
$missing = @($required | Where-Object { -not [Environment]::GetEnvironmentVariable($_) })
if ($missing.Count -gt 0) {
    throw "Missing required env vars for native run: $($missing -join ', '). Populate provider/.env first."
}

if ($env:PROVIDER_PUBLIC_BASE_URL -match "0\.0\.0\.0") {
    Write-Warning "PROVIDER_PUBLIC_BASE_URL currently points at 0.0.0.0. Set -PublicBaseUrl to the real external URL if you want remote login / OIDC to be correct."
}

$python = Resolve-Python -Override $PythonExe -ProjectRoot $projectRoot

Set-Location $projectRoot
Write-Host "==> running provider natively"
Write-Host "    python: $python"
Write-Host "    config: $Config"
Write-Host ("    bind  : {0}:{1}" -f $BindHost, $Port)

& $python -m provider --config $Config --host $BindHost --port $Port --log-level $LogLevel
exit $LASTEXITCODE