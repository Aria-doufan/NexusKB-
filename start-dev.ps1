param(
    [switch]$Migrate,
    [switch]$SkipElasticsearch,
    [string]$CondaEnv = "nexuskb",
    [string]$RedisServerPath = "D:\Tools\Redis-7.4.8-Windows-x64-msys2\redis-server.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$DjangoDir = Join-Path $ProjectRoot "DjangoUserService"
$BackendDir = Join-Path $ProjectRoot "backend"
$FrontendDir = Join-Path $ProjectRoot "front"
$ElasticsearchComposeFile = Join-Path $ProjectRoot "docker-compose.elasticsearch.yml"
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

function Test-RequiredPath {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Name not found: $Path"
    }
}

function Test-RequiredCommand {
    param(
        [string]$CommandName,
        [string]$InstallHint
    )

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "$CommandName was not found on PATH. $InstallHint"
    }
}

function Convert-ToSingleQuotedPowerShellString {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function New-PythonCommand {
    param([string]$Command)

    if (-not [string]::IsNullOrWhiteSpace($Script:PythonExecutable)) {
        if ($Command.StartsWith("python ")) {
            return "& $(Convert-ToSingleQuotedPowerShellString $Script:PythonExecutable) $($Command.Substring(7))"
        }

        return "& $(Convert-ToSingleQuotedPowerShellString $Script:PythonExecutable) -m $Command"
    }

    if (-not [string]::IsNullOrWhiteSpace($CondaEnv)) {
        return "conda run --no-capture-output -n $(Convert-ToSingleQuotedPowerShellString $CondaEnv) $Command"
    }

    return $Command
}

function Get-CondaEnvironmentPath {
    param([string]$EnvironmentName)

    $envList = conda env list | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith("#") }
    foreach ($line in $envList) {
        $parts = $line -split "\s+"
        if ($parts[0] -eq $EnvironmentName) {
            return $parts[-1]
        }
    }

    throw "Conda environment not found: $EnvironmentName"
}

function New-ServiceCommand {
    param(
        [string]$WorkingDirectory,
        [string[]]$Commands,
        [switch]$UseVenv
    )

    $parts = @(
        "Set-Location -LiteralPath $(Convert-ToSingleQuotedPowerShellString $WorkingDirectory)",
        "`$env:PYTHONUTF8 = '1'"
    )

    if ($UseVenv -and [string]::IsNullOrWhiteSpace($CondaEnv) -and (Test-Path -LiteralPath $VenvActivate)) {
        $parts += ". $(Convert-ToSingleQuotedPowerShellString $VenvActivate)"
    }

    if ($UseVenv -and -not [string]::IsNullOrWhiteSpace($Script:LibMagicDir) -and (Test-Path -LiteralPath $Script:LibMagicDir)) {
        $parts += "`$env:PATH = $(Convert-ToSingleQuotedPowerShellString $Script:LibMagicDir) + ';' + `$env:PATH"
    }

    $parts += $Commands
    return $parts -join "; "
}

function Start-ServiceWindow {
    param(
        [string]$Title,
        [string]$Command
    )

    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command", "`$Host.UI.RawUI.WindowTitle = '$Title'; $Command"
    )
}

function Test-PortListening {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Assert-PortAvailable {
    param(
        [int]$Port,
        [string]$ServiceName
    )

    if (Test-PortListening $Port) {
        throw "$ServiceName port $Port is already in use. Stop the existing process or change the configured port before running this script."
    }
}

function Wait-PortListening {
    param(
        [int]$Port,
        [int]$TimeoutSeconds = 10
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortListening $Port) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }

    return $false
}

function Test-DockerComposeAvailable {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return $false
    }

    $output = & docker compose version 2>$null
    return $LASTEXITCODE -eq 0
}

function Start-ElasticsearchIfNeeded {
    if ($SkipElasticsearch) {
        Write-Host "Skipping Elasticsearch Docker startup." -ForegroundColor Yellow
        return
    }

    if (Test-PortListening 9200) {
        Write-Host "Elasticsearch is already listening on port 9200." -ForegroundColor Green
        return
    }

    Test-RequiredPath $ElasticsearchComposeFile "Elasticsearch Docker Compose file"

    if (-not (Test-DockerComposeAvailable)) {
        throw "Docker Compose is not available. Start Docker Desktop and ensure 'docker compose version' works, or rerun with -SkipElasticsearch."
    }

    Write-Host "Starting Elasticsearch with Docker Compose..." -ForegroundColor Cyan
    & docker compose -f $ElasticsearchComposeFile up -d
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Elasticsearch with Docker Compose. Check Docker Desktop and docker-compose.elasticsearch.yml, or rerun with -SkipElasticsearch."
    }

    if (-not (Wait-PortListening -Port 9200 -TimeoutSeconds 60)) {
        throw "Elasticsearch did not start listening on port 9200 within 60 seconds. Check 'docker compose -f docker-compose.elasticsearch.yml logs elasticsearch'."
    }

    Write-Host "Elasticsearch is listening on port 9200." -ForegroundColor Green
}

function Start-RedisIfNeeded {
    if (Test-PortListening 6379) {
        Write-Host "Redis is already listening on port 6379." -ForegroundColor Green
        return
    }

    $redisCommand = Get-Command redis-server -ErrorAction SilentlyContinue
    $redisPath = if ($redisCommand) { $redisCommand.Source } else { $RedisServerPath }

    if (-not (Test-Path -LiteralPath $redisPath)) {
        throw "Redis server not found. Install Redis, add redis-server to PATH, or pass -RedisServerPath '<path-to-redis-server.exe>'."
    }

    Write-Host "Starting Redis on port 6379..." -ForegroundColor Cyan
    Start-ServiceWindow -Title "NexusKB Redis :6379" -Command "& $(Convert-ToSingleQuotedPowerShellString $redisPath)"

    if (-not (Wait-PortListening -Port 6379 -TimeoutSeconds 10)) {
        throw "Redis did not start listening on port 6379 within 10 seconds. Check the Redis window for errors."
    }

    Write-Host "Redis is listening on port 6379." -ForegroundColor Green
}

function Test-EnvFile {
    param(
        [string]$Path,
        [string]$ExamplePath,
        [string]$Name
    )

    if (Test-Path -LiteralPath $Path) {
        return
    }

    if (Test-Path -LiteralPath $ExamplePath) {
        Write-Host "Warning: $Name not found. Copy $ExamplePath to $Path and configure it if startup fails." -ForegroundColor Yellow
        return
    }

    Write-Host "Warning: $Name not found, and no example file was found at $ExamplePath." -ForegroundColor Yellow
}

function Test-ProjectPreflight {
    Test-RequiredPath $DjangoDir "Django service directory"
    Test-RequiredPath $BackendDir "FastAPI backend directory"
    Test-RequiredPath $FrontendDir "Frontend directory"
    if (-not $SkipElasticsearch) {
        Test-RequiredPath $ElasticsearchComposeFile "Elasticsearch Docker Compose file"
    }

    Test-RequiredCommand "npm" "Install Node.js, then run npm install in the front directory."

    if (-not [string]::IsNullOrWhiteSpace($CondaEnv)) {
        Test-RequiredCommand "conda" "Start Anaconda PowerShell Prompt or pass -CondaEnv '' to use .venv/PATH Python."
        $condaEnvPath = Get-CondaEnvironmentPath $CondaEnv
        $Script:PythonExecutable = Join-Path $condaEnvPath "python.exe"
        Test-RequiredPath $Script:PythonExecutable "Python executable for conda environment '$CondaEnv'"
        $Script:LibMagicDir = Join-Path $condaEnvPath "Lib\site-packages\magic\libmagic"
    }

    if ([string]::IsNullOrWhiteSpace($CondaEnv) -and -not (Test-Path -LiteralPath $VenvActivate)) {
        Write-Host "Warning: .venv was not found. The script will use python from PATH." -ForegroundColor Yellow
    }

    Test-EnvFile -Path (Join-Path $DjangoDir ".env") -ExamplePath (Join-Path $DjangoDir ".env.example") -Name "DjangoUserService\.env"
    Test-EnvFile -Path (Join-Path $BackendDir ".env") -ExamplePath (Join-Path $BackendDir ".env.example") -Name "backend\.env"

    if (-not (Test-Path -LiteralPath (Join-Path $FrontendDir "node_modules"))) {
        throw "Frontend dependencies are not installed. Run 'npm install' in $FrontendDir before starting the project."
    }

    if (-not (Test-PortListening 3306)) {
        throw "MySQL is not listening on port 3306. Start MySQL before running this script."
    }

    Assert-PortAvailable -Port 8000 -ServiceName "FastAPI"
    Assert-PortAvailable -Port 8001 -ServiceName "Django"
    Assert-PortAvailable -Port 3000 -ServiceName "Frontend"
}

$Script:LibMagicDir = $null
$Script:PythonExecutable = $null

Test-ProjectPreflight

Write-Host "Starting NexusKB development services..." -ForegroundColor Cyan
if (-not [string]::IsNullOrWhiteSpace($CondaEnv)) {
    Write-Host "Using conda environment: $CondaEnv" -ForegroundColor Cyan
    Write-Host "Using Python executable: $Script:PythonExecutable" -ForegroundColor Cyan
}
Write-Host "MySQL is listening on port 3306." -ForegroundColor Green
Start-ElasticsearchIfNeeded
Start-RedisIfNeeded

$djangoCommands = @()
if ($Migrate) {
    $djangoCommands += New-PythonCommand "python manage.py migrate"
}
$djangoCommands += New-PythonCommand "python manage.py runserver 8001"

$djangoCommand = New-ServiceCommand -WorkingDirectory $DjangoDir -Commands $djangoCommands -UseVenv
$backendCommand = New-ServiceCommand -WorkingDirectory $BackendDir -Commands @(New-PythonCommand "uvicorn main:app --reload") -UseVenv
$frontendCommand = New-ServiceCommand -WorkingDirectory $FrontendDir -Commands @("npm run dev")

Start-ServiceWindow -Title "NexusKB Django :8001" -Command $djangoCommand
Start-ServiceWindow -Title "NexusKB FastAPI :8000" -Command $backendCommand
Start-ServiceWindow -Title "NexusKB Frontend :3000" -Command $frontendCommand

Write-Host "Started development services in separate PowerShell windows." -ForegroundColor Green
Write-Host "Frontend: http://127.0.0.1:3000" -ForegroundColor Green
Write-Host "FastAPI docs: http://127.0.0.1:8000/docs" -ForegroundColor Green
