param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [string]$HostAddress = '127.0.0.1',
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'

$projectDirectory = $PSScriptRoot
$urlHost = if ($HostAddress -match ':' -and -not $HostAddress.StartsWith('[')) { "[$HostAddress]" } else { $HostAddress }
$applicationUrl = "http://${urlHost}:${Port}"
$healthUrl = "$applicationUrl/api/health"

function Show-LauncherError {
    param([string]$Message)
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        $Message,
        'CinematicSRT v2.0',
        'OK',
        'Error'
    ) | Out-Null
}

function Get-V2HealthState {
    param([string]$Url)
    try {
        $resp = Invoke-RestMethod -Uri $Url -TimeoutSec 2
        if ($resp -and $resp.service -eq 'CinematicSRT v2.0') {
            if ($resp.status -eq 'ok') { return 'ready' }
            if ($resp.status -eq 'shutting_down') { return 'shutting_down' }
        }
        return 'other'
    } catch {
        $bodyText = $null
        $statusCode = $null
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
            $bodyText = $_.ErrorDetails.Message
        }
        $exResp = $_.Exception.Response
        if ($exResp) {
            try { $statusCode = [int]$exResp.StatusCode } catch {}
            if (-not $bodyText -and ($exResp.PSObject.Methods.Name -contains 'GetResponseStream')) {
                try {
                    $stream = $exResp.GetResponseStream()
                    if ($stream) {
                        $reader = New-Object System.IO.StreamReader($stream)
                        $bodyText = $reader.ReadToEnd()
                        $reader.Dispose()
                    }
                } catch {}
            }
        }
        if ($bodyText -and ($statusCode -eq 503 -or $statusCode -eq $null)) {
            $parsed = $bodyText | ConvertFrom-Json -ErrorAction SilentlyContinue
            if ($parsed -and $parsed.service -eq 'CinematicSRT v2.0' -and $parsed.status -eq 'shutting_down') {
                return 'shutting_down'
            }
        }
        return 'unreachable'
    }
}

function Test-V2Healthy {
    param([string]$Url)
    return ((Get-V2HealthState -Url $Url) -eq 'ready')
}

$parentDir = Split-Path -Path $projectDirectory -Parent
$pythonCandidates = @(
    (Join-Path $projectDirectory 'venv\Scripts\python.exe'),
    $(if ($parentDir) { Join-Path $parentDir 'cinematic\cinematic-srt\venv\Scripts\python.exe' }),
    (Join-Path $env:USERPROFILE 'Desktop\cinematic\cinematic-srt\venv\Scripts\python.exe')
) | Where-Object { $_ } | Select-Object -Unique

$pythonExecutable = $null
foreach ($candidatePy in $pythonCandidates) {
    if (Test-Path -LiteralPath $candidatePy -PathType Leaf) {
        $pythonExecutable = $candidatePy
        break
    }
}
if (-not $pythonExecutable) {
    $cmdPy = Get-Command python -ErrorAction SilentlyContinue
    if ($cmdPy) {
        $pythonExecutable = $cmdPy.Source
    }
}

if (-not $pythonExecutable) {
    Show-LauncherError "لم يتم العثور على بيئة بايثون (Python) الخاصة ببرنامج CinematicSRT v2.0.`nتأكد من وجود البيئة الافتراضية أو تشغيل setup.bat أولاً."
    exit 1
}

$mutex = New-Object System.Threading.Mutex($false, "Local\CinematicSRT_v2_Launcher_$Port")
$acquired = $false
try {
    try {
        $acquired = $mutex.WaitOne(15000)
    } catch [System.Threading.AbandonedMutexException] {
        $acquired = $true
    }

    $healthState = Get-V2HealthState -Url $healthUrl
    $serverReady = ($healthState -eq 'ready')

    if (-not $serverReady) {
        if (-not $acquired) {
            Show-LauncherError "تعذر الحصول على قفل التشغيل لأن مثيلاً آخر يستغرق وقتاً طويلاً للبدء."
            exit 1
        }

        $busyListener = $null
        $maxWaitAttempts = if ($healthState -eq 'shutting_down') { 16 } else { 8 }
        for ($waitAttempt = 1; $waitAttempt -le $maxWaitAttempts; $waitAttempt++) {
            $busyListener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $busyListener) {
                break
            }
            $healthState = Get-V2HealthState -Url $healthUrl
            if ($healthState -eq 'ready') {
                $serverReady = $true
                break
            }
            if ($healthState -eq 'shutting_down' -and $maxWaitAttempts -lt 16) {
                $maxWaitAttempts = 16
            }
            if ($waitAttempt -lt $maxWaitAttempts) {
                Start-Sleep -Milliseconds 400
            }
        }

        if (-not $serverReady -and $busyListener) {
            Show-LauncherError "المنفذ $Port مستخدم حالياً من برنامج آخر (PID $($busyListener.OwningProcess)).`nيرجى إغلاق البرنامج الآخر ثم المحاولة مجدداً."
            exit 1
        }
    }

    if (-not $serverReady) {
        $tempDir = Join-Path $projectDirectory 'temp'
        New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
        $outLog = Join-Path $tempDir "v2_server_${Port}.out.log"
        $errLog = Join-Path $tempDir "v2_server_${Port}.err.log"

        $spawnScript = @'
import subprocess, sys
cwd, out_path, err_path, host, port = sys.argv[1:6]
with open(out_path, 'w', encoding='utf-8', buffering=1) as fout, open(err_path, 'w', encoding='utf-8', buffering=1) as ferr:
    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'app:app', '--host', host, '--port', port],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=fout,
        stderr=ferr,
        close_fds=True,
        creationflags=0x08000200,
    )
sys.stdout.write('\nCINEMATIC_PID=' + str(proc.pid) + '\n')
sys.stdout.flush()
'@
        $oldEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $spawnLines = & $pythonExecutable -c $spawnScript $projectDirectory $outLog $errLog $HostAddress "$Port" 2>&1
            $spawnExit = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldEap
        }
        $spawnText = (($spawnLines | ForEach-Object { $_.ToString() }) -join "`n").Trim()
        $pidMatch = [regex]::Match($spawnText, '(?m)^CINEMATIC_PID=(\d+)\r?$')
        $serverPid = 0
        if ($pidMatch.Success) {
            [void][int]::TryParse($pidMatch.Groups[1].Value, [ref]$serverPid)
        }
        if ($spawnExit -ne 0 -or -not $pidMatch.Success -or $serverPid -le 0) {
            if ($serverPid -gt 0) {
                Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
            }
            Show-LauncherError "تعذر بدء تشغيل عملية خادم CinematicSRT v2.0.`n$spawnText"
            exit 1
        }

        foreach ($attempt in 1..60) {
            Start-Sleep -Milliseconds 500
            if (Test-V2Healthy -Url $healthUrl) {
                $serverReady = $true
                break
            }
            $alive = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
            if (-not $alive) {
                $errTail = ''
                if (Test-Path -LiteralPath $errLog) {
                    $errTail = (Get-Content -LiteralPath $errLog -Tail 8 -ErrorAction SilentlyContinue) -join "`n"
                }
                Show-LauncherError "توقف خادم CinematicSRT v2.0 أثناء البدء (PID $serverPid).`n$errTail"
                exit 1
            }
        }
    }

    if ($serverReady) {
        if (-not $NoBrowser) {
            Start-Process $applicationUrl
        }
    } else {
        Show-LauncherError "تعذر تشغيل خادم CinematicSRT v2.0 خلال 30 ثانية.`nتحقق من ملفات البرنامج وسجل: $projectDirectory\temp"
        exit 1
    }
} finally {
    if ($acquired) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
