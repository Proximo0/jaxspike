<#
Builds the ops board's data files (JSON, UTF-8 without BOM) for any game project from local run logs.

Usage (PowerShell 5+):
  .\make_ops_data.ps1 -Out .\data -ProjectName "My game ops" -Machine "MSI laptop" -Gpu "RTX 5050 Laptop GPU" `
      -Engine "Godot 4.7.2" -SoakSetup "High graphics, 1080p" `
      -SoakLog $env:TEMP\soak.log -HitchLog $env:TEMP\hitch.log -RunnerDir C:\actions-runner `
      -ModelDirs @{ "Weapons" = "C:\path\export\weapons"; "Gore" = "C:\path\export\gore" } -DefaultBudget 3000

Every input is optional: a dataset whose input is missing is skipped (the board shows that card as empty).
Hand-kept files (jobs.json, perf_heavy.json, script_cost.json, build_parts.json) are left alone if they exist.
See README.md for each dataset's columns.
#>
param(
    [Parameter(Mandatory)] [string]$Out,
    [string]$ProjectName = "Game ops",
    [string]$Machine = $env:COMPUTERNAME,
    [string]$Gpu = "",
    [string]$Engine = "",
    [string]$SoakSetup = "",
    [string]$SoakLog = "",
    [string]$HitchLog = "",
    [string]$RunnerDir = "",
    [hashtable]$ModelDirs = @{},
    [hashtable]$Budgets = @{},
    [int]$DefaultBudget = 3000,
    [string]$RepoDir = "",
    [string]$MainBranch = "main",
    [string]$CodeDir = "game"
)

$enc = New-Object Text.UTF8Encoding $false
New-Item -ItemType Directory -Force $Out | Out-Null
function Save($name, $rows) {
    [IO.File]::WriteAllText((Join-Path $Out $name), (ConvertTo-Json @($rows) -Compress -Depth 5), $enc)
    Write-Host ("wrote {0} ({1} rows)" -f $name, @($rows).Count)
}

Save 'project.json' @([ordered]@{ name = $ProjectName; machine = $Machine; gpu = $Gpu; engine = $Engine; soak_setup = $SoakSetup })

# Soak: lines "SOAK fight=N seed=S ticks=T secs=W fps=F p95_ms=P max_ms=M static_mb=a->b video_mb=c->d nodes=n objects=o"
if ($SoakLog -and (Test-Path $SoakLog)) {
    $txt = (Get-Content $SoakLog -Raw) -replace "\r?\n(?=video_mb)", " "
    $rows = foreach ($m in [regex]::Matches($txt, 'SOAK fight=(\d+) seed=(\d+) ticks=(\d+) secs=([\d.]+) fps=([\d.]+) p95_ms=([\d.]+) max_ms=([\d.]+) static_mb=([\d.]+)->([\d.]+)\s+video_mb=([\d.]+)->([\d.]+) nodes=(\d+) objects=(\d+)')) {
        $g = $m.Groups
        [ordered]@{ fight = [int]$g[1].Value; seed = [int]$g[2].Value; game_seconds = [math]::Round([int]$g[3].Value / 60, 1);
            fps = [double]$g[5].Value; p95_frame_ms = [double]$g[6].Value; worst_frame_ms = [double]$g[7].Value;
            static_mem_mb = [double]$g[9].Value; video_mem_mb = [double]$g[11].Value; nodes = [int]$g[12].Value; objects = [int]$g[13].Value }
    }
    Save 'soak_fights.json' $rows
}

# Hitches: "HITCH fight=N build_ms=X" and "HITCH fight=N frame=F tick=T ms=X objects=O events=..."
if ($HitchLog -and (Test-Path $HitchLog)) {
    $log = Get-Content $HitchLog
    $frames = foreach ($l in $log) {
        if ($l -match 'HITCH fight=(\d+) frame=(\d+) tick=\d+ ms=([\d.]+) objects=\d+ events=(.*)$') {
            $f = [int]$matches[1]; $fr = [int]$matches[2]; $ev = $matches[4]
            $cause = if ($ev) { 'First-time effect (' + ($ev -replace 'gore:', '' -replace ':', ' ') + ')' }
                elseif ($fr -le 2 -and $f -eq 1) { 'First frame, first match (shader compiles)' }
                elseif ($fr -le 2) { 'First frame of the match' } else { 'Mid-match' }
            [ordered]@{ fight = $f; frame = $fr; ms = [double]$matches[3]; cause = $cause }
        }
    }
    $starts = foreach ($l in $log) {
        if ($l -match 'HITCH fight=(\d+) build_ms=([\d.]+)') {
            $f = [int]$matches[1]
            $ff = $frames | Where-Object { $_.fight -eq $f -and $_.frame -le 2 } | Select-Object -First 1
            [ordered]@{ fight = $f; build_ms = [double]$matches[2]; first_frame_ms = $(if ($ff) { $ff.ms } else { 0 }) }
        }
    }
    Save 'hitch_frames.json' $frames
    Save 'match_start.json' $starts
}

# GitHub Actions self-hosted runner: one row per Worker_*.log, plus the latest perf_*.json in its workspace.
if ($RunnerDir -and (Test-Path "$RunnerDir\_diag")) {
    $runs = foreach ($f in Get-ChildItem "$RunnerDir\_diag" -Filter Worker_*.log | Sort-Object Name) {
        $t = Get-Content $f.FullName
        $res = (($t | Select-String 'Job result after all job steps finish: (\w+)' | Select-Object -Last 1).Matches.Groups | Select-Object -Last 1).Value
        $failstep = ''
        for ($i = 0; $i -lt $t.Count; $i++) {
            if ($t[$i] -match 'Step result: Failed') {
                for ($j = $i; $j -ge 0; $j--) { if ($t[$j] -match "Processing step: DisplayName='([^']+)'") { $failstep = $matches[1]; break } }
                break
            }
        }
        $st = [datetime]::ParseExact(($f.BaseName -replace 'Worker_', ''), 'yyyyMMdd-HHmmss-\u\t\c', $null)
        [ordered]@{ started = $st.ToString('yyyy-MM-ddTHH:mm:ssZ'); result = $(if ($res) { $res } else { 'Running' });
            failed_step = $(if ($failstep) { $failstep } else { '-' }); cause = '-' }
    }
    Save 'runner_jobs.json' $runs
    $ws = Get-ChildItem "$RunnerDir\_work" -Recurse -Depth 3 -Filter 'perf_*.json' -ErrorAction SilentlyContinue
    if ($ws) {
        $perf = foreach ($p in $ws | Sort-Object Name) {
            $j = Get-Content $p.FullName -Raw | ConvertFrom-Json
            [ordered]@{ run = $p.LastWriteTimeUtc.ToString('yyyy-MM-ddTHH:mm:ssZ'); tier = $(switch ([int]$j.quality) { 0 { 'Low' } 1 { 'Medium' } default { 'High' } });
                old_counter_ms = [math]::Round($j.process_ms.mean, 2); physics_ms = [math]::Round($j.physics_ms.mean, 2); effects_ms = [math]::Round($j.fx_ms.mean, 2);
                draw_calls = [math]::Round($j.draw_calls.mean); triangles = [math]::Round($j.primitives.mean); video_mem_mb = $j.video_mem_mb }
        }
        Save 'runner_perf.json' $perf
    }
}

# Models: triangle count straight from each .glb, grouped by folder.
function Get-GlbTris($path) {
    $data = [IO.File]::ReadAllBytes($path)
    $jlen = [BitConverter]::ToInt32($data, 12)
    $doc = [Text.Encoding]::UTF8.GetString($data, 20, $jlen) | ConvertFrom-Json
    $tris = 0
    foreach ($mesh in $doc.meshes) { foreach ($prim in $mesh.primitives) {
        if ($null -ne $prim.indices) { $tris += [int]($doc.accessors[$prim.indices].count / 3) }
        else { $tris += [int]($doc.accessors[$prim.attributes.POSITION].count / 3) } } }
    return $tris
}
if ($ModelDirs.Count -gt 0) {
    $models = foreach ($g in $ModelDirs.Keys) {
        foreach ($glb in Get-ChildItem $ModelDirs[$g] -Filter *.glb -ErrorAction SilentlyContinue | Sort-Object Name) {
            [ordered]@{ model = $glb.BaseName; group = $g; triangles = Get-GlbTris $glb.FullName; size_kb = [math]::Round($glb.Length / 1024, 1);
                budget = $(if ($Budgets.ContainsKey($g)) { $Budgets[$g] } else { $DefaultBudget }) }
        }
    }
    Save 'models.json' $models
}

# Build status from a git clone: recent commits on the main branch, branches waiting to merge, and which asset folders
# the game code actually loads (a folder nobody references is built but not in the game yet).
if ($RepoDir -and (Test-Path "$RepoDir\.git")) {
    Push-Location $RepoDir
    try {
        git fetch -q origin 2>$null
        $commits = foreach ($l in (git log "origin/$MainBranch" -25 --format='%h|%cI|%s')) {
            $p = $l -split '\|', 3
            [ordered]@{ sha = $p[0]; date = ([datetime]$p[1]).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); subject = $p[2];
                kind = $(if ($p[2] -like 'Merge*') { 'Merge' } else { 'Change' }) }
        }
        Save 'builds.json' $commits
        $branches = foreach ($r in (git for-each-ref refs/remotes/origin --format='%(refname:short)')) {
            if ($r -match 'HEAD$' -or $r -eq "origin/$MainBranch") { continue }
            $ahead = [int](git rev-list --count "origin/$MainBranch..$r")
            if ($ahead -eq 0) { continue }
            $last = (git log -1 --format='%cI|%s' $r) -split '\|', 2
            [ordered]@{ branch = $r -replace '^origin/', ''; ahead = $ahead; last_commit = ([datetime]$last[0]).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); subject = $last[1] }
        }
        Save 'branches.json' ($branches | Sort-Object { $_.last_commit } -Descending)
        $groups = git ls-tree -r --name-only "origin/$MainBranch" assets | Where-Object { $_ -match '\.(glb|png)$' } |
            ForEach-Object { ($_ -split '/')[1] } | Sort-Object -Unique
        $assets = foreach ($g in $groups) {
            $n = @(git ls-tree -r --name-only "origin/$MainBranch" "assets/$g" | Where-Object { $_ -match '\.(glb|png)$' }).Count
            $refs = @(git grep -l "assets/$g" "origin/$MainBranch" -- $CodeDir "*.godot" "*.cfg" 2>$null).Count
            [ordered]@{ group = $g; files_on_main = $n; code_files_using = $refs; status = $(if ($refs -gt 0) { 'In game' } else { 'On main, not wired in' }) }
        }
        Save 'asset_status.json' $assets
    } finally { Pop-Location }
}

foreach ($hand in 'jobs.json', 'perf_heavy.json', 'script_cost.json', 'build_parts.json', 'backlog.json') {
    if (-not (Test-Path (Join-Path $Out $hand))) { Save $hand @() }
}
