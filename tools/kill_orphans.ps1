# Stop orphaned Jaxspike work processes left behind when a Claude session crashes: pytest runs, soaks, benches,
# test servers, headless Godot and Blender renders. Run from PowerShell:  tools\kill_orphans.ps1
# Add -KeepServers to leave net.server processes (for example a playtest server on port 8766) running.
param([switch]$KeepServers)
$killed = 0
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='Godot_v4.7.2-stable_win64_console.exe' OR Name='Godot_v4.7.2-stable_win64.exe' OR Name='blender.exe'") {
    $c = $p.CommandLine
    if (-not $c) { continue }
    $heavy = ($c -match 'pytest|soak_|bench[/\\]|server_load|world_scale|--headless|-s tests|blender')
    $server = ($c -match 'net\.server')
    if ($server -and $KeepServers) { continue }
    if ($heavy -or $server) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; $killed++ } catch {}
    }
}
$free = [int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
"KILL_ORPHANS stopped=$killed free_mb=$free"
