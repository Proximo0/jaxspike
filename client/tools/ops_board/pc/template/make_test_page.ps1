# Builds <DataDir>\test.html: the board page (template\index.html) with a mocked `dash` API and the data files from
# make_ops_data.ps1, so the page can be checked in any browser before it is published. Serve the folder over http
# (file:// blocks scripts in some viewers), open test.html, and read window.__errors in the console.
param([Parameter(Mandatory)] [string]$DataDir, [string]$Page = (Join-Path $PSScriptRoot "index.html"))
$d = $DataDir
$enc = New-Object Text.UTF8Encoding $false
$map = [ordered]@{ soak = 'soak_fights'; perf = 'perf_heavy'; scripts = 'script_cost'; models = 'models'; jobs = 'jobs';
    match_start = 'match_start'; build_parts = 'build_parts'; hitch_frames = 'hitch_frames'; runner_jobs = 'runner_jobs'; runner_perf = 'runner_perf'; project = 'project'; backlog = 'backlog'; branches = 'branches'; asset_status = 'asset_status'; builds = 'builds' }
$js = "window.__DATA={" + (($map.GetEnumerator() | ForEach-Object { "'$($_.Key)':" + [IO.File]::ReadAllText("$d\$($_.Value).json") }) -join ',') + "};"
$page = ConvertTo-Json ([IO.File]::ReadAllText($Page))
$tokens = ':root{--color-fg:#222;--color-fg-muted:#666;--color-bg:#fff;--color-panel:#f7f7f5;--color-border-line:#ddd;--color-ok:#1a7f37;--color-warn:#9a6700;--color-bad:#cf222e;--cds-gap-xs:4px;--cds-gap-sm:8px;--cds-gap-md:12px;--cds-gap-lg:20px;--cds-pad-xs:4px;--cds-pad-sm:8px;--cds-pad-lg:16px;--cds-radius:8px;--cds-font-size-caption:12px;--cds-font-size-body:14px;--cds-font-size-heading:16px;--cds-font-size-title:28px;--cds-font-weight-medium:500;--cds-chart-grid:#eee;--cds-chart-axis:#bbb;--cds-chart-muted:#ccc;--cds-chart-reference:#888;--cds-chart-status-critical:#cf222e;--cds-surface-popover:#fff;--cds-border:#ccc;--cds-shadow-popover:0 2px 8px rgba(0,0,0,.15);--cds-text-secondary:#666;--font-anthropic-sans:system-ui;--font-anthropic-serif:Georgia}body{margin:16px;background:#fff}'
$mock = @'
window.__errors=[];window.addEventListener('error',e=>window.__errors.push(String(e.message)));
const calcs={},subs=[];
const rowsOf=id=>window.__DATA[id]||(calcs[id]&&calcs[id].fn(...calcs[id].inputs.map(rowsOf)));
window.dash={colors:['#2f6fde','#d97706','#16a34a','#9333ea','#e11d48','#0891b2','#65a30d','#db2777'],
 calc(id,o){calcs[id]=o},loader(){},setLink(){},refresh(){},params(){return{}},setParams(){},resetParams(){},
 data(id){ try{ const r=rowsOf(id); if(r) return {status:'ok',data:r}; }catch(e){window.__errors.push('calc '+id+': '+e.message);return {status:'error',data:[]}} window.__errors.push('missing '+id); return {status:'missing',data:[]} },
 onData(fn){subs.push(fn); try{fn()}catch(e){window.__errors.push('onData: '+e.message)}}};
window.addEventListener('resize',()=>subs.forEach(f=>{try{f()}catch(e){window.__errors.push('onData: '+e.message)}}));
'@
$html = "<!doctype html><html><head><meta charset=`"utf-8`"><title>Dash test</title><style>$tokens</style>" +
    "<script src=`"https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js`"></script></head><body><div id=`"dash-root`"></div><script>$js`n$mock`n" +
    "const root=document.getElementById('dash-root');const tpl=document.createElement('div');tpl.innerHTML=$page;" +
    "[...tpl.childNodes].forEach(n=>{if(n.tagName==='SCRIPT'){const s=document.createElement('script');s.textContent=n.textContent;root.appendChild(s);}else root.appendChild(n);});</script></body></html>"
[IO.File]::WriteAllText("$d\test.html", $html, $enc)

