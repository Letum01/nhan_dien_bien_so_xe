param(
    [Parameter(Mandatory=$true)][string]$Destination,
    [switch]$IncludeModels,
    [switch]$ConfirmModelSharingRights
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$dest = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $dest) { throw 'Destination must not already exist.' }
if ($dest.StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Choose a destination outside the project.'
}
if ($IncludeModels -and -not $ConfirmModelSharingRights) {
    throw 'Only include model binaries when you have verified redistribution rights. Add -ConfirmModelSharingRights only then.'
}
$paths = @(& git -C $root ls-files)
if ($LASTEXITCODE -ne 0) { throw 'Cannot enumerate release source via Git.' }
# Include handoff changes even before they are committed; never enumerate all files.
$paths += @('prepare_handoff.ps1','CAI_DAT_VA_CHAY.bat','HUONG_DAN_CAI_DAT_TU_DONG.txt')
$paths = @($paths | Where-Object { $_ -notmatch '^reports[\\/]' } | Sort-Object -Unique)
$spec = Get-Content -LiteralPath (Join-Path $root 'configs\runtime-inference.json') -Raw | ConvertFrom-Json
if ($IncludeModels) {
    foreach ($entry in $spec.artifacts.PSObject.Properties) {
        $src = Join-Path $root $entry.Name
        if (!(Test-Path -LiteralPath $src -PathType Leaf)) { throw "Missing artifact: $src" }
        if ((Get-Item -LiteralPath $src).Length -ne $entry.Value.bytes -or
            (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value.sha256) {
            throw "Artifact hash/size mismatch: $src"
        }
        $paths += $entry.Name
    }
}
$paths = @($paths | Sort-Object -Unique)
# Validate every path BEFORE copying. Source-only handoffs deliberately omit models.
foreach ($name in $paths) {
    if ($name -match '(^|[\\/])(\.git|\.venv[^\\/]*|data|external_datasets)([\\/]|$)' -or
        $name -eq 'alpr-local-env.bat' -or $name -match '\.(sqlite3|db|zip|log|pyc)$') { throw "Forbidden handoff path: $name" }
    if (!(Test-Path -LiteralPath (Join-Path $root $name) -PathType Leaf)) { throw "Missing source: $name" }
}
New-Item -ItemType Directory -Path $dest | Out-Null
$manifest = @()
foreach ($name in $paths) {
    $src = Join-Path $root $name
    $target = Join-Path $dest $name
    New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
    Copy-Item -LiteralPath $src -Destination $target
    $hash = (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $hash) { throw "Copy mismatch: $name" }
    $manifest += [pscustomobject]@{path=$name;bytes=(Get-Item -LiteralPath $target).Length;sha256=$hash.ToLowerInvariant()}
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $dest 'HANDOFF_MANIFEST.json') -Encoding UTF8
Write-Output "Prepared $($paths.Count) verified files at $dest. Models included: $IncludeModels"
Write-Output 'Recipient: extract first, then CAI_DAT_VA_CHAY.bat. Later use START_ALPR.bat.'