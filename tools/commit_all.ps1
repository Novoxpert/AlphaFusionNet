param(
  [string]$Branch = "",
  [int]$MaxFileMB = 80,
  [switch]$NoSubmodulePush
)

$ErrorActionPreference = "Stop"

function Get-DefaultRemote([string]$dir){
  $remotes = @(& git -C $dir remote 2>$null) | ForEach-Object { $_.Trim() } | Where-Object { $_ }
  if ($remotes -contains "origin") { return "origin" }
  if ($remotes.Count -gt 0) { return $remotes[0] }
  return $null
}

function Ensure-CredentialHelper([string]$dir){
  # Avoid the broken helper name "credential-manager-core" which triggers:
  # git: 'credential-manager-core' is not a git command.
  if (Get-Command git-credential-manager -ErrorAction SilentlyContinue) {
    & git -C $dir config --local credential.helper manager | Out-Null
    return
  }
  if (Get-Command git-credential-manager-core -ErrorAction SilentlyContinue) {
    # some setups still use manager-core, but if it doesn't exist, it will break
    & git -C $dir config --local credential.helper manager-core | Out-Null
    return
  }
  # no GCM -> do not set any helper here
  & git -C $dir config --local --unset credential.helper 2>$null | Out-Null
}

function Assert-NoSensitiveFiles([string]$dir, [string]$label){
  $bad = @()
  $tracked = @(& git -C $dir ls-files 2>$null)
  foreach($p in $tracked){
    $pp = $p.Replace("\","/").ToLowerInvariant()
    if ($pp -match '(^|/)\.env($|\.|_)') { $bad += $p; continue }
    if ($pp -match '\.(pem|key|pfx|p12)$') { $bad += $p; continue }
    if ($pp -match '(^|/)id_rsa($|\.|_)') { $bad += $p; continue }
    if ($pp -match '(token|secret|credential)') { } # heuristic only (not blocking)
  }
    $bad = @($bad) | Where-Object { $_ -notmatch '(^|[\\/])\.env(\.example|_example|\.sample|_sample|\.template|_template)$' }
  if ($bad.Count -gt 0){
    throw ("Possible sensitive file(s) tracked in {0}:`n- {1}`nRemove from git index (git rm --cached) before pushing." -f $label, ($bad -join "`n- "))
  }
}

function Assert-NoHugeTrackedFiles([string]$dir, [string]$label, [int]$maxBytes){
  $tracked = @(& git -C $dir ls-files 2>$null)
  foreach($p in $tracked){
    $full = Join-Path $dir $p
    if (Test-Path -LiteralPath $full){
      $fi = Get-Item -LiteralPath $full -Force -ErrorAction SilentlyContinue
      if ($fi -and -not $fi.PSIsContainer -and $fi.Length -gt $maxBytes){
        throw ("File too large in {0}: {1} ({2} bytes). Use LFS or ignore it." -f $label, $p, $fi.Length)
      }
    }
  }
}

function Ensure-Branch([string]$dir, [string]$branch){
  if ([string]::IsNullOrWhiteSpace($branch)) { return }
  & git -C $dir show-ref --verify --quiet ("refs/heads/$branch") 2>$null
  if ($LASTEXITCODE -eq 0) {
    & git -C $dir checkout $branch | Out-Null
  } else {
    & git -C $dir checkout -b $branch | Out-Null
  }
}

function CommitPush([string]$dir, [string]$msg, [switch]$noPush){
  & git -C $dir add -A | Out-Null

  & git -C $dir diff --cached --quiet
  $hasStaged = ($LASTEXITCODE -ne 0)

  if ($hasStaged){
    & git -C $dir commit -m $msg | Out-Null
  }

  if ($noPush) { return }

  $remote = Get-DefaultRemote $dir
  if (-not $remote) { throw ("No git remote found in: {0}" -f $dir) }

  $branch = (& git -C $dir rev-parse --abbrev-ref HEAD).Trim()

  $up = ""
  try { $up = (& git -C $dir rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null).Trim() } catch { $up = "" }

  if ([string]::IsNullOrWhiteSpace($up)) {
    & git -C $dir push -u $remote $branch | Out-Null
  } else {
    & git -C $dir push | Out-Null
  }
}

# ---- main ----
$repo = (& git rev-parse --show-toplevel).Trim()
Set-Location $repo

Ensure-CredentialHelper $repo
& git -C $repo config --local push.autoSetupRemote true | Out-Null

if ([string]::IsNullOrWhiteSpace($Branch)) {
  $Branch = (& git -C $repo branch --show-current).Trim()
}

$stamp = (Get-Date -Format "yyyy-MM-dd HH:mm")
$maxBytes = $MaxFileMB * 1MB

# 1) Commit/push dirty submodules first (so pointers can move)
$subLines = @(& git -C $repo submodule status --recursive 2>$null)
foreach($line in $subLines){
  $t = $line.Trim()
  if (-not $t) { continue }
  $parts = $t -split "\s+"
  if ($parts.Count -lt 2) { continue }
  $rel = $parts[1]
  $abs = Join-Path $repo $rel
  if (-not (Test-Path -LiteralPath $abs)) { continue }

  $dirtyCount = (@(& git -C $abs status --porcelain 2>$null) | Measure-Object).Count
  if ($dirtyCount -gt 0){
    Ensure-CredentialHelper $abs
    Ensure-Branch $abs $Branch
    Assert-NoSensitiveFiles $abs ("submodule " + $rel)
    Assert-NoHugeTrackedFiles $abs ("submodule " + $rel) $maxBytes
    CommitPush $abs ("wip: submodule {0} ({1})" -f $rel, $stamp) -noPush:$NoSubmodulePush
    Write-Host ("OK submodule: {0}" -f $rel) -ForegroundColor Green
  }
}

# 2) Commit/push main repo (including updated submodule pointers)
Ensure-Branch $repo $Branch
Assert-NoSensitiveFiles $repo "main repo"
Assert-NoHugeTrackedFiles $repo "main repo" $maxBytes
CommitPush $repo ("wip: save all changes ({0})" -f $stamp) -noPush:$false
Write-Host ("OK main repo: {0}" -f $repo) -ForegroundColor Green

Write-Host ("DONE. Everything committed/pushed on branch: {0}" -f $Branch) -ForegroundColor Cyan
