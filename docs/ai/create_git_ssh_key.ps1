# Creates a new SSH keypair for Git usage for the current Windows user.
# - Prefers Ed25519; falls back to RSA-4096.
# - Never overwrites existing keys.
# - Locks down permissions on the private key and .ssh folder.
# - Best-effort: starts ssh-agent and adds the key.
# - Prints the public key and tries to copy to clipboard.
# - Updates ~/.ssh/config for github.com and gitlab.com without duplicating blocks.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Section([string]$title) {
  Write-Host "" 
  Write-Host "=== $title ==="
}

function Get-ToolPath([string]$name) {
  $cmd = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($cmd) { return $cmd.Source }
  return $null
}

function Ensure-SshHostBlock {
  param(
    [string[]]$Lines,
    [Parameter(Mandatory)] [string]$HostName,
    [Parameter(Mandatory)] [string]$IdentityFile
  )

  $hostLine = "Host $HostName"
  $startIdx = -1
  for ($i = 0; $i -lt $Lines.Count; $i++) {
    if ($Lines[$i].Trim() -ieq $hostLine) { $startIdx = $i; break }
  }

  $desiredBlock = @(
    $hostLine,
    "  HostName $HostName",
    "  User git",
    "  IdentityFile $IdentityFile",
    "  IdentitiesOnly yes"
  )

  # Append new block if host not found
  if ($startIdx -lt 0) {
    if ($Lines.Count -gt 0 -and $Lines[-1].Trim().Length -gt 0) { $Lines += '' }
    $Lines += $desiredBlock
    return ,$Lines
  }

  # Determine end of host block (next Host or EOF)
  $endIdx = $Lines.Count - 1
  for ($i = $startIdx + 1; $i -lt $Lines.Count; $i++) {
    if ($Lines[$i] -match '^\s*Host\s+') { $endIdx = $i - 1; break }
  }

  # Update/insert IdentityFile within block
  $identityUpdated = $false
  for ($i = $startIdx; $i -le $endIdx; $i++) {
    if ($Lines[$i] -match '^\s*IdentityFile\s+') {
      $Lines[$i] = "  IdentityFile $IdentityFile"
      $identityUpdated = $true
      break
    }
  }

  if (-not $identityUpdated) {
    # Insert after User line if present, else after Host line
    $insertAt = $startIdx + 1
    for ($i = $startIdx; $i -le $endIdx; $i++) {
      if ($Lines[$i] -match '^\s*User\s+') { $insertAt = $i + 1; break }
    }
    $before = @()
    if ($insertAt -gt 0) { $before = $Lines[0..($insertAt-1)] }
    $after = @()
    if ($insertAt -lt $Lines.Count) { $after = $Lines[$insertAt..($Lines.Count-1)] }
    $Lines = $before + @("  IdentityFile $IdentityFile") + $after
  }

  return ,$Lines
}

Write-Section 'Environment & prerequisites'
Write-Host ('PowerShell: ' + $PSVersionTable.PSVersion)

$ssh = Get-ToolPath 'ssh'
$sshKeygen = Get-ToolPath 'ssh-keygen'
$sshAdd = Get-ToolPath 'ssh-add'

Write-Host ('ssh:        ' + ($(if ($null -ne $ssh) { $ssh } else { '<missing>' })))
Write-Host ('ssh-keygen:  ' + ($(if ($null -ne $sshKeygen) { $sshKeygen } else { '<missing>' })))
Write-Host ('ssh-add:     ' + ($(if ($null -ne $sshAdd) { $sshAdd } else { '<missing>' })))

if (-not $sshKeygen) {
  throw 'OpenSSH ssh-keygen is missing; cannot continue.'
}

Write-Section 'Create ~/.ssh and select a safe key filename'
$sshDir = Join-Path $env:USERPROFILE '.ssh'
if (-not (Test-Path $sshDir)) {
  New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
  Write-Host ('Created: ' + $sshDir)
} else {
  Write-Host ('Exists:  ' + $sshDir)
}

$keyPath = Join-Path $sshDir 'id_ed25519'
$pubPath = $keyPath + '.pub'

Write-Host ('Private key path: ' + $keyPath)
Write-Host ('Public  key path: ' + $pubPath)

Write-Section 'Generate keypair (Ed25519; fallback to RSA)'
$comment = "$env:USERNAME@$env:COMPUTERNAME"

# If the key already exists, do NOT create a second key on re-runs.
# This keeps the script idempotent and avoids producing extra keys.
if ((Test-Path $keyPath) -and (Test-Path $pubPath)) {
  Write-Host 'Key already exists; skipping key generation.'
  $code = 0
} else {

function Run-SshKeygenEd25519([string]$KeyPath, [string]$Comment) {
  # NOTE: Windows PowerShell 5.1 can drop empty-string arguments when invoking native EXEs.
  # Use Start-Process with a single argument string that includes: -N "" (empty passphrase)
  $argString = (@(
      '-t','ed25519',
      '-a','64',
      '-C', ('"' + $Comment + '"'),
      '-f', ('"' + $KeyPath + '"'),
      '-N','""'
    ) -join ' ')

  $p = Start-Process -FilePath $sshKeygen -ArgumentList $argString -NoNewWindow -Wait -PassThru
  return $p.ExitCode
}

function Run-SshKeygenRsa4096([string]$KeyPath, [string]$Comment) {
  $argString = (@(
      '-t','rsa',
      '-b','4096',
      '-a','64',
      '-C', ('"' + $Comment + '"'),
      '-f', ('"' + $KeyPath + '"'),
      '-N','""'
    ) -join ' ')

  $p = Start-Process -FilePath $sshKeygen -ArgumentList $argString -NoNewWindow -Wait -PassThru
  return $p.ExitCode
}


  $code = Run-SshKeygenEd25519 -KeyPath $keyPath -Comment $comment

if ($code -ne 0 -or -not (Test-Path $keyPath)) {
  Write-Warning "Ed25519 generation failed (exit $code). Trying RSA-4096..."
  $code = Run-SshKeygenRsa4096 -KeyPath $keyPath -Comment $comment
}

}

if (-not (Test-Path $keyPath) -or -not (Test-Path $pubPath)) {
  throw "Key generation failed; expected files not found: $keyPath and/or $pubPath"
}

Write-Host 'Key generation complete.'

Write-Section 'Fix permissions (private key + .ssh folder)'
# Use cmd.exe for icacls so quoting/arguments behave consistently on Windows PowerShell 5.1.
# Folder: full control for current user; optionally keep SYSTEM/Administrators.
cmd /c "icacls \"$sshDir\" /inheritance:r" | Out-Host
cmd /c "icacls \"$sshDir\" /grant:r \"$env:USERNAME:(F)\" \"NT AUTHORITY\\SYSTEM:(F)\" \"BUILTIN\\Administrators:(F)\"" | Out-Host

# Private key: full control for current user; optionally keep SYSTEM.
cmd /c "icacls \"$keyPath\" /inheritance:r" | Out-Host
cmd /c "icacls \"$keyPath\" /grant:r \"$env:USERNAME:(F)\" \"NT AUTHORITY\\SYSTEM:(F)\"" | Out-Host

Write-Section 'ssh-agent + ssh-add (best effort)'
$svc = Get-Service ssh-agent -ErrorAction SilentlyContinue
if ($null -eq $svc) {
  Write-Warning 'ssh-agent service not found.'
} else {
  Write-Host ("ssh-agent Status={0} StartType={1}" -f $svc.Status, $svc.StartType)
  try {
    Start-Service ssh-agent -ErrorAction Stop
    Write-Host 'ssh-agent started.'
  } catch {
    Write-Warning ('Could not start ssh-agent (likely disabled / needs admin): ' + $_.Exception.Message)
  }

  if ($sshAdd) {
    try {
      & $sshAdd $keyPath | Out-Host
    } catch {
      Write-Warning ('ssh-add failed (agent may not be running): ' + $_.Exception.Message)
    }
  }
}

Write-Section 'PUBLIC KEY (printed + clipboard)'
Write-Host ("PUBLIC KEY PATH: {0}" -f $pubPath)
$pub = Get-Content -Raw -Path $pubPath
Write-Host $pub

try {
  $pub | Set-Clipboard
  Write-Host 'Copied public key to clipboard.'
} catch {
  Write-Warning ('Could not copy to clipboard: ' + $_.Exception.Message)
}

Write-Section 'Update ~/.ssh/config for github.com + gitlab.com'
$configPath = Join-Path $sshDir 'config'
$identity = ('~/.ssh/' + (Split-Path -Leaf $keyPath))

$cfgLines = if (Test-Path $configPath) { Get-Content -Path $configPath } else { @() }
$cfgLines = Ensure-SshHostBlock -Lines $cfgLines -HostName 'github.com' -IdentityFile $identity
$cfgLines = Ensure-SshHostBlock -Lines $cfgLines -HostName 'gitlab.com' -IdentityFile $identity

Set-Content -Path $configPath -Value $cfgLines -Encoding ascii
Write-Host ('Updated: ' + $configPath)

Write-Section 'FINAL SUMMARY'
Write-Host ('Private key path: ' + $keyPath)
Write-Host ('Public  key path: ' + $pubPath)
Write-Host ('SSH config path : ' + $configPath)
Write-Host ''
Write-Host 'Next steps:'
Write-Host '1) Add the PUBLIC KEY above to your Git provider (GitHub/GitLab -> Settings -> SSH Keys).'
Write-Host '2) Test connectivity:'
Write-Host '   ssh -T git@github.com'
Write-Host '   ssh -T git@gitlab.com'
