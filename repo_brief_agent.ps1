<#
Repo Merge-Ready Brief Agent (ONE FILE)
- Windows-safe (no rg/sed/awk)
- Submodule-safe (incl. nested .gitmodules)
- NO-SKIP policy (blocker-aware; never pretends complete)
- Fork/ownership policy support via memory-bank/submodule_overrides.json
- Generates: .clinerules, memory-bank/*, docs/ai/* (health, trees, prompts, policy)
- Optionally commits ONLY allowed outputs (no push)
#>

[CmdletBinding()]
param(
  [switch]$Force = $false,
  [switch]$NoCommit = $false,
  [int]$MaxPass = 3,
  [int]$Jobs = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:GIT_TERMINAL_PROMPT = '0'   # avoid hangs on credential prompts

# ---------------- Helpers ----------------
function Write-Section([string]$t) {
  Write-Output ""
  Write-Output ("==== " + $t + " ====")
}

function Ensure-Dir([string]$p) {
  if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

function Write-Utf8([string]$path, [string]$content) {
  $dir = Split-Path -Parent $path
  if ($dir) { Ensure-Dir $dir }
  Set-Content -LiteralPath $path -Encoding utf8 -Value $content
}

function Append-Utf8([string]$path, [string]$content) {
  Add-Content -LiteralPath $path -Encoding utf8 -Value $content
}

function Run([string]$cmd, [string[]]$args, [string]$cwd = $null) {
  if ($cwd) { Push-Location $cwd }
  try {
    Write-Output ("> " + $cmd + " " + ($args -join " "))
    & $cmd @args
    if ($LASTEXITCODE -ne 0) {
      throw ("Command failed with exit code " + $LASTEXITCODE + ": " + $cmd + " " + ($args -join " "))
    }
  } finally {
    if ($cwd) { Pop-Location }
  }
}

function Try-Run([string]$cmd, [string[]]$args, [string]$cwd = $null) {
  try { Run $cmd $args $cwd; return $true } catch { return $false }
}

function Convert-GitUrlToHttps([string]$u) {
  if ([string]::IsNullOrWhiteSpace($u)) { return $u }
  $u = $u.Trim()
  if ($u -match '^git@github\.com:(.+)$') { return ('https://github.com/' + $Matches[1]) }
  if ($u -match '^ssh://git@github\.com/(.+)$') { return ('https://github.com/' + $Matches[1]) }
  if ($u -match '^git://github\.com/(.+)$') { return ('https://github.com/' + $Matches[1]) }
  return $u
}

function Load-Overrides([string]$root) {
  $p = Join-Path $root 'memory-bank/submodule_overrides.json'
  if (Test-Path -LiteralPath $p) {
    try { return (Get-Content -LiteralPath $p -Raw | ConvertFrom-Json) } catch { return $null }
  }
  return $null
}

function Apply-Overrides([string]$url, $ov) {
  if (-not $ov) { return $url }
  if (-not $ov.rewrites) { return $url }
  foreach ($r in $ov.rewrites) {
    if ($r.from -and $r.to) {
      if ($url.Trim() -eq $r.from.Trim()) { return $r.to.Trim() }
    }
  }
  return $url
}

function Is-GitWorkTree([string]$dir) {
  try {
    $v = (& git -C $dir rev-parse --is-inside-work-tree 2>$null).Trim()
    return ($v -eq 'true')
  } catch { return $false }
}

function Find-GitmodulesFiles([string]$root) {
  Get-ChildItem -LiteralPath $root -Recurse -Force -File -Filter '.gitmodules' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\\.git\\' }
}

function Rewrite-Gitmodules([string]$repoDir, [string]$gitmodulesPath, $ov, [string]$healthPath) {
  if (-not (Is-GitWorkTree $repoDir)) {
    Append-Utf8 $healthPath "### BLOCKER: .gitmodules found but directory is not a git work tree"
    Append-Utf8 $healthPath ("Path: " + $gitmodulesPath)
    Append-Utf8 $healthPath ("Dir : " + $repoDir)
    Append-Utf8 $healthPath "Fix: ensure this folder is a git clone (has .git) or is properly initialized as a submodule."
    Append-Utf8 $healthPath ""
    return $false
  }

  $lines = & git -C $repoDir config -f $gitmodulesPath --get-regexp url 2>$null
  foreach ($line in $lines) {
    $parts = $line -split '\s+', 2
    if ($parts.Count -lt 2) { continue }
    $key = $parts[0]
    $old = $parts[1]

    $https = Convert-GitUrlToHttps $old
    $final = Apply-Overrides $https $ov

    if ($final -ne $old) {
      Write-Output ("  url rewrite: " + $old + "  =>  " + $final)
      & git -C $repoDir config -f $gitmodulesPath $key $final | Out-Null
    }
  }
  return $true
}

function Submodule-Update([string]$repoDir, [int]$jobs, [string]$healthPath) {
  $ok = Try-Run 'git' @('submodule','sync','--recursive') $repoDir
  if (-not $ok) {
    Append-Utf8 $healthPath "### BLOCKER: git submodule sync failed"
    Append-Utf8 $healthPath ("RepoDir: " + $repoDir)
    Append-Utf8 $healthPath "Try manually: git -C <RepoDir> submodule sync --recursive"
    Append-Utf8 $healthPath ""
    return $false
  }

  $ok = Try-Run 'git' @('submodule','update','--init','--recursive','--jobs',"$jobs",'--progress') $repoDir
  if (-not $ok) {
    Append-Utf8 $healthPath "### BLOCKER: git submodule update failed"
    Append-Utf8 $healthPath ("RepoDir: " + $repoDir)
    Append-Utf8 $healthPath "Likely causes: private repo auth (HTTPS needs PAT/GCM), remaining SSH URLs, or network."
    Append-Utf8 $healthPath "Fix steps:"
    Append-Utf8 $healthPath "- Ensure URLs are HTTPS in ALL .gitmodules (this agent rewrites them)."
    Append-Utf8 $healthPath "- If private: configure Git Credential Manager or a PAT for HTTPS."
    Append-Utf8 $healthPath "- Re-run: git -C <RepoDir> submodule update --init --recursive --jobs 1 --progress"
    Append-Utf8 $healthPath ""
    return $false
  }
  return $true
}

function Get-SubmodulePaths([string]$root) {
  $paths = @()
  $lines = & git -C $root submodule status --recursive 2>$null
  foreach ($l in $lines) {
    # formats: " 817aecb path (heads/main)" or "-817aecb path" (not init)
    if ($l -match '^\s*[-+U ]?[0-9a-f]{8,40}\s+(.+?)(\s+\(.*\))?\s*$') {
      $p = $Matches[1].Trim()
      if ($p) { $paths += $p }
    }
  }
  return ($paths | Sort-Object -Unique)
}

function Build-TreeAll([string]$root, [string]$healthPath) {
  $set = New-Object 'System.Collections.Generic.HashSet[string]'

  # top repo files
  $top = & git -C $root ls-files 2>$null
  foreach ($f in $top) {
    $p = $f.Replace('\','/').Trim()
    if ($p) { $set.Add($p) | Out-Null }
  }

  # submodule tracked files
  $subs = Get-SubmodulePaths $root
  foreach ($sm in $subs) {
    $abs = Join-Path $root $sm
    if (-not (Test-Path -LiteralPath $abs)) {
      Append-Utf8 $healthPath ("### NOTE: submodule path missing on disk (not initialized?): " + $sm)
      continue
    }
    if (-not (Is-GitWorkTree $abs)) {
      Append-Utf8 $healthPath ("### NOTE: submodule path not a git work tree: " + $sm)
      continue
    }
    $files = & git -C $abs ls-files 2>$null
    foreach ($f in $files) {
      $p = ($sm.TrimEnd('\','/') + '/' + $f).Replace('\','/').Trim()
      if ($p) { $set.Add($p) | Out-Null }
    }
  }

  return ($set | Sort-Object)
}

# ---------------- Start ----------------
Write-Section "PRECHECK"
if (-not (Try-Run 'git' @('--version'))) { throw "git is required on PATH." }

$root = (& git rev-parse --show-toplevel).Trim()
if (-not $root) { throw "Not inside a git repository. Run from within your repo." }
$root = (Resolve-Path -LiteralPath $root).Path
Set-Location $root
Write-Output ("Repo root: " + $root)

# Local configs (per-repo)
Try-Run 'git' @('config','core.longpaths','true') | Out-Null
Try-Run 'git' @('config','credential.helper','manager-core') | Out-Null
Try-Run 'git' @('config','url."https://github.com/".insteadOf','git@github.com:') | Out-Null
Try-Run 'git' @('config','url."https://github.com/".insteadOf','ssh://git@github.com/') | Out-Null
Try-Run 'git' @('config','url."https://github.com/".insteadOf','git://github.com/') | Out-Null

Ensure-Dir (Join-Path $root 'docs/ai')
Ensure-Dir (Join-Path $root 'memory-bank')

# .clinerules (enhanced)
$clinerules = @'
# Repo Merge-Ready Brief Agent (Windows + Submodule + Fork Policy)

MODE:
- READ-ONLY by default.
- Do NOT modify application/source code unless user explicitly asks.

ALLOWED WRITES (ONLY):
- docs/ai/**
- memory-bank/**

NO-SKIP POLICY (CRITICAL):
- Never skip a failed step.
- If a command fails OR outputs look incomplete, STOP, diagnose, apply a safe fix, and re-run.
- If still blocked, produce a BLOCKER section with exact user actions and do NOT generate “complete” outputs.

WINDOWS / SHELL POLICY:
- Avoid Linux-only tools (sed/awk/find/grep/rg). Prefer: git commands, PowerShell built-ins, or Python.
- Avoid long PowerShell one-liners. If multi-step is needed, write a .ps1 under docs/ai/ and execute it.
- Always check exit codes and stop on non-zero.

GIT / SUBMODULE POLICY:
- Always verify repo root: git rev-parse --show-toplevel
- If .gitmodules exists anywhere, perform SUBMODULE BRING-UP before building trees/contracts.
- Rewrite git@github.com / ssh://git@github.com / git://github.com URLs to https://github.com
- Build TWO trees:
  - docs/ai/REPO_TREE.txt (top repo)
  - docs/ai/REPO_TREE_ALL.txt (includes submodule tracked files)
- If submodules fail to clone (auth/private), mark dossier INCOMPLETE and write BLOCKER steps.

FORK POLICY (EXTERNAL CODE):
- Prefer forks under the user's GitHub account/org for all external repos.
- If memory-bank/submodule_overrides.json exists, use it to rewrite submodule URLs to user's forks (HTTPS).
- User edits happen on user's fork/branch; main repo only updates submodule pointers.
- Never push to upstream remotes. Only commit local docs outputs. Push only if user explicitly asks.

OUTPUT OWNERSHIP POLICY:
- All architecture/docs outputs MUST live in user's repo and be committed:
  - docs/ai/**
  - memory-bank/**

SECURITY:
- Never open/copy secrets: .env, *.key, *.pem, credentials, tokens.

DELIVERABLES:
- docs/ai/REPO_HEALTH.md
- docs/ai/REPO_TREE.txt
- docs/ai/REPO_TREE_ALL.txt
- docs/ai/AGENT_PROMPTS.md
- docs/ai/SUBMODULE_FORK_POLICY.md
- memory-bank/*.md
'@
if ($Force -or -not (Test-Path -LiteralPath (Join-Path $root '.clinerules'))) {
  Write-Utf8 (Join-Path $root '.clinerules') $clinerules
}

# Memory-bank templates (non-destructive)
function Ensure-Template([string]$path, [string]$content) {
  if (-not (Test-Path -LiteralPath $path)) { Write-Utf8 $path $content }
}
Ensure-Template (Join-Path $root 'memory-bank/projectbrief.md') @'
# Project Brief
- What is this repository?
- What is the new target architecture / merge goal (5–10 lines)?
(Write your merge goal here. The agent will treat this as ground truth.)
'@
Ensure-Template (Join-Path $root 'memory-bank/productContext.md') @'
# Product Context
- Who is the user/customer?
- What problem does this system solve?
- Success criteria / non-goals
'@
Ensure-Template (Join-Path $root 'memory-bank/systemPatterns.md') @'
# System Patterns
- Key architectural patterns
- Module boundaries
- Data flow patterns
- Conventions
'@
Ensure-Template (Join-Path $root 'memory-bank/techContext.md') @'
# Tech Context
- Languages, frameworks
- Databases, queues
- Infrastructure, CI/CD
- Local dev / run instructions
'@
Ensure-Template (Join-Path $root 'memory-bank/activeContext.md') @'
# Active Context
- What is currently being worked on?
- What changed recently?
- Open questions / blockers
'@
Ensure-Template (Join-Path $root 'memory-bank/progress.md') @'
# Progress Log
- Dated notes of what the agent discovered
- Decisions
- Next steps
'@
Ensure-Template (Join-Path $root 'memory-bank/submodule_overrides.json') @'
{
  "rewrites": [
    { "from": "https://github.com/OWNER/REPO.git", "to": "https://github.com/YOUR_ORG_OR_USER/REPO.git" }
  ]
}
'@

# Health report start
$healthPath = Join-Path $root 'docs/ai/REPO_HEALTH.md'
Write-Utf8 $healthPath "## Repo Health`n"
Append-Utf8 $healthPath ("Repo root: " + $root)
Append-Utf8 $healthPath ""
Append-Utf8 $healthPath "### git remote -v"
Append-Utf8 $healthPath ((& git -C $root remote -v) -join "`n")
Append-Utf8 $healthPath ""
Append-Utf8 $healthPath "### git status --porcelain (snapshot)"
Append-Utf8 $healthPath ((& git -C $root status --porcelain) -join "`n")
Append-Utf8 $healthPath ""

# Submodule bring-up
Write-Section "SUBMODULE BRING-UP"
$ov = Load-Overrides $root
$blockers = $false

for ($pass = 1; $pass -le $MaxPass; $pass++) {
  Write-Output ("-- Pass " + $pass + " --")
  $gms = Find-GitmodulesFiles $root
  foreach ($gm in $gms) {
    $repoDir = Split-Path -Parent $gm.FullName
    $ok = Rewrite-Gitmodules $repoDir $gm.FullName $ov $healthPath
    if ($ok) {
      $ok2 = Submodule-Update $repoDir $Jobs $healthPath
      if (-not $ok2) { $blockers = $true }
    } else {
      $blockers = $true
    }
  }
}

# Trees
Write-Section "TREES"
$treePath = Join-Path $root 'docs/ai/REPO_TREE.txt'
(& git -C $root ls-files | Sort-Object) | Set-Content -LiteralPath $treePath -Encoding utf8

$treeAllPath = Join-Path $root 'docs/ai/REPO_TREE_ALL.txt'
$all = Build-TreeAll $root $healthPath
$all | Set-Content -LiteralPath $treeAllPath -Encoding utf8

Write-Output ("Wrote: " + $healthPath)
Write-Output ("Wrote: " + $treePath)
Write-Output ("Wrote: " + $treeAllPath + " lines=" + (Get-Content -LiteralPath $treeAllPath).Count)

# Submodule fork policy doc
Write-Section "SUBMODULE FORK POLICY"
$policyPath = Join-Path $root 'docs/ai/SUBMODULE_FORK_POLICY.md'
Write-Utf8 $policyPath @'
# Submodule + Fork Policy

## Policy
- External code should be consumed via Submodule or Fork.
- Best practice: rewrite each submodule URL to your fork (HTTPS).
- Make changes on your fork/branch.
- In your main repo, only update submodule pointers.
- Keep architecture outputs only in this repo: docs/ai/** and memory-bank/** and commit them.
- If a submodule is private: HTTPS requires credentials (Git Credential Manager / PAT). This is an access constraint, not an agent bug.

## Overrides
Edit: memory-bank/submodule_overrides.json
'@
Append-Utf8 $policyPath "`n## Discovered .gitmodules`n"

$gms2 = Find-GitmodulesFiles $root
foreach ($gm in $gms2) {
  $repoDir = Split-Path -Parent $gm.FullName
  Append-Utf8 $policyPath ("### " + ($gm.FullName.Substring($root.Length).TrimStart('\','/').Replace('\','/')))
  if (Is-GitWorkTree $repoDir) {
    $lines = & git -C $repoDir config -f $gm.FullName --get-regexp url 2>$null
    foreach ($line in $lines) {
      $parts = $line -split '\s+', 2
      if ($parts.Count -lt 2) { continue }
      $key = $parts[0]
      $url = $parts[1]
      $https = Convert-GitUrlToHttps $url
      $rec = Apply-Overrides $https $ov
      Append-Utf8 $policyPath ("- " + $key + " = " + $https + "  (recommended: " + $rec + ")")
    }
  } else {
    Append-Utf8 $policyPath "- BLOCKER: directory is not a git work tree; cannot update submodules here."
  }
  Append-Utf8 $policyPath ""
}
Write-Output ("Wrote: " + $policyPath)

# Agent prompts for Cline (Plan/Act)
Write-Section "AGENT PROMPTS"
$promptsPath = Join-Path $root 'docs/ai/AGENT_PROMPTS.md'
Write-Utf8 $promptsPath @'
# Cline Agent Prompts (Plan + Act)

## PLAN PROMPT (paste in Cline Plan Mode)
You are "Repo Merge-Ready Brief Agent" on Windows (VS Code + Cline). Produce a complete repo dossier usable by another AI for merging and architecture work.

Hard rules (from .clinerules):
- READ-ONLY by default. Write ONLY to docs/ai/** and memory-bank/**.
- NO-SKIP: never skip failed steps. Failures are blocking until fixed or reported as BLOCKER.
- Windows-safe: avoid Linux commands and avoid long PowerShell one-liners. If needed, write helper .ps1 under docs/ai/ and execute it.
- Submodule-safe: if any .gitmodules exists, submodules must be initialized/updated first. Tree must include submodule tracked files.

Fork & ownership policy:
- Prefer forks under the user's GitHub account/org for external code. If memory-bank/submodule_overrides.json exists, use it to rewrite submodule URLs to the user's forks (HTTPS).
- User edits happen on user's fork/branch; main repo updates only submodule pointers.
- All architecture outputs must live in user's repo (docs/ai/** and memory-bank/**) and be commit-ready.
- Private submodules require credentials (Git Credential Manager / PAT). If blocked, report exact remediation.

PLAN:
0) Bootstrap validation (blocking):
   - Open docs/ai/REPO_HEALTH.md and docs/ai/REPO_TREE_ALL.txt.
   - If REPO_HEALTH.md contains BLOCKER, stop and output remediation steps.

1) Inventory:
   - Use docs/ai/REPO_TREE_ALL.txt as authoritative file inventory (includes submodules).
   - Determine repo type (monorepo vs multi-service vs nested repos).

2) Read high-signal files first:
   - README*, docs, package manifests, Docker/compose/k8s, CI, configs, entrypoints, migrations/models, API specs.
   - Identify modules and boundaries with paths + entrypoints.

3) Extract contracts (no rg required):
   - Use git grep / Select-String / Python to find:
     APIs (routes/controllers/OpenAPI)
     Events/queues (topics/consumers/producers)
     DB (migrations/schema/models/ORM)
     Config keys/feature flags

4) Produce deliverables:
   - docs/ai/REPO_DOSSIER.md (structured)
   - docs/ai/REPO_DOSSIER.json (stable schema)
   - Update memory-bank/*.md (durable summaries)

At the end of PLAN:
- Provide a numbered Act checklist.
- List 10–20 first files to open (paths).
- List assumptions (ask zero questions unless blocked).

## ACT PROMPT (paste in Cline Act Mode)
Execute the approved plan now with strict NO-SKIP behavior.

A) Blocking validation:
- Open docs/ai/REPO_HEALTH.md and docs/ai/REPO_TREE_ALL.txt.
- If REPO_HEALTH.md contains BLOCKER, STOP and output:
  1) exact root cause
  2) exact commands to fix (HTTPS vs SSH, credentials/PAT, access)
  3) re-run instructions
Do not proceed with incomplete trees.

B) Dossier generation (MUST):
Create docs/ai/REPO_DOSSIER.md with these headings:
1) Executive Summary (product goal + merge goal 5–10 lines)
2) Repo Type & Layout
3) Module Map (name, responsibility, key paths, entrypoints, dependencies)
4) Key Data Flows (end-to-end narratives)
5) Inputs/Outputs (Contracts):
   - APIs (routes + request/response if discoverable)
   - Events/Queues (topics + payload fields if discoverable)
   - DB (schema/models/migrations)
   - File-based I/O (important files)
6) Tech Stack Inventory (language/framework/DB/queue/infra/CI)
7) Top 20 Important Files (path + why)
8) Risks & Constraints (incl. submodule/private/auth, windows/linux issues)
9) Suggested Read Order
10) Open Questions / Unknowns

Create docs/ai/REPO_DOSSIER.json with stable schema:
{
  "product_goal": "",
  "merge_goal": "",
  "repo_type": "",
  "tech_stack": { "languages":[], "frameworks":[], "db":[], "queue":[], "infra":[], "ci":[] },
  "modules": [{ "name":"","paths":[],"entrypoints":[],"responsibility":"","depends_on":[] }],
  "contracts": { "apis":[], "events":[], "db":[], "files":[] },
  "critical_flows": [],
  "important_files": [{ "path":"","why":"" }],
  "risks": [],
  "read_order": [],
  "submodules": [{ "path":"","url":"","status":"","recommended_fork_url":"" }]
}

Update memory-bank/*.md accordingly.

Tools:
- Do NOT require rg. Use git grep / Select-String / Python.
- Avoid complex PowerShell one-liners; write helper scripts under docs/ai when needed.
- Never modify application code. Never expose secrets.

NO-SKIP CONTRACT:
If you cannot read a submodule file or the file inventory is incomplete, you MUST:
1) explain the exact reason,
2) run diagnostics (submodule status, .gitmodules URLs, directory existence),
3) apply safe fixes (HTTPS rewrite + sync + update),
4) re-run the failed step,
5) if still blocked, produce a BLOCKER section with exact user actions.
You are NOT allowed to continue with partial outputs as if they were complete.
'@
Write-Output ("Wrote: " + $promptsPath)

# Commit outputs (safe allowlist) unless NoCommit
Write-Section "COMMIT OUTPUTS (SAFE)"
if (-not $NoCommit) {
  $status = & git -C $root status --porcelain
  if (-not $status) {
    Write-Output "No changes to commit."
  } else {
    $ok = $true
    foreach ($line in $status) {
      $path = $line.Substring(3).Trim().Replace('\','/')
      $allowed = $false
      if ($path -eq '.clinerules') { $allowed = $true }
      if ($path -eq 'repo_brief_agent.ps1') { $allowed = $true }
      if ($path.StartsWith('docs/ai/')) { $allowed = $true }
      if ($path.StartsWith('memory-bank/')) { $allowed = $true }

      if (-not $allowed) {
        $ok = $false
        Write-Output ("Refusing to commit due to unrelated change: " + $path)
      }
    }

    if ($ok) {
      Run 'git' @('add','.clinerules','repo_brief_agent.ps1','docs/ai','memory-bank') $root
      Run 'git' @('commit','-m','docs: repo brief agent outputs (health/trees/policy/prompts)') $root
      Write-Output "Committed outputs (no push)."
    } else {
      Write-Output "Commit skipped. Clean unrelated changes or commit them separately."
    }
  }
} else {
  Write-Output "NoCommit specified; skipping commit."
}

Write-Section "NEXT"
if ($blockers) {
  Write-Output 'BLOCKER detected in docs/ai/REPO_HEALTH.md — fix access/credentials/submodules first.'
}
Write-Output '1) Open docs/ai/REPO_HEALTH.md'
Write-Output '2) Open docs/ai/AGENT_PROMPTS.md'
Write-Output '3) Paste PLAN prompt into Cline Plan Mode, then paste ACT prompt into Cline Act Mode.'
Write-Output '4) Cline will produce docs/ai/REPO_DOSSIER.md and docs/ai/REPO_DOSSIER.json.'
