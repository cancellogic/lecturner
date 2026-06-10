# =============================================================================
# setup_windows.ps1 — Windows bootstrap for Crane + lecturner
#
# ROADMAP:
#   1. Check system build requirements (MSVC build tools, git, ffmpeg, cmake,
#      clang/
#      LLVM, Rust, hf CLI); detect CUDA toolkit; warn if not in Dev Prompt
#   2. Ask user: CUDA or CPU  (Metal is Apple-only)
#   3. Download model assets:
#        - Qwen3-4B            → checkpoints\Qwen3-4B\
#        - Qwen3-TTS CustomVoice → Qwen3-TTS-12Hz-1.7B-CustomVoice\
#        - Whisper ggml binary  → models\ggml-medium.en.bin
#        - CMU dict             → cmudict.dict  (root of work dir)
#   4. Clone + build lucasjinreal/Crane with the chosen feature flag
#      CUDA builds REQUIRE the x64 Native Tools Command Prompt for VS;
#      script warns but proceeds — build will fail cleanly if env is wrong.
#   5. Clone + build lecturner with the SAME feature flag.  Cargo features
#      replaced the old Cargo.toml whisper-rs line patching, so the repo
#      stays clean and re-runs can git pull without conflict.
#   6. Write lecturner.toml.  Paths use forward slashes (TOML-safe, and
#      Rust's Path accepts them on Windows).
#
# Run in PowerShell 7+ as Administrator from your install directory.
# =============================================================================

#Requires -Version 7.0
# ^ Load-bearing beyond the obvious: PS7's `Set-Content -Encoding UTF8` writes
#   UTF-8 *without* a BOM.  Under PS 5.1 the same line writes a BOM, which the
#   toml parser rejects — so this guard is what keeps Step 6's output valid.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Ok   { param($Msg) Write-Host "[✓] $Msg" -ForegroundColor Green  }
function Write-Warn { param($Msg) Write-Host "[!] $Msg" -ForegroundColor Yellow }
function Write-Die  { param($Msg) Write-Host "[✗] $Msg" -ForegroundColor Red; exit 1 }

$WorkDir = (Get-Location).Path

# ── Step 1: System build requirements ────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host " Step 1 — Checking system build requirements"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Windows version
$WinVer = [System.Environment]::OSVersion.Version
Write-Ok "Windows $($WinVer.Major).$($WinVer.Minor) Build $($WinVer.Build)"

# MSVC build tools — lecturner README explicitly requires these on Windows.
# GNU toolchain has linker compatibility issues with CUDA crates.
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $VsWhere) {
    $VsInstall = & $VsWhere -latest -property installationPath 2>$null
    if ($VsInstall) {
        Write-Ok "MSVC build tools: $VsInstall"
    } else {
        Write-Warn "Visual Studio detected but C++ workload may be missing."
    }
} else {
    Write-Die (
        "MSVC build tools not found.`n" +
        "Install 'Desktop development with C++' from:`n" +
        "https://visualstudio.microsoft.com/visual-cpp-build-tools/`n" +
        "Then re-run this script."
    )
}

# Developer Command Prompt check — VCINSTALLDIR is set by vcvarsall.bat.
# Plain PowerShell will work for CPU builds; CUDA builds need the Dev Prompt.
if (-not $env:VCINSTALLDIR) {
    Write-Warn (
        "Not running in a Visual Studio Developer Command Prompt.`n" +
        "  CPU builds: fine as-is.`n" +
        "  CUDA builds: close this and re-open from:`n" +
        "    Start → 'x64 Native Tools Command Prompt for VS 20xx'`n" +
        "    then run: pwsh setup_windows.ps1"
    )
}

# git
# NB on the PATH refresh pattern used after each winget install below: it
# REPLACES the session PATH from the registry, discarding session-only
# additions.  It works here because every installer we invoke (winget
# packages, rustup-init) writes its PATH entry to the registry — but if you
# add a session-only `$env:PATH +=` above one of these refreshes, it will be
# silently lost.  Appending instead of replacing would be more robust.
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Warn "git not found — installing via winget…"
    winget install --id Git.Git -e --source winget
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","User")
}
Write-Ok "git: $(git --version)"

# ffmpeg — lecturner calls it as a subprocess for WAV→MP3 transcoding
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warn "ffmpeg not found — installing via winget…"
    winget install --id Gyan.FFmpeg -e --source winget
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","User")
}
Write-Ok "ffmpeg: $(ffmpeg -version 2>&1 | Select-Object -First 1)"

# cmake — whisper-rs-sys compiles whisper.cpp from source at build time and
# its build system is CMake.  The VS Build Tools C++ workload can bundle
# CMake, but it is only on PATH inside a Developer Command Prompt — checking
# here makes the lecturner build work from any shell.
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Warn "cmake not found — installing via winget…"
    winget install --id Kitware.CMake -e --source winget
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
}
Write-Ok "cmake: $(cmake --version 2>&1 | Select-Object -First 1)"

# clang/LLVM — whisper-rs bindgen calls clang directly for CUDA builds
if (-not (Get-Command clang -ErrorAction SilentlyContinue)) {
    Write-Warn "clang not found — installing LLVM via winget…"
    winget install --id LLVM.LLVM -e --source winget
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","User")
}
Write-Ok "clang: $(clang --version 2>&1 | Select-Object -First 1)"

# Rust 1.88+ — force MSVC toolchain target
if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
    Write-Warn "rustup not found — downloading installer…"
    $RustupInstaller = "$env:TEMP\rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $RustupInstaller
    & $RustupInstaller -y --default-host x86_64-pc-windows-msvc --default-toolchain stable
    $env:PATH += ";$env:USERPROFILE\.cargo\bin"
} else {
    rustup update stable
}
Write-Ok "Rust: $(rustc --version)"

# hf CLI — huggingface_hub ≥ 1.0
if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    Write-Warn "'hf' CLI not found — installing via pip…"
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Warn "Python not found — installing via winget…"
        winget install --id Python.Python.3.11 -e --source winget
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH","User")
    }
    python -m pip install --user -U huggingface_hub
    $PyScripts = python -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user'))"
    $env:PATH += ";$PyScripts"
}
if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    Write-Die "'hf' still not on PATH — add your Python user Scripts dir to PATH and re-run."
}
Write-Ok "hf CLI: $(hf --version 2>&1 | Select-Object -First 1)"

# CUDA toolkit check — non-fatal; shapes the backend menu below
$CudaAvailable = $false
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $CudaVer = [regex]::Match((nvcc --version | Out-String), 'release ([\d.]+)').Groups[1].Value
    Write-Ok "CUDA toolkit: $CudaVer"
    $CudaAvailable = $true
} else {
    Write-Warn "nvcc not found — CUDA option unavailable. Install from https://developer.nvidia.com/cuda-downloads"
}

# ── Step 2: Backend selection ─────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host " Step 2 — Choose inference backend"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

if ($CudaAvailable) {
    Write-Host "  NVIDIA GPU with CUDA detected. Options:"
    Write-Host "    1) CUDA  (recommended — fastest; requires Dev Command Prompt)"
    Write-Host "    2) CPU   (slower, works anywhere)"
    $BackendChoice = Read-Host "Enter choice [1-2] (default: 1)"
    if ([string]::IsNullOrWhiteSpace($BackendChoice)) { $BackendChoice = "1" }
} else {
    Write-Host "  No CUDA detected — using CPU."
    $BackendChoice = "2"
}

switch ($BackendChoice) {
    "1" { $CraneFeature = "cuda"; $BackendLabel = "CUDA" }
    "2" { $CraneFeature = "";     $BackendLabel = "CPU"  }
    default {
        Write-Warn "Unrecognised — defaulting to CPU."
        $CraneFeature = ""; $BackendLabel = "CPU"
    }
}
Write-Ok "Selected backend: $BackendLabel"

# ── Step 3: Download model assets ─────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host " Step 3 — Downloading model assets"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "  Qwen3-4B ~8 GB · TTS model ~3 GB · hf download is resume-safe."
Write-Host ""

$CheckpointsDir = "$WorkDir\checkpoints"
New-Item -ItemType Directory -Force -Path $CheckpointsDir | Out-Null

# Qwen3-4B — crane_llm_model in toml
$QwenLlmDir = "$CheckpointsDir\Qwen3-4B"
if (-not (Test-Path $QwenLlmDir)) {
    Write-Host "Downloading Qwen3-4B…"
    hf download Qwen/Qwen3-4B --local-dir $QwenLlmDir
    Write-Ok "Qwen3-4B → $QwenLlmDir"
} else {
    Write-Ok "Qwen3-4B already present"
}

# Qwen3-TTS — crane_tts_model in toml
$QwenTtsDir = "$WorkDir\Qwen3-TTS-12Hz-1.7B-CustomVoice"
if (-not (Test-Path $QwenTtsDir)) {
    Write-Host "Downloading Qwen3-TTS-12Hz-1.7B-CustomVoice…"
    hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir $QwenTtsDir
    Write-Ok "Qwen3-TTS → $QwenTtsDir"
} else {
    Write-Ok "Qwen3-TTS already present"
}

# Whisper ggml binary — whisper_model + whisper_model_dir (two separate toml keys)
$ModelsDir = "$WorkDir\models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
$WhisperBin = "$ModelsDir\ggml-medium.en.bin"
if (-not (Test-Path $WhisperBin)) {
    Write-Host "Downloading Whisper ggml-medium.en.bin…"
    Invoke-WebRequest `
        -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin" `
        -OutFile $WhisperBin
    Write-Ok "Whisper → $WhisperBin"
} else {
    Write-Ok "Whisper already present"
}

# CMU dict — work dir root by convention; not a toml key
$CmuDict = "$WorkDir\cmudict.dict"
if (-not (Test-Path $CmuDict)) {
    Write-Host "Downloading CMU Pronouncing Dictionary…"
    Invoke-WebRequest `
        -Uri "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict" `
        -OutFile $CmuDict
    Write-Ok "CMU dict → $CmuDict"
} else {
    Write-Ok "CMU dict already present"
}

# ── Step 4: Clone + build Crane ───────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host " Step 4 — Building Crane"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ($BackendLabel -eq "CUDA") {
    Write-Host "  Reminder: CUDA builds need the x64 Native Tools Command Prompt."
    Write-Host "  If this step fails with linker errors, that's why."
}
Write-Host ""

$CraneDir = "$WorkDir\Crane"
if (-not (Test-Path "$CraneDir\.git")) {
    git clone https://github.com/lucasjinreal/Crane.git $CraneDir
} else {
    Write-Ok "Crane repo already present — pulling latest…"
    git -C $CraneDir pull --ff-only
}

$CargoArgs = @("build", "--manifest-path", "$CraneDir\Cargo.toml", "-p", "crane-oai", "--release")
if ($CraneFeature) { $CargoArgs += @("--features", $CraneFeature) }
Write-Host "Building crane-oai [feature: $(if ($CraneFeature) { $CraneFeature } else { '(none — CPU)' })]…"
& cargo @CargoArgs

$CraneBin = "$CraneDir\target\release\crane-oai.exe"
Write-Ok "Crane built → $CraneBin"

# ── Step 5: Clone + build lecturner ───────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host " Step 5 — Building lecturner"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$LecturnerDir = "$WorkDir\lecturner"
if (-not (Test-Path "$LecturnerDir\.git")) {
    git clone https://github.com/cancellogic/lecturner.git $LecturnerDir
} else {
    Write-Ok "lecturner repo already present — pulling latest…"
    git -C $LecturnerDir pull --ff-only
}

# Backend selection is a cargo feature (cuda / none = CPU) — the same flag we
# just passed to Crane.  No Cargo.toml patching, so the repo stays clean and
# the pull above never hits a dirty-tree conflict on re-runs.
$CargoToml = "$LecturnerDir\Cargo.toml"
$LectArgs  = @("build", "--manifest-path", $CargoToml, "--release")
if ($CraneFeature) { $LectArgs += @("--features", $CraneFeature) }
Write-Host "Building lecturner [feature: $(if ($CraneFeature) { $CraneFeature } else { '(none — CPU)' })]…"
& cargo @LectArgs

$LecturnerBin = "$LecturnerDir\target\release\lecturner.exe"
Write-Ok "lecturner built → $LecturnerBin"

# ── Step 6: Write lecturner.toml ──────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host " Step 6 — Writing lecturner.toml"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Use forward slashes throughout — TOML-safe and Rust's Path accepts them on Windows.
# The real lecturner.toml shows Windows paths with forward slashes are fine.
$CraneBinToml   = $CraneBin   -replace '\\', '/'
$QwenLlmToml    = $QwenLlmDir -replace '\\', '/'
$QwenTtsToml    = $QwenTtsDir -replace '\\', '/'
$ModelsDirToml  = $ModelsDir  -replace '\\', '/'

$TomlOut = "$LecturnerDir\lecturner.toml"
@"
# lecturner.toml — generated by setup_windows.ps1
# Edit as needed; CLI flags override any value here.
# Paths use forward slashes — safe in TOML and accepted by Rust on Windows.

[lecturner]

# ── Input / output ────────────────────────────────────────────────────────────
input   = "text.txt"
out_dir = "audio_out"

# ── Splitting ─────────────────────────────────────────────────────────────────
max_words = 200
min_chars = 10

# ── Output / timing ───────────────────────────────────────────────────────────
merge            = true
to_mp3           = true
sentence_gap_ms  = 180
paragraph_gap_ms = 360
rest_ms          = 300

# ── ffmpeg ────────────────────────────────────────────────────────────────────
ffmpeg_bin = "ffmpeg"

# ── PDF ripping ───────────────────────────────────────────────────────────────
skip_refs     = true
skip_captions = true

# ── Whisper validation ────────────────────────────────────────────────────────
validate           = true
whisper_model      = "ggml-medium.en.bin"
whisper_model_dir  = "$ModelsDirToml"
validate_threshold = 0.18

# ── LLM text cleanup ──────────────────────────────────────────────────────────
llm_clean         = true
crane_llm_bin     = "$CraneBinToml"
crane_llm_model   = "$QwenLlmToml"
crane_llm_port    = 8101
crane_llm_timeout = 60

# ── TTS ───────────────────────────────────────────────────────────────────────
crane_tts_bin     = "$CraneBinToml"
crane_tts_model   = "$QwenTtsToml"
crane_tts_port    = 8102
crane_tts_timeout = 60
crane_tts_voice   = "Aiden"
crane_tts_instruct = "read clearly and calmly at a medium pace."
"@ | Set-Content -Path $TomlOut -Encoding UTF8

Write-Ok "lecturner.toml written → $TomlOut"

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host " All done!  Backend: $BackendLabel"
Write-Host ""
Write-Host " Crane:      $CraneBin"
Write-Host " lecturner:  $LecturnerBin"
Write-Host " Config:     $TomlOut"
Write-Host ""
Write-Host " Quick start:"
Write-Host "   cd $LecturnerDir"
Write-Host "   New-Item -ItemType Directory -Force -Path batch\in"
Write-Host "   Copy-Item C:\your\paper.pdf batch\in\"
Write-Host "   .\target\release\lecturner.exe --batch-pdf batch"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
