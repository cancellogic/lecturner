#!/usr/bin/env bash
# =============================================================================
# setup_linux.sh — Linux bootstrap for Crane + lecturner
#
# ROADMAP:
#   1. Check system build requirements (build-essential, libclang-dev, clang,
#      ffmpeg, cmake, Rust, hf CLI); probe nvcc.  apt-only — dies with a pointer to
#      the README's manual steps on other distros (per README: Debian/Ubuntu
#      or manual install)
#   2. Ask user: CUDA or CPU  (Metal is Apple-only)
#   3. Download model assets:
#        - Qwen3-4B            → checkpoints/Qwen3-4B/
#        - Qwen3-TTS CustomVoice → Qwen3-TTS-12Hz-1.7B-CustomVoice/
#        - Whisper ggml binary  → models/ggml-medium.en.bin
#        - CMU dict             → cmudict.dict  (work dir; copied into
#          the lecturner repo root in Step 5 — compile-time embed)
#   4. Clone + build lucasjinreal/Crane with the chosen feature flag
#      (CUDA builds need nvcc and the CUDA headers reachable — put
#       /usr/local/cuda/bin on PATH first if the build can't find them)
#   5. Clone + build lecturner with the SAME feature flag.  Cargo features
#      replaced the old comment-out-the-right-line Cargo.toml patching, so
#      the repo stays clean and re-runs can git pull without conflict.
#   6. Write lecturner.toml with absolute paths
#
# Tested on Ubuntu 22.04 / Debian 12.  Run from your install directory.
# Line endings must stay LF — add `*.sh text eol=lf` to .gitattributes;
# CRLF here breaks bash with "syntax error near unexpected token".
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
die()  { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

# Refuse to run as root.  Run as a normal user with sudo rights — the script
# invokes sudo itself only for apt.  Under a root shell, rustup/cargo/pipx
# would all install into /root instead of your home, and everything built or
# downloaded would be root-owned.
if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    die "Don't run this as root/sudo — run as your normal user.
    If you hit 'permission denied' launching the script, the execute bit
    is missing:  chmod +x setup_linux.sh   (or run:  bash setup_linux.sh)"
fi

WORK_DIR="$(pwd)"

# ── Step 1: System build requirements ────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 1 — Checking system build requirements"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# apt-only, honestly.  The old version detected dnf/pacman but then fed them
# apt package names (build-essential doesn't exist on Fedora or Arch), so it
# failed anyway — just later and more confusingly.  README policy stands:
# Debian/Ubuntu via this script, everything else manual.
if ! command -v apt-get &>/dev/null; then
    die "No apt-get found — this script supports Debian/Ubuntu only.
    For other distros, install manually (see README 'Quick Start'):
      build tools + libclang  (Fedora: gcc gcc-c++ make clang-devel;
                               Arch: base-devel clang)
      cmake (whisper-rs builds whisper.cpp with it), ffmpeg, git, curl,
      Rust 1.88+, and the 'hf' CLI (pipx install huggingface_hub)"
fi
PKG_INSTALL="sudo apt-get install -y"
PKG_UPDATE="sudo apt-get update -qq"

# lecturner README specifies: apt install build-essential libclang-dev
# libclang-dev is required by whisper-rs bindgen; cmake is required because
# whisper-rs-sys compiles whisper.cpp from source at build time.
NEED_UPDATE=false
for tool in git curl gcc cmake; do
    if ! command -v "$tool" &>/dev/null; then
        warn "Missing: $tool"
        NEED_UPDATE=true
    fi
done
if ! dpkg -s libclang-dev &>/dev/null 2>&1; then
    warn "libclang-dev not found"
    NEED_UPDATE=true
fi
if $NEED_UPDATE; then
    $PKG_UPDATE
    $PKG_INSTALL build-essential libclang-dev git curl cmake
fi
ok "Core build tools present"

# clang on PATH — whisper-rs build.rs invokes it directly
if ! command -v clang &>/dev/null; then
    $PKG_INSTALL clang
fi
ok "clang: $(clang --version | head -1)"

# ffmpeg — lecturner calls it as a subprocess for WAV→MP3 transcoding
if ! command -v ffmpeg &>/dev/null; then
    warn "ffmpeg not found — installing…"
    $PKG_INSTALL ffmpeg
fi
ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# Rust 1.88+ required per lecturner README
if command -v rustup &>/dev/null; then
    rustup update stable
    ok "Rust: $(rustc --version)"
else
    warn "rustup not found — installing…"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    ok "Rust installed: $(rustc --version)"
fi

# hf CLI — huggingface_hub ≥ 1.0.
# pipx, not pip3 --user: Debian 12 and Ubuntu 23.04+ mark the system Python
# externally managed (PEP 668), so bare pip3 refuses to install.  pipx is the
# blessed path; `pip3 install --user --break-system-packages huggingface_hub`
# is the manual fallback if pipx is unavailable.
if ! command -v hf &>/dev/null; then
    warn "'hf' CLI not found — installing via pipx…"
    if ! command -v pipx &>/dev/null; then
        $PKG_INSTALL pipx || die "Could not install pipx — install it manually, then re-run."
    fi
    pipx install huggingface_hub \
        || die "pipx install failed — try: pip3 install --user --break-system-packages huggingface_hub"
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v hf &>/dev/null \
    || die "'hf' not on PATH — add ~/.local/bin to PATH and re-run."
ok "hf CLI: $(hf --version 2>&1 | head -1)"

# CUDA toolkit check — non-fatal; shapes the backend menu below
CUDA_AVAILABLE=false
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
    ok "CUDA toolkit: $CUDA_VER"
    CUDA_AVAILABLE=true
else
    warn "nvcc not found — CUDA option will not be offered."
    warn "Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads if you have an NVIDIA GPU."
fi

# ── Step 2: Backend selection ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 2 — Choose inference backend"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if $CUDA_AVAILABLE; then
    echo "  NVIDIA GPU with CUDA detected. Options:"
    echo "    1) CUDA  (recommended — fastest)"
    echo "    2) CPU   (slower, no GPU required)"
    read -rp "Enter choice [1-2] (default: 1): " BACKEND_CHOICE
    BACKEND_CHOICE=${BACKEND_CHOICE:-1}
    case "$BACKEND_CHOICE" in
        1) BACKEND_FEATURE="cuda"; BACKEND_LABEL="CUDA" ;;
        2) BACKEND_FEATURE="";     BACKEND_LABEL="CPU"  ;;
        *) warn "Unrecognised — defaulting to CUDA."
           BACKEND_FEATURE="cuda"; BACKEND_LABEL="CUDA" ;;
    esac
else
    echo "  No CUDA detected — using CPU."
    BACKEND_FEATURE=""
    BACKEND_LABEL="CPU"
fi
ok "Selected backend: $BACKEND_LABEL"

# ── Step 3: Download model assets ─────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 3 — Downloading model assets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Qwen3-4B ~8 GB · TTS model ~3 GB · hf download is resume-safe."
echo ""

# Qwen3-4B — crane_llm_model in toml
QWEN_LLM_DIR="$WORK_DIR/checkpoints/Qwen3-4B"
mkdir -p "$(dirname "$QWEN_LLM_DIR")"
if [[ ! -d "$QWEN_LLM_DIR" ]]; then
    echo "Downloading Qwen3-4B…"
    hf download Qwen/Qwen3-4B --local-dir "$QWEN_LLM_DIR"
    ok "Qwen3-4B → $QWEN_LLM_DIR"
else
    ok "Qwen3-4B already present"
fi

# Qwen3-TTS — crane_tts_model in toml
QWEN_TTS_DIR="$WORK_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice"
if [[ ! -d "$QWEN_TTS_DIR" ]]; then
    echo "Downloading Qwen3-TTS-12Hz-1.7B-CustomVoice…"
    hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir "$QWEN_TTS_DIR"
    ok "Qwen3-TTS → $QWEN_TTS_DIR"
else
    ok "Qwen3-TTS already present"
fi

# Whisper ggml binary — whisper_model + whisper_model_dir in toml (two separate keys)
MODELS_DIR="$WORK_DIR/models"
mkdir -p "$MODELS_DIR"
WHISPER_BIN="$MODELS_DIR/ggml-medium.en.bin"
if [[ ! -f "$WHISPER_BIN" ]]; then
    echo "Downloading Whisper ggml-medium.en.bin…"
    curl -L -o "$WHISPER_BIN" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin"
    ok "Whisper → $WHISPER_BIN"
else
    ok "Whisper already present"
fi

# CMU Pronouncing Dictionary — goes in work dir root by convention (not a toml key)
CMUDICT="$WORK_DIR/cmudict.dict"
if [[ ! -f "$CMUDICT" ]]; then
    echo "Downloading CMU Pronouncing Dictionary…"
    curl -L -o "$CMUDICT" \
        "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    ok "CMU dict → $CMUDICT"
else
    ok "CMU dict already present"
fi

# ── Step 4: Clone + build Crane ───────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 4 — Building Crane"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$BACKEND_LABEL" == "CUDA" ]]; then
    echo "  CUDA build: if you hit linker errors, make sure CUDA is on PATH:"
    echo "    export PATH=/usr/local/cuda/bin:\$PATH"
    echo "  and CUDA lib is findable:"
    echo "    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi
echo ""

CRANE_DIR="$WORK_DIR/Crane"
if [[ ! -d "$CRANE_DIR/.git" ]]; then
    git clone https://github.com/lucasjinreal/Crane.git "$CRANE_DIR"
else
    ok "Crane repo already present — pulling latest…"
    git -C "$CRANE_DIR" pull --ff-only
fi

echo "Building crane-oai [feature: ${BACKEND_FEATURE:-'(none — CPU)'}]…"
if [[ -n "$BACKEND_FEATURE" ]]; then
    cargo build \
        --manifest-path "$CRANE_DIR/Cargo.toml" \
        -p crane-oai --release \
        --features "$BACKEND_FEATURE"
else
    cargo build \
        --manifest-path "$CRANE_DIR/Cargo.toml" \
        -p crane-oai --release
fi

CRANE_BIN="$CRANE_DIR/target/release/crane-oai"
ok "Crane built → $CRANE_BIN"

# ── Step 5: Clone + build lecturner ───────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 5 — Building lecturner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

LECTURNER_DIR="$WORK_DIR/lecturner"
if [[ ! -d "$LECTURNER_DIR/.git" ]]; then
    git clone https://github.com/cancellogic/lecturner.git "$LECTURNER_DIR"
else
    ok "lecturner repo already present — pulling latest…"
    git -C "$LECTURNER_DIR" pull --ff-only
fi

# cmudict.dict is embedded into the binary at COMPILE time via
# include_str!("../cmudict.dict") in src/main.rs — resolved relative to the
# source file, so the dict must sit in the LECTURNER REPO ROOT before cargo
# runs.  Step 3 downloaded it to the work dir; fresh clones lack it.
if [[ ! -f "$LECTURNER_DIR/cmudict.dict" ]]; then
    cp "$CMUDICT" "$LECTURNER_DIR/cmudict.dict"
    ok "cmudict.dict → lecturner repo root (compile-time embed)"
fi

# Backend selection is a cargo feature (cuda / metal / none = CPU) — same
# flag we just passed to Crane.  No Cargo.toml patching, so the repo stays
# clean and the pull above never hits a dirty-tree conflict on re-runs.
CARGO_TOML="$LECTURNER_DIR/Cargo.toml"
echo "Building lecturner [feature: ${BACKEND_FEATURE:-'(none — CPU)'}]…"
if [[ -n "$BACKEND_FEATURE" ]]; then
    cargo build \
        --manifest-path "$CARGO_TOML" \
        --release \
        --features "$BACKEND_FEATURE"
else
    cargo build \
        --manifest-path "$CARGO_TOML" \
        --release
fi

LECTURNER_BIN="$LECTURNER_DIR/target/release/lecturner"
ok "lecturner built → $LECTURNER_BIN"

# ── Step 6: Write lecturner.toml ──────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 6 — Writing lecturner.toml"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TOML_OUT="$LECTURNER_DIR/lecturner.toml"
cat > "$TOML_OUT" <<TOML
# lecturner.toml — generated by setup_linux.sh
# Edit as needed; CLI flags override any value here.

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
whisper_model_dir  = "$MODELS_DIR"
validate_threshold = 0.18

# ── LLM text cleanup ──────────────────────────────────────────────────────────
llm_clean         = true
crane_llm_bin     = "$CRANE_BIN"
crane_llm_model   = "$QWEN_LLM_DIR"
crane_llm_port    = 8101
crane_llm_timeout = 60

# ── TTS ───────────────────────────────────────────────────────────────────────
crane_tts_bin     = "$CRANE_BIN"
crane_tts_model   = "$QWEN_TTS_DIR"
crane_tts_port    = 8102
crane_tts_timeout = 60
crane_tts_voice   = "Aiden"
crane_tts_instruct = "read clearly and calmly at a medium pace."
TOML

ok "lecturner.toml written → $TOML_OUT"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " All done!  Backend: $BACKEND_LABEL"
echo ""
echo " Crane:      $CRANE_BIN"
echo " lecturner:  $LECTURNER_BIN"
echo " Config:     $TOML_OUT"
echo ""
echo " Quick start:"
echo "   cd $LECTURNER_DIR"
echo "   mkdir -p batch/in"
echo "   cp /your/paper.pdf batch/in/"
echo "   ./target/release/lecturner --batch-pdf batch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
