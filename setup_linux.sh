#!/usr/bin/env bash
# =============================================================================
# setup_linux.sh — Linux bootstrap for Crane + lecturner
#
# ROADMAP:
#   1. Check system build requirements (build-essential, libclang-dev, clang,
#      ffmpeg, Rust, hf CLI); detect package manager; probe nvcc
#   2. Ask user: CUDA or CPU  (Metal is Apple-only)
#   3. Download model assets:
#        - Qwen3-4B            → checkpoints/Qwen3-4B/
#        - Qwen3-TTS CustomVoice → Qwen3-TTS-12Hz-1.7B-CustomVoice/
#        - Whisper ggml binary  → models/ggml-medium.en.bin
#        - CMU dict             → cmudict.dict  (root of work dir)
#   4. Clone + build lucasjinreal/Crane with the chosen feature flag
#      (CUDA builds need the MSVC-equivalent: run from a shell where
#       nvcc and the CUDA headers are on PATH — source cuda's env if needed)
#   5. Clone lecturner, patch Cargo.toml whisper-rs line, build, write toml
#
# Tested on Ubuntu 22.04 / Debian 12.  Run from your install directory.
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
die()  { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

WORK_DIR="$(pwd)"

# ── Step 1: System build requirements ────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 1 — Checking system build requirements"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Detect package manager
if command -v apt-get &>/dev/null; then
    PKG_INSTALL="sudo apt-get install -y"
    PKG_UPDATE="sudo apt-get update -qq"
elif command -v dnf &>/dev/null; then
    PKG_INSTALL="sudo dnf install -y"
    PKG_UPDATE="sudo dnf check-update || true"
elif command -v pacman &>/dev/null; then
    PKG_INSTALL="sudo pacman -S --noconfirm"
    PKG_UPDATE="sudo pacman -Sy"
else
    warn "Unknown package manager — install deps manually if the build fails."
    PKG_INSTALL="echo 'Please install manually:'"
    PKG_UPDATE=":"
fi

# lecturner README specifies: apt install build-essential libclang-dev
# libclang-dev is required by whisper-rs bindgen for both CUDA and Metal paths.
NEED_UPDATE=false
for tool in git curl gcc; do
    if ! command -v "$tool" &>/dev/null; then
        warn "Missing: $tool"
        NEED_UPDATE=true
    fi
done
if ! (dpkg -s libclang-dev &>/dev/null 2>&1 || rpm -q clang-devel &>/dev/null 2>&1); then
    warn "libclang-dev not found"
    NEED_UPDATE=true
fi
if $NEED_UPDATE; then
    $PKG_UPDATE
    $PKG_INSTALL build-essential libclang-dev git curl
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

# hf CLI — huggingface_hub ≥ 1.0
if ! command -v hf &>/dev/null; then
    warn "'hf' CLI not found — installing via pip3…"
    pip3 install --user -U huggingface_hub \
        || die "pip3 failed — install python3-pip first."
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
        1) CRANE_FEATURE="cuda"; WHISPER_FEATURE="cuda"; BACKEND_LABEL="CUDA" ;;
        2) CRANE_FEATURE="";     WHISPER_FEATURE="cpu";  BACKEND_LABEL="CPU"  ;;
        *) warn "Unrecognised — defaulting to CUDA."
           CRANE_FEATURE="cuda"; WHISPER_FEATURE="cuda"; BACKEND_LABEL="CUDA" ;;
    esac
else
    echo "  No CUDA detected — using CPU."
    CRANE_FEATURE=""
    WHISPER_FEATURE="cpu"
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

echo "Building crane-oai [feature: ${CRANE_FEATURE:-'(none — CPU)'}]…"
if [[ -n "$CRANE_FEATURE" ]]; then
    cargo build \
        --manifest-path "$CRANE_DIR/Cargo.toml" \
        -p crane-oai --release \
        --features "$CRANE_FEATURE"
else
    cargo build \
        --manifest-path "$CRANE_DIR/Cargo.toml" \
        -p crane-oai --release
fi

CRANE_BIN="$CRANE_DIR/target/release/crane-oai"
ok "Crane built → $CRANE_BIN"

# ── Step 5: Clone + patch + build lecturner ───────────────────────────────────
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

# Patch the whisper-rs feature line in Cargo.toml.
#
# Real Cargo.toml ships with cuda active, metal and cpu commented:
#   whisper-rs = { version = "0.16", features = ["cuda"] }
#   # whisper-rs = { version = "0.16", features = ["metal"] }
#   # whisper-rs = { version = "0.16" }
#
# Strategy: comment all three whisper-rs lines, then uncomment the right one.
# GNU sed uses -i '' with no extension argument; we write .bak explicitly.

CARGO_TOML="$LECTURNER_DIR/Cargo.toml"
cp "$CARGO_TOML" "$CARGO_TOML.bak"
echo "Patching $CARGO_TOML for backend: $BACKEND_LABEL…"

# Step A — comment every active whisper-rs line (idempotent on re-run)
sed -i 's|^whisper-rs |# whisper-rs |' "$CARGO_TOML"

# Step B — uncomment the right one
case "$WHISPER_FEATURE" in
    cuda)
        sed -i 's|^# \(whisper-rs = { version = "0.16", features = \["cuda"\].*\)|\1|' "$CARGO_TOML"
        ;;
    cpu)
        # CPU line has no features key at all — match exactly to avoid hitting cuda/metal
        sed -i 's|^# \(whisper-rs = { version = "0.16" }\)|\1|' "$CARGO_TOML"
        ;;
esac
ok "Cargo.toml patched — active whisper-rs line:"
grep '^whisper-rs' "$CARGO_TOML" || warn "  No active line found — check $CARGO_TOML"

cargo build \
    --manifest-path "$CARGO_TOML" \
    --release

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
