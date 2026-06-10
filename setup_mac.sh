#!/usr/bin/env bash
# =============================================================================
# setup_mac.sh — macOS bootstrap for Crane + lecturner
#
# ROADMAP:
#   1. Check system build requirements (Xcode CLT, clang, Homebrew, Rust,
#      ffmpeg, hf CLI)
#   2. Ask user: Apple Metal or CPU  (CUDA not available on macOS)
#   3. Download model assets:
#        - Qwen3-4B            → checkpoints/Qwen3-4B/
#        - Qwen3-TTS CustomVoice → Qwen3-TTS-12Hz-1.7B-CustomVoice/
#        - Whisper ggml binary  → models/ggml-medium.en.bin
#        - CMU dict             → cmudict.dict  (root of work dir)
#   4. Clone + build lucasjinreal/Crane — NO feature flag on macOS:
#      crane-core hardwires Metal (Apple Silicon) / Accelerate (Intel)
#      via target-specific deps; crane-oai has no "metal" feature
#   5. Clone + build lecturner with the SAME feature flag.  Cargo features
#      replaced the old comment-out-the-right-line Cargo.toml patching, so
#      the repo stays clean and re-runs can git pull without conflict.
#   6. Write lecturner.toml with absolute paths
#
# Run from the directory you want everything installed into.
# Requires macOS 12+.
# Line endings must stay LF — add `*.sh text eol=lf` to .gitattributes;
# CRLF here breaks bash with "syntax error near unexpected token".
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
die()  { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

# Refuse to run as root.  rustup, cargo, and Homebrew are all per-user tools:
# under sudo, rustup provisions a second toolchain into /var/root/.cargo (then
# errors), downloads land root-owned, and Homebrew refuses root outright.
# Nothing in this script needs elevation on macOS.
if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    die "Don't run this as root/sudo — everything here is user-level.
    If you hit 'permission denied' launching the script, the execute bit
    is missing:  chmod +x setup_mac.sh   (or run:  bash setup_mac.sh)
    If a root shell already created /var/root/.rustup or /var/root/.cargo,
    remove them:  sudo rm -rf /var/root/.rustup /var/root/.cargo"
fi

WORK_DIR="$(pwd)"

# ── Step 1: System build requirements ────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 1 — Checking system build requirements"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# macOS version — Metal requires 12+
MACOS_VER=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VER" | cut -d. -f1)
[[ "$MACOS_MAJOR" -ge 12 ]] \
    && ok "macOS $MACOS_VER" \
    || warn "macOS $MACOS_VER is below 12 — Metal may not work; CPU still fine"

# Xcode CLT — provides clang, which whisper-rs bindgen needs for Metal builds
if xcode-select -p &>/dev/null; then
    ok "Xcode CLT: $(xcode-select -p)"
else
    warn "Xcode command-line tools not found — installing…"
    xcode-select --install
    echo "Re-run this script after the Xcode CLT installer completes."
    exit 1
fi

command -v clang &>/dev/null || die "clang not found even after CLT check — install Xcode from the App Store."
ok "clang: $(clang --version | head -1)"

# Homebrew
command -v brew &>/dev/null || die "Homebrew not found — install from https://brew.sh then re-run."
ok "Homebrew: $(brew --version | head -1)"

# ffmpeg — called as a subprocess by lecturner for WAV→MP3 transcoding
if ! command -v ffmpeg &>/dev/null; then
    warn "ffmpeg not found — installing via Homebrew…"
    brew install ffmpeg
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

# hf CLI — huggingface_hub ≥ 1.0 renamed huggingface-cli to hf
if ! command -v hf &>/dev/null; then
    warn "'hf' CLI not found — installing via pip3…"
    pip3 install --user -U huggingface_hub
    # macOS pip user bin lives under ~/Library/Python/x.y/bin
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    export PATH="$HOME/Library/Python/$PY_VER/bin:$PATH"
fi
command -v hf &>/dev/null \
    || die "'hf' not on PATH after install — add ~/Library/Python/x.y/bin to your PATH and re-run."
ok "hf CLI: $(hf --version 2>&1 | head -1)"

# ── Step 2: Backend selection ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 2 — Choose inference backend"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ARCH=$(uname -m)
# NB: this choice now applies ONLY to lecturner's Whisper validation
# (whisper-rs gates Metal behind a feature flag).  Crane ignores it —
# crane-core hardwires Metal/Accelerate by build target on macOS.
if [[ "$ARCH" == "arm64" ]]; then
    echo "  Apple Silicon detected. Whisper validation backend:"
    echo "    1) Apple Metal  (recommended — ~6x faster than CPU on M-series)"
    echo "    2) CPU          (slower, works everywhere)"
    read -rp "Enter choice [1-2] (default: 1): " BACKEND_CHOICE
    BACKEND_CHOICE=${BACKEND_CHOICE:-1}
    case "$BACKEND_CHOICE" in
        1) BACKEND_FEATURE="metal"; BACKEND_LABEL="Apple Metal" ;;
        2) BACKEND_FEATURE="";      BACKEND_LABEL="CPU" ;;
        *) warn "Unrecognised — defaulting to Metal."
           BACKEND_FEATURE="metal"; BACKEND_LABEL="Apple Metal" ;;
    esac
else
    warn "Intel Mac — Metal not supported; using CPU."
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

# Qwen3-4B — LLM text cleanup; lecturner.toml key: crane_llm_model
QWEN_LLM_DIR="$WORK_DIR/checkpoints/Qwen3-4B"
mkdir -p "$(dirname "$QWEN_LLM_DIR")"
if [[ ! -d "$QWEN_LLM_DIR" ]]; then
    echo "Downloading Qwen3-4B…"
    hf download Qwen/Qwen3-4B --local-dir "$QWEN_LLM_DIR"
    ok "Qwen3-4B → $QWEN_LLM_DIR"
else
    ok "Qwen3-4B already present at $QWEN_LLM_DIR"
fi

# Qwen3-TTS — speech synthesis; lecturner.toml key: crane_tts_model
QWEN_TTS_DIR="$WORK_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice"
if [[ ! -d "$QWEN_TTS_DIR" ]]; then
    echo "Downloading Qwen3-TTS-12Hz-1.7B-CustomVoice…"
    hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir "$QWEN_TTS_DIR"
    ok "Qwen3-TTS → $QWEN_TTS_DIR"
else
    ok "Qwen3-TTS already present at $QWEN_TTS_DIR"
fi

# Whisper ggml binary — lecturner.toml keys: whisper_model + whisper_model_dir
MODELS_DIR="$WORK_DIR/models"
mkdir -p "$MODELS_DIR"
WHISPER_BIN="$MODELS_DIR/ggml-medium.en.bin"
if [[ ! -f "$WHISPER_BIN" ]]; then
    echo "Downloading Whisper ggml-medium.en.bin…"
    curl -L -o "$WHISPER_BIN" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin"
    ok "Whisper → $WHISPER_BIN"
else
    ok "Whisper already present at $WHISPER_BIN"
fi

# CMU Pronouncing Dictionary — goes in work dir root (not referenced in toml;
# whisper validation finds it by convention next to the binary)
CMUDICT="$WORK_DIR/cmudict.dict"
if [[ ! -f "$CMUDICT" ]]; then
    echo "Downloading CMU Pronouncing Dictionary…"
    curl -L -o "$CMUDICT" \
        "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    ok "CMU dict → $CMUDICT"
else
    ok "CMU dict already present at $CMUDICT"
fi

# ── Step 4: Clone + build Crane ───────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 4 — Building Crane"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CRANE_DIR="$WORK_DIR/Crane"
if [[ ! -d "$CRANE_DIR/.git" ]]; then
    git clone https://github.com/lucasjinreal/Crane.git "$CRANE_DIR"
else
    ok "Crane repo already present — pulling latest…"
    git -C "$CRANE_DIR" pull --ff-only
fi

# Crane gets NO feature flag on macOS — and this is correct, not an omission.
# crane-core's Cargo.toml selects the backend by build target:
#   macos + aarch64 → candle with "metal"      (automatic, not a feature)
#   macos + x86_64  → candle with "accelerate" (automatic)
#   elsewhere       → CPU unless --features cuda
# crane-oai's only named features are cuda / cudnn / mkl; passing
# "--features metal" fails with: "the package 'crane-oai' does not contain
# this feature".  Verified against crane-core/Cargo.toml 2026-06.
# ($BACKEND_FEATURE still matters — lecturner's whisper-rs DOES gate Metal
#  behind an explicit feature.  Two upstreams, two conventions.)
echo "Building crane-oai (backend auto-selected by target: Metal on Apple Silicon)…"
cargo build \
    --manifest-path "$CRANE_DIR/Cargo.toml" \
    -p crane-oai --release

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

# Backend selection for lecturner IS a cargo feature (metal / none = CPU) —
# unlike Crane above, whisper-rs gates Metal explicitly.  No Cargo.toml
# patching, so the repo stays clean and the pull above never hits a
# dirty-tree conflict on re-runs.
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

# Keys and structure match the real lecturner.toml exactly.
# whisper_model + whisper_model_dir are separate (not a single path).
# cmudict.dict is not a toml key — it goes in the work dir root by convention.
TOML_OUT="$LECTURNER_DIR/lecturner.toml"
cat > "$TOML_OUT" <<TOML
# lecturner.toml — generated by setup_mac.sh
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
