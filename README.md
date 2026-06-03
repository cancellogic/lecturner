# Lecturner

Turns PDFs and text files into narrated MP3 audiobooks. Keep up with professional papers on your commute. Powered by Qwen3-TTS, Qwen3-4B, Whisper, pdfplumber, and ffmpeg via [lucasjinreal Crane](https://github.com/lucasjinreal/Crane).

---

## What it does

Lecturner takes a PDF or text file and produces a narrated MP3, paragraph by paragraph:

1. **Rip** — extract clean prose from a PDF via `pdf_rip.py` (pdfplumber), handling two-column layouts, dropping references and captions
2. **Clean** — rewrite extracted prose for natural spoken delivery using Qwen3-4B via Crane
3. **Speak** — synthesise each paragraph via Qwen3-TTS CustomVoice (Crane)
4. **Validate** — transcribe each WAV with Whisper and check phoneme error rate; quarantine glitched chunks automatically
5. **Merge** — concatenate paragraph WAVs with tuned silence gaps, transcode to MP3 via ffmpeg

Drop a stack of PDFs in a folder and run overnight. Wake up with a playlist. Yes, it is a little slow. Note that adding files to the batch after a batch has started results in ignored the new files until the next batch run.
---

## Quick Start
**Before you begin you need**
- **Windows**: MSVC build tools (Visual Studio Build Tools, C++ workload) + CUDA toolkit
- **macOS**: `xcode-select --install`
- **Linux**: `apt install build-essential libclang-dev` + CUDA toolkit if using GPU

**1. Install system tools**
- [Rust](https://rustup.rs)
- [Python 3.8+](https://python.org)
- [ffmpeg](https://ffmpeg.org) on PATH
- `pip install pdfplumber`
- You may need xcode or clang or gcc  for CUDA and building rust's Crane

**2. Build Crane** (the inference engine — not mine to distribute)
```bash
git clone https://github.com/lucasjinreal/Crane
cd Crane

# Windows / Linux with CUDA:
cargo build -p crane-oai --release --features cuda

# macOS Apple Silicon (Metal):
cargo build -p crane-oai --release --features metal

# CPU only:
cargo build -p crane-oai --release
```

**3. Download models**
```bash
# LLM text cleanup
hf download Qwen/Qwen3-4B --local-dir checkpoints/Qwen3-4B

# TTS speech synthesis
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir Qwen3-TTS-12Hz-1.7B-CustomVoice

# Whisper audio validation
mkdir models
curl -L -o models/ggml-medium.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin

# CMU Pronouncing Dictionary (phoneme validation)
curl -L -o cmudict.dict \
  https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
```

**4. Build Lecturner**

Edit `Cargo.toml` first — uncomment the `whisper-rs` line that matches your platform (CUDA / Metal / CPU), then:

```bash
git clone https://github.com/cancellogic/lecturner
cd lecturner
cargo build --release
```

**5. Configure**

Edit `lecturner.toml` — at minimum set the four paths that point to your Crane binary and model directories:

```toml
[lecturner]
crane_llm_bin   = "/path/to/Crane/target/release/crane-oai"
crane_llm_model = "checkpoints/Qwen3-4B"

crane_tts_bin   = "/path/to/Crane/target/release/crane-oai"
crane_tts_model = "Qwen3-TTS-12Hz-1.7B-CustomVoice"
```

**6. Run**
```bash
# Drop PDFs or text files into batch/in/ then:
lecturner --batch-pdf batch
```
---

## Hardware requirements

| Platform | Minimum | Recommended |
|---|---|---|
| Windows | 16 GB RAM, NVIDIA GPU 8 GB VRAM | 24 GB VRAM |
| macOS Apple Silicon | 16 GB unified memory | 24 GB |
| Linux | 16 GB RAM, NVIDIA GPU 8 GB VRAM | 24 GB VRAM |

Qwen3-4B and Qwen3-TTS-1.7B run sequentially, not concurrently — peak VRAM is ~8 GB.
CPU-only is supported but synthesis will be slow.

---

## Usage

### Single PDF
```bash
lecturner --rip-pdf paper.pdf
```
Rips → cleans → speaks → validates → produces `audio_out/combined.mp3`.

### Text file
```bash
lecturner --input myarticle.txt
```
Skips the rip step. LLM cleanup runs if `llm_clean = true` in config.

### Rip only (inspect before committing)
```bash
lecturner --rip-pdf paper.pdf --rip-pdf-only
```
Extracts prose to `text.txt` and stops. Read it before a multi-hour synthesis run.

### Batch overnight
```bash
lecturner --batch-pdf batch
```
Processes every `.pdf` and `.txt` in `batch/in/` unattended. On first run it creates the directory tree — just drop files into `batch/in/` and fire it.

Output layout:
```
batch/
  in/                  drop files here before starting
  audio/               paper.mp3  (or paper_part.mp3 if chunks were quarantined)
  text_completed/      paper.txt  (cleaned prose)
  pdf_completed/       input file moves here on success
  pdf_errored/         input file moves here on hard failure
```

### Repair quarantined chunks
```bash
lecturner --fix-quarantine
lecturner --merge-only
```
Re-synthesises chunks that failed Whisper validation, then rebuilds the merged output.

---

## Validation

When `validate = true` in `lecturner.toml`, each synthesised WAV is transcribed by Whisper and compared phoneme-by-phoneme against the source text. Chunks exceeding `validate_threshold` (default 0.18) are quarantined rather than included in the final merge. A threshold of 0.18 catches genuine synthesis glitches while comfortably passing technical terminology, dates, and abbreviations.

Typical clean-run PER scores on technical prose: 0.000–0.062.

---

## Model licenses

Lecturner is MIT licensed. The models it uses have their own licenses:

- **Qwen3-4B / Qwen3-TTS** — Tongyi Qianwen License; permissive for personal and research use, restrictions apply to commercial deployment at scale. Review before commercial use.
- **Whisper** — MIT
- **cmudict** — Public domain (Carnegie Mellon University)
- **Crane** — check [Crane's repository](https://github.com/lucasjinreal/Crane) for its current license

---

## Acknowledgements

Built with [Crane](https://github.com/lucasjinreal/Crane) by lucasjinreal —
without Crane's Qwen3-TTS and Qwen3-4B inference this project would not exist.

Pair programmed with Claude Sonnet (Anthropic).

Crane and Claude got me through early-days AI model Python and clang and C++ and fickle version/subversion dependency hell. Some regret that Python is still used to rip PDF text (sorry pdfplumber, you are a great program). And regret for the users that you need to break out grep, curl, hf, and download the really important bits that make this program work. Be sure to edit `lecturner.toml` file paths to point at those important bits. Cheers!

---

## License

MIT — see `LICENSE`. And I love a good shout out.
