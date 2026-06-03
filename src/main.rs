//! lecturner — paragraph-by-paragraph audiobook pipeline.
//!
//! Reads `text.txt`, optionally rips a PDF and rewrites prose via an LLM,
//! splits on blank lines into paragraphs, sends each chunk to a Crane
//! (crane-oai) TTS server running Qwen3-TTS at POST /v1/audio/speech,
//! saves `paragraph_NNN.wav`, validates transcription with Whisper, then
//! optionally merges into `combined.wav` and transcodes to `combined.mp3`
//! via ffmpeg.
//!
//! Config file (`lecturner.toml` next to the binary) provides defaults so you
//! don't have to retype them.  CLI flags always win over the config.
//!
//! Minimal launch once lecturner.toml is set up:
//!   cargo run --release

mod records;
use records::{Boundary, ChunkRecord, ChunkStatus, RunRecord};

use std::{
    fs,
    path::PathBuf,
    process::{Child, Command},
    thread,
    time::Duration,
};

use anyhow::{Context, Result};
use clap::Parser;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

// ─── Config file shape ─────────────────────────────────────────────────────────
//
// lecturner.toml — every field is optional; missing = use the hard-coded default.
//
// [lecturner]
// input            = "text.txt"
// out_dir          = "audio_out"
// rest_ms          = 500
// min_chars        = 10
// max_words        = 200
// merge            = true
// to_mp3           = true
// sentence_gap_ms  = 180
// paragraph_gap_ms = 360
// ffmpeg_bin       = "ffmpeg"
// crane_tts_voice  = "Aiden"

#[derive(Deserialize, Default, Debug)]
struct LecturnerTomlConfig {
    lecturner: Option<LecturnerConfig>,
}

#[derive(Deserialize, Default, Debug)]
struct LecturnerConfig {
    input:                Option<String>,
    out_dir:              Option<String>,
    rest_ms:              Option<u64>,
    min_chars:            Option<usize>,
    max_words:            Option<usize>,
    merge:                Option<bool>,
    to_mp3:               Option<bool>,
    sentence_gap_ms:      Option<u32>,
    paragraph_gap_ms:     Option<u32>,
    ffmpeg_bin:           Option<String>,
    python_bin:           Option<String>,   // for pdf_rip.py (pdf_rip.py stays Python forever)
    rip_pdf:              Option<String>,
    skip_refs:            Option<bool>,
    skip_captions:        Option<bool>,
    // ── Whisper validation ─────────────────────────────────────────────────────
    validate:             Option<bool>,
    whisper_model:        Option<String>,
    whisper_model_dir:    Option<String>,
    validate_threshold:   Option<f32>,
    // ── LLM text cleanup (Track A-revised) ────────────────────────────────────
    // Defaults to true when rip_pdf is set, false otherwise — see resolve_config.
    llm_clean:            Option<bool>,
    crane_llm_bin:        Option<String>,   // path to crane-oai binary
    crane_llm_model:      Option<String>,   // path to Qwen3-4B dir
    crane_llm_port:       Option<u16>,      // default 8101
    crane_llm_timeout:    Option<u64>,      // startup wait, seconds; default 60
    // ── TTS (Track B-revised: Crane + Qwen3-TTS) ──────────────────────────────
    crane_tts_bin:        Option<String>,   // path to crane-oai binary (may share with LLM)
    crane_tts_model:      Option<String>,   // path to Qwen3-TTS checkpoint dir
    crane_tts_port:       Option<u16>,      // default 8102 — separate from LLM (8101)
    crane_tts_timeout:    Option<u64>,      // startup wait, seconds; default 60
    crane_tts_voice:      Option<String>,   // named speaker; default "Aiden"
    crane_tts_language:   Option<String>,   // instruct field; None = omit from request
    crane_tts_instruct:   Option<String>,   // speaking style hint; None = omit
}

fn load_toml_config() -> LecturnerConfig {
    // Look for lecturner.toml next to the binary, then next to CWD.
    let candidates: Vec<PathBuf> = [
        std::env::current_exe().ok().and_then(|mut p| { p.pop(); Some(p.join("lecturner.toml")) }),
        Some(PathBuf::from("lecturner.toml")),
    ]
    .into_iter()
    .flatten()
    .collect();

    for path in &candidates {
        if path.exists() {
            let raw = fs::read_to_string(path).unwrap_or_default();
            match toml::from_str::<LecturnerTomlConfig>(&raw) {
                Ok(cfg) => {
                    println!("[lecturner] Config loaded from {}", path.display());
                    return cfg.lecturner.unwrap_or_default();
                }
                Err(e) => eprintln!("[lecturner] Warning: Could not parse {}: {e}", path.display()),
            }
        }
    }
    LecturnerConfig::default()
}

// ─── CLI ───────────────────────────────────────────────────────────────────────

/// Paragraph-by-paragraph audiobook pipeline: PDF/TXT → MP3.
#[derive(Parser, Debug)]
#[command(author, version, about, after_help = "Dustan's Gossip works text and pdfs to spoken audio for personal use.
Greatly assisted by Claude Sonnet 4.6 (pair programmer),
Qwen3-TTS CustomVoice and Qwen3-4B for voice and text processing.  
(you'll need to correct the file paths are in the lecturner.toml program settings file.  
Your user directory is not my user directory and all.  Windows Cuda centric build
that might work on linux and mac with 16gb+ memory. )

MODES
─────
  Single file (default)
    cargo run --release
    Reads lecturner.toml -> text.txt -> audio_out/combined.mp3
    Assumes text.txt already exists.  Use --rip-pdf to produce it first.

  Single PDF, one command
    lecturner --rip-pdf paper.pdf
    Rips paper.pdf -> text.txt, runs LLM cleanup, synthesises, produces
    audio_out/combined.mp3.  Set llm_clean = true in lecturner.toml (automatic
    when --rip-pdf is used).

  Rip only (no audio)
    lecturner --rip-pdf paper.pdf --rip-pdf-only
    Extracts prose to text.txt and stops.  Useful for inspecting the text
    before committing a multi-hour synthesis run.

  Batch overnight
    lecturner --batch-pdf batch
    Processes every .pdf and .txt found in batch\\in\\ unattended.
    Outputs:
      batch\\audio\\             rosencrantz_guildenstern.mp3
                                ophelia_part.mp3  (quarantines present)
                                hamlet_picturebook.txt  (no audio produced)
      batch\\text_completed\\   rosencrantz_guildenstern.txt
      batch\\pdf_completed\\    processed input files
      batch\\pdf_errored\\      failed input files (2 strikes = skipped)
    Omit the path to use a 'batch' directory next to the binary.

  Repair quarantined chunks
    lecturner --fix-quarantine
    Re-synthesises chunks that failed Whisper validation in a previous run.
    Run --merge-only afterwards to rebuild combined.wav with recovered chunks.

  Rebuild audio without re-synthesising
    lecturner --merge-only
    Re-merges existing paragraph WAVs from run.json.  Useful after
    --fix-quarantine or manual WAV edits.

VALIDATION
----------
  Set validate = true in lecturner.toml to enable Whisper PER checking.
  Chunks that exceed validate_threshold after one retry are quarantined
  (not included in the merge).  Use --fix-quarantine to retry them.

CONFIG PRIORITY
---------------
  CLI flag  >  lecturner.toml  >  hard-coded default
  lecturner.toml is searched next to the binary, then in the working directory.
")]
struct CliArgs {
    /// Input text file (paragraphs separated by blank lines)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Directory where paragraph_NNN.wav files will be written
    #[arg(short, long)]
    out_dir: Option<PathBuf>,

    /// Milliseconds to sleep between paragraph requests
    #[arg(long)]
    rest_ms: Option<u64>,

    /// Minimum non-whitespace characters for a paragraph to be voiced
    #[arg(long)]
    min_chars: Option<usize>,

    /// Maximum words per synthesis chunk (0 = no sub-splitting)
    #[arg(long)]
    max_words: Option<usize>,

    /// Merge all paragraph WAVs into combined.wav after synthesis
    #[arg(long)]
    merge: Option<bool>,

    /// Transcode combined.wav to combined.mp3 via ffmpeg after merging
    #[arg(long)]
    to_mp3: Option<bool>,

    /// Silence between sentence-boundary splits (ms)
    #[arg(long)]
    sentence_gap_ms: Option<u32>,

    /// Silence between paragraph-boundary splits (ms)
    #[arg(long)]
    paragraph_gap_ms: Option<u32>,

    /// Path to ffmpeg executable (default: "ffmpeg" on PATH)
    #[arg(long)]
    ffmpeg_bin: Option<PathBuf>,

    /// Rip this PDF to text.txt then synthesise (or use --rip-pdf-only to stop after ripping)
    #[arg(long)]
    rip_pdf: Option<PathBuf>,

    /// Drop References/Bibliography section when ripping PDF
    #[arg(long)]
    skip_refs: Option<bool>,

    /// Drop figure/table captions when ripping PDF
    #[arg(long)]
    skip_captions: Option<bool>,

    /// Rip PDF to text.txt then exit without synthesising
    #[arg(long)]
    rip_pdf_only: bool,

    /// Re-merge existing paragraph WAVs from a previous run using run.json.
    /// Does not re-synthesize anything.
    #[arg(long)]
    merge_only: bool,

    /// Re-synthesize quarantined chunks from a previous run using run.json,
    /// then remind you to run --merge-only to rebuild the combined output.
    #[arg(long)]
    fix_quarantine: bool,

    /// Batch-process every PDF in <path>/in/ unattended overnight.
    /// Outputs land in <path>/audio/, text_completed/, pdf_completed/, pdf_errored/.
    /// Omit the path to use a "batch" directory next to the binary.
    #[arg(long, num_args = 0..=1, default_missing_value = "batch")]
    batch_pdf: Option<PathBuf>,
}

// ─── Resolved configuration (TOML + CLI merged) ───────────────────────────────

struct Config {
    input:                PathBuf,
    out_dir:              PathBuf,
    rest_ms:              u64,
    min_chars:            usize,
    max_words:            usize,
    merge:                bool,
    to_mp3:               bool,
    sentence_gap_ms:      u32,
    paragraph_gap_ms:     u32,
    ffmpeg_bin:           PathBuf,
    python_bin:           String,           // for pdf_rip.py (pdf_rip.py stays Python forever)
    rip_pdf:              Option<PathBuf>,
    rip_pdf_only:         bool,
    skip_refs:            bool,
    skip_captions:        bool,
    // ── Whisper validation ─────────────────────────────────────────────────────
    validate:             bool,
    whisper_model:        String,
    whisper_model_dir:    PathBuf,
    validate_threshold:   f32,
    // ── LLM text cleanup ───────────────────────────────────────────────────────
    llm_clean:            bool,
    crane_llm_bin:        Option<PathBuf>,
    crane_llm_model:      Option<PathBuf>,
    crane_llm_port:       u16,
    crane_llm_timeout:    u64,
    // ── TTS (Track B-revised) ─────────────────────────────────────────────────
    crane_tts_bin:        Option<PathBuf>,
    crane_tts_model:      Option<PathBuf>,
    crane_tts_port:       u16,
    crane_tts_timeout:    u64,
    crane_tts_voice:      String,
    crane_tts_language:   Option<String>,
    crane_tts_instruct:   Option<String>,
    // ── Run modes ──────────────────────────────────────────────────────────────
    merge_only:           bool,
    fix_quarantine:       bool,
    // ── Batch mode ─────────────────────────────────────────────────────────────
    batch_pdf:            Option<PathBuf>,  // Some(dir) = batch root; None = single-file mode
}

/// Merge TOML defaults + CLI overrides into a single resolved Config.
/// Priority (highest→lowest): CLI flag → TOML value → hard-coded default.
fn resolve_config(cli: CliArgs, toml: LecturnerConfig) -> Config {
    // Helper macros that read "cli field or toml field or literal default".
    macro_rules! pick {
        ($cli:expr, $toml:expr, $default:expr) => {
            $cli.unwrap_or_else(|| $toml.unwrap_or($default))
        };
    }
    macro_rules! pick_str {
        ($cli:expr, $toml:expr, $default:expr) => {
            $cli.unwrap_or_else(|| $toml.unwrap_or_else(|| $default.to_owned()))
        };
    }
    macro_rules! pick_path {
        ($cli:expr, $toml:expr, $default:expr) => {
            $cli.unwrap_or_else(|| PathBuf::from($toml.unwrap_or_else(|| $default.to_owned())))
        };
    }

    // Compute llm_clean default before the struct literal moves cli.rip_pdf
    // and toml.rip_pdf.  Default = true iff either side has a PDF path set.
    let llm_clean_default = toml.llm_clean.unwrap_or_else(||
        cli.rip_pdf.is_some() || toml.rip_pdf.is_some()
    );

    Config {
        input:       pick_path!(cli.input,   toml.input,   "text.txt"),
        out_dir:     pick_path!(cli.out_dir, toml.out_dir, "audio_out"),
        rest_ms:              pick!(cli.rest_ms,              toml.rest_ms,              500u64),
        min_chars:            pick!(cli.min_chars,            toml.min_chars,            10usize),
        max_words:            pick!(cli.max_words,            toml.max_words,            200usize),
        merge:                pick!(cli.merge,                toml.merge,                true),
        to_mp3:               pick!(cli.to_mp3,               toml.to_mp3,               false),
        sentence_gap_ms:      pick!(cli.sentence_gap_ms,      toml.sentence_gap_ms,      180u32),
        paragraph_gap_ms:     pick!(cli.paragraph_gap_ms,     toml.paragraph_gap_ms,     360u32),
        ffmpeg_bin: cli.ffmpeg_bin
            .unwrap_or_else(|| PathBuf::from(
                toml.ffmpeg_bin.unwrap_or_else(|| "ffmpeg".to_owned())
            )),
        python_bin:  toml.python_bin.unwrap_or_else(|| "python".to_owned()),
        rip_pdf:      cli.rip_pdf.or_else(|| toml.rip_pdf.map(PathBuf::from)),
        rip_pdf_only: cli.rip_pdf_only,
        skip_refs:    pick!(cli.skip_refs,    toml.skip_refs,    true),
        skip_captions: pick!(cli.skip_captions, toml.skip_captions, true),
        validate:           pick!(None::<bool>,  toml.validate,           false),
        whisper_model:      pick_str!(None,      toml.whisper_model,      "ggml-medium.en.bin"),
        whisper_model_dir:  pick_path!(None,     toml.whisper_model_dir,  "models"),
        validate_threshold: pick!(None::<f32>,   toml.validate_threshold, 0.15f32),
        // llm_clean defaults to true when rip_pdf is set, false otherwise.
        // Default computed before this struct literal (above) to avoid
        // use-after-move on cli.rip_pdf and toml.rip_pdf.
        llm_clean: llm_clean_default,
        crane_llm_bin:     toml.crane_llm_bin.map(PathBuf::from),
        crane_llm_model:   toml.crane_llm_model.map(PathBuf::from),
        crane_llm_port:    toml.crane_llm_port.unwrap_or(8101),
        crane_llm_timeout: toml.crane_llm_timeout.unwrap_or(60),
        crane_tts_bin:     toml.crane_tts_bin.map(PathBuf::from),
        crane_tts_model:   toml.crane_tts_model.map(PathBuf::from),
        crane_tts_port:    toml.crane_tts_port.unwrap_or(8102),
        crane_tts_timeout: toml.crane_tts_timeout.unwrap_or(60),
        crane_tts_voice:   toml.crane_tts_voice.unwrap_or_else(|| "Aiden".to_owned()),
        crane_tts_language: toml.crane_tts_language,
        crane_tts_instruct: toml.crane_tts_instruct,
        merge_only:     cli.merge_only,
        fix_quarantine: cli.fix_quarantine,
        batch_pdf:      cli.batch_pdf,
    }
}

// ─── Wire types for POST /v1/audio/speech (Crane + Qwen3-TTS) ────────────────
//
// Crane's TTS endpoint speaks OpenAI audio/speech shape but with Qwen3-specific
// fields.  Chatterbox knobs (exaggeration, cfg_weight, temperature) are gone;
// voice selection is by named speaker string.
//
// `language` and `instruct` are optional — crane ignores them if absent, so we
// skip_serializing_if None rather than sending empty strings that might confuse
// the model.

#[derive(Serialize, Debug)]
struct SpeechRequest<'a> {
    model:           &'a str,   // "qwen3-tts" — crane identifies the loaded model
    input:           &'a str,
    voice:           &'a str,   // named speaker: Uncle_Fu, Aiden, Ryan, Vivian, ...
    response_format: &'a str,   // "wav"
    #[serde(skip_serializing_if = "Option::is_none")]
    language:        Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instruct:        Option<&'a str>,
}

#[derive(Deserialize, Debug)]
struct ServerError {
    error: Option<serde_json::Value>,
}

// ─── Split boundary tagging ────────────────────────────────────────────────────

/// A synthesis unit: the text to voice plus the silence that follows it.
/// `Boundary` is now defined in records.rs and imported above.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub text:              String,
    pub trailing_boundary: Option<Boundary>,
}

// ─── Text splitting ────────────────────────────────────────────────────────────

/// Split raw text into tagged synthesis chunks.
///
/// Approach — two passes:
///   Pass 1: split on \n\n → coarse paragraphs; tag each trailing edge as
///           Boundary::Paragraph (last paragraph: None).
///   Pass 2: any paragraph over max_words gets sub-split at the nearest
///           sentence boundary; those interior edges become Boundary::Sentence.
///           The final sub-chunk of each paragraph inherits the paragraph's
///           original trailing tag (preserving the longer pause at the end).
pub fn harvest_chunks(raw: &str, min_chars: usize, max_words: usize) -> Vec<Chunk> {
    let normalised = raw.replace("\r\n", "\n");
    let coarse_texts: Vec<&str> = normalised
        .split("\n\n")
        .map(|s| s.trim())
        .filter(|s| s.chars().filter(|c| !c.is_whitespace()).count() >= min_chars)
        .collect();

    let coarse_count = coarse_texts.len();
    let mut result: Vec<Chunk> = Vec::new();

    for (para_idx, para_text) in coarse_texts.iter().enumerate() {
        let para_trailing = if para_idx + 1 < coarse_count {
            Some(Boundary::Paragraph)
        } else {
            None
        };

        if max_words == 0 {
            result.push(Chunk { text: para_text.to_string(), trailing_boundary: para_trailing });
            continue;
        }

        let sub_texts  = sub_split_by_words(para_text, max_words);
        let sub_count  = sub_texts.len();
        for (sub_idx, sub_text) in sub_texts.into_iter().enumerate() {
            let trailing = if sub_idx + 1 == sub_count {
                para_trailing
            } else {
                Some(Boundary::Sentence)
            };
            result.push(Chunk { text: sub_text, trailing_boundary: trailing });
        }
    }

    result
}

/// Recursively sub-split a single text block at sentence punctuation until
/// every piece is at or below max_words.  Falls back to a word boundary if no
/// sentence terminus is found within 150 bytes of the target split point.
fn sub_split_by_words(text: &str, max_words: usize) -> Vec<String> {
    let mut results   = Vec::new();
    let mut remainder = text.to_owned();

    loop {
        if remainder.split_whitespace().count() <= max_words {
            if !remainder.trim().is_empty() {
                results.push(remainder.trim().to_owned());
            }
            break;
        }

        let split_hint = byte_offset_after_nth_word(&remainder, max_words);
        let search_end = (split_hint + 150).min(remainder.len());
        let cut = find_sentence_boundary(&remainder, split_hint, search_end)
            .or_else(|| {
                remainder[..split_hint]
                    .rfind(|c: char| c.is_whitespace())
                    .map(|p| p + 1)
            })
            .unwrap_or(split_hint);

        results.push(remainder[..cut].trim().to_owned());
        remainder = remainder[cut..].trim().to_owned();
    }

    results
}

/// Return the byte offset immediately after the Nth whitespace-delimited word.
fn byte_offset_after_nth_word(text: &str, n: usize) -> usize {
    let mut words_seen = 0;
    let mut in_word    = false;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if in_word {
                words_seen += 1;
                if words_seen == n { return i; }
            }
            in_word = false;
        } else {
            in_word = true;
        }
    }
    text.len()
}

/// Scan forward from `start` to `end` for a sentence-terminal character
/// (. ! ?) followed by whitespace or end-of-string, and return the byte
/// offset just past it (so the split includes the punctuation).
fn find_sentence_boundary(text: &str, start: usize, end: usize) -> Option<usize> {
    let slice = &text[start..end];
    for (rel, c) in slice.char_indices() {
        if matches!(c, '.' | '!' | '?') {
            let after = start + rel + c.len_utf8();
            let next  = text[after..].chars().next();
            if next.map_or(true, |nc| nc.is_whitespace()) {
                return Some(after);
            }
        }
    }
    None
}

// ─── WAV header parsing ───────────────────────────────────────────────────────

/// Decoded fields from a WAV/RIFF header we need for merge and validation.
pub struct WavHeader {
    pub fmt_tag:        u16,  // 1 = integer PCM, 3 = float32 PCM
    pub channels:       u16,
    pub sample_rate:    u32,
    pub bits_per_sample: u16,
    pub data_offset:    usize, // byte offset to first PCM sample
    pub data_len:       u32,   // byte length of PCM payload
}

/// Parse the RIFF/WAV header of a byte slice.
///
/// Approach — linear scan of RIFF chunks:
///   1. Verify "RIFF" + "WAVE" magic.
///   2. Walk sub-chunks by their 4-byte tag + 4-byte little-endian size.
///   3. "fmt " chunk → decode format fields.
///   4. "data" chunk → record offset + length; stop scanning.
///   Supports both PCM (fmt_tag=1) and IEEE float (fmt_tag=3).
///   Extra fmt bytes (extensible WAV) are skipped cleanly.
pub fn parse_wav_header(data: &[u8]) -> Result<WavHeader> {
    anyhow::ensure!(data.len() >= 44, "Too short to be a WAV");
    anyhow::ensure!(&data[0..4] == b"RIFF", "Not a RIFF file");
    anyhow::ensure!(&data[8..12] == b"WAVE", "Not a WAVE file");

    let mut pos       = 12usize;
    let mut fmt_tag   = 0u16;
    let mut channels  = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data_offset = 0usize;
    let mut data_len  = 0u32;
    let mut found_fmt  = false;
    let mut found_data = false;

    while pos + 8 <= data.len() {
        let tag      = &data[pos..pos+4];
        let chunk_sz = u32::from_le_bytes(data[pos+4..pos+8].try_into().unwrap()) as usize;
        pos += 8;

        if tag == b"fmt " {
            anyhow::ensure!(chunk_sz >= 16, "fmt chunk too small");
            fmt_tag         = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap());
            channels        = u16::from_le_bytes(data[pos+2..pos+4].try_into().unwrap());
            sample_rate     = u32::from_le_bytes(data[pos+4..pos+8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(data[pos+14..pos+16].try_into().unwrap());
            anyhow::ensure!(
                fmt_tag == 1 || fmt_tag == 3,
                "Only PCM WAV supported (fmt_tag={fmt_tag})"
            );
            found_fmt = true;
        } else if tag == b"data" {
            data_offset = pos;
            data_len    = chunk_sz as u32;
            found_data  = true;
            break;
        }

        pos += chunk_sz + (chunk_sz % 2); // RIFF chunks are word-aligned
    }

    anyhow::ensure!(found_fmt,  "No fmt  chunk found in WAV");
    anyhow::ensure!(found_data, "No data chunk found in WAV");
    Ok(WavHeader { fmt_tag, channels, sample_rate, bits_per_sample, data_offset, data_len })
}

/// Write a minimal 44-byte WAV header into `out`.
/// fmt_tag: 1 = integer PCM, 3 = IEEE float32 PCM.
/// The header layout is identical for both; only the format field differs.
fn write_wav_header(out: &mut Vec<u8>, fmt_tag: u16, ch: u16, sr: u32, bps: u16, pcm_len: u32) {
    let byte_rate   = sr * ch as u32 * bps as u32 / 8;
    let block_align = ch * bps / 8;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + pcm_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&fmt_tag.to_le_bytes());
    out.extend_from_slice(&ch.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bps.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&pcm_len.to_le_bytes());
}

/// Generate zero-filled PCM silence for the requested duration.
///
/// Approach: PCM silence = all-zero bytes.
///   total_bytes = floor(sample_rate × duration_ms / 1000) × bytes_per_frame
///   aligned to frame boundary to prevent join artefacts.
fn silence_bytes(sr: u32, ch: u16, bps: u16, ms: u32) -> Vec<u8> {
    let bytes_per_frame = ch as u32 * (bps as u32 / 8);
    let frames          = (sr * ms) / 1000;
    vec![0u8; (frames * bytes_per_frame) as usize]
}

/// Merge WAV files with semantically-sized silence gaps between them.
///
/// Approach:
///   1. Parse all headers; assert uniform sample-rate/channels/depth.
///   2. Pre-compute silence pads for Sentence and Paragraph boundaries.
///   3. Pre-calculate total PCM length to write a correct header upfront.
///   4. Stream each chunk's PCM then its trailing silence into one buffer.
pub fn merge_wavs(
    wav_paths:       &[PathBuf],
    boundaries:      &[Option<Boundary>],
    output_path:     &PathBuf,
    sentence_gap_ms: u32,
    paragraph_gap_ms: u32,
) -> Result<()> {
    anyhow::ensure!(!wav_paths.is_empty(), "Nothing to merge");
    anyhow::ensure!(wav_paths.len() == boundaries.len(), "Lengths must match");

    let loaded: Vec<(WavHeader, Vec<u8>)> = wav_paths.iter()
        .map(|p| {
            let b = fs::read(p).with_context(|| format!("Cannot read {}", p.display()))?;
            let h = parse_wav_header(&b).with_context(|| format!("Bad WAV: {}", p.display()))?;
            Ok((h, b))
        })
        .collect::<Result<_>>()?;

    let ref_hdr = &loaded[0].0;
    for (i, (h, _)) in loaded.iter().enumerate().skip(1) {
        anyhow::ensure!(
            h.channels == ref_hdr.channels
                && h.sample_rate == ref_hdr.sample_rate
                && h.bits_per_sample == ref_hdr.bits_per_sample,
            "File {} format mismatch: {}ch {}Hz {}bit vs {}ch {}Hz {}bit",
            i+1, h.channels, h.sample_rate, h.bits_per_sample,
            ref_hdr.channels, ref_hdr.sample_rate, ref_hdr.bits_per_sample
        );
    }

    let sentence_pad  = silence_bytes(ref_hdr.sample_rate, ref_hdr.channels, ref_hdr.bits_per_sample, sentence_gap_ms);
    let paragraph_pad = silence_bytes(ref_hdr.sample_rate, ref_hdr.channels, ref_hdr.bits_per_sample, paragraph_gap_ms);

    let total_pcm: u32 = loaded.iter().zip(boundaries.iter())
        .map(|((h, _), b)| {
            let pad = match b {
                Some(Boundary::Sentence)  => sentence_pad.len() as u32,
                Some(Boundary::Paragraph) => paragraph_pad.len() as u32,
                None                      => 0,
            };
            h.data_len.saturating_add(pad)
        })
        .fold(0u32, |acc, n| acc.saturating_add(n));

    let mut out: Vec<u8> = Vec::with_capacity(44 + total_pcm as usize);
    write_wav_header(&mut out, ref_hdr.fmt_tag, ref_hdr.channels, ref_hdr.sample_rate, ref_hdr.bits_per_sample, total_pcm);

    for ((h, bytes), boundary) in loaded.iter().zip(boundaries.iter()) {
        out.extend_from_slice(&bytes[h.data_offset..h.data_offset + h.data_len as usize]);
        match boundary {
            Some(Boundary::Sentence)  => out.extend_from_slice(&sentence_pad),
            Some(Boundary::Paragraph) => out.extend_from_slice(&paragraph_pad),
            None                      => {}
        }
    }

    fs::write(output_path, &out)
        .with_context(|| format!("Cannot write: {}", output_path.display()))?;

    println!(
        "[lecturner] Merged {} file(s) → {} ({:.1} MB)",
        loaded.len(), output_path.display(),
        out.len() as f64 / 1_048_576.0,
    );
    Ok(())
}

// ─── ffmpeg MP3 export ────────────────────────────────────────────────────────

/// Transcode `wav_path` → sibling `.mp3` using ffmpeg.
///
/// Approach: shell out to `ffmpeg -y -i <wav> -codec:a libmp3lame -q:a 2 <mp3>`
///   -y          overwrite without prompting
///   -q:a 2      variable bitrate quality 2 (~190 kbps) — fine for speech
/// The MP3 file lands next to the WAV with the same stem.
fn wav_to_mp3(ffmpeg_bin: &PathBuf, wav_path: &PathBuf) -> Result<PathBuf> {
    let mp3_path = wav_path.with_extension("mp3");
    let status = Command::new(ffmpeg_bin)
        .args([
            "-y", "-i",
            wav_path.to_str().context("Non-UTF8 WAV path")?,
            "-codec:a", "libmp3lame",
            "-q:a", "2",
            mp3_path.to_str().context("Non-UTF8 MP3 path")?,
        ])
        .status()
        .with_context(|| format!(
            "Failed to launch ffmpeg ({}).  Is it on PATH or set via --ffmpeg-bin?",
            ffmpeg_bin.display()
        ))?;
    anyhow::ensure!(status.success(), "ffmpeg exited with {status}");
    Ok(mp3_path)
}

// ─── Per-chunk synthesis ──────────────────────────────────────────────────────

/// Ensure the chunk ends with sentence-terminal punctuation.
///
/// Qwen3-TTS uses end-of-sentence cues to know when to stop generating.
/// Without one it idles for several seconds of silence before the EOS token
/// fires.  Appending a period when one is absent costs nothing audible and
/// reliably kills the dead-air tail.
fn ensure_terminal_punctuation(text: &str) -> std::borrow::Cow<'_, str> {
    let trimmed = text.trim_end();
    if trimmed.ends_with(['.', '!', '?', '\u{2026}', '"', '\'']) {
        std::borrow::Cow::Borrowed(text)
    } else {
        std::borrow::Cow::Owned(format!("{trimmed}."))
    }
}

fn synthesise_chunk(client: &Client, cfg: &Config, text: &str) -> Result<Vec<u8>> {
    let url        = format!("{}/v1/audio/speech", crane_tts_url(cfg));
    let punctuated = ensure_terminal_punctuation(text);
    let body = SpeechRequest {
        model:           "qwen3-tts",
        input:           &punctuated,
        voice:           &cfg.crane_tts_voice,
        response_format: "wav",
        language:        cfg.crane_tts_language.as_deref(),
        instruct:        cfg.crane_tts_instruct.as_deref(),
    };

    let response = client.post(&url)
        .json(&body)
        .send()
        .with_context(|| format!("HTTP POST to {url} failed — is the server up?"))?;

    if !response.status().is_success() {
        let status = response.status();
        let raw    = response.text().unwrap_or_default();
        if let Ok(e) = serde_json::from_str::<ServerError>(&raw) {
            anyhow::bail!("Server returned {status}: {:?}", e.error);
        }
        anyhow::bail!("Server returned {status}: {raw}");
    }

    Ok(response.bytes().context("Failed to read WAV bytes")?.to_vec())
}

// ─── Whisper validation ────────────────────────────────────────────────────────

// CMU Pronouncing Dictionary bundled at compile time.
// Drop `cmudict.dict` (public domain, from NLTK or cmudict.sourceforge.net)
// into the project root before building.  `cargo build` will fail with a clear
// error if the file is absent — preferable to silent degradation.
//
// Format: one entry per line — WORD  P1 P2 P3 ...
//   HELLO  HH AH0 L OW1
// Variant numbering (HELLO(2)) is stripped on load; first variant wins.
static CMUDICT_RAW: &str = include_str!("../cmudict.dict");

/// Load the Whisper model once at startup; keep the context alive for all chunks.
///
/// Approach:
///   1. Verify model file exists on disk — emit a download URL if not.
///   2. Call WhisperContext::new_with_params; whisper-rs picks the CUDA backend
///      automatically when the cuda feature is enabled and a GPU is visible.
pub fn load_whisper_model(model_path: &std::path::Path) -> Result<WhisperContext> {
    anyhow::ensure!(
        model_path.exists(),
        "Whisper model not found: {}.\n\
         Download: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin\n\
         Place it in the models/ directory (or set whisper_model_dir in lecturner.toml).",
        model_path.display()
    );
    let params = WhisperContextParameters::default();
    WhisperContext::new_with_params(
        model_path.to_str().context("Non-UTF8 model path")?,
        params,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load whisper model: {e:?}"))
}

/// Transcribe a WAV byte blob to text using the loaded Whisper context.
///
/// Approach:
///   1. Parse the WAV header to locate the PCM data chunk and read format fields.
///   2. Decode samples to f32:
///        fmt_tag=1 (int16 PCM)  → cast i16 → f32, scale by 1/32768
///        fmt_tag=3 (float32 PCM) → reinterpret bytes directly
///   3. Resample from hdr.sample_rate → 16 000 Hz via linear interpolation.
///      The source rate is read from the header rather than assumed — Qwen3-TTS
///      outputs at 12 000 Hz, Chatterbox at 24 000 Hz; both are handled.
///   4. Run whisper-rs full inference with greedy sampling.
///   5. Concatenate segment texts and return.
pub fn transcribe_wav(state: &mut WhisperState, wav_bytes: &[u8]) -> Result<String> {
    const WHISPER_HZ: u32 = 16_000;

    let hdr  = parse_wav_header(wav_bytes)?;
    let data = &wav_bytes[hdr.data_offset..hdr.data_offset + hdr.data_len as usize];

    // Decode PCM to f32 — handle both int16 and float32 WAVs.
    let src_samples: Vec<f32> = match hdr.fmt_tag {
        1 => {
            // Integer PCM: int16 little-endian, scale to [-1.0, 1.0].
            anyhow::ensure!(data.len() % 2 == 0, "int16 WAV data length not a multiple of 2");
            data.chunks_exact(2)
                .map(|b| i16::from_le_bytes(b.try_into().unwrap()) as f32 / 32768.0_f32)
                .collect()
        }
        3 => {
            // Float32 PCM: already in [-1.0, 1.0].
            anyhow::ensure!(data.len() % 4 == 0, "f32 WAV data length not a multiple of 4");
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        tag => anyhow::bail!("Unsupported WAV fmt_tag={tag}; expected 1 (int16) or 3 (f32)"),
    };

    // Resample src_rate → 16 000 Hz via linear interpolation.
    // ratio = src_rate / WHISPER_HZ; output index i maps to source position i * ratio.
    // Works for downsample (24kHz→16kHz, ratio=1.5) and upsample (12kHz→16kHz, ratio=0.75).
    let src_rate = hdr.sample_rate;
    let resampled: Vec<f32> = if src_rate == WHISPER_HZ {
        src_samples
    } else {
        let ratio   = src_rate as f32 / WHISPER_HZ as f32;
        let dst_len = (src_samples.len() as f32 / ratio).floor() as usize;
        let mut out = Vec::with_capacity(dst_len);
        for i in 0..dst_len {
            let t    = i as f32 * ratio;
            let lo   = t.floor() as usize;
            let hi   = (lo + 1).min(src_samples.len() - 1);
            let frac = t - lo as f32;
            out.push(src_samples[lo] + frac * (src_samples[hi] - src_samples[lo]));
        }
        out
    };

    // State is reused across calls — GPU buffers stay allocated, no re-init per chunk.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    state.full(params, &resampled)
        .map_err(|e| anyhow::anyhow!("Whisper inference: {e:?}"))?;

    // full_n_segments() returns c_int directly (no Result) in whisper-rs 0.16.
    // as_iter() is the idiomatic 0.16 path: yields WhisperSegment<'_> with
    // Display impl for text.
    let pieces: Vec<String> = state.as_iter()
        .map(|seg| seg.to_string().trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();
    Ok(pieces.join(" "))
}

/// Normalize text for phoneme comparison: to lowercase, strip punctuation,
/// collapse whitespace.  Also folds Unicode punctuation to ASCII first.
pub fn normalize_for_comparison(text: &str) -> String {
    let ascii_ish = text
        .replace('\u{2018}', "'").replace('\u{2019}', "'")
        .replace('\u{201C}', "\"").replace('\u{201D}', "\"")
        .replace('\u{2014}', " ").replace('\u{2013}', " ");
    ascii_ish
        .chars()
        .filter(|c| c.is_alphabetic() || c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Lazy-initialized CMU dict.  Parsed once, reused for all chunks.
fn cmudict() -> &'static std::collections::HashMap<String, Vec<String>> {
    use std::sync::OnceLock;
    static DICT: OnceLock<std::collections::HashMap<String, Vec<String>>> = OnceLock::new();
    DICT.get_or_init(|| {
        let mut map = std::collections::HashMap::new();
        for line in CMUDICT_RAW.lines() {
            if line.starts_with(";;;") || line.is_empty() { continue; }
            let mut parts = line.splitn(2, "  ");
            if let (Some(word_raw), Some(phones_raw)) = (parts.next(), parts.next()) {
                let word = word_raw.split('(').next().unwrap_or(word_raw).to_lowercase();
                map.entry(word).or_insert_with(|| {
                    phones_raw
                        .split_whitespace()
                        .map(|p| p.trim_end_matches(|c: char| c.is_ascii_digit()).to_owned())
                        .collect()
                });
            }
        }
        map
    })
}

/// Map a slice of normalized words to a flat phoneme token string.
///
/// Approach:
///   Known words → CMU phonemes joined by spaces, then "|" as word boundary.
///   OOV words   → each character uppercased, same "|" boundary.
///   Using character-level tokens for OOV keeps the edit distance metric
///   uniform across both paths (consistent token space).
///
///   Example: "hello world" → "HH AH L OW | W ER L D |"
pub fn words_to_phonemes(words: &[&str]) -> String {
    let dict = cmudict();
    let mut out = String::new();
    for word in words {
        if let Some(phones) = dict.get(*word) {
            out.push_str(&phones.join(" "));
        } else {
            let spelled: Vec<String> = word.chars()
                .map(|c| c.to_uppercase().to_string())
                .collect();
            out.push_str(&spelled.join(" "));
        }
        out.push_str(" | ");
    }
    out
}

/// Compute normalized phoneme edit distance in [0.0, 1.0].
///
/// Approach — Wagner-Fischer Levenshtein on whitespace-split phoneme tokens,
/// normalized by max(orig_len, xscr_len).
///   0.0 = phonemically identical
///   1.0 = completely different (or one side empty)
///
/// The "|" word-boundary markers participate in the distance so that
/// word-count mismatches register even when stray phonemes accidentally align.
pub fn phoneme_error_rate(original: &str, transcription: &str) -> f32 {
    let orig_tok: Vec<&str> = original.split_whitespace().collect();
    let xscr_tok: Vec<&str> = transcription.split_whitespace().collect();
    let m = orig_tok.len();
    let n = xscr_tok.len();
    if m == 0 && n == 0 { return 0.0; }
    if m == 0 || n == 0 { return 1.0; }

    // Wagner-Fischer with O(n) space (two rolling rows)
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if orig_tok[i-1] == xscr_tok[j-1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j-1] + 1).min(prev[j-1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n] as f32 / m.max(n) as f32
}

/// Sanitize a synthesis chunk: fold Unicode punctuation → ASCII, strip
/// non-printable non-ASCII characters, collapse whitespace.
///
/// Returns (sanitized_text, was_changed).
pub fn sanitize_chunk(text: &str) -> (String, bool) {
    let folded = text
        .replace('\u{2018}', "'").replace('\u{2019}', "'")
        .replace('\u{201C}', "\"").replace('\u{201D}', "\"")
        .replace('\u{2014}', " — ").replace('\u{2013}', " - ")
        .replace('\u{2026}', "...");
    let stripped: String = folded
        .chars()
        .filter(|c| c.is_ascii_graphic() || *c == ' ' || *c == '\n' || *c == '\t')
        .collect();
    let clean: String = stripped.split_whitespace().collect::<Vec<_>>().join(" ");
    let was_changed = clean != text.split_whitespace().collect::<Vec<_>>().join(" ");
    (clean, was_changed)
}

/// Outcome of a fully-exhausted retry sequence — enough detail to write a
/// useful quarantine sidecar and for the caller to decide next steps.
pub struct ValidationFailure {
    pub original_text: String,
    pub transcription: String,
    pub phoneme_error: f32,
    pub attempts:      u32,
}

impl std::fmt::Display for ValidationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PER={:.3} after {} attempt(s)\n  original:      {}\n  transcription: {}",
            self.phoneme_error, self.attempts,
            &self.original_text.chars().take(120).collect::<String>(),
            &self.transcription.chars().take(120).collect::<String>(),
        )
    }
}

/// Validate a synthesized WAV against its source text and retry once on failure.
///
/// Approach — two-attempt pipeline:
///   Attempt 0 (free):    validate the WAV already received from Crane TTS.
///   Attempt 1 (sanitize): fold Unicode punctuation → ASCII and re-synthesize.
///                         Qwen3-TTS has no temperature knob; sanitized text
///                         is the only retry lever available.  If the text was
///                         already clean, the model's own sampling produces a
///                         different take on the same input.
///   Quarantine:          if both attempts exceed the threshold, write a .bad.txt
///                        sidecar and return Err(ValidationFailure).
///
/// Takes `original_text: &str` directly rather than `&Chunk` so this function
/// is reusable from both the normal synthesis loop and --fix-quarantine.
fn validate_and_retry(
    client:        &Client,
    wstate:        &mut WhisperState,
    cfg:           &Config,
    original_text: &str,
    first_wav:     Vec<u8>,
    bad_path:      &std::path::Path,
) -> Result<Vec<u8>, ValidationFailure> {
    let (sanitized_text, text_was_dirty) = sanitize_chunk(original_text);
    if text_was_dirty {
        eprintln!("  [validate] Input had Unicode punctuation — sanitized copy ready for retry.");
    }

    let mut attempts    = 0u32;
    let mut current_wav = first_wav;

    loop {
        attempts += 1;

        let transcription = transcribe_wav(wstate, &current_wav).unwrap_or_default();

        // Write transcription sidecar every attempt so we can inspect what
        // Whisper actually heard regardless of pass/fail outcome.
        // Path: same stem as bad_path with .transcription.txt extension.
        let xscr_path = bad_path.with_extension("transcription.txt");
        let _ = std::fs::write(&xscr_path, &transcription);

        let orig_norm  = normalize_for_comparison(original_text);
        let xscr_norm  = normalize_for_comparison(&transcription);
        let orig_words: Vec<&str> = orig_norm.split_whitespace().collect();
        let xscr_words: Vec<&str> = xscr_norm.split_whitespace().collect();
        let per = phoneme_error_rate(
            &words_to_phonemes(&orig_words),
            &words_to_phonemes(&xscr_words),
        );
        eprintln!(
            "  [validate] attempt={attempts}  PER={per:.3}  threshold={:.3}",
            cfg.validate_threshold
        );

        if per <= cfg.validate_threshold {
            return Ok(current_wav);
        }

        if attempts >= 2 {
            let sidecar = format!(
                "ORIGINAL:\n{}\n\nTRANSCRIPTION:\n{}\n\nPHONEME_ERROR: {:.4}\nATTEMPTS: {}\n",
                original_text, transcription, per, attempts,
            );
            if let Err(e) = std::fs::write(bad_path, &sidecar) {
                eprintln!("  [validate] Could not write quarantine file: {e}");
            } else {
                eprintln!("  [validate] Quarantined → {}", bad_path.display());
            }
            return Err(ValidationFailure {
                original_text: original_text.to_owned(),
                transcription,
                phoneme_error: per,
                attempts,
            });
        }

        // One retry: use sanitized text if the original had Unicode punctuation.
        // Qwen3-TTS has no temperature knob — the only lever we have is the
        // sanitized text path.  If the text was clean, we re-send as-is;
        // the model's own sampling will produce a different take.
        let retry_text = if text_was_dirty { &sanitized_text } else { original_text };
        eprintln!(
            "  [validate] Retrying  text={}",
            if text_was_dirty { "sanitized" } else { "original" }
        );

        let url        = format!("{}/v1/audio/speech", crane_tts_url(cfg));
        let retry_body = SpeechRequest {
            model:           "qwen3-tts",
            input:           retry_text,
            voice:           &cfg.crane_tts_voice,
            response_format: "wav",
            language:        cfg.crane_tts_language.as_deref(),
            instruct:        cfg.crane_tts_instruct.as_deref(),
        };
        match client.post(&url).json(&retry_body).send().and_then(|r| r.bytes()) {
            Ok(b)  => current_wav = b.to_vec(),
            Err(e) => {
                eprintln!("  [validate] Retry HTTP error: {e:#}");
                current_wav = Vec::new();
            }
        }
    }
}

// ─── PDF ripping ──────────────────────────────────────────────────────────────

fn locate_pdf_rip_script(explicit: &Option<PathBuf>) -> Option<PathBuf> {
    if let Some(p) = explicit { return Some(p.clone()); }
    let name = "pdf_rip.py";
    if let Ok(mut self_path) = std::env::current_exe() {
        self_path.pop();
        let candidate = self_path.join(name);
        if candidate.exists() { return Some(candidate); }
    }
    let cwd_candidate = PathBuf::from(name);
    if cwd_candidate.exists() { return Some(cwd_candidate); }
    None
}

/// Run pdf_rip.py to extract prose from a PDF into the configured input text file.
///
/// Approach:
///   1. Verify pdf_rip.py is locatable.
///   2. Spawn Python with the script; capture stdout for progress lines.
///   3. Detect exit code 1 = missing pdfplumber dep → print friendly install hint.
///   4. Detect exit code 2 = extraction error → surface stderr.
///   5. On success, lecturner's normal input file now contains the extracted prose.
fn rip_pdf(cfg: &Config, pdf_path: &PathBuf) -> Result<()> {
    let script = locate_pdf_rip_script(&None)
        .context("Cannot find pdf_rip.py.  Put it next to lecturner.exe or in the working directory.")?;

    let out_path = cfg.input.to_str().context("Non-UTF8 output path")?;
    let pdf_str  = pdf_path.to_str().context("Non-UTF8 PDF path")?;

    println!("[lecturner] Ripping PDF: {} → {}", pdf_path.display(), cfg.input.display());

    let mut cmd = Command::new(&cfg.python_bin);
    cmd.arg(&script)
       .arg("--input").arg(pdf_str)
       .arg("--output").arg(out_path);

    if !cfg.skip_refs     { cmd.arg("--no-skip-refs"); }
    if !cfg.skip_captions { cmd.arg("--no-skip-captions"); }

    let output = cmd.output()
        .with_context(|| format!("Failed to spawn '{}'", cfg.python_bin))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() { println!("{line}"); }

    match output.status.code() {
        Some(0) => Ok(()),
        Some(1) => {
            if stdout.contains("GOSSIP_MISSING_DEP:pdfplumber") {
                anyhow::bail!(
                    "pdfplumber is not installed.  Fix:\n  {} -m pip install pdfplumber",
                    cfg.python_bin
                );
            }
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("pdf_rip.py failed:\n{stderr}");
        }
        Some(2) | _ => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("pdf_rip.py extraction error:\n{stderr}");
        }
    }
}

// ─── LLM text cleanup (Track A-revised) ──────────────────────────────────────
//
// Crane-oai exposes POST /v1/chat/completions (OpenAI-compat shape).
// We run Qwen2.5-Instruct as a rewrite pass on each paragraph before chunking.
// The same reqwest blocking client used for TTS handles this call.

/// Request body for POST /v1/chat/completions.
/// We only need the fields crane-oai actually reads; the rest are optional.
#[derive(Serialize)]
struct ChatRequest<'a> {
    model:       &'a str,
    messages:    &'a [ChatMessage<'a>],
    max_tokens:  u32,
    temperature: f32,
    stream:      bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role:    &'a str,
    content: &'a str,
}

/// Response shape — we only destructure down to the text we need.
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

/// System prompt for the paragraph rewrite pass.
///
/// /no_think at the top disables Qwen3 chain-of-thought mode — we want
/// fast deterministic rewrites, not reasoned essays.
/// DROP return value lets the model cleanly discard non-prose artifacts
/// without silently corrupting the paragraph stream.
const LLM_CLEAN_SYSTEM: &str = "\
/no_think
You are a prose-cleanup assistant preparing academic or technical text for \
text-to-speech narration. \
You will be given one paragraph of text extracted from a PDF. \
Some paragraphs may contain extraction artifacts from imperfect PDF parsing. \
Rewrite the paragraph so it reads naturally when spoken aloud. \
Follow these rules exactly:
- If the paragraph is clearly not prose — an image credit line, a URL, a lone \
  figure number, a table of contents entry, a page number, a section heading, \
  or an isolated proper noun with no surrounding sentence — return the single \
  word DROP and nothing else.
- If the paragraph reads as two or more unrelated sentence fragments \
  interleaved mid-sentence (column extraction artifact), reconstruct the most \
  coherent single reading you can from the available words. If no coherent \
  reading is possible, return DROP.
- Remove image and figure credit lines in any form: \
  \"Credit: ...\", \"Image courtesy of ...\", \"Reproduced with permission\", \
  \"| Full Image\", \"Full Image\", and similar attribution fragments embedded \
  anywhere in the paragraph.
- Replace figure and table references with natural language: \
  \"As shown in Figure 3\" → \"As shown in the figure\"; \
  \"(see Table 2)\" → remove the parenthetical entirely.
- Replace equation references: \"from Equation 7\" → \"from the equation above\".
- Replace section cross-references: \"see Section 4.2\" → \"as discussed earlier\".
- Expand citation brackets: \"Smith et al. [14]\" → \"Smith and colleagues\"; \
  \"[3,7,14]\" → remove entirely.
- Spell out units where natural: \"3.2 km/s\" → \"3.2 kilometers per second\"; \
  \"ΔV\" → \"delta-V\".
- Fix soft-hyphenation artifacts from line breaks: \"perturba-tion\" → \"perturbation\".
- Do NOT summarize, shorten, or add any commentary.
- Do NOT change technical facts, numerical values, or sentence meaning.
- Do NOT add phrases like \"Here is the rewritten paragraph:\".
- Return ONLY the rewritten paragraph text, or the single word DROP. Nothing else.";

/// Crane-oai URL for the LLM cleanup server.
fn crane_llm_url(cfg: &Config) -> String {
    format!("http://127.0.0.1:{}", cfg.crane_llm_port)
}

/// Crane-oai URL for the TTS server (separate port from LLM crane).
fn crane_tts_url(cfg: &Config) -> String {
    format!("http://127.0.0.1:{}", cfg.crane_tts_port)
}

/// Locate the crane-oai binary: explicit config path first, then PATH.
fn locate_crane_bin(explicit: &Option<PathBuf>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        if p.exists() { return Some(p.clone()); }
    }
    // Try common binary names on PATH.
    for name in &["crane-oai", "crane-oai.exe"] {
        if let Ok(output) = Command::new("where").arg(name).output()
            .or_else(|_| Command::new("which").arg(name).output())
        {
            if output.status.success() {
                let path_str = String::from_utf8_lossy(&output.stdout);
                let first_line = path_str.lines().next().unwrap_or("").trim();
                if !first_line.is_empty() {
                    return Some(PathBuf::from(first_line));
                }
            }
        }
    }
    None
}

/// Launch crane-oai as the LLM cleanup server if not already responding.
/// Returns a child handle so main() can kill it on exit.
///
/// Approach:
///   1. Check /health on crane_llm_url — if alive, reuse (user may have
///      pre-launched it with a larger model or custom flags).
///   2. Otherwise locate crane-oai binary and crane_llm_model path.
///   3. Spawn: crane-oai --model-path <model> --model-type qwen25 --port <port>
///   4. Poll /health until ready or timeout.
fn ensure_crane_llm_running(cfg: &Config, client: &Client) -> Result<Option<Child>> {
    let base_url = crane_llm_url(cfg);
    let health   = format!("{}/health", base_url);

    // Already up — maybe the user launched it manually or it's from a prior run.
    if matches!(client.get(&health).timeout(Duration::from_secs(2)).send(), Ok(_)) {
        println!("[lecturner] Crane LLM server already running at {}", base_url);
        return Ok(None);
    }

    let bin = locate_crane_bin(&cfg.crane_llm_bin).context(
        "Cannot find crane-oai binary.  Set crane_llm_bin in lecturner.toml \
         or place crane-oai on PATH."
    )?;

    let model_path = cfg.crane_llm_model.as_ref().context(
        "crane_llm_model not set in lecturner.toml.  \
         Point it at your Qwen2.5-Instruct checkpoint directory."
    )?;

    anyhow::ensure!(
        model_path.exists(),
        "crane_llm_model path does not exist: {}",
        model_path.display()
    );

    println!(
        "[lecturner] Launching Crane LLM: {} --model-path {} --model-type qwen3 --port {}",
        bin.display(), model_path.display(), cfg.crane_llm_port
    );

    let child = Command::new(&bin)
        .arg("--model-path").arg(model_path)
        .arg("--model-type").arg("qwen3")
        .arg("--port").arg(cfg.crane_llm_port.to_string())
        .spawn()
        .with_context(|| format!("Failed to spawn crane-oai ({})", bin.display()))?;

    // Poll until ready.  Qwen3-4B loads in roughly 15-30s on an RTX 4080.
    print!("[lecturner] Waiting for Crane LLM to load model");
    use std::io::Write;
    let poll_steps = cfg.crane_llm_timeout / 2;
    for _ in 0..poll_steps {
        if matches!(client.get(&health).timeout(Duration::from_secs(2)).send(), Ok(_)) {
            println!(" ready!");
            return Ok(Some(child));
        }
        print!(".");
        let _ = std::io::stdout().flush();
        thread::sleep(Duration::from_secs(2));
    }
    println!();
    anyhow::bail!(
        "Crane LLM server did not become ready within {}s.  \
         Check that crane-oai can load the model at '{}'.",
        cfg.crane_llm_timeout,
        model_path.display()
    );
}

/// Launch crane-oai as the TTS server if not already responding.
/// Returns a child handle so main() can kill it on exit.
///
/// Approach — mirrors ensure_crane_llm_running but targets the TTS checkpoint
/// and uses crane_tts_port / crane_tts_timeout.  Runs on the same binary
/// (crane-oai) with --model-type qwen3-tts.
///   1. Health-check crane_tts_url — reuse if already alive.
///   2. Locate binary (crane_tts_bin config → PATH fallback).
///   3. Spawn: crane-oai --model-path <tts_model> --model-type qwen3-tts --port <tts_port>
///   4. Poll /health until ready or crane_tts_timeout expires.
fn ensure_crane_tts_running(cfg: &Config, client: &Client) -> Result<Option<Child>> {
    let base_url       = crane_tts_url(cfg);
    let uncle_fu_check = format!("{}/health", base_url); // polling the voice butler

    if matches!(client.get(&uncle_fu_check).timeout(Duration::from_secs(2)).send(), Ok(_)) {
        println!("[lecturner] Crane TTS server already running at {}", base_url);
        return Ok(None);
    }

    // TTS bin may differ from LLM bin if the user has separate deployments;
    // fall back to the LLM bin path before checking PATH.
    let bin = locate_crane_bin(&cfg.crane_tts_bin)
        .or_else(|| locate_crane_bin(&cfg.crane_llm_bin))
        .context(
            "Cannot find crane-oai binary for TTS.  Set crane_tts_bin in lecturner.toml \
             or place crane-oai on PATH."
        )?;

    let model_path = cfg.crane_tts_model.as_ref().context(
        "crane_tts_model not set in lecturner.toml.  \
         Point it at your Qwen3-TTS checkpoint directory."
    )?;

    anyhow::ensure!(
        model_path.exists(),
        "crane_tts_model path does not exist: {}",
        model_path.display()
    );

    println!(
        "[lecturner] Launching Crane TTS: {} --model-path {} --model-type qwen3-tts --port {}",
        bin.display(), model_path.display(), cfg.crane_tts_port
    );

    let child = Command::new(&bin)
        .arg("--model-path").arg(model_path)
        .arg("--model-type").arg("qwen3-tts")
        .arg("--port").arg(cfg.crane_tts_port.to_string())
        .spawn()
        .with_context(|| format!("Failed to spawn crane-oai TTS ({})", bin.display()))?;

    // Qwen3-TTS-1.7B loads faster than the 4B LLM, but give it the same
    // timeout headroom in case of VRAM pressure from other processes.
    print!("[lecturner] Waiting for Crane TTS to load model");
    use std::io::Write;
    let poll_steps = cfg.crane_tts_timeout / 2;
    for _ in 0..poll_steps {
        if matches!(client.get(&uncle_fu_check).timeout(Duration::from_secs(2)).send(), Ok(_)) {
            println!(" ready!");
            return Ok(Some(child));
        }
        print!(".");
        let _ = std::io::stdout().flush();
        thread::sleep(Duration::from_secs(2));
    }
    println!();
    anyhow::bail!(
        "Crane TTS server did not become ready within {}s.  \
         Check that crane-oai can load the model at '{}'.",
        cfg.crane_tts_timeout,
        model_path.display()
    );
}
///
/// Approach:
///   POST /v1/chat/completions with system prompt + paragraph as user content.
///   If the call fails or returns empty text, log a warning and return the
///   original paragraph unchanged — LLM failure must never silently eat content.
fn clean_paragraph(
    paragraph: &str,
    cfg:       &Config,
    client:    &Client,
) -> String {
    let url = format!("{}/v1/chat/completions", crane_llm_url(cfg));

    // Model name doesn't matter to crane-oai — it serves whatever was loaded.
    // We send a non-empty string so the JSON is valid.
    let messages = [
        ChatMessage { role: "system", content: LLM_CLEAN_SYSTEM },
        ChatMessage { role: "user",   content: paragraph },
    ];
    let body = ChatRequest {
        model:       "qwen25",
        messages:    &messages,
        max_tokens:  2048,
        temperature: 0.2,   // low temperature = deterministic rewrites, not creative leaps
        stream:      false,
    };

    let response = match client.post(&url).json(&body).send() {
        Ok(r)  => r,
        Err(e) => {
            eprintln!("  [llm_clean] HTTP error, keeping original: {e:#}");
            return paragraph.to_owned();
        }
    };

    if !response.status().is_success() {
        eprintln!("  [llm_clean] Server returned {}, keeping original", response.status());
        return paragraph.to_owned();
    }

    let chat_resp: ChatResponse = match response.json() {
        Ok(r)  => r,
        Err(e) => {
            eprintln!("  [llm_clean] Bad JSON response, keeping original: {e:#}");
            return paragraph.to_owned();
        }
    };

    let raw = chat_resp.choices
        .into_iter()
        .next()
        .map(|c| c.message.content.trim().to_owned())
        .unwrap_or_default();

    // Strip Qwen3 think block — /no_think produces an empty one but it must
    // not leak into the cleaned paragraph text.  Also handles the edge case
    // where the model ignores /no_think and actually reasons; we keep the
    // answer after </think> and discard whatever it thought.
    let cleaned = if let Some(after) = raw.find("</think>") {
        raw[after + "</think>".len()..].trim().to_owned()
    } else {
        raw
    };

    if cleaned.is_empty() {
        eprintln!("  [llm_clean] Empty response, keeping original");
        paragraph.to_owned()
    } else {
        cleaned
    }
}

/// Run the full LLM cleanup pass on raw text (paragraphs separated by \n\n).
///
/// Approach — paragraph-by-paragraph to stay within a 2.5B model's quality
/// envelope and allow incremental progress reporting:
///   1. Split on \n\n, filter min_chars (same logic as harvest_chunks).
///   2. For each paragraph: call clean_paragraph; accumulate stats.
///   3. Write text_cleaned.txt as a sidecar next to text.txt.
///   4. Return the cleaned text and an LlmCleanRecord for run.json.
///
/// Failure policy: any paragraph that fails (HTTP error, empty response) is
/// kept verbatim from the original — we never silently drop content.
fn llm_clean_text(
    raw:    &str,
    cfg:    &Config,
    client: &Client,
) -> Result<(String, records::LlmCleanRecord)> {
    use records::LlmCleanRecord;

    // Split the same way harvest_chunks does so paragraph count matches.
    let normalised = raw.replace("\r\n", "\n");
    let paragraphs: Vec<&str> = normalised
        .split("\n\n")
        .map(|s| s.trim())
        .filter(|s| s.chars().filter(|c| !c.is_whitespace()).count() >= cfg.min_chars)
        .collect();

    let total = paragraphs.len();
    println!("[lecturner] LLM cleanup: {} paragraph(s) via Crane at {}",
        total, crane_llm_url(cfg));

    let words_before: usize = paragraphs.iter()
        .map(|p| p.split_whitespace().count())
        .sum();

    let mut cleaned_paragraphs: Vec<String> = Vec::with_capacity(total);
    let mut n_changed = 0usize;

    let mut n_dropped = 0usize;

    for (i, &para) in paragraphs.iter().enumerate() {
        let ordinal = i + 1;
        let preview: String = para.chars().take(60).collect();
        print!("[lecturner] [llm {ordinal}/{total}] \"{preview}…\" ");
        use std::io::Write;
        let _ = std::io::stdout().flush();

        let cleaned = clean_paragraph(para, cfg, client);

        // Model signals non-prose artifact — drop silently, never pass downstream.
        if cleaned.trim().eq_ignore_ascii_case("drop") {
            n_dropped += 1;
            println!("✗ dropped");
            continue;
        }

        let changed = cleaned != para;
        if changed { n_changed += 1; }
        println!("{}", if changed { "✎" } else { "·" });

        cleaned_paragraphs.push(cleaned);
    }

    let cleaned_text = cleaned_paragraphs.join("\n\n");
    let words_after: usize = cleaned_paragraphs.iter()
        .map(|p| p.split_whitespace().count())
        .sum();

    // Write sidecar so the original text.txt is always recoverable.
    let cleaned_path = cfg.input.with_file_name(
        cfg.input.file_stem()
            .map(|s| format!("{}_cleaned.txt", s.to_string_lossy()))
            .unwrap_or_else(|| "text_cleaned.txt".to_owned())
    );
    fs::write(&cleaned_path, &cleaned_text)
        .with_context(|| format!("Cannot write {}", cleaned_path.display()))?;
    println!("[lecturner] LLM cleanup done: {n_changed}/{total} rewritten, \
        {n_dropped} dropped → {}", cleaned_path.display());

    let record = LlmCleanRecord {
        model:        "qwen3".to_owned(),
        words_before,
        words_after,
        changed:      n_changed > 0,
    };

    Ok((cleaned_text, record))
}

// ─── --merge-only mode ────────────────────────────────────────────────────────

/// Re-merge all Ok WAVs from a previous run without re-synthesizing anything.
///
/// Approach:
///   1. Load run.json from out_dir.
///   2. Collect mergeable records in ordinal order.
///   3. Verify each WAV file exists on disk (skip with warning if absent).
///   4. Call merge_wavs with the recovered paths and boundaries.
///   5. Optionally transcode to MP3.
fn run_merge_only(cfg: &Config) -> Result<()> {
    let run = RunRecord::load(&cfg.out_dir)
        .context("Cannot load run.json — has a synthesis run completed?")?;

    let mut wav_paths:  Vec<PathBuf>          = Vec::new();
    let mut boundaries: Vec<Option<Boundary>> = Vec::new();

    for rec in run.chunks.iter().filter(|r| r.is_mergeable()) {
        let wav_path = cfg.out_dir.join(&rec.wav);
        if !wav_path.exists() {
            eprintln!("[lecturner] Warning: {} listed as Ok but not on disk — skipping", rec.wav);
            continue;
        }
        wav_paths.push(wav_path);
        boundaries.push(rec.trailing_boundary);
    }

    if wav_paths.is_empty() {
        anyhow::bail!("No mergeable WAVs found in run.json");
    }

    println!("[lecturner] --merge-only: {} WAV(s) to merge", wav_paths.len());
    let combined_path = cfg.out_dir.join("combined.wav");
    merge_wavs(&wav_paths, &boundaries, &combined_path, cfg.sentence_gap_ms, cfg.paragraph_gap_ms)?;

    if cfg.to_mp3 && combined_path.exists() {
        match wav_to_mp3(&cfg.ffmpeg_bin, &combined_path) {
            Ok(mp3_path) => println!("[lecturner] MP3 written → {}", mp3_path.display()),
            Err(e)       => eprintln!("[lecturner] ffmpeg transcode failed: {e:#}"),
        }
    }
    Ok(())
}

// ─── --fix-quarantine mode ────────────────────────────────────────────────────

/// Re-synthesize quarantined chunks from a previous run.
///
/// Approach:
///   1. Load run.json from out_dir.
///   2. Find all Quarantined records (Failed records are skipped — server problem).
///   3. For each: apply aggressive sanitization, synthesize, validate if enabled.
///   4. On success: write WAV, update record status to Ok in the RunRecord.
///   5. Save updated run.json after each chunk (partial progress is preserved).
///   6. Print a reminder to run --merge-only to rebuild combined output.
fn run_fix_quarantine(cfg: &Config, client: &Client, wstate: &mut Option<WhisperState>) -> Result<()> {
    let mut run = RunRecord::load(&cfg.out_dir)
        .context("Cannot load run.json — has a synthesis run completed?")?;

    let retry_indices: Vec<usize> = run.chunks.iter()
        .enumerate()
        .filter(|(_, r)| r.needs_retry())
        .map(|(i, _)| i)
        .collect();

    if retry_indices.is_empty() {
        println!("[lecturner] No quarantined chunks found in run.json.");
        return Ok(());
    }

    println!("[lecturner] --fix-quarantine: {} chunk(s) to retry", retry_indices.len());

    let mut recovered = 0usize;
    let mut still_bad = 0usize;

    for idx in retry_indices {
        let rec           = &run.chunks[idx];
        let ordinal       = rec.ordinal;
        let wav_name      = rec.wav.clone();
        let original_text = match rec.quarantined_text() {
            Some(t) => t.to_owned(),
            None    => {
                eprintln!("[lecturner] Chunk {ordinal}: Failed record (server error), skipping.");
                still_bad += 1;
                continue;
            }
        };

        println!("[lecturner] Retrying chunk {ordinal}: \"{}…\"",
            original_text.chars().take(60).collect::<String>());

        // Aggressive sanitization pass before re-synthesis.
        let (sanitized, was_dirty) = sanitize_chunk(&original_text);
        let retry_text = if was_dirty { &sanitized } else { &original_text };
        if was_dirty {
            eprintln!("  [fix] Applied sanitization pass.");
        }

        let wav_path = cfg.out_dir.join(&wav_name);
        let bad_path = cfg.out_dir.join(format!("paragraph_{ordinal:03}.bad.txt"));

        match synthesise_chunk(client, cfg, retry_text) {
            Ok(wav_bytes) => {
                // Validate if enabled, otherwise accept directly.
                let final_wav = if let Some(ref mut ws) = wstate {
                    match validate_and_retry(client, ws, cfg, retry_text, wav_bytes, &bad_path) {
                        Ok(v)    => v,
                        Err(fail) => {
                            eprintln!("  ✗ Chunk {ordinal} still quarantined after fix attempt: {fail}");
                            still_bad += 1;
                            continue;
                        }
                    }
                } else {
                    wav_bytes
                };

                fs::write(&wav_path, &final_wav)
                    .with_context(|| format!("Cannot write {}", wav_path.display()))?;

                // Remove the stale .bad.txt sidecar if it exists.
                let _ = fs::remove_file(&bad_path);

                // Update the record in place and save immediately.
                run.chunks[idx].status = ChunkStatus::Ok;
                run.save(&cfg.out_dir)?;

                println!("  ✓ Chunk {ordinal} recovered → {wav_name}");
                recovered += 1;
            }
            Err(e) => {
                eprintln!("  ✗ Chunk {ordinal} synthesis failed: {e:#}");
                still_bad += 1;
            }
        }

        if cfg.rest_ms > 0 {
            thread::sleep(Duration::from_millis(cfg.rest_ms));
        }
    }

    println!(
        "\n[lecturner] Fix complete: {} recovered, {} still quarantined.",
        recovered, still_bad
    );
    if recovered > 0 {
        println!("[lecturner] Run with --merge-only to rebuild combined.wav with recovered chunks.");
    }
    Ok(())
}

// ─── main ─────────────────────────────────────────────────────────────────────


// ─── Batch mode ───────────────────────────────────────────────────────────────

/// What a single PDF job produced — drives naming and directory routing.
#[derive(Debug, Clone, PartialEq)]
enum JobOutcome {
    /// All chunks passed validation (or validation was off).
    Clean,
    /// Some chunks were quarantined but enough audio exists to ship a partial MP3.
    Part,
    /// Synthesis started but produced zero usable WAVs.
    Picturebook,
    /// Hard failure before synthesis could produce anything (LLM crash, rip fail, etc.).
    HardFail { reason: String },
}

/// One line appended to batch.log per event.  JSONL — one object per line.
///
/// Design: minimal fields, human-readable, enough for lecturner to count failures
/// per filename across a crash-interrupted run.
#[derive(Serialize, Deserialize, Debug)]
struct BatchLogEntry {
    ts:      String,
    pdf:     String,   // filename only, e.g. "RosenCrantz_Guildenstern.pdf"
    event:   String,   // "start" | "complete" | "fail"
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome: Option<String>,  // on "complete": "ok" | "part" | "picturebook"
    #[serde(skip_serializing_if = "Option::is_none")]
    mp3:     Option<String>,  // on "complete": output filename
    #[serde(skip_serializing_if = "Option::is_none")]
    reason:  Option<String>,  // on "fail": short description
}

/// Append-only batch log — one JSONL file, wiped when the batch finishes cleanly.
struct BatchLog {
    path: PathBuf,
}

impl BatchLog {
    fn open(batch_dir: &PathBuf) -> Result<Self> {
        let path = batch_dir.join("batch.log");
        Ok(BatchLog { path })
    }

    /// Append one entry.  Each entry is a single JSON line terminated by \n.
    fn append(&self, entry: &BatchLogEntry) -> Result<()> {
        use std::io::Write;
        let line = serde_json::to_string(entry)? + "
";
        let mut file = std::fs::OpenOptions::new()
            .create(true).append(true).open(&self.path)?;
        file.write_all(line.as_bytes())?;
        Ok(())
    }

    /// Count "fail" events per PDF filename across all existing log entries.
    /// Missing or unreadable log = no prior failures.
    fn failure_counts(&self) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        let Ok(raw) = fs::read_to_string(&self.path) else { return counts; };
        for line in raw.lines() {
            if let Ok(entry) = serde_json::from_str::<BatchLogEntry>(line) {
                if entry.event == "fail" {
                    *counts.entry(entry.pdf).or_insert(0) += 1;
                }
            }
        }
        counts
    }

    /// Wipe the log file.  Called when the batch loop exits cleanly.
    fn clear(&self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Derive the output stem from any inbox filename (PDF or TXT).
/// "RosenCrantz_Guildenstern.pdf" | "RosenCrantz_Guildenstern.txt" → "rosencrantz_guildenstern"
fn input_stem(path: &PathBuf) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Collect inbox jobs sorted alphabetically. Accepts .pdf and .txt.
fn collect_inbox_jobs(inbox: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut jobs: Vec<PathBuf> = fs::read_dir(inbox)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
             .map(|x| x.eq_ignore_ascii_case("pdf") || x.eq_ignore_ascii_case("txt"))
             .unwrap_or(false)
        })
        .collect();
    jobs.sort();
    Ok(jobs)
}

/// Run the full pipeline for one inbox job and return what happened.
///
/// Approach — wraps the existing single-file pipeline end-to-end:
///   1. Acquire raw text: PDF → rip via pdf_rip.py; TXT → read directly.
///   2. LLM cleanup pass → cleaned text written to text_completed/.
///   3. Chunk + synthesise into work_dir WAVs (per-job TTS server lifecycle).
///   4. Merge WAVs → work_dir/combined.wav → MP3 in audio/.
///   5. Classify outcome (Clean / Part / Picturebook).
///   6. Wipe work_dir regardless of outcome.
fn run_single_job(
    job_path:   &PathBuf,
    batch_dirs: &BatchDirs,
    cfg:        &Config,
    client:     &Client,
    wstate:     &mut Option<WhisperState>,
) -> JobOutcome {
    let stem     = input_stem(job_path);
    let work_dir = batch_dirs.work.join(&stem);
    let text_out = batch_dirs.text_completed.join(format!("{stem}.txt"));

    // ── Create work dir ───────────────────────────────────────────────────────
    if let Err(e) = fs::create_dir_all(&work_dir) {
        return JobOutcome::HardFail { reason: format!("Cannot create work dir: {e}") };
    }

    // ── Step 1: Acquire raw text ───────────────────────────────────────────────
    let is_pdf = job_path.extension()
        .map(|x| x.eq_ignore_ascii_case("pdf"))
        .unwrap_or(false);

    let raw_text = if is_pdf {
        let raw_txt_path = work_dir.join("raw.txt");
        let rip_cfg = PartialConfig { input: raw_txt_path.clone() };
        if let Err(e) = rip_pdf_to(cfg, job_path, &rip_cfg) {
            return JobOutcome::HardFail { reason: format!("PDF rip failed: {e:#}") };
        }
        match fs::read_to_string(&raw_txt_path) {
            Ok(t)  => t,
            Err(e) => return JobOutcome::HardFail { reason: format!("Cannot read ripped text: {e}") },
        }
    } else {
        match fs::read_to_string(job_path) {
            Ok(t)  => t,
            Err(e) => return JobOutcome::HardFail { reason: format!("Cannot read input file: {e}") },
        }
    };

    // ── Step 2: LLM cleanup ───────────────────────────────────────────────────
    let (synthesis_text, _llm_record) = if cfg.llm_clean {
        let llm_client = match Client::builder().timeout(Duration::from_secs(300)).build() {
            Ok(c)  => c,
            Err(e) => return JobOutcome::HardFail { reason: format!("HTTP client: {e}") },
        };
        let mut crane_task = match ensure_crane_llm_running(cfg, &llm_client) {
            Ok(t)  => t,
            Err(e) => return JobOutcome::HardFail { reason: format!("Crane LLM: {e:#}") },
        };
        let result = llm_clean_text_to(&raw_text, cfg, &llm_client, &work_dir);
        if let Some(ref mut task) = crane_task {
            println!("[batch] Shutting down Crane LLM (pid {})…", task.id());
            let _ = task.kill();
            let _ = task.wait();
        }
        match result {
            Ok((cleaned, record)) => (cleaned, Some(record)),
            Err(e) => return JobOutcome::HardFail { reason: format!("LLM cleanup: {e:#}") },
        }
    } else {
        (raw_text, None)
    };

    // Write cleaned (or raw) text to text_completed/ now — survives even if
    // synthesis fails completely.
    if let Err(e) = fs::write(&text_out, &synthesis_text) {
        eprintln!("[batch] Warning: could not write {}: {e}", text_out.display());
    }

    // ── Step 3: Chunk + synthesise ────────────────────────────────────────────
    let chunks = harvest_chunks(&synthesis_text, cfg.min_chars, cfg.max_words);
    if chunks.is_empty() {
        wipe_work_dir(&work_dir);
        return JobOutcome::Picturebook;
    }

    // Start TTS crane for this job; shut it down before cleanup so the next
    // job's LLM startup doesn't fight it for VRAM.
    let mut tts_task = match ensure_crane_tts_running(cfg, client) {
        Ok(t)  => t,
        Err(e) => {
            wipe_work_dir(&work_dir);
            return JobOutcome::HardFail { reason: format!("Crane TTS: {e:#}") };
        }
    };

    let total = chunks.len();
    let mut good_paths:      Vec<PathBuf>          = Vec::new();
    let mut good_boundaries: Vec<Option<Boundary>> = Vec::new();
    let mut n_failed      = 0usize;
    let mut n_quarantined = 0usize;

    for (idx, chunk) in chunks.iter().enumerate() {
        let ordinal  = idx + 1;
        let wav_name = format!("paragraph_{ordinal:03}.wav");
        let wav_path = work_dir.join(&wav_name);

        let word_count = chunk.text.split_whitespace().count();
        let preview: String = chunk.text.chars().take(80).collect();
        let ellipsis = if chunk.text.len() > 80 { "…" } else { "" };
        println!(
            "[batch] [{stem} {ordinal}/{total}] {word_count}w \u{2192} {wav_name}\n  \"{preview}{ellipsis}\""
        );

        match synthesise_chunk(client, cfg, &chunk.text) {
            Ok(first_wav) => {
                let final_wav = if let Some(ref mut ws) = wstate {
                    let bad_path = work_dir.join(format!("paragraph_{ordinal:03}.bad.txt"));
                    match validate_and_retry(client, ws, cfg, &chunk.text, first_wav, &bad_path) {
                        Ok(v) => v,
                        Err(fail) => {
                            eprintln!("  ✗ [{stem}] Chunk {ordinal} quarantined: {fail}");
                            n_quarantined += 1;
                            continue;
                        }
                    }
                } else {
                    first_wav
                };
                if let Err(e) = fs::write(&wav_path, &final_wav) {
                    eprintln!("  ✗ [{stem}] Cannot write {wav_name}: {e}");
                    n_failed += 1;
                    continue;
                }
                println!("  ✓ {wav_name} ({} bytes)", final_wav.len());
                good_paths.push(wav_path);
                good_boundaries.push(chunk.trailing_boundary);
            }
            Err(e) => {
                eprintln!("  ✗ [{stem}] Chunk {ordinal} synthesis failed: {e:#}");
                n_failed += 1;
            }
        }

        if ordinal < total && cfg.rest_ms > 0 {
            thread::sleep(Duration::from_millis(cfg.rest_ms));
        }
    }

    // Shut down TTS crane before merge — frees VRAM for next job's LLM.
    if let Some(ref mut task) = tts_task {
        println!("[batch] Stopping Crane TTS (pid {})…", task.id());
        let _ = task.kill();
        let _ = task.wait();
    }

    // ── Step 4: Merge + transcode ─────────────────────────────────────────────
    let n_bad    = n_failed + n_quarantined;
    let outcome  = classify_outcome(good_paths.len(), n_bad, total);
    let mp3_stem = match &outcome {
        JobOutcome::Clean => format!("{stem}.mp3"),
        JobOutcome::Part  => format!("{stem}_part.mp3"),
        JobOutcome::Picturebook => {
            let marker = batch_dirs.audio.join(format!("{stem}_picturebook.txt"));
            let _ = fs::write(&marker,
                format!("lecturner batch: no audio produced for {stem}
                         chunks={total}, failed={n_failed}, quarantined={n_quarantined}
"));
            wipe_work_dir(&work_dir);
            return JobOutcome::Picturebook;
        }
        JobOutcome::HardFail { .. } => unreachable!(),
    };

    let combined_wav = work_dir.join("combined.wav");
    if let Err(e) = merge_wavs(&good_paths, &good_boundaries, &combined_wav,
                                cfg.sentence_gap_ms, cfg.paragraph_gap_ms) {
        eprintln!("[batch] Merge failed for {stem}: {e:#}");
        wipe_work_dir(&work_dir);
        return JobOutcome::HardFail { reason: format!("merge failed: {e:#}") };
    }

    let mp3_dest = batch_dirs.audio.join(&mp3_stem);
    match wav_to_mp3(&cfg.ffmpeg_bin, &combined_wav) {
        Ok(tmp_mp3) => {
            if let Err(e) = fs::rename(&tmp_mp3, &mp3_dest) {
                eprintln!("[batch] Cannot move MP3 to audio/: {e}");
            } else {
                println!("[batch] ✓ {}", mp3_dest.display());
            }
        }
        Err(e) => eprintln!("[batch] ffmpeg failed for {stem}: {e:#}"),
    }

    wipe_work_dir(&work_dir);
    outcome
}

/// Classify outcome from synthesis counts.
fn classify_outcome(n_good: usize, n_bad: usize, total: usize) -> JobOutcome {
    if n_good == 0          { return JobOutcome::Picturebook; }
    if n_bad  == 0          { return JobOutcome::Clean; }
    let _ = total;          // available for future threshold tuning
    JobOutcome::Part
}

/// Thin config shim — the parts of Config that rip_pdf_to and llm_clean_text_to
/// need overridden per-job without cloning the entire Config.
struct PartialConfig {
    input: PathBuf,
}

/// rip_pdf variant that writes to an explicit output path (PartialConfig.input)
/// rather than cfg.input, so batch jobs each get their own work directory.
fn rip_pdf_to(cfg: &Config, pdf_path: &PathBuf, dest: &PartialConfig) -> Result<()> {
    let script = locate_pdf_rip_script(&None)
        .context("Cannot find pdf_rip.py.")?;
    let out_path = dest.input.to_str().context("Non-UTF8 output path")?;
    let pdf_str  = pdf_path.to_str().context("Non-UTF8 PDF path")?;

    println!("[batch] Ripping: {} → {}", pdf_path.display(), dest.input.display());

    let mut cmd = Command::new(&cfg.python_bin);
    cmd.arg(&script)
       .arg("--input").arg(pdf_str)
       .arg("--output").arg(out_path);
    if !cfg.skip_refs     { cmd.arg("--no-skip-refs"); }
    if !cfg.skip_captions { cmd.arg("--no-skip-captions"); }

    let output = cmd.output()
        .with_context(|| format!("Failed to spawn '{}'", cfg.python_bin))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() { println!("{line}"); }

    match output.status.code() {
        Some(0) => Ok(()),
        Some(1) => {
            if stdout.contains("GOSSIP_MISSING_DEP:pdfplumber") {
                anyhow::bail!("pdfplumber not installed: {} -m pip install pdfplumber", cfg.python_bin);
            }
            anyhow::bail!("pdf_rip.py failed:
{}", String::from_utf8_lossy(&output.stderr))
        }
        _ => anyhow::bail!("pdf_rip.py error:
{}", String::from_utf8_lossy(&output.stderr)),
    }
}

/// llm_clean_text variant that writes the sidecar into work_dir instead of
/// next to cfg.input, keeping each job's files self-contained.
fn llm_clean_text_to(
    raw:      &str,
    cfg:      &Config,
    client:   &Client,
    work_dir: &PathBuf,
) -> Result<(String, records::LlmCleanRecord)> {
    use records::LlmCleanRecord;
    let normalised  = raw.replace("\r\n", "\n");
    let paragraphs: Vec<&str> = normalised
        .split("\n\n")
        .map(|s| s.trim())
        .filter(|s| s.chars().filter(|c| !c.is_whitespace()).count() >= cfg.min_chars)
        .collect();
    let total        = paragraphs.len();
    let words_before = paragraphs.iter().map(|p| p.split_whitespace().count()).sum();
    let mut cleaned_paragraphs = Vec::with_capacity(total);
    let mut n_changed = 0usize;
    let mut n_dropped = 0usize;

    for (i, &para) in paragraphs.iter().enumerate() {
        let ordinal = i + 1;
        let preview: String = para.chars().take(60).collect();
        print!("[batch] [llm {ordinal}/{total}] \"{preview}…\" ");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let cleaned = clean_paragraph(para, cfg, client);
        if cleaned.trim().eq_ignore_ascii_case("drop") {
            n_dropped += 1;
            println!("✗ dropped");
            continue;
        }
        let changed = cleaned != para;
        if changed { n_changed += 1; }
        println!("{}", if changed { "✎" } else { "·" });
        cleaned_paragraphs.push(cleaned);
    }

    let cleaned_text = cleaned_paragraphs.join("\n\n");
    let words_after  = cleaned_paragraphs.iter().map(|p| p.split_whitespace().count()).sum();
    let sidecar      = work_dir.join("cleaned.txt");
    fs::write(&sidecar, &cleaned_text)
        .with_context(|| format!("Cannot write {}", sidecar.display()))?;
    println!("[batch] LLM cleanup done: {n_changed}/{total} rewritten, {n_dropped} dropped");

    Ok((cleaned_text, LlmCleanRecord {
        model:        "qwen3".to_owned(),
        words_before,
        words_after,
        changed:      n_changed > 0,
    }))
}

/// Delete the work directory tree and everything in it.
/// Errors are logged but do not propagate — cleanup failure is never fatal.
fn wipe_work_dir(work_dir: &PathBuf) {
    if work_dir.exists() {
        if let Err(e) = fs::remove_dir_all(work_dir) {
            eprintln!("[batch] Warning: could not wipe work dir {}: {e}", work_dir.display());
        }
    }
}

/// All batch subdirectories in one place — constructed once, passed by ref.
struct BatchDirs {
    inbox:          PathBuf,
    pdf_completed:  PathBuf,
    pdf_errored:    PathBuf,
    text_completed: PathBuf,
    audio:          PathBuf,
    work:           PathBuf,
}

impl BatchDirs {
    fn create(batch_root: &PathBuf) -> Result<Self> {
        let dirs = BatchDirs {
            inbox:          batch_root.join("in"),
            pdf_completed:  batch_root.join("pdf_completed"),
            pdf_errored:    batch_root.join("pdf_errored"),
            text_completed: batch_root.join("text_completed"),
            audio:          batch_root.join("audio"),
            work:           batch_root.join("work"),
        };
        for d in [&dirs.inbox, &dirs.pdf_completed, &dirs.pdf_errored,
                  &dirs.text_completed, &dirs.audio, &dirs.work] {
            fs::create_dir_all(d)
                .with_context(|| format!("Cannot create batch dir: {}", d.display()))?;
        }
        Ok(dirs)
    }
}

/// Collect PDF paths from inbox, sorted alphabetically for deterministic ordering.
/// Main batch loop.
///
/// Approach:
///   1. Create subdirectory tree under batch_root.
///   2. Open (or resume) batch.log and tally prior failure counts.
///   3. For each PDF in inbox/ (alphabetical):
///        a. Skip if failure_count >= 2 (double-fail rule).
///        b. Log "start".
///        c. Run run_single_job() — catches its own panics via Result.
///        d. On Clean or Part: move PDF to pdf_completed/, log "complete".
///        e. On Picturebook or HardFail: move PDF to pdf_errored/, log "fail".
///   4. Wipe batch.log on clean exit.
fn run_batch(batch_root: PathBuf, cfg: &Config, wstate: &mut Option<WhisperState>) -> Result<()> {
    let dirs    = BatchDirs::create(&batch_root)?;
    let log     = BatchLog::open(&batch_root)?;
    let client  = Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .context("Failed to build HTTP client")?;

    let prior_failures = log.failure_counts();

    let jobs = collect_inbox_jobs(&dirs.inbox)?;
    if jobs.is_empty() {
        println!("[batch] No jobs found in {} (.pdf and .txt)", dirs.inbox.display());
        log.clear();
        return Ok(());
    }
    println!("[batch] {} job(s) queued in {}", jobs.len(), dirs.inbox.display());

    let ts = || chrono::Utc::now().to_rfc3339();

    for job_path in &jobs {
        let filename = job_path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_owned());

        // ── Double-fail skip ───────────────────────────────────────────────────
        let prior = *prior_failures.get(&filename).unwrap_or(&0);
        if prior >= 2 {
            println!("[batch] Skipping '{filename}' — {prior} prior failures in this session.");
            continue;
        }

        println!("
[batch] ══ Starting: {filename} ══");
        let _ = log.append(&BatchLogEntry {
            ts: ts(), pdf: filename.clone(), event: "start".into(),
            outcome: None, mp3: None, reason: None,
        });

        // ── Run the job ────────────────────────────────────────────────────────
        let outcome = run_single_job(&job_path, &dirs, cfg, &client, wstate);

        // ── Route based on outcome ─────────────────────────────────────────────
        match &outcome {
            JobOutcome::Clean | JobOutcome::Part | JobOutcome::Picturebook => {
                let (event, outcome_str, mp3_name) = match &outcome {
                    JobOutcome::Clean =>
                        ("complete", "ok",          Some(format!("{}.mp3", input_stem(&job_path)))),
                    JobOutcome::Part =>
                        ("complete", "part",        Some(format!("{}_part.mp3", input_stem(&job_path)))),
                    JobOutcome::Picturebook =>
                        ("complete", "picturebook", Some(format!("{}_picturebook.txt", input_stem(&job_path)))),
                    _ => unreachable!(),
                };
                let dest = dirs.pdf_completed.join(&filename);
                if let Err(e) = fs::rename(&job_path, &dest) {
                    eprintln!("[batch] Could not move to completed/: {e}");
                }
                let _ = log.append(&BatchLogEntry {
                    ts: ts(), pdf: filename.clone(), event: event.into(),
                    outcome: Some(outcome_str.into()), mp3: mp3_name, reason: None,
                });
                println!("[batch] ✓ '{filename}' → completed/ [{outcome_str}]");
            }
            JobOutcome::HardFail { reason } => {
                let dest = dirs.pdf_errored.join(&filename);
                if let Err(e) = fs::rename(&job_path, &dest) {
                    eprintln!("[batch] Could not move to errored/: {e}");
                }
                let _ = log.append(&BatchLogEntry {
                    ts: ts(), pdf: filename.clone(), event: "fail".into(),
                    outcome: None, mp3: None, reason: Some(reason.clone()),
                });
                eprintln!("[batch] ✗ '{filename}' → errored/ [{reason}]");
            }
        }
    }

    println!("\n[batch] All done.");
    log.clear();
    Ok(())
}

fn main() -> Result<()> {
    let cli  = CliArgs::parse();
    let toml = load_toml_config();
    let cfg  = resolve_config(cli, toml);

    fs::create_dir_all(&cfg.out_dir)
        .with_context(|| format!("Cannot create: {}", cfg.out_dir.display()))?;

    // ── Early-exit modes: --merge-only and --fix-quarantine ───────────────────
    // These operate on a previous run's run.json and do not need the full
    // synthesis pipeline.  --fix-quarantine does need the server and optionally
    // Whisper; --merge-only needs neither.

    // ── Batch mode ───────────────────────────────────────────────────────────────
    if let Some(batch_root) = cfg.batch_pdf.clone() {
        // Whisper is loaded once here and shared across all jobs in the batch.
        let whisper_ctx: Option<WhisperContext> = if cfg.validate {
            let model_path = cfg.whisper_model_dir.join(&cfg.whisper_model);
            println!("[lecturner] Loading Whisper model: {}", model_path.display());
            let ctx = load_whisper_model(&model_path)?;
            println!("[lecturner] Whisper ready — threshold={:.3}", cfg.validate_threshold);
            Some(ctx)
        } else { None };
        let mut whisper_state: Option<WhisperState> = match whisper_ctx {
            Some(ref ctx) => Some(
                ctx.create_state()
                   .map_err(|e| anyhow::anyhow!("Whisper create_state: {e:?}"))?
            ),
            None => None,
        };
        return run_batch(batch_root, &cfg, &mut whisper_state);
    }

    if cfg.merge_only {
        return run_merge_only(&cfg);
    }

    if cfg.fix_quarantine {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .context("Failed to build HTTP client")?;
        let mut server_task = ensure_crane_tts_running(&cfg, &client)?;

        let whisper_ctx: Option<WhisperContext> = if cfg.validate {
            let model_path = cfg.whisper_model_dir.join(&cfg.whisper_model);
            println!("[lecturner] Loading Whisper model: {}", model_path.display());
            let ctx = load_whisper_model(&model_path)?;
            println!("[lecturner] Whisper ready.");
            Some(ctx)
        } else {
            None
        };
        let mut whisper_state: Option<WhisperState> = match whisper_ctx {
            Some(ref ctx) => Some(
                ctx.create_state()
                   .map_err(|e| anyhow::anyhow!("Whisper create_state: {e:?}"))?
            ),
            None => None,
        };

        run_fix_quarantine(&cfg, &client, &mut whisper_state)?;

        if let Some(ref mut task) = server_task {
            let _ = task.kill();
            let _ = task.wait();
        }
        return Ok(());
    }

    // ── 0. PDF rip (if requested) ─────────────────────────────────────────────
    if let Some(ref pdf_path) = cfg.rip_pdf.clone() {
        rip_pdf(&cfg, pdf_path)?;
        if cfg.rip_pdf_only {
            println!("[lecturner] PDF ripped to '{}' — exiting (--rip-pdf-only).", cfg.input.display());
            return Ok(());
        }
    }

    // ── 0b. LLM cleanup pass (if enabled) ─────────────────────────────────────
    // Rewrites the extracted prose paragraph-by-paragraph using Qwen2.5 via
    // crane-oai before chunking.  Runs only when llm_clean = true (automatic
    // when --rip-pdf is used; opt-in otherwise via lecturner.toml).
    // On completion the cleaned text replaces raw_text for all downstream steps;
    // the LlmCleanRecord is stored in run.json.
    let (raw_text, llm_clean_record) = {
        let maybe_raw = fs::read_to_string(&cfg.input)
            .with_context(|| format!("Cannot open: {}", cfg.input.display()))?;

        if cfg.llm_clean {
            // Build a short-timeout client for LLM health checks, then a
            // longer one for actual generation (2.5B model, 300s ceiling).
            let llm_client = Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .context("Failed to build LLM HTTP client")?;

            let mut crane_task = ensure_crane_llm_running(&cfg, &llm_client)?;
            let (cleaned, record) = llm_clean_text(&maybe_raw, &cfg, &llm_client)?;

            // Crane LLM is only needed for the cleanup pass; shut it down now
            // so TTS crane can claim the GPU headroom before loading.
            if let Some(ref mut task) = crane_task {
                println!("[lecturner] Shutting down Crane LLM (pid {})…", task.id());
                let _ = task.kill();
                let _ = task.wait();
            }

            (cleaned, Some(record))
        } else {
            (maybe_raw, None)
        }
    };

    // ── 1. Split text into chunks ──────────────────────────────────────────────

    let chunks = harvest_chunks(&raw_text, cfg.min_chars, cfg.max_words);
    if chunks.is_empty() {
        anyhow::bail!(
            "No usable chunks in '{}' (min_chars={})",
            cfg.input.display(), cfg.min_chars
        );
    }
    println!(
        "[lecturner] {} chunk(s) from '{}' (max_words={}, voice={})",
        chunks.len(), cfg.input.display(), cfg.max_words,
        cfg.crane_tts_voice,
    );

    // ── 2. HTTP client ─────────────────────────────────────────────────────────
    let client = Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .context("Failed to build HTTP client")?;

    // ── 3. Ensure Crane TTS server is running ──────────────────────────────────
    let mut server_task = ensure_crane_tts_running(&cfg, &client)?;

    // ── 4. Load Whisper model (if validation enabled) ─────────────────────────
    let whisper_ctx: Option<WhisperContext> = if cfg.validate {
        let model_path = cfg.whisper_model_dir.join(&cfg.whisper_model);
        println!("[lecturner] Loading Whisper model: {}", model_path.display());
        let ctx = load_whisper_model(&model_path)?;
        println!("[lecturner] Whisper ready — validation threshold={:.3}", cfg.validate_threshold);
        Some(ctx)
    } else {
        None
    };
    // WhisperState holds ~500 MB of GPU compute buffers.  Created once here,
    // passed by &mut through the call chain — never re-allocated per chunk.
    let mut whisper_state: Option<WhisperState> = match whisper_ctx {
        Some(ref ctx) => Some(
            ctx.create_state()
               .map_err(|e| anyhow::anyhow!("Whisper create_state: {e:?}"))?
        ),
        None => None,
    };

    // ── 5. Synthesise each chunk ───────────────────────────────────────────────
    let total = chunks.len();
    let mut run_record = RunRecord::new(
        cfg.input.to_string_lossy().into_owned(),
        cfg.rip_pdf.as_ref().map(|p| p.to_string_lossy().into_owned()),
    );
    run_record.llm_clean = llm_clean_record;
    let mut good_paths:      Vec<PathBuf>          = Vec::new();
    let mut good_boundaries: Vec<Option<Boundary>> = Vec::new();
    let mut n_skipped    = 0usize;
    let mut n_quarantined = 0usize;

    for (idx, chunk) in chunks.iter().enumerate() {
        let ordinal  = idx + 1;
        let wav_name = format!("paragraph_{ordinal:03}.wav");
        let wav_path = cfg.out_dir.join(&wav_name);

        let word_count = chunk.text.split_whitespace().count();
        let preview: String = chunk.text.chars().take(80).collect();
        let ellipsis = if chunk.text.len() > 80 { "…" } else { "" };
        let gap_label = match chunk.trailing_boundary {
            Some(Boundary::Paragraph) => format!("→ {}ms para gap",     cfg.paragraph_gap_ms),
            Some(Boundary::Sentence)  => format!("→ {}ms sentence gap", cfg.sentence_gap_ms),
            None                      => "→ end".to_string(),
        };
        println!(
            "[lecturner] [{ordinal}/{total}] {word_count}w {gap_label} → {wav_name}\n  \"{preview}{ellipsis}\""
        );

        match synthesise_chunk(&client, &cfg, &chunk.text) {
            Ok(first_wav) => {
                let final_wav = if let Some(ref mut wstate) = whisper_state {
                    let bad_path = cfg.out_dir.join(format!("paragraph_{ordinal:03}.bad.txt"));
                    match validate_and_retry(
                        &client, wstate, &cfg,
                        &chunk.text,   // <-- &str now, not &Chunk
                        first_wav, &bad_path,
                    ) {
                        Ok(validated_wav) => validated_wav,
                        Err(fail) => {
                            eprintln!("  ✗ Chunk {ordinal} quarantined: {fail}");
                            n_quarantined += 1;
                            run_record.chunks.push(ChunkRecord {
                                ordinal,
                                wav: wav_name,
                                trailing_boundary: chunk.trailing_boundary,
                                status: ChunkStatus::Quarantined {
                                    original_text: chunk.text.clone(),
                                },
                            });
                            let _ = run_record.save(&cfg.out_dir);
                            continue;
                        }
                    }
                } else {
                    first_wav
                };

                fs::write(&wav_path, &final_wav)
                    .with_context(|| format!("Cannot write {}", wav_path.display()))?;
                println!("  ✓ {} ({} bytes)", wav_path.display(), final_wav.len());

                run_record.chunks.push(ChunkRecord {
                    ordinal,
                    wav: wav_name,
                    trailing_boundary: chunk.trailing_boundary,
                    status: ChunkStatus::Ok,
                });
                let _ = run_record.save(&cfg.out_dir);

                good_paths.push(wav_path);
                good_boundaries.push(chunk.trailing_boundary);
            }
            Err(e) => {
                eprintln!("  ✗ Chunk {ordinal} synthesis failed: {e:#}");
                n_skipped += 1;
                run_record.chunks.push(ChunkRecord {
                    ordinal,
                    wav: wav_name,
                    trailing_boundary: chunk.trailing_boundary,
                    status: ChunkStatus::Failed { reason: format!("{e:#}") },
                });
                let _ = run_record.save(&cfg.out_dir);
            }
        }

        if ordinal < total && cfg.rest_ms > 0 {
            thread::sleep(Duration::from_millis(cfg.rest_ms));
        }
    }

    // ── 6. Merge WAVs ──────────────────────────────────────────────────────────
    let combined_path = cfg.out_dir.join("combined.wav");
    if cfg.merge && !good_paths.is_empty() {
        if let Err(e) = merge_wavs(
            &good_paths, &good_boundaries, &combined_path,
            cfg.sentence_gap_ms, cfg.paragraph_gap_ms,
        ) {
            eprintln!("[lecturner] Merge failed: {e:#}");
        }
    } else if cfg.merge {
        eprintln!("[lecturner] Nothing to merge — all chunks failed.");
    }

    // ── 7. MP3 transcode ───────────────────────────────────────────────────────
    if cfg.to_mp3 && cfg.merge && combined_path.exists() {
        match wav_to_mp3(&cfg.ffmpeg_bin, &combined_path) {
            Ok(mp3_path) => println!("[lecturner] MP3 written → {}", mp3_path.display()),
            Err(e)       => eprintln!("[lecturner] ffmpeg transcode failed: {e:#}"),
        }
    }

    // ── 8. Summary ─────────────────────────────────────────────────────────────
    println!(
        "\n[lecturner] Done. {} WAV(s) in '{}', {} failed, {} quarantined.",
        good_paths.len(), cfg.out_dir.display(), n_skipped, n_quarantined,
    );
    if n_quarantined > 0 {
        println!("[lecturner] Run with --fix-quarantine to retry bad chunks, then --merge-only to rebuild.");
    }

    // ── 9. Shut down server if we launched it ──────────────────────────────────
    if let Some(ref mut task) = server_task {
        println!("[lecturner] Stopping Crane TTS server (pid {})…", task.id());
        let _ = task.kill();
        let _ = task.wait();
    }

    if n_skipped > 0 { std::process::exit(1); }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_line_yields_paragraph_boundary() {
        let chunks = harvest_chunks("First.\n\nSecond.", 3, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].trailing_boundary, Some(Boundary::Paragraph));
        assert_eq!(chunks[1].trailing_boundary, None);
    }

    #[test]
    fn word_split_yields_sentence_boundaries() {
        let long: String = (0..250).map(|i| format!("word{i} ")).collect();
        let chunks = harvest_chunks(long.trim(), 1, 200);
        assert!(chunks.len() >= 2);
        for c in &chunks[..chunks.len()-1] {
            assert_eq!(c.trailing_boundary, Some(Boundary::Sentence));
        }
        assert_eq!(chunks.last().unwrap().trailing_boundary, None);
    }

    #[test]
    fn paragraph_boundary_propagates_to_last_sub_chunk() {
        let long: String = (0..250).map(|i| format!("word{i} ")).collect();
        let raw  = format!("{}\n\nShort second.", long.trim());
        let chunks = harvest_chunks(&raw, 3, 200);
        let last_of_first = chunks.iter()
            .take_while(|c| !c.text.contains("Short second"))
            .last()
            .unwrap();
        assert_eq!(last_of_first.trailing_boundary, Some(Boundary::Paragraph));
    }

    #[test]
    fn last_chunk_has_no_trailing_boundary() {
        let chunks = harvest_chunks("One.\n\nTwo.\n\nThree.", 1, 0);
        assert_eq!(chunks.last().unwrap().trailing_boundary, None);
    }

    #[test]
    fn silence_length_mono() {
        assert_eq!(silence_bytes(24000, 1, 16, 100).len(), 4800);
    }

    #[test]
    fn silence_length_stereo() {
        assert_eq!(silence_bytes(24000, 2, 16, 100).len(), 9600);
    }

    #[test]
    fn wav_header_roundtrip() {
        let pcm = vec![0u8; 200];
        let mut wav = Vec::new();
        write_wav_header(&mut wav, 1, 1, 24000, 16, pcm.len() as u32);
        wav.extend_from_slice(&pcm);
        let hdr = parse_wav_header(&wav).unwrap();
        assert_eq!(hdr.channels, 1);
        assert_eq!(hdr.sample_rate, 24000);
        assert_eq!(hdr.data_len, 200);
    }

    #[test]
    fn wav_header_accepts_float32_format() {
        let pcm = vec![0u8; 200];
        let mut wav = Vec::new();
        write_wav_header(&mut wav, 3, 1, 24000, 32, pcm.len() as u32);
        wav[20] = 3;
        wav[21] = 0;
        wav.extend_from_slice(&pcm);
        let hdr = parse_wav_header(&wav).unwrap();
        assert_eq!(hdr.bits_per_sample, 32);
    }
}