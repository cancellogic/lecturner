//! records.rs — persistent run and chunk outcome types for lecturner.
//!
//! `RunRecord` is written to `out_dir/run.json` after each chunk is processed.
//! `--merge-only` and `--fix-quarantine` read this file instead of re-parsing
//! `text.txt`, so every subsequent operation is self-contained.
//!
//! Design constraints:
//!   - Every subsequent operation works from `run.json` alone.
//!   - JSON is human-readable without a viewer (serde tag = discriminant).
//!   - Filenames are relative to `out_dir` — directory is moveable.
//!   - `original_text` lives only on `Quarantined` — ok chunks don't need it.
//!   - Quality data (PER, transcription) stays in `.bad.txt`; `run.json` is
//!     operational state, not a quality report.
//!   - Run-level metadata wraps the chunk vec so future pipeline stages
//!     (LLM cleanup, etc.) have a home without restructuring chunk records.

use chrono::Utc;
use serde::{Deserialize, Serialize};

// ─── Boundary ─────────────────────────────────────────────────────────────────

/// The silence gap that trails a synthesis chunk.
///
/// Persisted so `--merge-only` can reconstruct correct gap durations
/// without re-reading the source text.
/// Absence in `ChunkRecord.trailing_boundary` means final chunk — no
/// trailing silence inserted.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum Boundary {
    Paragraph,
    Sentence,
}

// ─── ChunkStatus ──────────────────────────────────────────────────────────────

/// Outcome variants for a single synthesis chunk.
///
/// `#[serde(tag = "status")]` produces flat, readable JSON:
///   {"status": "Ok"}
///   {"status": "Quarantined", "original_text": "..."}
///   {"status": "Failed",      "reason": "..."}
///   {"status": "Skipped",     "reason": "..."}
///
/// `Quarantined` carries `original_text` so `--fix-quarantine` needs
/// nothing beyond `run.json`.
///
/// `Failed` is distinct from `Quarantined`: a network or IO error before
/// Whisper validation ran is a server problem, not a text problem.
/// Fix-quarantine skips these with a message rather than retrying.
///
/// `Skipped` covers pre-synthesis rejection: chunk below `min_chars`,
/// or in future any pipeline stage (LLM cleanup, etc.) refusing the input.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "status")]
pub enum ChunkStatus {
    /// WAV written and passed validation (or validation disabled).
    Ok,

    /// Synthesis succeeded but validation failed after all retry attempts
    /// including sub-partition.  WAV is NOT on disk.
    /// `original_text` is pre-sanitization; fix-quarantine applies its own
    /// sanitization pass on retry.
    Quarantined { original_text: String },

    /// Hard failure before validation ran (HTTP error, IO error, etc.).
    /// WAV is NOT on disk.  Re-run the full job when the server is healthy.
    Failed { reason: String },

    /// Chunk rejected before synthesis ever ran.
    /// Not a retry candidate — the input text itself is the problem.
    Skipped { reason: String },
}

// ─── ChunkRecord ──────────────────────────────────────────────────────────────

/// Outcome of processing one synthesis chunk.
/// Written in ordinal order; the full vec is rewritten after each chunk
/// so a partial run always leaves a valid file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChunkRecord {
    /// 1-based position in the synthesis run — matches `paragraph_NNN.wav`.
    pub ordinal: usize,

    /// WAV filename relative to `out_dir`, e.g. `"paragraph_003.wav"`.
    /// Present on all records; file may not exist on disk for non-Ok status.
    pub wav: String,

    /// Silence gap that follows this chunk in the merged output.
    /// `None` for the final chunk.
    pub trailing_boundary: Option<Boundary>,

    /// What actually happened when we tried to synthesize this chunk.
    pub status: ChunkStatus,
}

impl ChunkRecord {
    /// True if this record's WAV exists and is ready to include in a merge.
    pub fn is_mergeable(&self) -> bool {
        matches!(self.status, ChunkStatus::Ok)
    }

    /// True if this record is a candidate for `--fix-quarantine`.
    /// `Failed` chunks are excluded — they need a server fix, not a text fix.
    pub fn needs_retry(&self) -> bool {
        matches!(self.status, ChunkStatus::Quarantined { .. })
    }

    /// Extract the original text for re-synthesis.
    /// `None` for everything except `Quarantined`.
    pub fn quarantined_text(&self) -> Option<&str> {
        match &self.status {
            ChunkStatus::Quarantined { original_text } => Some(original_text),
            _ => None,
        }
    }
}

// ─── LlmCleanRecord ───────────────────────────────────────────────────────────

/// Metadata from an LLM text-cleanup pass, if one was run.
/// Placeholder shape — fields will grow when that feature is built.
/// Lives on `RunRecord` so chunk records are unaffected by its addition.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LlmCleanRecord {
    /// Model identifier, e.g. `"qwen2.5:7b"`.
    pub model:        String,
    pub words_before: usize,
    pub words_after:  usize,
    /// True if the model actually changed the text.
    pub changed:      bool,
}

// ─── RunRecord ────────────────────────────────────────────────────────────────

/// Top-level record written to `out_dir/run.json`.
///
/// Wraps the chunk vec in run-level context so future pipeline stages
/// have a natural home without restructuring chunk records.
/// `--merge-only` and `--fix-quarantine` deserialize this and work from
/// `chunks` — the other fields are invisible to those operations.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RunRecord {
    /// RFC 3339 timestamp of when this run started.
    pub timestamp:  String,

    /// Source text file as configured (relative path).
    pub input_file: String,

    /// Source PDF if `--rip-pdf` was used; `None` otherwise.
    pub pdf_ripped: Option<String>,

    /// LLM cleanup metadata if `--llm-clean` was run; `None` otherwise.
    pub llm_clean:  Option<LlmCleanRecord>,

    /// One record per chunk, written incrementally during synthesis.
    pub chunks:     Vec<ChunkRecord>,
}

impl RunRecord {
    pub fn new(input_file: String, pdf_ripped: Option<String>) -> Self {
        RunRecord {
            timestamp:  Utc::now().to_rfc3339(),
            input_file,
            pdf_ripped,
            llm_clean:  None,
            chunks:     Vec::new(),
        }
    }

    /// Persist the current state of this record to `run.json` in `out_dir`.
    /// Rewrites the full file — called after each chunk so partial runs
    /// leave a valid file.
    ///
    /// Atomicity: written to `run.json.tmp` then renamed over `run.json`.
    /// Rename within one directory is atomic on every platform we support
    /// (Rust's `fs::rename` replaces an existing destination on Windows too),
    /// so a crash or power loss mid-write can never leave a truncated
    /// `run.json` — at worst it leaves a stale `.tmp` beside a good file.
    pub fn save(&self, out_dir: &std::path::Path) -> anyhow::Result<()> {
        let path = out_dir.join("run.json");
        let tmp  = out_dir.join("run.json.tmp");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Load a previous run record from `out_dir/run.json`.
    pub fn load(out_dir: &std::path::Path) -> anyhow::Result<Self> {
        let path = out_dir.join("run.json");
        let raw  = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&raw)?)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trips the atomic save path twice: the second save exercises
    /// rename-over-existing-destination, which is the Windows-sensitive case.
    #[test]
    fn save_load_roundtrip_and_overwrite() {
        let dir = std::env::temp_dir().join(format!(
            "lecturner_records_test_{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let mut rec = RunRecord::new("text.txt".into(), None);
        rec.chunks.push(ChunkRecord {
            ordinal: 1,
            wav: "paragraph_001.wav".into(),
            trailing_boundary: Some(Boundary::Paragraph),
            status: ChunkStatus::Ok,
        });
        rec.save(&dir).unwrap();

        rec.chunks.push(ChunkRecord {
            ordinal: 2,
            wav: "paragraph_002.wav".into(),
            trailing_boundary: None,
            status: ChunkStatus::Quarantined { original_text: "oops".into() },
        });
        rec.save(&dir).unwrap(); // overwrite via rename

        let loaded = RunRecord::load(&dir).unwrap();
        assert_eq!(loaded.chunks.len(), 2);
        assert_eq!(loaded.chunks[1].quarantined_text(), Some("oops"));
        assert!(!dir.join("run.json.tmp").exists(), "tmp file should not linger");

        let _ = std::fs::remove_dir_all(&dir);
    }
}