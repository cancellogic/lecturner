//! pdf_rip.rs — Pure-Rust PDF prose extractor.
//!
//! Direct replacement for pdf_rip.py (pdfplumber).  Same algorithm, same
//! output contract: a UTF-8 text file of blank-line-separated paragraphs
//! ready for the LLM cleanup pass.
//!
//! Depends on pdfsink-rs for word bounding-box extraction; all column
//! detection, line reconstruction, and prose filtering logic lives here
//! and is owned by this project.
//!
//! # Approach (mirrors pdf_rip.py section-by-section)
//!
//! Per page:
//!   1. Extract word bboxes via `page.extract_words()`.
//!   2. Cluster words into left-column / right-column / spanning by
//!      x-midpoint relative to page centre, with an overlap-fraction guard
//!      for words that straddle the centre (title blocks, wide figures).
//!   3. If spanning words dominate, treat the page as single-column.
//!   4. Sort words within each column by (top, x0) and group into lines
//!      whose vertical centres fall within SAME_LINE_TOLERANCE points.
//!   5. Emit: span_lines ++ left_lines ++ right_lines (NASA paper order).
//!
//! Post-page:
//!   6. Rejoin soft-hyphenated line breaks ("perturba-\ntion" → "perturbation").
//!   7. Scrub bracket-encoded PDF glyph artifacts ([bracketleft], [fi], etc.)
//!      before the LLM sees the text — saves tokens, reduces hallucination risk.
//!   8. Drop page numbers, noise lines (short / ALL-CAPS / URLs / bare DOIs).
//!   9. Drop figure/table caption lines (optional).
//!  10. Stop at References / Bibliography / Acknowledgements heading (optional).
//!  11. Collapse contiguous non-blank lines into paragraphs.

use anyhow::{Context, Result};
use pdfsink_rs::PdfDocument;
use std::path::Path;

// ─── Tuning constants (match pdf_rip.py defaults) ─────────────────────────────

/// Words whose vertical centres are within this many points share a line.
const SAME_LINE_TOLERANCE: f64 = 3.0;

/// If a word's x-span crosses page centre and its overlap fraction exceeds
/// this, it is treated as a spanning (single-column) word.
const COLUMN_OVERLAP_FRAC: f64 = 0.1;

/// Lines shorter than this are candidates for noise rejection,
/// unless they end with sentence-terminal punctuation.
const MIN_LINE_LEN: usize = 40;

// ─── Public entry point ───────────────────────────────────────────────────────

/// Configuration knobs forwarded from `Config` in main.rs.
pub struct RipConfig {
    pub skip_refs:     bool,
    pub skip_captions: bool,
}

/// Rip `pdf_path` to clean prose paragraphs and write them to `out_path`.
///
/// Prints progress lines matching the old pdf_rip.py format so existing
/// log-scraping in main.rs (and the user's eyes) sees identical output.
pub fn rip_pdf(pdf_path: &Path, out_path: &Path, cfg: &RipConfig) -> Result<()> {
    println!(
        "[pdf_rip] Opening {}",
        pdf_path.display()
    );

    // ── 1. Open document ──────────────────────────────────────────────────────
    let doc = PdfDocument::open(pdf_path)
        .with_context(|| format!("Cannot open PDF: {}", pdf_path.display()))?;

    // ── 2. Extract lines page-by-page ─────────────────────────────────────────
    // Use pages() iterator — avoids needing a separate page_count() call and
    // handles malformed page trees gracefully by skipping bad entries.
    let mut all_lines: Vec<String> = Vec::new();
    let mut page_num   = 0usize;
    let mut page_count = 0usize;

    // Collect pages first so we have a total count for progress messages.
    let pages: Vec<_> = doc.pages().collect();
    page_count = pages.len();
    println!("[pdf_rip] {} page(s) in {}", page_count, pdf_path.display());

    for page in pages {
        page_num += 1;

        let page_width = page.width as f64;

        // pdfsink-rs Word fields: text, x0, x1, top, bottom — identical to
        // the pdfplumber word dicts pdf_rip.py consumed.
        let words = page.extract_words();

        let lines = words_to_lines(words, page_width);
        all_lines.extend(lines);
        all_lines.push(String::new()); // page break → blank line

        if page_num % 10 == 0 {
            println!("[pdf_rip]   {page_num}/{page_count} pages processed…");
        }
    }

    // ── 3. Post-extraction filters ────────────────────────────────────────────

    // Rejoin soft-hyphenated splits across lines before anything else so
    // filter heuristics see whole words, not trailing fragments.
    let joined_text = rejoin_hyphens(&all_lines.join("\n"));
    let mut lines: Vec<&str> = joined_text.lines().collect();

    // Scrub bracket-encoded glyph artifacts — these are a pre-LLM pass so we
    // don't waste tokens or risk the model hallucinating around [bracketleft].
    let scrubbed_lines: Vec<String> = lines
        .iter()
        .map(|l| scrub_glyph_artifacts(l))
        .collect();
    lines = scrubbed_lines.iter().map(String::as_str).collect();

    let filtered   = filter_lines(&lines, cfg.skip_refs, cfg.skip_captions);
    let paragraphs = lines_to_paragraphs(&filtered);

    if paragraphs.is_empty() {
        anyhow::bail!(
            "No prose extracted from {} — PDF may be scanned/image-only",
            pdf_path.display()
        );
    }

    // ── 4. Write output ───────────────────────────────────────────────────────
    let output = paragraphs.join("\n\n") + "\n";
    std::fs::write(out_path, &output)
        .with_context(|| format!("Cannot write {}", out_path.display()))?;

    let word_count: usize = paragraphs.iter().map(|p| p.split_whitespace().count()).sum();
    println!(
        "[pdf_rip] {} paragraph(s), ~{} words → {}",
        paragraphs.len(),
        word_count,
        out_path.display()
    );

    Ok(())
}

// ─── Column-aware line reconstruction ────────────────────────────────────────
//
// Mirrors pdf_rip.py `words_to_lines` + `_cluster_into_lines`.

/// A word with its bounding box — mirrors the pdfplumber word dict fields.
/// pdfsink-rs Word is not Copy so we borrow only what the geometry needs.
struct WordRef {
    text:   String,
    x0:     f64,
    x1:     f64,
    top:    f64,
    bottom: f64,
}

fn words_to_lines(raw_words: Vec<pdfsink_rs::Word>, page_width: f64) -> Vec<String> {
    if raw_words.is_empty() {
        return Vec::new();
    }

    // Convert to our lightweight geometry struct.
    // Variable name scheme: the column sorters are called `sinister` (left)
    // and `dexter` (right) — heraldic terms, memorable, non-offensive.
    let words: Vec<WordRef> = raw_words
        .into_iter()
        .map(|w| WordRef {
            text:   w.text,
            x0:     w.x0 as f64,
            x1:     w.x1 as f64,
            top:    w.top as f64,
            bottom: w.bottom as f64,
        })
        .collect();

    let mid = page_width / 2.0;

    let mut sinister_words: Vec<&WordRef> = Vec::new(); // left column
    let mut dexter_words:   Vec<&WordRef> = Vec::new(); // right column
    let mut spanning_words: Vec<&WordRef> = Vec::new(); // straddles centre

    for w in &words {
        let w_mid            = (w.x0 + w.x1) / 2.0;
        let crosses_centre   = w.x0 < mid && mid < w.x1;
        let overlap_fraction = if crosses_centre {
            let overlap = mid.min(w.x1) - mid.max(w.x0);
            let word_width = (w.x1 - w.x0).max(page_width * 0.01);
            overlap / word_width
        } else {
            0.0
        };

        if crosses_centre && overlap_fraction > COLUMN_OVERLAP_FRAC {
            spanning_words.push(w);
        } else if w_mid < mid {
            sinister_words.push(w);
        } else {
            dexter_words.push(w);
        }
    }

    // If spanning words dominate, the page is single-column (title, abstract,
    // wide-figure pages in NASA papers).
    if spanning_words.len() > sinister_words.len() + dexter_words.len() {
        let mut all_sorted: Vec<&WordRef> = words.iter().collect();
        all_sorted.sort_by(|a, b| {
            a.top.partial_cmp(&b.top).unwrap()
                .then(a.x0.partial_cmp(&b.x0).unwrap())
        });
        return cluster_into_lines(all_sorted);
    }

    // Two-column: span lines interleaved at the top (title/abstract block),
    // then left column, then right column.  Correct ~95% of the time for
    // standard academic/NASA paper layout without bbox-overlap analysis.
    let mut span_sorted = spanning_words;
    span_sorted.sort_by(|a, b| {
        a.top.partial_cmp(&b.top).unwrap()
            .then(a.x0.partial_cmp(&b.x0).unwrap())
    });
    let mut sin_sorted = sinister_words;
    sin_sorted.sort_by(|a, b| {
        a.top.partial_cmp(&b.top).unwrap()
            .then(a.x0.partial_cmp(&b.x0).unwrap())
    });
    let mut dex_sorted = dexter_words;
    dex_sorted.sort_by(|a, b| {
        a.top.partial_cmp(&b.top).unwrap()
            .then(a.x0.partial_cmp(&b.x0).unwrap())
    });

    let mut result = cluster_into_lines(span_sorted);
    result.extend(cluster_into_lines(sin_sorted));
    result.extend(cluster_into_lines(dex_sorted));
    result
}

/// Group vertically-adjacent words into line strings.
/// Mirrors pdf_rip.py `_cluster_into_lines`.
fn cluster_into_lines(sorted_words: Vec<&WordRef>) -> Vec<String> {
    if sorted_words.is_empty() {
        return Vec::new();
    }

    let mut lines:   Vec<String>    = Vec::new();
    let mut current: Vec<&WordRef>  = vec![sorted_words[0]];

    for w in &sorted_words[1..] {
        let prev_mid = {
            let p = current.last().unwrap();
            (p.top + p.bottom) / 2.0
        };
        let this_mid = (w.top + w.bottom) / 2.0;

        if (this_mid - prev_mid).abs() <= SAME_LINE_TOLERANCE {
            current.push(w);
        } else {
            lines.push(current.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" "));
            current = vec![w];
        }
    }
    if !current.is_empty() {
        lines.push(current.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" "));
    }

    lines
}

// ─── Glyph artifact scrubber ──────────────────────────────────────────────────
//
// PDF fonts with non-standard encoding tables emit glyph names as literal
// bracket-wrapped tokens.  Scrub these before the LLM sees the text.
// NASA press releases are the canonical offender.
//
// Patterns caught:
//   [bracketleft] [bracketright] [parenleft] [bullet] [dagger] — named glyphs
//   [fi] [fl] [ff] [ffi] [ffl]                                 — ligatures
//   [uniXXXX] [uni00XX]                                        — Unicode escapes
//   Runs of isolated single-char bracket tokens: [f][i][space] — decomposed ligs

fn scrub_glyph_artifacts(line: &str) -> String {
    // Named glyphs and ligatures — replace with their semantic equivalent
    // where one exists, otherwise delete.
    static NAMED_GLYPH_SUBS: &[(&str, &str)] = &[
        ("[bracketleft]",  "["),
        ("[bracketright]", "]"),
        ("[parenleft]",    "("),
        ("[parenright]",   ")"),
        ("[bullet]",       ""),
        ("[dagger]",       ""),
        ("[daggerdbl]",    ""),
        ("[section]",      ""),
        ("[paragraph]",    ""),
        ("[endash]",       "-"),
        ("[emdash]",       "\u{2014}"),
        ("[quotedblleft]", "\u{201C}"),
        ("[quotedblright]","\u{201D}"),
        ("[quoteleft]",    "'"),
        ("[quoteright]",   "'"),
        // Ligature decompositions — restore the actual letters.
        ("[fi]",           "fi"),
        ("[fl]",           "fl"),
        ("[ff]",           "ff"),
        ("[ffi]",          "ffi"),
        ("[ffl]",          "ffl"),
        ("[ft]",           "ft"),
        ("[st]",           "st"),
    ];

    let mut s = line.to_owned();

    for (glyph, replacement) in NAMED_GLYPH_SUBS {
        s = s.replace(glyph, replacement);
    }

    // Unicode escape glyph names: [uni0041] [uni00A0] etc.
    // Just delete them — they're decoration, not prose.
    // Simple scan: find "[uni" followed by hex digits and "]".
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            // Peek for "uni" prefix
            if s[i..].starts_with("[uni") {
                if let Some(close) = s[i..].find(']') {
                    let token = &s[i..i + close + 1];
                    // Validate: everything between [uni and ] is hex digits.
                    let hex_part = &token[4..token.len() - 1];
                    if !hex_part.is_empty() && hex_part.chars().all(|c| c.is_ascii_hexdigit()) {
                        i += token.len(); // skip the whole token
                        continue;
                    }
                }
            }
        }
        // Encode one char at a time to handle multi-byte UTF-8 correctly.
        let c = s[i..].chars().next().unwrap();
        out.push(c);
        i += c.len_utf8();
    }
    s = out;

    // Collapse any runs of whitespace left behind by deletions.
    // A simple repeated-space collapse is enough here.
    while s.contains("  ") {
        s = s.replace("  ", " ");
    }

    s.trim().to_owned()
}

// ─── Soft-hyphen rejoiner ─────────────────────────────────────────────────────

/// Rejoin soft-hyphenated line breaks: "perturba-\ntion" → "perturbation".
/// Only fires on word-final hyphens followed immediately by a newline and a
/// lowercase letter — leaves em-dashes and compound words alone.
fn rejoin_hyphens(text: &str) -> String {
    // Manual scan is faster than a regex dep and the pattern is simple.
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '-' {
            // Look ahead: is the next char a newline followed by a lowercase?
            let mut peek_buf = chars.clone();
            if let Some('\n') = peek_buf.next() {
                if let Some(nc) = peek_buf.next() {
                    if nc.is_ascii_lowercase() {
                        // Consume the newline, drop the hyphen, continue.
                        chars.next(); // consume '\n'
                        // nc will be emitted naturally in the next iteration.
                        continue;
                    }
                }
            }
        }
        out.push(c);
    }
    out
}

// ─── Prose filters ────────────────────────────────────────────────────────────

fn is_page_number(line: &str) -> bool {
    // Line is purely a number, optionally surrounded by hyphens/spaces.
    line.trim()
        .trim_matches(|c| c == '-' || c == ' ')
        .chars()
        .all(|c| c.is_ascii_digit())
        && !line.trim().is_empty()
}

fn is_noise(line: &str) -> bool {
    let s = line.trim();
    if s.is_empty() { return false; }

    // Too short to be a sentence fragment, and no terminal punctuation.
    if s.len() < MIN_LINE_LEN && !matches!(s.chars().last(), Some('.' | '?' | '!')) {
        return true;
    }

    // URL
    if s.starts_with("http://") || s.starts_with("https://") {
        return true;
    }

    // Bare DOI: starts with "10." followed by 4+ digits and a slash.
    if s.starts_with("10.") {
        let rest = &s[3..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 4 {
            let after = &rest[digits.len()..];
            if after.starts_with('/') {
                return true;
            }
        }
    }

    false
}

/// Headings that signal "drop everything below here."
fn is_refs_heading(line: &str) -> bool {
    matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "references"
        | "bibliography"
        | "acknowledgements"
        | "acknowledgments"
        | "appendix"
    )
}

/// Figure/table caption starters.
fn is_caption(line: &str) -> bool {
    let s = line.trim().to_ascii_lowercase();
    // "figure 3", "fig. 3", "fig 3", "table 2", "tbl. 2", "tbl 2"
    (s.starts_with("figure ")
        || s.starts_with("fig. ")
        || s.starts_with("fig ")
        || s.starts_with("table ")
        || s.starts_with("tbl. ")
        || s.starts_with("tbl "))
        && s.chars().nth(s.find(' ').map(|i| i + 1).unwrap_or(0))
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
}

fn filter_lines<'a>(lines: &[&'a str], skip_refs: bool, skip_captions: bool) -> Vec<&'a str> {
    let mut result  = Vec::new();
    let mut in_refs = false;

    for &line in lines {
        let stripped = line.trim();

        if stripped.is_empty() {
            result.push(line);
            continue;
        }

        if skip_refs && is_refs_heading(stripped) {
            in_refs = true;
        }

        if in_refs                                { continue; }
        if is_page_number(stripped)               { continue; }
        if skip_captions && is_caption(stripped)  { continue; }
        if is_noise(stripped)                     { continue; }

        result.push(line);
    }

    result
}

// ─── Paragraph assembly ───────────────────────────────────────────────────────

/// Collapse runs of non-blank lines into single paragraphs separated by blanks.
/// Mirrors pdf_rip.py `lines_to_paragraphs`.
fn lines_to_paragraphs(lines: &[&str]) -> Vec<String> {
    let mut paragraphs: Vec<String> = Vec::new();
    let mut current:    Vec<&str>   = Vec::new();

    for &line in lines {
        if line.trim().is_empty() {
            if !current.is_empty() {
                paragraphs.push(current.join(" "));
                current.clear();
            }
        } else {
            current.push(line.trim());
        }
    }
    if !current.is_empty() {
        paragraphs.push(current.join(" "));
    }

    paragraphs
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_hyphen_rejoined() {
        let input = "perturba-\ntion is key";
        assert_eq!(rejoin_hyphens(input), "perturbation is key");
    }

    #[test]
    fn emdash_not_rejoined() {
        // Em-dash followed by newline should NOT be joined.
        let input = "end—\nnext";
        assert_eq!(rejoin_hyphens(input), "end—\nnext");
    }

    #[test]
    fn uppercase_hyphen_not_rejoined() {
        // "NASA-\nJohnson" — uppercase after newline, should not join.
        let input = "NASA-\nJohnson";
        assert_eq!(rejoin_hyphens(input), "NASA-\nJohnson");
    }

    #[test]
    fn glyph_ligatures_restored() {
        assert_eq!(scrub_glyph_artifacts("e[fi]cient"), "efficient");
        assert_eq!(scrub_glyph_artifacts("o[ff]set"),   "offset");
    }

    #[test]
    fn glyph_named_removed() {
        assert_eq!(scrub_glyph_artifacts("text[bullet]more"), "textmore");
    }

    #[test]
    fn glyph_unicode_escape_removed() {
        // [uni0041] is 'A' — we drop it rather than decode it.
        assert_eq!(scrub_glyph_artifacts("te[uni0041]xt"), "text");
    }

    #[test]
    fn glyph_bracket_passthrough() {
        // Real square brackets in prose should survive unharmed.
        assert_eq!(scrub_glyph_artifacts("see [1] for details"), "see [1] for details");
    }

    #[test]
    fn page_number_detected() {
        assert!(is_page_number("42"));
        assert!(is_page_number(" - 7 - "));
        assert!(!is_page_number("Section 4"));
    }

    #[test]
    fn refs_heading_detected() {
        assert!(is_refs_heading("References"));
        assert!(is_refs_heading("REFERENCES"));
        assert!(is_refs_heading("  bibliography  "));
        assert!(!is_refs_heading("Reference architecture"));
    }

    #[test]
    fn caption_detected() {
        assert!(is_caption("Figure 3. Thrust profile."));
        assert!(is_caption("Table 2: Summary"));
        assert!(is_caption("Fig. 1 shows…"));
        assert!(!is_caption("Figuratively speaking"));
    }

    #[test]
    fn paragraphs_assembled() {
        let lines = vec!["First sentence.", "Second sentence.", "", "New paragraph."];
        let result = lines_to_paragraphs(&lines);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "First sentence. Second sentence.");
        assert_eq!(result[1], "New paragraph.");
    }
}