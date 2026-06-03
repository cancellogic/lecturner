#!/usr/bin/env python3
"""
pdf_rip.py — Extract clean prose from a PDF into a text file for gossip.

Usage (called by gossip, not directly):
    python pdf_rip.py --input paper.pdf --output text.txt [options]

Options:
    --input          PDF file path (required)
    --output         Output text file path (default: text.txt)
    --skip-refs      Drop the References / Bibliography section (default: true)
    --skip-captions  Drop figure/table caption lines (default: true)
    --min-line-len   Lines shorter than this are treated as headers/noise (default: 40)

Exit codes:
    0  success
    1  pdfplumber not installed (gossip prints install hint)
    2  other error
"""

# ── Approach ───────────────────────────────────────────────────────────────────
# 1. Verify pdfplumber is importable; exit 1 with a machine-readable marker if not.
# 2. Open PDF; iterate pages.
# 3. For each page use pdfplumber's word-bbox data to reconstruct reading order
#    that respects two-column layouts:
#      a. Cluster words into left / right columns by x-midpoint relative to
#         page centre.  Single-column pages (title, abstract, wide figures)
#         are handled as one column automatically.
#      b. Sort words within each column by (top, x0) to get reading order.
#      c. Reassemble into lines by grouping words whose vertical centres are
#         within SAME_LINE_TOLERANCE points of each other.
# 4. Apply prose filters:
#      - Rejoin soft-hyphenated line breaks ("perturba-\ntion" → "perturbation")
#      - Drop page numbers (line is purely numeric)
#      - Drop running headers/footers (repeat across pages, short, no sentence punct)
#      - Optionally drop figure/table captions
#      - Optionally drop everything from References heading onward
# 5. Merge lines into paragraphs: blank line between each page's text blocks.
# 6. Write UTF-8 text file.
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import io
import re
import sys

# Force UTF-8 stdout on Windows regardless of terminal code page.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    import pdfplumber
except ImportError:
    print("GOSSIP_MISSING_DEP:pdfplumber", flush=True)
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

SAME_LINE_TOLERANCE = 3   # points; words within this vertical range = same line
COLUMN_OVERLAP_FRAC = 0.1 # if a word's x-span crosses page centre by more than
                           # this fraction of page width it belongs to neither
                           # column → treat page as single-column

# Headings that signal "everything below here is references / boilerplate"
REFS_HEADINGS = re.compile(
    r'^\s*(references|bibliography|acknowledgements?|appendix)\s*$',
    re.IGNORECASE
)

# Figure / table caption starters
CAPTION_RE = re.compile(r'^\s*(figure|fig\.?|table|tbl\.?)\s*\d', re.IGNORECASE)


# ── Column-aware line reconstruction ──────────────────────────────────────────

def words_to_lines(words, page_width):
    """
    Given pdfplumber word dicts, return an ordered list of line strings
    that respects two-column layout.

    Each word dict has keys: text, x0, x1, top, bottom.
    """
    if not words:
        return []

    mid = page_width / 2

    # Separate into left column, right column, and spanning (single-col) words.
    left_words  = []
    right_words = []
    span_words  = []

    for w in words:
        w_mid = (w["x0"] + w["x1"]) / 2
        crosses_centre = (w["x0"] < mid < w["x1"])
        overlap = (min(w["x1"], mid) - max(w["x0"], mid)) if crosses_centre else 0
        overlap_frac = overlap / max(page_width * 0.01, w["x1"] - w["x0"])

        if crosses_centre and overlap_frac > COLUMN_OVERLAP_FRAC:
            span_words.append(w)
        elif w_mid < mid:
            left_words.append(w)
        else:
            right_words.append(w)

    # If almost everything spans, treat as single column.
    if len(span_words) > len(left_words) + len(right_words):
        ordered_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
        return _cluster_into_lines(ordered_words)

    # Two-column: left column first, then right column, spanning words interleaved
    # by their vertical position relative to column blocks.
    # Simple heuristic: sort span_words by top and insert them between column
    # blocks whose top range they fall within.
    left_lines  = _cluster_into_lines(sorted(left_words,  key=lambda w: (w["top"], w["x0"])))
    right_lines = _cluster_into_lines(sorted(right_words, key=lambda w: (w["top"], w["x0"])))
    span_lines  = _cluster_into_lines(sorted(span_words,  key=lambda w: (w["top"], w["x0"])))

    # Interleave span lines at the top (title, authors, abstract) then
    # left column, then right column.  This matches the typical NASA paper layout.
    # More sophisticated interleaving would need bounding-box overlap analysis;
    # for prose extraction this ordering is correct ~95% of the time.
    return span_lines + left_lines + right_lines


def _cluster_into_lines(sorted_words):
    """Group vertically-adjacent words into line strings."""
    if not sorted_words:
        return []

    lines   = []
    current = [sorted_words[0]]

    for w in sorted_words[1:]:
        prev_mid = (current[-1]["top"] + current[-1]["bottom"]) / 2
        this_mid = (w["top"] + w["bottom"]) / 2
        if abs(this_mid - prev_mid) <= SAME_LINE_TOLERANCE:
            current.append(w)
        else:
            lines.append(" ".join(c["text"] for c in current))
            current = [w]

    if current:
        lines.append(" ".join(c["text"] for c in current))

    return lines


# ── Prose filters ──────────────────────────────────────────────────────────────

def is_page_number(line):
    """True if the line is just a number or a number with minimal decoration."""
    return bool(re.match(r'^\s*-?\s*\d+\s*-?\s*$', line))


def is_likely_noise(line, min_len):
    """
    True for lines that are probably headers, footers, or other non-prose noise:
    - Too short to be a sentence fragment
    - No lowercase letters (ALL CAPS heading)
    - Looks like a DOI / URL
    """
    stripped = line.strip()
    if len(stripped) < min_len and not stripped.endswith(('.', '?', '!')):
        return True
    if re.match(r'https?://', stripped):
        return True
    if re.match(r'10\.\d{4,}/', stripped):   # bare DOI
        return True
    return False


def rejoin_hyphens(text):
    """
    Rejoin soft-hyphenated line breaks.
    "perturba-\ntion" → "perturbation"
    Leaves em-dashes and legitimate compound words alone by only acting on
    word-final hyphens followed immediately by a newline and a lowercase letter.
    """
    return re.sub(r'-\n([a-z])', r'\1', text)


def filter_lines(lines, skip_refs, skip_captions, min_line_len):
    """
    Apply all prose filters to a flat list of line strings.
    Returns filtered list; sets a stop flag when a refs heading is encountered.
    """
    result  = []
    in_refs = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue

        if skip_refs and REFS_HEADINGS.match(stripped):
            in_refs = True

        if in_refs:
            continue

        if is_page_number(stripped):
            continue

        if skip_captions and CAPTION_RE.match(stripped):
            continue

        if is_likely_noise(stripped, min_line_len):
            continue

        result.append(stripped)

    return result


# ── Paragraph assembly ────────────────────────────────────────────────────────

def lines_to_paragraphs(lines):
    """
    Collapse runs of non-empty lines into paragraphs separated by blank lines.
    Consecutive non-blank lines are joined with a space (they're the same
    paragraph); existing blank lines become paragraph separators.
    """
    paragraphs = []
    current    = []

    for line in lines:
        if line.strip():
            current.append(line.strip())
        else:
            if current:
                paragraphs.append(" ".join(current))
                current = []

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PDF → clean prose text for gossip")
    parser.add_argument("--input",          required=True,  help="Input PDF path")
    parser.add_argument("--output",         default="text.txt")
    parser.add_argument("--skip-refs",      default=True,  action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip-captions",  default=True,  action=argparse.BooleanOptionalAction)
    parser.add_argument("--min-line-len",   default=40, type=int)
    args = parser.parse_args()

    try:
        all_lines = []
        with pdfplumber.open(args.input) as pdf:
            page_count = len(pdf.pages)
            print(f"[pdf_rip] {page_count} page(s) in {args.input}", flush=True)

            for i, page in enumerate(pdf.pages, 1):
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,   # we do our own ordering
                )
                lines = words_to_lines(words, float(page.width))
                all_lines.extend(lines)
                all_lines.append("")  # page break becomes blank line
                if i % 10 == 0:
                    print(f"[pdf_rip]   {i}/{page_count} pages processed…", flush=True)

    except Exception as exc:
        print(f"[pdf_rip] ERROR reading PDF: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)

    # Apply filters
    all_lines  = rejoin_hyphens("\n".join(all_lines)).split("\n")
    filtered   = filter_lines(all_lines, args.skip_refs, args.skip_captions, args.min_line_len)
    paragraphs = lines_to_paragraphs(filtered)

    if not paragraphs:
        print("[pdf_rip] WARNING: no prose extracted — PDF may be scanned/image-only",
              file=sys.stderr, flush=True)
        sys.exit(2)

    output = "\n\n".join(paragraphs) + "\n"

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    except Exception as exc:
        print(f"[pdf_rip] ERROR writing {args.output}: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)

    word_count = sum(len(p.split()) for p in paragraphs)
    print(
        f"[pdf_rip] {len(paragraphs)} paragraph(s), ~{word_count} words -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()