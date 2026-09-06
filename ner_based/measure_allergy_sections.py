#!/usr/bin/env python
"""Measure how often MEDICATION spans sit under an allergy header.

Decides one thing: whether the medspaCy sectionizer earns its place for the 17
MEDICATION features. A drug named in an allergy list is a drug the patient did
NOT receive, so a span there is a false positive for any medication feature.

Reads the verified pre-annotation JSONs only -- no model, no GPU, no pipeline
run. The verified JSONs already carry the note text and per-span char offsets.

RUN ON A MINERVA COMPUTE NODE. The JSONs contain note text (PHI). Only the
aggregate table printed at the end is safe to copy out; --dump-samples writes
note text and must stay on the enclave filesystem.

Usage:
    python measure_allergy_sections.py --glob 'annotations/batch*/verified/*.json'
    python measure_allergy_sections.py --dump-samples /sc/arion/work/.../allergy_samples.txt
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict

# The top-level source-text key, tried in order. Verified JSONs are documented
# as note_id / patient_id / source text / spans, but the key name is checked at
# runtime rather than assumed -- see the offset self-check below.
TEXT_KEYS = ("text", "note_text", "source_text", "NOTE_TEXT", "note", "source")
ID_KEYS = ("note_id", "NOTE_ID", "id")
PATIENT_KEYS = ("patient_id", "PATIENT_ID", "mrn")

# Loose: the word appears anywhere in the lookback window.
ALLERGY = re.compile(r"\ballerg(?:y|ies|ic)\b", re.I)

# Header-anchored: at the start of the note or after a flattened line break
# (3+ spaces -- the AIR.MS export has no newlines), optionally preceded by a
# qualifier ("Drug Allergies:", "Allergies / Intolerances:").
ALLERGY_HDR = re.compile(
    r"(?:\A|\s{3,})(?:[A-Za-z/&]+[ ]){0,2}allerg(?:y|ies)\b[ ]*:?",
    re.I,
)

# A subsequent header closes the allergy region: flattened line break followed
# by a capitalised label and a colon.
NEXT_HDR = re.compile(
    # capitalised label + colon, optionally numbered ("12. Local Anesthesia Given:")
    r"\s{3,}(?:\d{1,2}[.)][ ]*)?[A-Z][A-Za-z0-9/&'-]{1,30}(?:[ ][A-Za-z0-9/&'-]{1,30}){0,5}:"
    # medication-list headers carry no colon in the Epic export
    r"|\s{3,}\S[^\r\n]{0,60}?[Mm]edications?\b"
)

# The benign case seen in the gold pilot: an allergy header with no drug named.
NO_KNOWN = re.compile(
    r"\b(nkda|nkfa|no known (drug |food )?allerg|denies allerg|"
    r"none known|no allerg|no reported allerg)",
    re.I,
)

WINDOWS = (60, 100, 150, 300, 600)


def first_key(obj, keys):
    for k in keys:
        if k in obj and isinstance(obj[k], (str, int)):
            return obj[k]
    return None


def load_note(path):
    with open(path) as fh:
        obj = json.load(fh)
    text = None
    for k in TEXT_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            text = v
            break
    return obj, text


def governed_by_header(text, start, headers, window):
    """True if an allergy header opens a region that still contains `start`."""
    best = None
    for h in headers:
        if h.end() <= start:
            best = h
        else:
            break
    if best is None:
        return None
    if start - best.end() > window:
        return None
    if NEXT_HDR.search(text, best.end(), start):
        return None
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="annotations/batch*/verified/*.json")
    ap.add_argument("--label", default="MEDICATION")
    ap.add_argument("--dump-samples", default=None,
                    help="write up to --n-samples context snippets here (PHI: keep on Minerva)")
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--report-window", type=int, default=300,
                    help="which window drives the headline number and samples")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        sys.exit(f"no files matched {args.glob!r}")

    notes = spans_total = 0
    no_text = no_spans = offset_mismatch = 0
    notes_with_hdr = notes_hdr_no_known = 0

    loose = {w: 0 for w in WINDOWS}
    header = {w: 0 for w in WINDOWS}
    hit_notes = defaultdict(set)
    hit_patients = defaultdict(set)
    all_notes, all_patients = set(), set()
    drug_counter = Counter()
    samples = []

    for path in files:
        obj, text = load_note(path)
        if text is None:
            no_text += 1
            continue
        spans = obj.get("spans") or []
        if not spans:
            no_spans += 1
        notes += 1
        nid = first_key(obj, ID_KEYS) or path
        pid = first_key(obj, PATIENT_KEYS)
        all_notes.add(nid)
        if pid is not None:
            all_patients.add(pid)

        headers = list(ALLERGY_HDR.finditer(text))
        if headers:
            notes_with_hdr += 1
            if any(NO_KNOWN.search(text[h.end():h.end() + 120]) for h in headers):
                notes_hdr_no_known += 1

        for s in spans:
            if s.get("label") != args.label:
                continue
            spans_total += 1
            st, en = s.get("start_char"), s.get("end_char")
            if st is None or text[st:en] != s.get("text"):
                offset_mismatch += 1
                continue

            for w in WINDOWS:
                if ALLERGY.search(text, max(0, st - w), st):
                    loose[w] += 1
                h = governed_by_header(text, st, headers, w)
                if h is not None:
                    header[w] += 1
                    hit_notes[w].add(nid)
                    if pid is not None:
                        hit_patients[w].add(pid)
                    if w == args.report_window:
                        drug_counter[s["text"].lower()] += 1
                        if args.dump_samples and len(samples) < args.n_samples:
                            samples.append(
                                f"[{nid}] ...{text[max(0, h.start()):st]}"
                                f"<<{s['text']}>>{text[en:en + 60]}...\n"
                            )

    pct = lambda n, d: f"{100.0 * n / d:6.2f}%" if d else "   n/a"
    W = args.report_window

    print(f"\nfiles matched            {len(files):>8}")
    print(f"notes read               {notes:>8}"
          f"   (skipped: no text {no_text}, empty spans {no_spans})")
    print(f"{args.label} spans        {spans_total:>8}"
          f"   (offset mismatches: {offset_mismatch})")
    print(f"notes with allergy header{notes_with_hdr:>8}   {pct(notes_with_hdr, notes)}")
    print(f"  ...of which 'no known' {notes_hdr_no_known:>8}   {pct(notes_hdr_no_known, notes_with_hdr)}")

    print(f"\n{'window':>8} {'loose':>10} {'':>8} {'header-anchored':>16} {'':>8}")
    for w in WINDOWS:
        print(f"{w:>8} {loose[w]:>10} {pct(loose[w], spans_total)} "
              f"{header[w]:>16} {pct(header[w], spans_total)}")

    print(f"\n--- headline (window={W}, header-anchored) ---")
    print(f"affected {args.label} spans  {header[W]:>8}   {pct(header[W], spans_total)}")
    print(f"affected notes           {len(hit_notes[W]):>8}   {pct(len(hit_notes[W]), len(all_notes))}")
    if all_patients:
        print(f"affected patients        {len(hit_patients[W]):>8}"
              f"   {pct(len(hit_patients[W]), len(all_patients))} of {len(all_patients)}")

    if drug_counter:
        print(f"\ntop drugs named under an allergy header:")
        for name, n in drug_counter.most_common(25):
            print(f"  {n:>5}  {name}")

    if args.dump_samples and samples:
        with open(args.dump_samples, "w") as fh:
            fh.writelines(samples)
        print(f"\nwrote {len(samples)} context snippets to {args.dump_samples}  (PHI -- keep on Minerva)")


if __name__ == "__main__":
    main()
