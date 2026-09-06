#!/usr/bin/env python
"""Draw a blind adjudication sample for the airms_allergy component.

Gold cannot measure this axis: allergy documentation is patient-level, so a
four-patient gold set caps it at a handful of instances regardless of how many
notes are annotated. This samples the corpus instead.

Two samples:

  precision -- MEDICATION spans the component flagged is_allergy, stratified
               across the two firing mechanisms (header region, inline cue),
               which fail differently.
  recall    -- notes containing an allergy header where the component flagged
               nothing. Reading these surfaces terminator and scope failures.

Both are written WITHOUT the component's verdict, so adjudication is blind.
Mark the `judgment` column yourself, then join with the key file.

Usage:
  python sample_allergy.py --roots annotations/batch01_v2/verified \\
      annotations/batch02/verified annotations/batch03/verified \\
      --model models/ner_full/model-best --n-precision 40 --n-recall 20

Output (all PHI -- keep on Minerva):
  allergy_precision_sample.csv   adjudicate this: fill `judgment`
  allergy_recall_sample.csv      adjudicate this: fill `n_missed`
  allergy_sample_key.csv         the withheld verdicts; do not open first
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))
from src.ner.allergy import find_allergy_headers  # noqa: E402
from src.ner.assertion import build_assertion_pipeline  # noqa: E402

CONTEXT = 300


def note_text(obj):
    best = ""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            if len(cur) > len(best):
                best = cur
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return best


def window(text, start, end, n=CONTEXT):
    a, b = max(0, start - n), min(len(text), end + n)
    return re.sub(r"\s+", " ", text[a:start]).strip(), \
           text[start:end], \
           re.sub(r"\s+", " ", text[end:b]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--model", default="models/ner_full/model-best")
    ap.add_argument("--n-precision", type=int, default=40)
    ap.add_argument("--n-recall", type=int, default=20)
    ap.add_argument("--max-notes", type=int, default=1200,
                    help="cap on notes scanned; the corpus is 20,937")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    files = []
    for root in args.roots:
        files.extend(glob.glob(f"{root}/*.json"))
    files = sorted(set(files))
    rng.shuffle(files)
    files = files[: args.max_notes]
    print(f"scanning {len(files)} notes from {len(args.roots)} directories")

    nlp = build_assertion_pipeline(args.model)

    flagged, unflagged_with_header = [], []
    for i, f in enumerate(files, 1):
        if i % 200 == 0:
            print(f"  {i}/{len(files)}")
        note = os.path.basename(f)[:-5]
        try:
            text = note_text(json.load(open(f)))
        except Exception:
            continue
        if not text:
            continue
        doc = nlp(text)

        hits = [e for e in doc.ents if getattr(e._, "is_allergy", False)]
        for e in hits:
            flagged.append({
                "note_id": note,
                "span": e.text,
                "start": e.start_char,
                "end": e.end_char,
                "source": getattr(e._, "allergy_source", ""),
            })
        if not hits and find_allergy_headers(text):
            unflagged_with_header.append({"note_id": note, "n_meds": sum(
                1 for e in doc.ents if e.label_ == "MEDICATION")})

    print(f"\nflagged spans found: {len(flagged)}")
    print(f"notes with an allergy header and no flag: {len(unflagged_with_header)}")

    # stratify the precision sample across firing mechanisms
    by_source = {}
    for r in flagged:
        by_source.setdefault(r["source"], []).append(r)
    print("by mechanism:", {k: len(v) for k, v in by_source.items()})

    # Take every span from the rare mechanisms, top up from the common one:
    # 'cue' and 'header+cue' are where the component is most likely to fail,
    # so under-sampling them would hide exactly what this measures.
    picked = []
    rare = [s for s in by_source if s != 'header']
    for src in rare:
        picked.extend(by_source[src])
    common = by_source.get('header', [])
    rng.shuffle(common)
    picked.extend(common[: max(0, args.n_precision - len(picked))])
    rng.shuffle(picked)
    picked = picked[: args.n_precision]

    out = Path(args.outdir)
    with open(out / "allergy_precision_sample.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "note_id", "before", "SPAN", "after", "judgment"])
        for n, r in enumerate(picked, 1):
            b, sp, a = window(json_text_cache(r["note_id"], files), r["start"], r["end"])
            w.writerow([n, r["note_id"], b, sp, a, ""])

    with open(out / "allergy_sample_key.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "note_id", "span", "start", "end", "source"])
        for n, r in enumerate(picked, 1):
            w.writerow([n, r["note_id"], r["span"], r["start"], r["end"], r["source"]])

    rng.shuffle(unflagged_with_header)
    with open(out / "allergy_recall_sample.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["note_id", "n_medication_spans", "n_missed_allergens", "notes"])
        for r in unflagged_with_header[: args.n_recall]:
            w.writerow([r["note_id"], r["n_meds"], "", ""])

    print(f"\nwrote {len(picked)} precision rows, "
          f"{min(len(unflagged_with_header), args.n_recall)} recall rows")
    print("ALL THREE FILES CONTAIN PHI -- keep on Minerva, keep gitignored")
    print("\nAdjudicate allergy_precision_sample.csv WITHOUT opening the key:")
    print("  judgment = allergen | administered | unclear")


_TEXT_CACHE = {}


def json_text_cache(note_id, files):
    if note_id not in _TEXT_CACHE:
        for f in files:
            if os.path.basename(f)[:-5] == note_id:
                _TEXT_CACHE[note_id] = note_text(json.load(open(f)))
                break
        else:
            _TEXT_CACHE[note_id] = ""
    return _TEXT_CACHE[note_id]


if __name__ == "__main__":
    main()
