#!/usr/bin/env python
"""
Convert an INCEpTION project export (WebAnno TSV 3.x) into a spaCy DocBin.

Produces the GOLD standard used to measure:
  - student-vs-gold  (spacy evaluate models/ner_*/model-best gold.spacy)
  - teacher-vs-gold  (separate span-comparison script)

Reads directly from the export .zip so nothing is unpacked into the repo.
Documents with zero annotations are skipped by default -- in a partly-annotated
project those are notes that were opened but not worked on, and including them
would look like "annotator found no entities" and destroy the recall figures.

Tokenization uses en_core_sci_sm so the gold DocBin aligns with the models,
exactly as build_splits.py does for the training data.

Usage
  python inception_to_docbin.py \
      --zip annotations/gold_export/gold25.zip \
      --out annotations/gold_export/gold.spacy \
      --spans-csv annotations/gold_export/gold_spans.csv
"""

import argparse
import csv
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

VALID = {"DISEASE", "MEDICATION", "PROCEDURE"}

# A cell looks like: "_"  |  "DISEASE"  |  "DISEASE[3]"  |  "DISEASE[3]|MEDICATION[4]"
CELL = re.compile(r"^(?P<label>[^\[\]|]*?)(?:\[(?P<cid>\d+)\])?$")


def parse_webanno_tsv(text):
    """Return [(label, start_char, end_char), ...] for one document.

    Single-token spans carry a bare label; multi-token spans carry a
    disambiguation id in brackets that is shared by every token of the span,
    so those are merged by (label, id).
    """
    chained = {}
    singles = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5 or "-" not in cols[1]:
            continue
        try:
            begin, end = (int(x) for x in cols[1].split("-", 1))
        except ValueError:
            continue
        for item in cols[4].split("|"):
            item = item.strip()
            if item in ("_", "", "*"):
                continue
            m = CELL.match(item)
            if not m:
                continue
            label = m.group("label").lstrip("*").strip()
            if not label:
                continue
            cid = m.group("cid")
            if cid is None:
                singles.append((label, begin, end))
            else:
                key = (label, cid)
                if key in chained:
                    lo, hi = chained[key]
                    chained[key] = (min(lo, begin), max(hi, end))
                else:
                    chained[key] = (begin, end)
    spans = list(singles)
    spans += [(label, lo, hi) for (label, _), (lo, hi) in chained.items()]
    spans.sort(key=lambda s: (s[1], s[2]))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="INCEpTION project export .zip")
    ap.add_argument("--out", required=True, help="output DocBin path")
    ap.add_argument("--annotator", default="admin", help="which annotator's TSV to read")
    ap.add_argument("--base-model", default="en_core_sci_sm")
    ap.add_argument("--spans-csv", default=None, help="optional span dump for inspection")
    ap.add_argument("--keep-empty", action="store_true",
                    help="also include documents with zero annotations")
    args = ap.parse_args()

    import spacy
    from spacy.tokens import DocBin
    from spacy.util import filter_spans

    try:
        nlp = spacy.load(args.base_model, disable=["ner", "parser", "tagger"])
        tok_src = args.base_model
    except Exception:
        nlp = spacy.blank("en")
        tok_src = "blank:en (WARNING: will not align with scispaCy-trained models)"
    print(f"tokenizer: {tok_src}")

    zf = zipfile.ZipFile(args.zip)
    tsvs = [n for n in zf.namelist()
            if n.startswith("annotation/") and n.endswith(f"/{args.annotator}.tsv")]
    if not tsvs:
        raise SystemExit(f"no annotation/*/{args.annotator}.tsv found in the zip")
    print(f"found {len(tsvs)} annotation files\n")

    stats = Counter()
    label_counts = Counter()
    rows = []
    docs = []

    for tsv_name in sorted(tsvs):
        doc_name = tsv_name.split("/")[1]          # e.g. 109981187.txt
        note_id = doc_name.rsplit(".", 1)[0]
        src_name = f"source/{doc_name}"
        if src_name not in zf.namelist():
            print(f"  !! no source text for {doc_name}; skipped")
            stats["missing_source"] += 1
            continue

        text = zf.read(src_name).decode("utf-8")
        raw = zf.read(tsv_name).decode("utf-8")
        parsed = parse_webanno_tsv(raw)

        if not parsed and not args.keep_empty:
            stats["skipped_unannotated"] += 1
            continue

        doc = nlp.make_doc(text)
        spans = []
        for label, begin, end in parsed:
            if label not in VALID:
                stats["invalid_label"] += 1
                continue
            if begin >= end or end > len(text):
                stats["bad_offsets"] += 1
                continue
            sp = doc.char_span(begin, end, label=label, alignment_mode="expand")
            if sp is None:
                stats["alignment_failed"] += 1
                continue
            spans.append(sp)
            rows.append({"note_id": note_id, "label": label,
                         "start": begin, "end": end,
                         "text": text[begin:end].replace("\n", " ")})

        kept = filter_spans(spans)
        stats["dropped_overlapping"] += len(spans) - len(kept)
        doc.ents = kept
        doc.user_data["note_id"] = note_id
        docs.append(doc)
        label_counts.update(e.label_ for e in kept)
        stats["spans_kept"] += len(kept)
        print(f"  {note_id}: {len(parsed):>4} parsed -> {len(kept):>4} entities "
              f"({len(text):>6} chars)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    DocBin(store_user_data=True, docs=docs).to_disk(out)

    print(f"\nwrote {out}: {len(docs)} docs, {stats['spans_kept']} entities")
    print("label distribution:", dict(label_counts))
    print("stats:", dict(stats))

    if args.spans_csv:
        with open(args.spans_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["note_id", "label", "start", "end", "text"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.spans_csv} ({len(rows)} rows) -- CONTAINS PHI, keep it ignored")

    if stats.get("alignment_failed"):
        print("\nNOTE: alignment failures mean a span boundary fell inside a token "
              "that scispaCy tokenizes differently. 'expand' should prevent this; "
              "a nonzero count is worth investigating before trusting the gold set.")


if __name__ == "__main__":
    main()
