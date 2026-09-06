#!/usr/bin/env python
"""Score the assertion layer against gold attributes.

Gold spans are injected directly as doc.ents, bypassing NER, so this measures
the assertion components (ConText + terminators + allergy) in isolation rather
than confounding them with NER recall.

Axis -> flag mapping:
    polarity=negated       <- ent._.is_negated
    certainty=uncertain    <- ent._.is_uncertain
    certainty=hypothetical <- ent._.is_hypothetical
    temporality=historical <- ent._.is_historical
    experiencer=other      <- ent._.is_family
    allergy=yes            <- ent._.is_allergy

RUN ON A MINERVA COMPUTE NODE (reads note text: PHI).

Usage:
    python score_assertions.py --export /tmp/ga --model models/ner_full/model-best
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LABEL = re.compile(r"^(DISEASE|MEDICATION|PROCEDURE)(\[\d+\])?$")
STRIP = re.compile(r"\[\d+\]$")

# axis -> (non-default value, extension attribute)
AXES = [
    ("polarity", "negated", "is_negated"),
    ("certainty", "uncertain", "is_uncertain"),
    ("certainty", "hypothetical", "is_hypothetical"),
    ("temporality", "historical", "is_historical"),
    ("experiencer", "other", "is_family"),
    ("allergy", "yes", "is_allergy"),
]

PIPES = ["medspacy_pyrush", "medspacy_sectionizer", "medspacy_context",
         "airms_allergy"]


def read_gold(export_dir):
    """-> {note_id: [(start, end, label, {axis: value}), ...]}"""
    out = {}
    for d in sorted(glob.glob(os.path.join(export_dir, "annotation", "*/"))):
        note = re.sub(r"\.(txt|tsv)$", "", os.path.basename(d.rstrip("/")))
        f = os.path.join(d, "admin.tsv")
        if not os.path.exists(f):
            f = os.path.join(d, "INITIAL_CAS.tsv")
        lines = open(f).read().split("\n")
        hdr = next(l for l in lines if l.startswith("#T_SP="))
        feats = hdr.split("=", 1)[1].split("|")[1:]
        idx = {n: 3 + i for i, n in enumerate(feats)}
        spans, order = {}, []
        for ln in lines:
            if not ln.strip() or ln.startswith("#"):
                continue
            cols = ln.split("\t")
            if len(cols) <= max(idx.values()):
                continue
            v = cols[idx["value"]]
            m = LABEL.match(v)
            if not m:
                continue
            beg, end = cols[1].split("-")
            key = v if m.group(2) else f"{beg}-{end}"
            attrs = {a: STRIP.sub("", cols[idx[a]])
                     for a in idx if a != "value"}
            if key in spans:
                s = spans[key]
                spans[key] = (s[0], int(end), s[2], s[3])
            else:
                spans[key] = (int(beg), int(end), m.group(1), attrs)
                order.append(key)
        out[note] = [spans[k] for k in order]
    return out


def read_text(export_dir, note, text_dir):
    for cand in (os.path.join(text_dir, note + ".txt"),
                 os.path.join(text_dir, note)):
        if os.path.exists(cand):
            return open(cand).read()
    src = os.path.join(export_dir, "source", note + ".txt")
    if os.path.exists(src):
        return open(src).read()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="unzipped export dir")
    ap.add_argument("--model", default="models/ner_full/model-best")
    ap.add_argument("--text-dir", default="annotations/gold_text")
    ap.add_argument("--errors-csv", default=None)
    args = ap.parse_args()

    from spacy.util import filter_spans
    from src.ner.assertion import build_assertion_pipeline

    nlp = build_assertion_pipeline(args.model)
    gold = read_gold(args.export)

    stats = {}          # (axis, value) -> [tp, fp, fn]
    for a, v, _ in AXES:
        stats.setdefault((a, v), [0, 0, 0])
    matched = skipped = 0
    rows = []

    for note, spans in sorted(gold.items()):
        text = read_text(args.export, note, args.text_dir)
        if text is None:
            print(f"  ! no text for {note}, skipped", file=sys.stderr)
            continue
        doc = nlp.make_doc(text)
        ents, keep = [], []
        for beg, end, label, attrs in spans:
            sp = doc.char_span(beg, end, label=label, alignment_mode="expand")
            if sp is None:
                skipped += 1
                continue
            ents.append(sp)
            keep.append((sp, attrs))
        ents = filter_spans(ents)
        doc.ents = ents
        kept = {(s.start_char, s.end_char): at for s, at in keep
                if s in ents}
        for name in PIPES:
            if name in nlp.pipe_names:
                doc = nlp.get_pipe(name)(doc)

        for ent in doc.ents:
            attrs = kept.get((ent.start_char, ent.end_char))
            if attrs is None:
                continue
            matched += 1
            for axis, value, flag in AXES:
                g = attrs.get(axis) == value
                p = bool(getattr(ent._, flag, False))
                s = stats[(axis, value)]
                if g and p:
                    s[0] += 1
                elif p and not g:
                    s[1] += 1
                elif g and not p:
                    s[2] += 1
                if g != p and args.errors_csv:
                    rows.append((note, ent.text, ent.label_, axis, value,
                                 "gold_only" if g else "pred_only"))

    print(f"\nspans matched: {matched}   (offset failures: {skipped})\n")
    print(f"{'axis':<13} {'value':<13} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'P':>7} {'R':>7} {'F1':>7}")
    for axis, value, _ in AXES:
        tp, fp, fn = stats[(axis, value)]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        n = tp + fn
        flag = "   <- too few gold" if n < 5 else ""
        print(f"{axis:<13} {value:<13} {tp:>4} {fp:>4} {fn:>4} "
              f"{p*100:>6.1f}% {r*100:>6.1f}% {f*100:>6.1f}%{flag}")

    if args.errors_csv and rows:
        import csv
        with open(args.errors_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["note_id", "text", "label", "axis", "value", "side"])
            w.writerows(rows)
        print(f"\nwrote {len(rows)} disagreements to {args.errors_csv}"
              "  -- CONTAINS PHI, keep it ignored")


if __name__ == "__main__":
    main()
