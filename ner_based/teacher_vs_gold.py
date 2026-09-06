#!/usr/bin/env python
"""
Teacher-vs-gold evaluation: how good is the silver standard?

Compares llama3.1's verified pre-annotations against the hand-annotated gold
DocBin, on exactly the notes present in gold. This is the number that makes the
student's score interpretable:

  student 40 recall vs teacher 45 recall  -> student recovers ~90% of an
                                             imperfect teacher; the ceiling is
                                             the teacher / annotation standard
  student 40 recall vs teacher 75 recall  -> distillation is losing information;
                                             a different problem entirely

Fairness note: teacher spans are put through the SAME token expansion and
overlap filtering that the gold DocBin went through (char_span with
alignment_mode="expand", then filter_spans). Otherwise the teacher would be
penalised for tokenization differences the gold set had smoothed away.

Usage
  python teacher_vs_gold.py \
      --gold annotations/gold_export/gold.spacy \
      --ann-dirs annotations/batch01_v2 annotations/batch02 annotations/batch03 \
      --errors-csv annotations/gold_export/teacher_errors.csv
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

VALID = {"DISEASE", "MEDICATION", "PROCEDURE"}
TEXT_KEYS = ("text", "note_text", "source_text", "NOTE_TEXT")
SPAN_KEYS = ("spans", "entities", "verified_spans", "verified")


def pick(d, keys):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--ann-dirs", nargs="+", required=True)
    ap.add_argument("--base-model", default="en_core_sci_sm")
    ap.add_argument("--errors-csv", default=None)
    ap.add_argument("--max-examples", type=int, default=25,
                    help="disagreement examples to print")
    args = ap.parse_args()

    import spacy
    from spacy.tokens import DocBin
    from spacy.util import filter_spans

    try:
        nlp = spacy.load(args.base_model, disable=["ner", "parser", "tagger"])
    except Exception:
        nlp = spacy.blank("en")
        print("WARNING: falling back to blank:en tokenizer")

    gold_docs = list(DocBin(store_user_data=True)
                     .from_disk(args.gold).get_docs(nlp.vocab))
    print(f"gold: {len(gold_docs)} docs, "
          f"{sum(len(d.ents) for d in gold_docs)} entities\n")

    # locate the teacher's verified JSON for each gold note
    index = {}
    for ad in args.ann_dirs:
        for f in Path(ad).glob("verified/*.json"):
            index.setdefault(f.stem, f)

    tp = Counter(); fp = Counter(); fn = Counter()
    tot = Counter()
    rows = []
    examples = []

    for doc in gold_docs:
        note_id = str(doc.user_data.get("note_id", ""))
        gold_set = {(e.start_char, e.end_char, e.label_) for e in doc.ents}

        src = index.get(note_id)
        if src is None:
            print(f"  !! no teacher annotation found for {note_id}")
            for s, e, lab in gold_set:
                fn[lab] += 1
            continue

        d = json.load(open(src))
        raw = pick(d, SPAN_KEYS) or []
        spans = []
        for s in raw:
            lab = s.get("label")
            if lab not in VALID:
                continue
            sp = doc.char_span(s["start_char"], s["end_char"], label=lab,
                               alignment_mode="expand")
            if sp is not None:
                spans.append(sp)
        kept = filter_spans(spans)
        teach_set = {(sp.start_char, sp.end_char, sp.label_) for sp in kept}

        tot["gold"] += len(gold_set)
        tot["teacher"] += len(teach_set)

        for key in gold_set & teach_set:
            tp[key[2]] += 1
        for key in teach_set - gold_set:
            fp[key[2]] += 1
        for key in gold_set - teach_set:
            fn[key[2]] += 1

        # classify the disagreements for error analysis
        gold_by_offset = {(s, e): lab for s, e, lab in gold_set}
        teach_by_offset = {(s, e): lab for s, e, lab in teach_set}
        for (s, e), lab in teach_by_offset.items():
            if (s, e, lab) in gold_set:
                continue
            if (s, e) in gold_by_offset:
                kind = f"label: gold={gold_by_offset[(s, e)]}"
            elif any(not (e <= gs or s >= ge) for gs, ge in gold_by_offset):
                kind = "boundary"
            else:
                kind = "spurious"
            rows.append({"note_id": note_id, "side": "teacher_only", "kind": kind,
                         "label": lab, "start": s, "end": e,
                         "text": doc.text[s:e].replace("\n", " ")})
        for (s, e), lab in gold_by_offset.items():
            if (s, e, lab) in teach_set:
                continue
            if (s, e) in teach_by_offset:
                continue  # already recorded as a label disagreement
            overlaps = any(not (e <= ts or s >= te) for ts, te in teach_by_offset)
            kind = "boundary" if overlaps else "missed"
            rows.append({"note_id": note_id, "side": "gold_only", "kind": kind,
                         "label": lab, "start": s, "end": e,
                         "text": doc.text[s:e].replace("\n", " ")})

    print(f"teacher spans on these notes: {tot['teacher']}")
    print(f"gold spans on these notes   : {tot['gold']}\n")

    print("== teacher vs gold ==")
    print(f"{'label':<12} {'P':>7} {'R':>7} {'F':>7}   {'tp':>5} {'fp':>5} {'fn':>5}")
    for lab in sorted(VALID):
        p, r, f = prf(tp[lab], fp[lab], fn[lab])
        print(f"{lab:<12} {p:7.2f} {r:7.2f} {f:7.2f}   "
              f"{tp[lab]:5d} {fp[lab]:5d} {fn[lab]:5d}")
    P, R, F = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    print(f"{'OVERALL':<12} {P:7.2f} {R:7.2f} {F:7.2f}   "
          f"{sum(tp.values()):5d} {sum(fp.values()):5d} {sum(fn.values()):5d}")

    kinds = Counter((r["side"], r["kind"]) for r in rows)
    print("\n== disagreement taxonomy ==")
    for (side, kind), n in kinds.most_common():
        print(f"  {side:<13} {kind:<22} {n}")

    print(f"\n== up to {args.max_examples} disagreement examples ==")
    for r in rows[:args.max_examples]:
        print(f"  [{r['side']:<12} {r['kind']:<20}] {r['label']:<10} "
              f"{r['text'][:60]!r}")

    if args.errors_csv:
        with open(args.errors_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["note_id", "side", "kind", "label",
                                               "start", "end", "text"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.errors_csv} ({len(rows)} rows) -- CONTAINS PHI")


if __name__ == "__main__":
    main()
