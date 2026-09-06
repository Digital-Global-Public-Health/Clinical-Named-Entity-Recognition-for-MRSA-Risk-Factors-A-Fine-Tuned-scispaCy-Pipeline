#!/usr/bin/env python
"""
Diagnose the teacher's 321 missed entities.

Question: is the recall gap because the human annotated the SAME entities more
often (occurrence density), or because the teacher never tagged certain entity
STRINGS at all (systematic failure)? These have completely different fixes.

  density   -> a counting artifact of guideline §3.9 (tag every occurrence);
               the teacher is better than the raw recall figure suggests
  systematic-> the prompt genuinely fails on those entities; the fix is a better
               prompt and a re-run, which is the single biggest lever available

Splits every gold-only miss into:
  A. FIRST-OCCURRENCE MISS  - the teacher tagged this string nowhere in the note
                              => genuine failure
  B. REPEAT-OCCURRENCE MISS - the teacher tagged this string elsewhere in the
                              note, just fewer times => density artifact

Then recomputes recall counting each distinct string once per note ("type-level
recall"), which removes the density effect entirely.

Usage
  python analyse_teacher_misses.py \
      --gold annotations/gold_export/gold.spacy \
      --errors annotations/gold_export/teacher_errors_v3.csv \
      --ann-dirs annotations/batch01_v2 annotations/batch02 annotations/batch03
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

VALID = {"DISEASE", "MEDICATION", "PROCEDURE"}
SPAN_KEYS = ("spans", "entities", "verified_spans", "verified")


def pick(d, keys):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None


def norm(s):
    return " ".join(s.lower().split())


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--errors", required=True, help="teacher_errors_v3.csv")
    ap.add_argument("--ann-dirs", nargs="+", required=True)
    ap.add_argument("--base-model", default="en_core_sci_sm")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    import pandas as pd
    import spacy
    from spacy.tokens import DocBin

    try:
        nlp = spacy.load(args.base_model, disable=["ner", "parser", "tagger"])
    except Exception:
        nlp = spacy.blank("en")

    gold_docs = list(DocBin(store_user_data=True)
                     .from_disk(args.gold).get_docs(nlp.vocab))

    # ---- teacher's raw span strings per note (before any offset matching) ----
    index = {}
    for ad in args.ann_dirs:
        for f in Path(ad).glob("verified/*.json"):
            index.setdefault(f.stem, f)

    teacher_strings = {}      # note_id -> Counter of (norm_text, label)
    for doc in gold_docs:
        nid = str(doc.user_data.get("note_id", ""))
        c = Counter()
        src = index.get(nid)
        if src:
            for s in (pick(json.load(open(src)), SPAN_KEYS) or []):
                if s.get("label") in VALID:
                    c[(norm(s["text"]), s["label"])] += 1
        teacher_strings[nid] = c

    gold_strings = {}
    for doc in gold_docs:
        nid = str(doc.user_data.get("note_id", ""))
        gold_strings[nid] = Counter((norm(e.text), e.label_) for e in doc.ents)

    # ---- A vs B on the actual misses ---------------------------------------
    err = pd.read_csv(args.errors, dtype={"note_id": str})
    missed = err[(err.side == "gold_only") & (err.kind == "missed")]

    first_miss, repeat_miss = [], []
    for r in missed.itertuples():
        key = (norm(str(r.text)), r.label)
        if teacher_strings.get(str(r.note_id), Counter()).get(key, 0) > 0:
            repeat_miss.append(r)
        else:
            first_miss.append(r)

    n = len(missed)
    print("=" * 68)
    print(f"MISSED gold spans: {n}")
    print(f"  A. teacher never tagged this string in the note : "
          f"{len(first_miss):>4}  ({100*len(first_miss)/max(n,1):.1f}%)  <- genuine failure")
    print(f"  B. teacher tagged it elsewhere, fewer times      : "
          f"{len(repeat_miss):>4}  ({100*len(repeat_miss)/max(n,1):.1f}%)  <- density artifact")
    print("=" * 68)

    print("\nBy label (genuine failures only):")
    for lab, c in Counter(r.label for r in first_miss).most_common():
        print(f"  {lab:<12} {c}")

    print(f"\nTop {args.top} entity strings the teacher NEVER tagged:")
    for (txt, lab), c in Counter(
            (str(r.text).strip(), r.label) for r in first_miss).most_common(args.top):
        print(f"  {c:>3}x  {lab:<11} {txt[:55]!r}")

    # ---- type-level recall (each distinct string counted once per note) -----
    tp = Counter(); fp = Counter(); fn = Counter()
    for nid, gset in gold_strings.items():
        tset = teacher_strings.get(nid, Counter())
        g, t = set(gset), set(tset)
        for k in g & t:
            tp[k[1]] += 1
        for k in t - g:
            fp[k[1]] += 1
        for k in g - t:
            fn[k[1]] += 1

    print("\n" + "=" * 68)
    print("TYPE-LEVEL teacher vs gold  (each distinct string once per note,")
    print("string match not offset match -- density effect removed)")
    print("=" * 68)
    print(f"{'label':<12} {'P':>7} {'R':>7} {'F':>7}   {'tp':>5} {'fp':>5} {'fn':>5}")
    for lab in sorted(VALID):
        p, r, f = prf(tp[lab], fp[lab], fn[lab])
        print(f"{lab:<12} {p:7.2f} {r:7.2f} {f:7.2f}   "
              f"{tp[lab]:5d} {fp[lab]:5d} {fn[lab]:5d}")
    P, R, F = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    print(f"{'OVERALL':<12} {P:7.2f} {R:7.2f} {F:7.2f}   "
          f"{sum(tp.values()):5d} {sum(fp.values()):5d} {sum(fn.values()):5d}")

    print("\nCompare with the span-level figure (P 75.94 / R 35.25 / F 48.15).")
    print("If type-level recall is barely higher, density is NOT the explanation")
    print("and the prompt has a real coverage problem.")

    # ---- occurrence ratio on entities BOTH sides found ----------------------
    ratios = []
    for nid, gset in gold_strings.items():
        tset = teacher_strings.get(nid, Counter())
        for k in set(gset) & set(tset):
            ratios.append((gset[k], tset[k]))
    if ratios:
        gtot = sum(a for a, _ in ratios)
        ttot = sum(b for _, b in ratios)
        multi = sum(1 for a, b in ratios if a > b)
        print(f"\nOn strings BOTH sides tagged ({len(ratios)} distinct):")
        print(f"  gold occurrences {gtot} vs teacher {ttot} "
              f"(ratio {gtot/max(ttot,1):.2f}x)")
        print(f"  human tagged more occurrences in {multi} of {len(ratios)} cases")


if __name__ == "__main__":
    main()
