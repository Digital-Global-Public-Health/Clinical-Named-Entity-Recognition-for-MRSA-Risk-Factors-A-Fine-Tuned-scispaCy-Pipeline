#!/usr/bin/env python
"""Score NER models against the gold DocBin.

Reports exact-match precision/recall/F1 per label, overall, and split by
pilot notes (annotated with INCEpTION recommenders active) vs extension
notes (no machine assistance) — the two halves are not methodologically
identical and a large gap between them is a confound, not a finding.

Matching is on (start_char, end_char, label). Partial-overlap scores are
reported alongside, since boundary disagreement and missed entities are
different failure modes and PROCEDURE recall is the question of interest.

Usage:
  python eval_gold.py --gold annotations/gold_export/gold.spacy --models models/ner_*
"""
import argparse
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import spacy
from spacy.tokens import DocBin

# registers airms.scispacy_tokenizer.v1, referenced by the trained configs
sys.path.insert(0, str(Path(__file__).parent / "configs"))
import custom_code  # noqa: F401

PILOT = {"109981187", "110220141", "110577574",
         "110721936", "136602524", "141317604"}
LABELS = ["DISEASE", "MEDICATION", "PROCEDURE"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] == b[2]


def score(pairs, key=lambda n: True):
    """pairs: list of (note_id, gold_set, pred_set) of (start, end, label)."""
    exact = defaultdict(lambda: [0, 0, 0])   # label -> [tp, fp, fn]
    partial = defaultdict(lambda: [0, 0, 0])

    for note_id, gold, pred in pairs:
        if not key(note_id):
            continue
        for lab in LABELS:
            g = {s for s in gold if s[2] == lab}
            p = {s for s in pred if s[2] == lab}
            tp = len(g & p)
            exact[lab][0] += tp
            exact[lab][1] += len(p) - tp
            exact[lab][2] += len(g) - tp

            # partial: greedy one-to-one on overlap
            # sorted so greedy assignment is deterministic: longest
            # span first, then by start offset
            gl = sorted(g, key=lambda x: (-(x[1]-x[0]), x[0]))
            pl = sorted(p, key=lambda x: (-(x[1]-x[0]), x[0]))
            used_g, used_p = set(), set()
            for i, ps in enumerate(pl):
                for j, gs in enumerate(gl):
                    if j in used_g:
                        continue
                    if overlaps(ps, gs):
                        used_g.add(j)
                        used_p.add(i)
                        break
            ptp = len(used_p)
            partial[lab][0] += ptp
            partial[lab][1] += len(pl) - ptp
            partial[lab][2] += len(gl) - len(used_g)
    return exact, partial


def report(name, exact, partial, n_notes):
    print(f"\n=== {name}  ({n_notes} notes) ===")
    print(f"{'label':<12} {'gold':>6} {'pred':>6} "
          f"{'P':>7} {'R':>7} {'F1':>7}   {'pP':>7} {'pR':>7} {'pF1':>7}")
    tot = [0, 0, 0]
    ptot = [0, 0, 0]
    for lab in LABELS:
        tp, fp, fn = exact[lab]
        p, r, f = prf(tp, fp, fn)
        pp, pr_, pf = prf(*partial[lab])
        print(f"{lab:<12} {tp+fn:>6} {tp+fp:>6} "
              f"{p:>7.3f} {r:>7.3f} {f:>7.3f}   {pp:>7.3f} {pr_:>7.3f} {pf:>7.3f}")
        for i in range(3):
            tot[i] += exact[lab][i]
            ptot[i] += partial[lab][i]
    p, r, f = prf(*tot)
    pp, pr_, pf = prf(*ptot)
    print(f"{'OVERALL':<12} {tot[0]+tot[2]:>6} {tot[0]+tot[1]:>6} "
          f"{p:>7.3f} {r:>7.3f} {f:>7.3f}   {pp:>7.3f} {pr_:>7.3f} {pf:>7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", action="store_true",
                    help="also report pilot vs extension separately")
    args = ap.parse_args()

    blank = spacy.blank("en")
    gold_docs = list(DocBin().from_disk(args.gold).get_docs(blank.vocab))
    print(f"gold: {len(gold_docs)} docs, "
          f"{sum(len(d.ents) for d in gold_docs)} entities")
    print("      ", dict(Counter(e.label_ for d in gold_docs for e in d.ents)))

    # note_id is not stored in the DocBin; recover by matching char length
    # against the gold spans CSV is fragile, so use document order instead —
    # inception_to_docbin writes docs in sorted filename order.
    names = sorted(PILOT | {
        "109752688", "146810273", "204449710", "206602154", "236578963",
        "244607651", "259029479", "259491978", "259765625", "71778772"})
    if len(names) != len(gold_docs):
        print(f"!! {len(names)} known ids but {len(gold_docs)} docs — "
              "check the id list")
        return

    for mpath in args.models:
        mdir = Path(mpath)
        if (mdir / "model-best").exists():
            mdir = mdir / "model-best"
        if not (mdir / "config.cfg").exists():
            print(f"\n!! {mpath}: no config.cfg (looked in {mdir}) — skipped")
            continue
        nlp = spacy.load(mdir)

        pairs = []
        for name, gd in zip(names, gold_docs):
            pd_ = nlp(gd.text)
            gold = {(e.start_char, e.end_char, e.label_) for e in gd.ents}
            pred = {(e.start_char, e.end_char, e.label_) for e in pd_.ents}
            pairs.append((name, gold, pred))

        e, p = score(pairs)
        report(mpath, e, p, len(pairs))

        if args.split:
            e, p = score(pairs, key=lambda n: n in PILOT)
            report(f"{mpath}  [pilot, recommenders active]", e, p, 6)
            e, p = score(pairs, key=lambda n: n not in PILOT)
            report(f"{mpath}  [extension, no assistance]", e, p, 10)


if __name__ == "__main__":
    main()
