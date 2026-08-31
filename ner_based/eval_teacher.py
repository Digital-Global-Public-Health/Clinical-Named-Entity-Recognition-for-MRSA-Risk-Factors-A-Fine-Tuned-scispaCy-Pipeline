#!/usr/bin/env python
"""Score the teacher (llama) pre-annotations against the gold set.

The student can only learn what the teacher proposed. If the teacher's recall
against gold is low, no amount of silver data fixes it — the ceiling is set
upstream of training. This script measures that ceiling.

Reads the verified pre-annotation JSONs (the same files INCEpTION was fed for
the silver corpus) and scores them exactly as eval_gold.py scores a model:
exact and partial match, per label, with a pilot/extension split.

Usage:
  python eval_teacher.py --gold /tmp/gold16.spacy
"""
import argparse
import glob
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import spacy
from spacy.tokens import DocBin

PILOT = {"109981187", "110220141", "110577574",
         "110721936", "136602524", "141317604"}
EXTENSION = {"109752688", "146810273", "204449710", "206602154", "236578963",
             "244607651", "259029479", "259491978", "259765625", "71778772"}
LABELS = ["DISEASE", "MEDICATION", "PROCEDURE"]

START_KEYS = ("start", "start_char", "begin", "char_start", "offset_start")
END_KEYS = ("end", "end_char", "char_end", "offset_end")
LABEL_KEYS = ("label", "type", "entity_type", "tag", "entity")
TEXT_KEYS = ("text", "span", "surface", "mention", "string")


def pick(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def find_spans(obj, depth=0):
    """Locate the list of span dicts anywhere in the JSON."""
    if depth > 4:
        return None
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            d = obj[0]
            if pick(d, LABEL_KEYS) is not None and (
                pick(d, START_KEYS) is not None or pick(d, TEXT_KEYS) is not None
            ):
                return obj
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            got = find_spans(v, depth + 1)
            if got is not None:
                return got
    return None


def note_text(obj):
    """Longest string value in the JSON is the note text."""
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


def load_teacher(note_id, roots):
    for root in roots:
        hits = glob.glob(f"{root}/{note_id}.json")
        if hits:
            obj = json.load(open(hits[0]))
            spans = find_spans(obj)
            if spans is None:
                return None, None, hits[0]
            return spans, note_text(obj), hits[0]
    return None, None, None


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] == b[2]


def score(pairs, key=lambda n: True):
    exact = defaultdict(lambda: [0, 0, 0])
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
            partial[lab][0] += len(used_p)
            partial[lab][1] += len(pl) - len(used_p)
            partial[lab][2] += len(gl) - len(used_g)
    return exact, partial


def report(name, exact, partial, n):
    print(f"\n=== {name}  ({n} notes) ===")
    print(f"{'label':<12} {'gold':>6} {'pred':>6} "
          f"{'P':>7} {'R':>7} {'F1':>7}   {'pP':>7} {'pR':>7} {'pF1':>7}")
    tot, ptot = [0, 0, 0], [0, 0, 0]
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
    ap.add_argument("--roots", nargs="+",
                    default=["annotations/batch01/verified",
                             "annotations/batch02/verified",
                             "annotations/batch03/verified",
                             "annotations/batch04/verified"])
    args = ap.parse_args()

    blank = spacy.blank("en")
    gold_docs = list(DocBin().from_disk(args.gold).get_docs(blank.vocab))
    names = sorted(PILOT | EXTENSION)
    if len(names) != len(gold_docs):
        print(f"!! {len(names)} ids vs {len(gold_docs)} docs")
        return

    pairs, missing, schema_shown = [], [], False
    for note_id, gd in zip(names, gold_docs):
        spans, ttext, path = load_teacher(note_id, args.roots)
        if spans is None:
            missing.append(note_id)
            continue
        if not schema_shown:
            print(f"detected schema from {path}:")
            print("  ", json.dumps(spans[0], ensure_ascii=False)[:300])
            schema_shown = True

        pred = set()
        for s in spans:
            lab = pick(s, LABEL_KEYS)
            if lab not in LABELS:
                continue
            b, e = pick(s, START_KEYS), pick(s, END_KEYS)
            if b is None or e is None:
                # offsets absent: locate the verbatim span in the gold text
                t = pick(s, TEXT_KEYS)
                if not t:
                    continue
                i = gd.text.find(t)
                if i < 0:
                    continue
                b, e = i, i + len(t)
            pred.add((int(b), int(e), lab))

        # offsets may be relative to the teacher's own copy of the text
        if ttext and ttext != gd.text and pred:
            ok = sum(1 for b, e, _ in pred if gd.text[b:e].strip())
            if ok < len(pred) * 0.5:
                print(f"!! {note_id}: teacher offsets do not align with gold "
                      "text — check the source of the JSON")

        gold = {(e.start_char, e.end_char, e.label_) for e in gd.ents}
        pairs.append((note_id, gold, pred))

    if missing:
        print(f"!! no teacher JSON for: {', '.join(missing)}")
    if not pairs:
        print("no notes scored — check --roots")
        return

    n_pred = sum(len(p) for _, _, p in pairs)
    n_gold = sum(len(g) for _, g, _ in pairs)
    print(f"\nteacher spans: {n_pred}   gold spans: {n_gold}   "
          f"density ratio gold/teacher: {n_gold/max(n_pred,1):.2f}")
    print("teacher labels:",
          dict(Counter(l for _, _, p in pairs for _, _, l in p)))

    e, p = score(pairs)
    report("teacher (llama) vs gold", e, p, len(pairs))
    e, p = score(pairs, key=lambda n: n in PILOT)
    report("teacher  [pilot]", e, p, sum(1 for n, _, _ in pairs if n in PILOT))
    e, p = score(pairs, key=lambda n: n not in PILOT)
    report("teacher  [extension]", e, p,
           sum(1 for n, _, _ in pairs if n not in PILOT))


if __name__ == "__main__":
    main()
